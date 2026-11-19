"""Paired validation scoring and unity-gain blind listening exports."""
import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.compare_experiments import export_audio_bundle
from src.evaluation.metrics import compute_framewise_metrics, compute_leakage_diagnostics
from src.inference.separate_track import load_separation_model, separate_waveform

from .data import BASELINE, SAMPLE_RATE, STEMS, bind_run, read_segment, reserve_run
from .inference import separate
from .model import SubbandSeparator


def score_song(track, references, estimates, mixture):
    vocals, instrumental = [compute_framewise_metrics(ref, est)
                            for ref, est in zip(references, estimates, strict=True)]
    bleed = compute_leakage_diagnostics(*references, *estimates)["vocal_to_inst_bleed_coeff"]
    ref_side = .5 * (references[:, 0] - references[:, 1])
    est_side = .5 * (estimates[:, 0] - estimates[:, 1])
    ref_rms = np.sqrt(np.mean(ref_side.astype(np.float64)**2, axis=1))
    est_rms = np.sqrt(np.mean(est_side.astype(np.float64)**2, axis=1))
    side_error = np.abs(20 * np.log10(np.maximum(est_rms, 1e-12) / np.maximum(ref_rms, 1e-12)))
    side_error[ref_rms < 10**(-50 / 20)] = np.nan
    return {"track": track, "vocal_sdr": vocals["median_sdr_db"],
            "instrumental_sdr": instrumental["median_sdr_db"],
            "vocal_si_sdr": vocals["median_si_sdr_db"], "bleed": bleed,
            "consistency": float(np.abs(estimates.sum(0) - mixture).max()),
            "consistency_limit": 2e-5 * max(1., float(np.abs(mixture).max())),
            "side_error": side_error.tolist(),
            "clipped": np.mean(np.abs(estimates) > 1, axis=(1, 2)).tolist()}


def write_blind_bundle(output, identifier, mixture, references, baseline, candidate, swap):
    a, b = (candidate, baseline) if swap else (baseline, candidate)
    directory = reserve_run(Path(output) / identifier)
    arrays = {"mixture": mixture.T}
    for label, pair in (("reference", references), ("A", a), ("B", b)):
        for stem, audio in zip(STEMS[1:], pair, strict=True):
            arrays[f"{label}_{stem}"] = audio.T
    return export_audio_bundle(directory, arrays, SAMPLE_RATE)


def _aligned(baseline, candidate):
    a = {row["track"]: row for row in baseline}
    b = {row["track"]: row for row in candidate}
    if (len(a) != len(baseline) or len(b) != len(candidate) or set(a) != set(b) or len(a) != 15):
        raise ValueError("Expected the same 15 distinct validation tracks")
    keys = sorted(a)
    return [a[key] for key in keys], [b[key] for key in keys]


def _values(rows, key):
    return np.array([row[key] for row in rows], dtype=np.float64)


def _median(values):
    values = np.asarray(values)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


def grade(baseline, candidate, listening=None):
    baseline, candidate = _aligned(baseline, candidate)
    keys = ("vocal_sdr", "instrumental_sdr", "vocal_si_sdr", "bleed")
    medians = {key: _median(_values(candidate, key)) for key in keys}
    deltas = {key: _values(candidate, key) - _values(baseline, key) for key in keys}
    rng = np.random.default_rng(42)
    intervals = {}
    for key, values in deltas.items():
        finite = values[np.isfinite(values)]
        samples = rng.choice(finite, size=(10000, len(finite))) if len(finite) else np.empty((0, 0))
        intervals[key] = (np.quantile(np.median(samples, axis=1), [.025, .975]).tolist()
                          if len(finite) else [float("nan")] * 2)
    clipped_a, clipped_b = _values(baseline, "clipped"), _values(candidate, "clipped")
    side_a, side_b = _values(baseline, "side_error"), _values(candidate, "side_error")
    side_ok = all(np.array_equal(np.isfinite(a), np.isfinite(b))
                  and (not np.isfinite(a).any() or _median(b) <= _median(a) + .25)
                  for a, b in zip(side_a.T, side_b.T))
    wins = int((deltas["vocal_sdr"] > 0).sum())
    large_losses = int(((deltas["vocal_sdr"] < -1) | (deltas["instrumental_sdr"] < -1)).sum())
    gates = {
        "vocal_sdr": medians["vocal_sdr"] >= 7.2415,
        "instrumental_sdr": medians["instrumental_sdr"] >= 13.0874,
        "vocal_si_sdr": medians["vocal_si_sdr"] >= 6.173,
        "bleed": (medians["bleed"] <= .1839 and _median(np.abs(_values(candidate, "bleed")))
                  < _median(np.abs(_values(baseline, "bleed")))),
        "track_reliability": wins >= 10 and intervals["vocal_sdr"][0] > 0 and large_losses <= 2,
        "mixture_consistency": all(row["consistency"] <= row.get("consistency_limit", 2e-5)
                                   for row in candidate),
        "stereo": side_ok,
        "clipping": bool((np.median(clipped_b, axis=0) <= np.median(clipped_a, axis=0) + .0001).all()
                         and not ((clipped_a == 0) & (clipped_b > .0001)).any()),
    }
    if listening is not None:
        gates["listening"] = (listening["clearer"] >= 8 and listening["lower_bleed"] >= 8
                              and listening["instrumentals_equal_or_better"] >= 10
                              and not listening["material_artifact"])
    decision = ("NO-GO" if not all(gates.values()) else
                "PENDING_LISTENING" if listening is None else "GO")
    return {"decision": decision, "gates": gates, "medians": medians, "vocal_wins": wins,
            "large_losses": large_losses, "bootstrap_seed": 42, "paired_95_ci": intervals,
            "deltas": [{key: float(deltas[key][i]) for key in keys} | {"track": row["track"]}
                       for i, row in enumerate(baseline)]}


def evaluate_selected(store, manifest_path, output, device):
    """Evaluate the committed best checkpoint on the original validation songs."""
    _, validation, manifest = bind_run(manifest_path, store.fingerprint, store.fingerprint["config"])
    listening_tracks = {row["track"]: row for row in manifest["listening"]}
    if len(listening_tracks) != 12 or not set(listening_tracks) <= set(manifest["validation"]):
        raise ValueError("Expected 12 prespecified listening tracks")
    output = reserve_run(output)
    state = store.load("best")
    model = SubbandSeparator().to(device)
    model.load_state_dict(state["model"])
    baseline, _, _ = load_separation_model(BASELINE, device)
    rows_base, rows_native, rows_matched, blind_key = [], [], [], []
    rng = np.random.default_rng(42)
    for track in validation:
        mix, vocal, instrumental = (read_segment(track / f"{stem}.wav", 0, -1) for stem in STEMS)
        references = np.stack((vocal, instrumental))
        pred = separate_waveform(baseline, mix, device=device, segment_length_seconds=6., overlap_ratio=.5)
        old = np.stack((pred["vocals"], pred["instrumental"]))
        native = separate(model, mix, device)
        matched = separate(model, mix, device, chunk_samples=6 * SAMPLE_RATE)
        for rows, pair in ((rows_base, old), (rows_native, native), (rows_matched, matched)):
            rows.append(score_song(track.name, references, pair, mix))
        if track.name in listening_tracks:
            row = listening_tracks[track.name]
            start = row["start_sample"]
            take = slice(start, start + row["samples"])
            identifier = f"clip{len(blind_key) + 1:02d}"
            swapped = bool(rng.integers(2))
            write_blind_bundle(output / "listen", identifier, mix[..., take], references[..., take],
                               old[..., take], native[..., take], swap=swapped)
            blind_key.append({"clip": identifier, **row, "A": "Exp08" if swapped else "Exp04"})
    result = {"selected_epoch": state["epoch"], "baseline": rows_base, "native": rows_native,
              "matched_six_seconds": rows_matched, "grade": grade(rows_base, rows_native),
              "matched_sensitivity": grade(rows_base, rows_matched)}
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(_json_finite(result), handle, indent=2, allow_nan=False)
    with (output / "blind_key.json").open("x", encoding="utf-8") as handle:
        json.dump(blind_key, handle, indent=2)
    return result


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
