"""Read-only comparison on the frozen Exp04/Exp08 development split."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

from config import PROCESSED_DIR
from src.evaluation.compare_experiments import export_audio_bundle
from src.inference.separate_track import load_separation_model, separate_waveform
from src.experiments.exp08_subband.checkpointing import Checkpoints
from src.experiments.exp08_subband.data import BASELINE, STEMS, bind_run, read_segment, sha256
from src.experiments.exp08_subband.evaluate import _json_finite, score_song
from src.experiments.exp08_subband.inference import separate
from src.experiments.exp08_subband.model import SubbandSeparator
from src.experiments.exp08_subband.train import configure_determinism

RUN = "full_20261111_seed42"
PRIOR = ROOT / "reports/exp08_subband" / RUN
MODELS = ROOT / "models/experiments/exp08_subband" / RUN
NAMES = ("exp04", "exp08", "hybrid")
BANDS = (0, 80, 250, 1000, 4000, 8000, 22050)
RATE = 44100


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_finite(payload), handle, indent=2, allow_nan=False)


def protected():
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for scope in ("models", "reports")
            for p in sorted((ROOT / scope).rglob("*"))
            if p.is_file() and HERE not in p.parents}


def median(values):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(valid)) if valid else None


def blend(mixture, exp04, exp08):
    if exp04.shape != exp08.shape or exp04.shape != (2, *mixture.shape):
        raise ValueError("Expected aligned two-source stereo estimates")
    vocal = .5 * exp04[0] + .5 * exp08[0]
    return np.stack((vocal, mixture - vocal))


def gram_band(waves, device):
    """STFT Gram matrices, from aligned full tracks with bounded memory."""
    size, hop, block = 4096, 2048, 128
    length = waves[0].shape[-1]
    bins = np.fft.rfftfreq(size, 1 / RATE)
    indices = np.searchsorted(bins, BANDS)
    indices[-1] = len(bins)
    matrices = np.zeros((len(BANDS) - 1, len(waves), len(waves)), dtype=np.float64)
    window = torch.hann_window(size, device=device)
    weight = torch.full((size // 2 + 1,), 2**.5, device=device)
    weight[[0, -1]] = 1.
    frames = length // hop + 1
    for first in range(0, frames, block):
        n = min(frames - first, block)
        start = first * hop - hop
        stop = start + (n - 1) * hop + size
        wave = np.stack([x[:, max(0, start):min(length, stop)] for x in waves])
        if start < 0 or stop > length:
            wave = np.pad(wave, ((0, 0), (0, 0), (max(0, -start), max(0, stop - length))))
        tensor = torch.from_numpy(wave).to(device)
        z = torch.fft.rfft(tensor.unfold(-1, size, hop) * window) * weight
        block_grams = []
        for lo, hi in zip(indices, indices[1:]):
            a = torch.view_as_real(z[..., lo:hi]).reshape(len(waves), -1)
            block_grams.append(a @ a.T)
        matrices += torch.stack(block_grams).double().cpu().numpy()
    return matrices


def gram_wave(waves):
    length = waves[0].shape[-1]
    total = np.zeros((len(waves), len(waves)), dtype=np.float64)
    per_second = []
    block = RATE * 10
    for start in range(0, length, block):
        w = np.stack([x[:, start:start + block] for x in waves]).astype(np.float64)
        n = w.shape[-1] // RATE
        if n:
            frames = w[..., :n * RATE].reshape(len(waves), 2, n, RATE)
            rows = frames.transpose(2, 0, 1, 3).reshape(n, len(waves), -1)
            g = rows @ rows.transpose(0, 2, 1)
            total += g.sum(axis=0)
            per_second.extend(g)
        if n * RATE < w.shape[-1]:
            tail = w[..., n * RATE:].reshape(len(waves), -1)
            total += tail @ tail.T
    return total, per_second


def fit_coefficients(g, estimate):
    reference = g[:2, :2]
    determinant = np.linalg.det(reference)
    if determinant <= 1e-8 * max(reference[0, 0] * reference[1, 1], 1e-16):
        return None
    coeff = np.linalg.solve(reference, g[:2, estimate])
    residual = max(0., float(g[estimate, estimate] - coeff @ g[:2, estimate]))
    return {"vocal": float(coeff[0]), "instrumental": float(coeff[1]),
            "residual_fraction_of_estimate": residual / max(float(g[estimate, estimate]), 1e-12)}


def spectral_sdr(g, target, estimate):
    error = float(g[target, target] + g[estimate, estimate] - 2 * g[target, estimate])
    return float(10 * np.log10(max(float(g[target, target]), 1e-12) / max(error, 1e-12)))


def error_cosine(g, target, a, b):
    den = np.array([g[x, x] + g[target, target] - 2 * g[x, target] for x in (a, b)])
    cross = g[a, b] - g[a, target] - g[b, target] + g[target, target]
    return float(np.clip(cross / np.sqrt(max(float(den.prod()), 1e-24)), -1, 1))


def analyses(references, estimates, device):
    waves = [*references, *(stem for pair in estimates for stem in pair)]
    all_g, frames = gram_wave(waves)
    bands = gram_band(waves, device)
    energy = bands.sum(axis=0).diagonal()[:2]
    results = {}
    for index, name in enumerate(NAMES):
        v, i = 2 + index * 2, 3 + index * 2
        active = [fit_coefficients(g, v) for g in frames
                  if g[0, 0] >= 2 * RATE * 10**(-45 / 10)]
        gains = [row["vocal"] for row in active if row is not None]
        results[name] = {
            "joint_vocal": fit_coefficients(all_g, v), "joint_instrumental": fit_coefficients(all_g, i),
            "active_frame_count": len(gains),
            "active_frame_vocal_gain_median": median(gains),
            "active_frame_vocal_gain_below_0_8_pct": 100 * sum(a < .8 for a in gains) / len(gains) if gains else None,
            "bands": [{"vocal_sdr": spectral_sdr(g, 0, v) if g[0, 0] >= .0001 * energy[0] else None,
                       "instrumental_sdr": spectral_sdr(g, 1, i) if g[1, 1] >= .0001 * energy[1] else None,
                       "joint_vocal": fit_coefficients(g, v) if g[0, 0] >= .0001 * energy[0] else None,
                       "joint_instrumental": fit_coefficients(g, i) if g[0, 0] >= .0001 * energy[0] else None}
                      for g in bands],
        }
    correlations = {
        "waveform": [error_cosine(all_g, 0, 2, 4), error_cosine(all_g, 1, 3, 5)],
        "bands": [[error_cosine(g, 0, 2, 4) if g[0, 0] >= .0001 * energy[0] else None,
                   error_cosine(g, 1, 3, 5) if g[1, 1] >= .0001 * energy[1] else None]
                  for g in bands],
    }
    return results, correlations


def rms_gain(ref, estimate):
    ref_power = np.einsum("ct,ct->", ref, ref, dtype=np.float64)
    est_power = np.einsum("ct,ct->", estimate, estimate, dtype=np.float64)
    return float(10 * np.log10((est_power + 1e-8) / (ref_power + 1e-8)))


def side_sdr(ref, estimate):
    ref_side = .5 * (ref[0] - ref[1])
    if np.sqrt(np.mean(ref_side**2)) < 10**(-50 / 20):
        return None
    est_side = .5 * (estimate[0] - estimate[1])
    return -rms_gain(ref_side[None, :], (est_side - ref_side)[None, :])


def metrics(track, ref, estimate, mixture):
    row = score_song(track, ref, estimate, mixture)
    row["abs_bleed"] = abs(row["bleed"])
    row["bleed_energy_pct"] = 100 * row["bleed"]**2
    row["rms_gain_db"] = [rms_gain(r, e) for r, e in zip(ref, estimate)]
    row["side_sdr_db"] = [side_sdr(r, e) for r, e in zip(ref, estimate)]
    return row


def evaluate_replay(track, item, mixture, refs, estimates):
    start, length = item["start_sample"], item["samples"]
    take = slice(start, start + length)
    clip = item["clip"]
    prior_dir = PRIOR / "evaluation/listen" / clip
    names = {"exp04": "A" if item["A"] == "Exp04" else "B",
             "exp08": "A" if item["A"] == "Exp08" else "B"}
    checked = {}
    for name, pair in zip(NAMES[:2], estimates[:2], strict=True):
        for source, new in zip(STEMS[1:], pair, strict=True):
            old, rate = sf.read(prior_dir / f"{names[name]}_{source}.wav",
                                dtype="float32", always_2d=True)
            if rate != RATE or old.T.shape != new[:, take].shape:
                raise ValueError(f"Excerpt alignment failed for {track}")
            error = float(np.max(np.abs(old.T - new[:, take])))
            if error > 2e-6:
                raise ValueError(f"Excerpt sample replay failed for {track} {name} {source}: {error}")
            checked[f"{name}_{source}"] = error
    output = HERE / "listen" / clip
    output.mkdir(parents=True, exist_ok=False)
    arrays = {"mixture": mixture[:, take].T}
    for name, pair in zip(("reference", *NAMES), (refs, *estimates), strict=True):
        arrays.update({f"{name}_{source}": wave[:, take].T
                       for source, wave in zip(STEMS[1:], pair, strict=True)})
    paths = export_audio_bundle(output, arrays, RATE)
    for key, path in paths.items():
        sound, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != RATE or not np.array_equal(sound, arrays[key]):
            raise ValueError(f"Float WAV sample replay failed for {clip} {key}")
    return {"track": track, "clip": clip, "start_sample": start, "samples": length,
            "component_replay_max_abs": checked, "export_files": list(paths)}


def summary(tracks):
    scalar_keys = ("vocal_sdr", "instrumental_sdr", "vocal_si_sdr", "bleed", "abs_bleed", "bleed_energy_pct")
    all_metrics = {}
    pairwise = {}
    for name in NAMES:
        rows = [row["scores"][name] for row in tracks]
        values = {key: median([x[key] for x in rows]) for key in scalar_keys}
        for key in ("side_error", "side_sdr_db", "rms_gain_db", "clipped"):
            values[key] = [median([x[key][s] for x in rows]) for s in range(2)]
        values["peak_consistency"] = max(x["consistency"] for x in rows)
        all_metrics[name] = values
        if name != "exp04":
            rng = np.random.default_rng(42)
            pairwise[name] = {}
            for key in scalar_keys:
                delta = np.array([row["scores"][name][key] - row["scores"]["exp04"][key]
                                  for row in tracks], dtype=np.float64)
                boots = np.median(rng.choice(delta, (10000, len(delta))), axis=1)
                pairwise[name][key] = {"median_delta": float(np.median(delta)),
                                       "ci95": np.quantile(boots, [.025, .975]).tolist(),
                                       "positive": int((delta > 0).sum()), "negative": int((delta < 0).sum())}
    b, h = all_metrics["exp04"], all_metrics["hybrid"]
    d = pairwise["hybrid"]
    gates = {
        "vocal_sdr_improves": h["vocal_sdr"] > b["vocal_sdr"] and d["vocal_sdr"]["median_delta"] > 0,
        "vocal_bleed_improves": h["bleed"] < b["bleed"] and h["abs_bleed"] < b["abs_bleed"],
        "instrumental_sdr_noninferior": h["instrumental_sdr"] >= b["instrumental_sdr"] - .10
                                      and d["instrumental_sdr"]["median_delta"] >= -.10,
    }
    return {"metrics": all_metrics, "paired_vs_exp04": pairwise, "decision_gates": gates,
            "recommend_further_investigation": all(gates.values())}


def main():
    write(HERE / "protected_before.json", protected())
    config = read(MODELS / "commit.json")["fingerprint"]
    store = Checkpoints(MODELS, config)
    _, val, manifest = bind_run(PRIOR / "manifest.json", config, config["config"])
    assert [p.name for p in val] == manifest["validation"]
    assert all(p.parent.resolve() == (PROCESSED_DIR / "train").resolve() for p in val)
    assert len(val) == 15
    state = store.load("best")
    assert state["epoch"] == 53
    assert sha256(MODELS / "best.pt") == store._record()["best"]["sha256"]
    write(HERE / "provenance.json", {"exp04": str(BASELINE), "exp04_sha256": sha256(BASELINE),
          "exp08": str(MODELS / "best.pt"), "exp08_sha256": sha256(MODELS / "best.pt"),
          "exp08_epoch": state["epoch"], "manifest_sha256": sha256(PRIOR / "manifest.json"),
          "development_validation_tracks": [p.name for p in val]})
    configure_determinism(42)
    device = "cuda"
    exp08 = SubbandSeparator().to(device)
    exp08.load_state_dict(state["model"])
    exp04, _, _ = load_separation_model(BASELINE, device)
    previous = read(PRIOR / "evaluation/metrics.json")
    old_scores = {name: {row["track"]: row for row in previous[key]}
                  for name, key in (("exp04", "baseline"), ("exp08", "native"))}
    examples = {row["track"]: row for row in read(PRIOR / "evaluation/blind_key.json")}
    tracks, excerpts = [], []
    for track in val:
        mixture, v, i = (read_segment(track / f"{stem}.wav", 0, -1) for stem in STEMS)
        refs = np.stack((v, i))
        original = separate_waveform(exp04, mixture, device=device, segment_length_seconds=6., overlap_ratio=.5)
        baseline = np.stack((original["vocals"], original["instrumental"]))
        candidate = separate(exp08, mixture, device)
        hybrid = blend(mixture, baseline, candidate)
        estimates = (baseline, candidate, hybrid)
        scores = {name: metrics(track.name, refs, estimate, mixture) for name, estimate in zip(NAMES, estimates)}
        for name in NAMES[:2]:
            for key in ("vocal_sdr", "instrumental_sdr", "vocal_si_sdr", "bleed"):
                if abs(scores[name][key] - old_scores[name][track.name][key]) > 1e-4:
                    raise ValueError(f"Stored metric replay mismatch: {track.name} {name} {key}")
        if scores["hybrid"]["consistency"] > scores["hybrid"]["consistency_limit"]:
            raise ValueError(f"Hybrid mixture consistency failed for {track.name}")
        if track.name in examples:
            excerpts.append(evaluate_replay(track.name, examples[track.name], mixture, refs, estimates))
        source, correlation = analyses(refs, estimates, device)
        tracks.append({"track": track.name, "samples": mixture.shape[-1], "scores": scores,
                       "analysis": source, "exp04_exp08_error_cosine": correlation})
        print(f"{len(tracks):02d}/15 {track.name}", flush=True)
    assert len(excerpts) == 12
    write(HERE / "results.json", {"band_edges_hz": BANDS, "tracks": tracks, **summary(tracks)})
    write(HERE / "listening_manifest.json", excerpts)
    after = protected()
    if after != read(HERE / "protected_before.json"):
        raise ValueError("Protected checkpoint or prior report changed")
    write(HERE / "protected_after.json", after)
    print(f"Verified {len(after)} existing model and report files.", flush=True)


if __name__ == "__main__":
    main()
