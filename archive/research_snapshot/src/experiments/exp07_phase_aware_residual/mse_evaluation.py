"""Paired full-song evaluation of the selected deterministic MSE checkpoint."""

import csv
import json
from dataclasses import asdict

import numpy as np
import soundfile as sf
import torch

from config import SAMPLE_RATE
from src.audio.stft import compute_istft, compute_stft
from src.data.splits import create_train_validation_split
from src.evaluation.metrics import compute_leakage_diagnostics, evaluate_stem_pair

from .chunking import OverlapAdd, chunk_batch, chunk_starts
from .complex_ops import complex_additivity, make_vocal_ablation_stfts
from .config import NumericalTolerances, SuccessThresholds
from .inference import WaveformPair, measure_full_song_consistency
from .oracle_analysis import CHUNK_SAMPLES, _read_stems

METHODS = ("exp04", "exp07", "magnitude_only", "phase_only")
SOURCES = ("vocals", "instrumental")


def excerpt_start(reference, frames, rate=SAMPLE_RATE):
    if len(reference) <= frames:
        return 0
    power = np.square(reference, dtype=np.float64).sum(axis=1)
    cumulative = np.r_[0., np.cumsum(power)]
    starts = np.unique(np.r_[np.arange(0, len(reference) - frames + 1, rate), len(reference) - frames])
    return int(starts[np.argmax(cumulative[starts + frames] - cumulative[starts])])


def waveform_artifacts(wave):
    return {"waveform_peak": float(np.abs(wave).max()),
            "clipped_sample_pct": 100 * np.count_nonzero(np.abs(wave) > 1) / wave.size,
            "max_abs_channel_mean": float(np.abs(wave.mean(axis=0, dtype=np.float64)).max()),
            "peak_sample_difference": float(np.abs(np.diff(wave, axis=0)).max(initial=0.))}


@torch.inference_mode()
def paired_separation(model, mixture):
    n = len(mixture)
    starts = chunk_starts(n, CHUNK_SAMPLES, 0.5)
    ola = {name: OverlapAdd(4, n, CHUNK_SAMPLES, len(starts) > 1) for name in METHODS}
    samples = {name: [[], []] for name in METHODS}
    maxima = {name: [0., 0.] for name in METHODS}
    ratio_max = dict.fromkeys(METHODS, 0.)
    violations = dict.fromkeys(METHODS, 0)
    model.eval()
    for pos in range(0, len(starts), 4):
        offsets = starts[pos:pos + 4]
        audio = torch.from_numpy(chunk_batch(mixture.T, offsets, CHUNK_SAMPLES)).to(next(model.parameters()).device)
        x = compute_stft(audio)
        out = model.forward_stft(x)
        pairs = {name: (v, x - v) for name, v in
                 make_vocal_ablation_stfts(out.initial_vocal_stft, out.vocals_stft).items()}
        pairs["exp07"] = pairs.pop("full")
        pairs["exp04"] = (out.initial_vocal_stft, x * out.masks[:, 1])
        mag = x.abs()
        active = (mag > 0) & (mag >= .001 * mag.amax(dim=(-2, -1), keepdim=True))
        for name, (v, i) in pairs.items():
            error, bound = complex_additivity(x, v, i)
            ratio_max[name] = max(ratio_max[name], float((error / bound).max()))
            violations[name] += int((error > bound).sum())
            for k, spectrum in enumerate((v, i)):
                ratio = spectrum.abs() / mag.clamp_min(1e-12)
                maxima[name][k] = max(maxima[name][k], float(ratio.masked_fill(~active, 0).max()))
                samples[name][k].append(ratio.flatten()[::257][active.flatten()[::257]].cpu().numpy())
            waves = compute_istft(torch.cat((v, i), dim=1), length=CHUNK_SAMPLES)
            if not torch.isfinite(waves).all():
                raise FloatingPointError(f"{name} produced a nonfinite waveform")
            for wave, start in zip(waves.cpu().numpy(), offsets):
                ola[name].add(wave, start)
    results = {}
    for name in METHODS:
        v, i = np.split(ola[name].finish(), 2, axis=0)
        spectral = {"complex_max_error_to_bound": ratio_max[name], "complex_violations": violations[name]}
        for k, source in enumerate(SOURCES):
            sampled = np.concatenate(samples[name][k])
            spectral[source + "_amplification_p99"] = float(np.percentile(sampled, 99)) if sampled.size else 0.
            spectral[source + "_amplification_max"] = maxima[name][k]
        results[name] = (WaveformPair(v.T, i.T), spectral)
    return results, starts


def grade(records, summaries):
    t, numeric = SuccessThresholds(), NumericalTolerances()
    base, new = summaries["exp04"], summaries["exp07"]
    pairs = [(r["metrics"]["exp04"], r["metrics"]["exp07"]) for r in records]
    wins = sum(b["vocals_median_sdr_db"] > a["vocals_median_sdr_db"] for a, b in pairs)
    checks = {"vocals_sdr": new["vocals_median_sdr_db"] >= t.vocal_sdr_db,
              "instrumental_sdr": new["instrumental_median_sdr_db"] >= t.instrumental_sdr_db,
              "vocal_bleed": new["vocal_to_inst_bleed_coeff"] <= t.vocal_bleed_coeff,
              "vocal_track_wins": wins >= t.vocal_wins,
              "mixture_consistency": all(
                  b["global_nrmse_pct"] <= numeric.waveform_nrmse_pct and
                  max(b["global_max_abs"], b["max_seam_window_abs"]) <= numeric.waveform_max_abs and
                  b["complex_violations"] == 0 for _, b in pairs)}
    for source in SOURCES:
        key = source + "_energy_ratio_db"
        checks[source + "_gain"] = np.median([abs(b[key]) for _, b in pairs]) <= (
            np.median([abs(a[key]) for a, _ in pairs]) + t.gain_regression_db)
        key = source + "_clipped_sample_pct"
        checks[source + "_clipping"] = (new[key] - base[key] <= t.clipped_sample_increase_pp
                                        and all(not (a[key] <= t.newly_clipped_track_pct < b[key]) for a, b in pairs))
        key = source + "_amplification_p99"
        checks[source + "_spectral_amplification"] = (
            20 * np.log10(max(new[key], 1e-12) / max(base[key], 1e-12)) <= t.spectral_amplification_increase_db)
    checks = {k: bool(v) for k, v in checks.items()}
    return {"numeric_pass": all(checks.values()), "checks": checks, "vocal_track_wins": wins,
            "human_listening": "pending",
            "recommendation": ("Candidate for listening review; keep Exp04 preferred until approval"
                               if all(checks.values()) else "Keep Exp04 preferred; Exp07 missed numeric gates")}


def _score(pair, spectral, references, mixture, starts):
    row = dict(spectral) | asdict(measure_full_song_consistency(mixture, pair, starts))
    row.update(compute_leakage_diagnostics(*references, pair.vocals, pair.instrumental))
    for name, wave, ref in zip(SOURCES, (pair.vocals, pair.instrumental), references):
        row.update({name + "_" + key: value for key, value in
                    (evaluate_stem_pair(ref, wave, mixture) | waveform_artifacts(wave)).items()})
    return row


def evaluate(model, root, checkpoint_metadata):
    _, validation = create_train_validation_split()
    if len(validation) != 15:
        raise ValueError("Expected all 15 approved validation tracks")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, track in enumerate(validation, 1):
        mixture, *references = _read_stems(track)
        variants, starts = paired_separation(model, mixture)
        scores = {name: _score(pair, spectral, references, mixture, starts)
                  for name, (pair, spectral) in variants.items()}
        length = min(len(mixture), 20 * SAMPLE_RATE)
        start = excerpt_start(references[0], length)
        cut = slice(start, start + length)
        folder = root / "audio" / f"{index:02d}"
        folder.mkdir(parents=True)
        audio = dict(mixture=mixture, reference_vocals=references[0], reference_instrumental=references[1])
        for name, (pair, _) in variants.items():
            audio[name + "_vocals"] = pair.vocals
            audio[name + "_instrumental"] = pair.instrumental
        for name, wave in audio.items():
            sf.write(folder / f"{name}.wav", wave[cut], SAMPLE_RATE, subtype="FLOAT")
        rows.append({"track": track.name, "metrics": scores, "audio_dir": folder.relative_to(root).as_posix(),
                     "start_sample": start, "samples": length})
        delta = scores["exp07"]["vocals_median_sdr_db"] - scores["exp04"]["vocals_median_sdr_db"]
        print(f"Evaluation {index:02d}/15 {track.name}: vocal SDR delta {delta:+.3f} dB", flush=True)
        del variants, mixture, references, audio
    summaries = {}
    for name in METHODS:
        metrics = [row["metrics"][name] for row in rows]
        summaries[name] = {key: float(np.median([m[key] for m in metrics]))
                           for key in metrics[0] if all(np.isfinite(m[key]) for m in metrics)}
    decision = grade(rows, summaries)
    payload = {"checkpoint": checkpoint_metadata, "tracks": rows, "summary": summaries,
               "decision": decision, "chunk_samples": CHUNK_SAMPLES, "overlap": 0.5, "batch_size": 4}
    (root / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    flat = [{"track": row["track"], "method": name, **m} for row in rows for name, m in row["metrics"].items()]
    with (root / "per_track.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    labels = ("vocals_median_sdr_db", "instrumental_median_sdr_db", "vocals_median_si_sdr_db",
              "instrumental_median_si_sdr_db", "vocal_to_inst_bleed_coeff", "vocal_to_inst_bleed_energy_pct")
    lines = ["# Exp07 MSE validation", "", decision["recommendation"], "",
             "| Method | Vocal SDR | Instrumental SDR | Vocal SI-SDR | Instrumental SI-SDR | Bleed coeff | Bleed energy % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in METHODS:
        lines.append(f"| {name} | " + " | ".join(f"{summaries[name][key]:.4f}" for key in labels) + " |")
    lines.extend(["", f"Vocal SDR wins: {decision['vocal_track_wins']}/15.", "",
                  "SDR and SI-SDR use medians of track-level active-frame medians. Waveform peaks, clipping, "
                  "gain, DC and maximum sample differences are signal indicators. Listening remains pending.", "",
                  "## Gates", ""])
    lines.extend(f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in decision["checks"].items())
    lines.extend(["", "## Matched audio", ""])
    for row in rows:
        folder = row["audio_dir"]
        lines.append(f"- {row['track']} at {row['start_sample'] / SAMPLE_RATE:.2f} s: "
                     f"[Exp04 vocal]({folder}/exp04_vocals.wav) · [Exp07 vocal]({folder}/exp07_vocals.wav) · "
                     f"[Exp04 instrumental]({folder}/exp04_instrumental.wav) · [Exp07 instrumental]({folder}/exp07_instrumental.wav)")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision
