import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config import SAMPLE_RATE, SEGMENT_LENGTH_SECONDS
from src.audio.stft import compute_istft, compute_stft
from src.data.splits import create_train_validation_split
from src.evaluation.metrics import compute_leakage_diagnostics, compute_sdr, evaluate_stem_pair
from src.inference.separate_track import load_separation_model

from .chunking import OverlapAdd, chunk_batch, chunk_starts
from .complex_ops import project_equal_weight_targets, waveform_nrmse_pct
from .config import Exp07Paths, OracleThresholds
from .oracle_precision import ProjectionDiagnostics


METHODS = ("exp04", "mixture_phase", "unconstrained_complex", "constrained_complex")
CHUNK_SAMPLES = round(SEGMENT_LENGTH_SECONDS * SAMPLE_RATE)


@dataclass(frozen=True)
class SourcePair:
    vocals: torch.Tensor
    instrumental: torch.Tensor


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    vocal_paired_median_gain_db: float | None
    vocal_wins: int
    instrumental_median_gain_db: float | None
    failures: tuple[str, ...]


def _oracle_spectra(mixture, vocals, instrumental) -> dict[str, SourcePair]:
    vmag, imag = vocals.abs(), instrumental.abs()
    vocal_mask = vmag / (vmag + imag).clamp_min(1e-8)
    projected = project_equal_weight_targets(mixture, vocals, instrumental)
    return {
        "mixture_phase": SourcePair(mixture * vocal_mask, mixture * (1 - vocal_mask)),
        "unconstrained_complex": SourcePair(vocals, instrumental),
        "constrained_complex": SourcePair(projected.vocals, projected.instrumental),
    }


def compute_oracle_chunk(mixture, vocals, instrumental) -> dict[str, SourcePair]:
    with torch.autocast(mixture.device.type, enabled=False):
        return _oracle_spectra(*(compute_stft(wave.float()) for wave in (mixture, vocals, instrumental)))


def score_estimate(mixture, vocals_ref, instrumental_ref, vocals, instrumental) -> dict[str, float]:
    out = {}
    for name, ref, estimate in (("vocals", vocals_ref, vocals), ("instrumental", instrumental_ref, instrumental)):
        out.update({f"{name}_{key}": value for key, value in evaluate_stem_pair(ref, estimate, mixture).items()})
    out.update(compute_leakage_diagnostics(vocals_ref, instrumental_ref, vocals, instrumental))
    summed = vocals + instrumental
    out["mixture_nrmse_pct"] = waveform_nrmse_pct(summed, mixture)
    out["mixture_max_abs"] = float(np.max(np.abs(summed - mixture)))
    out["mixture_sdr_db"] = compute_sdr(mixture, summed)
    return out


def evaluate_oracle_gate(records, thresholds: OracleThresholds = OracleThresholds()) -> GateDecision:
    failures = []
    if len(records) != thresholds.tracks or len({row["track"] for row in records}) != thresholds.tracks:
        failures.append("Validation set must contain exactly 15 distinct tracks")
    vocals, instruments = [], []
    limits = thresholds.numerical
    numerical_limits = {
        "unconstrained_source_nrmse_pct": limits.waveform_nrmse_pct,
        "unconstrained_source_max_abs": limits.waveform_max_abs,
        "constrained_mixture_nrmse_pct": limits.waveform_nrmse_pct,
        "constrained_mixture_max_abs": limits.waveform_max_abs,
        "complex_max_error_to_bound": 1.0,
        "complex_violations": 0,
        "nonfinite_count": 0,
    }
    for record in records:
        phase = record["methods"]["mixture_phase"]
        complex_oracle = record["methods"]["constrained_complex"]
        vocals.append(complex_oracle["vocals_median_sdr_db"] - phase["vocals_median_sdr_db"])
        instruments.append((complex_oracle["instrumental_median_sdr_db"], phase["instrumental_median_sdr_db"]))
        for name, limit in numerical_limits.items():
            value = record["numerical"].get(name, float("nan"))
            if not np.isfinite(value) or not 0 <= value <= limit:
                failures.append(f"{record['track']}: {name}={value:.8g} exceeds {limit:g} or is invalid")
    vocal_gain = float(np.median(vocals)) if vocals and np.isfinite(vocals).all() else None
    inst_gain = None
    if instruments and np.isfinite(instruments).all():
        constrained_median, mixture_phase_median = np.median(instruments, axis=0)
        inst_gain = float(constrained_median - mixture_phase_median)
    wins = sum(gain > 0 and np.isfinite(gain) for gain in vocals)
    if vocal_gain is None or vocal_gain < thresholds.vocal_median_gain_db:
        failures.append("Median paired vocal SDR gain below 1.0 dB or undefined")
    if wins < thresholds.vocal_wins:
        failures.append("Fewer than 10 positive paired vocal SDR gains")
    if inst_gain is None or inst_gain < -thresholds.instrumental_max_loss_db:
        failures.append("Aggregate instrumental SDR loss above 0.5 dB or undefined")
    return GateDecision(not failures, vocal_gain, int(wins), inst_gain, tuple(failures))


def _read_stems(track: Path) -> tuple[np.ndarray, ...]:
    stems = []
    for name in ("mixture", "vocals", "instrumental"):
        wave, rate = sf.read(track / f"{name}.wav", dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE or wave.shape[1] != 2 or not np.isfinite(wave).all():
            raise ValueError(f"Expected finite 44.1 kHz stereo: {track.name}/{name}")
        if stems and stems[0].shape != wave.shape:
            raise ValueError(f"Stem lengths differ: {track.name}")
        stems.append(wave)
    return tuple(stems)


@torch.inference_mode()
def _run_oracles(stems, overlap, batch_size, device, model=None):
    total = stems[0].shape[0]
    starts = chunk_starts(total, CHUNK_SAMPLES, overlap)
    accumulators = {
        name: OverlapAdd(4, total, CHUNK_SAMPLES, len(starts) > 1 and overlap > 0)
        for name in (METHODS if model is not None else METHODS[1:])
    }
    precision = ProjectionDiagnostics()
    nonfinite = 0
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        waveforms = [torch.from_numpy(chunk_batch(wave.T, batch_starts, CHUNK_SAMPLES)).to(device)
                     for wave in stems]
        x, v, i = map(compute_stft, waveforms)
        pairs = _oracle_spectra(x, v, i)
        if model is not None:
            masks = model(x.abs())
            pairs["exp04"] = SourcePair(x * masks[:, 0], x * masks[:, 1])
        for name, pair in pairs.items():
            if name == "constrained_complex":
                precision.add(x, v, i, pair)
            spectra = torch.cat((pair.vocals, pair.instrumental), dim=1)
            if name == "exp04":
                audio = torch.cat([compute_istft(part, length=CHUNK_SAMPLES)
                                   for part in (pair.vocals, pair.instrumental)], dim=1)
            else:
                audio = compute_istft(spectra, length=CHUNK_SAMPLES)
            nonfinite += int((~torch.isfinite(spectra)).sum()) + int((~torch.isfinite(audio)).sum())
            for chunk, start in zip(audio.cpu().numpy(), batch_starts):
                accumulators[name].add(chunk, start)
    estimates = {name: accumulator.finish().T for name, accumulator in accumulators.items()}
    if any(estimate.shape != (total, 4) for estimate in estimates.values()):
        raise RuntimeError("Overlap-add must return two stereo stems at the original length")
    return estimates, precision.summary(), nonfinite


def _track_result(track, model, device, overlap, batch_size):
    mix, vocals, instrumental = _read_stems(track)
    estimates, precision, nonfinite = _run_oracles((mix, vocals, instrumental), overlap, batch_size, device, model=model)
    methods = {name: score_estimate(mix, vocals, instrumental, estimates[name][:, :2], estimates[name][:, 2:])
               for name in METHODS}
    free = estimates["unconstrained_complex"]
    constrained = estimates["constrained_complex"]
    numerical = {
        "unconstrained_source_nrmse_pct": max(waveform_nrmse_pct(free[:, :2], vocals),
                                             waveform_nrmse_pct(free[:, 2:], instrumental)),
        "unconstrained_source_max_abs": max(float(np.max(np.abs(free[:, :2] - vocals))),
                                            float(np.max(np.abs(free[:, 2:] - instrumental)))),
        "constrained_mixture_nrmse_pct": methods["constrained_complex"]["mixture_nrmse_pct"],
        "constrained_mixture_max_abs": methods["constrained_complex"]["mixture_max_abs"],
        "constrained_complex_max_abs": precision["fp32_max_abs"],
        "complex_max_error_to_bound": precision["max_error_to_bound"],
        "complex_violations": precision["complex_violations"],
        "nonfinite_count": nonfinite + sum(int((~np.isfinite(wave)).sum()) for wave in estimates.values()),
    }
    # Diagnose endpoint attenuation separately without excluding it from the gate.
    residual = np.abs(constrained[:, :2] + constrained[:, 2:] - mix)
    peak_sample = int(np.argmax(residual) // 2)
    numerical["constrained_mixture_peak_sample"] = peak_sample
    numerical["constrained_mixture_samples"] = len(mix)
    numerical["constrained_mixture_interior_max_abs"] = float(np.max(residual[8:-8], initial=0))
    half_seam = OracleThresholds().numerical.seam_window // 2
    seams = chunk_starts(len(mix), CHUNK_SAMPLES, overlap)
    numerical["constrained_mixture_seam_max_abs"] = max(
        (float(residual[max(0, start - half_seam):start + half_seam].max()) for start in seams[1:]),
        default=0.0,
    )
    return {
        "track": track.name, "methods": methods, "numerical": numerical, "projection_precision": precision,
        "stem_sum_nrmse_pct": waveform_nrmse_pct(vocals + instrumental, mix),
        "stem_sum_sdr_db": compute_sdr(mix, vocals + instrumental),
        "vocal_paired_gain_db": methods["constrained_complex"]["vocals_median_sdr_db"]
        - methods["mixture_phase"]["vocals_median_sdr_db"],
    }


def _median(values) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_reports(output: Path, tag: str, payload):
    output.mkdir(parents=True, exist_ok=False)
    records = payload["tracks"]
    (output / f"{tag}_summary.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False),
                                               encoding="utf-8")
    rows = [{"track": record["track"], "method": method, "stem_sum_nrmse_pct": record["stem_sum_nrmse_pct"],
             "stem_sum_sdr_db": record["stem_sum_sdr_db"], "vocal_paired_gain_db": record["vocal_paired_gain_db"],
             **record["numerical"], **record["projection_precision"], **metrics}
            for record in records for method, metrics in record["methods"].items()]
    with (output / f"{tag}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    gate = payload["gate"]
    lines = ["# Exp07 oracle headroom", "", f"**{payload['decision']}**", "",
             "Scores use the original stems. Values are medians across the 15 track medians.", "",
             "| Method | Vocals SDR | Vocals SI-SDR | Instrumental SDR | Instrumental SI-SDR |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, scores in payload["summary"].items():
        values = [scores[f"{source}_median_{metric}_db"] for source in ("vocals", "instrumental")
                  for metric in ("sdr", "si_sdr")]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f} dB" if value is not None else "n/a"
                                                for value in values) + " |")
    lines.extend(["", f"Paired vocal median gain: {gate['vocal_paired_median_gain_db']} dB.",
                  f"Vocal wins: {gate['vocal_wins']}/15.",
                  f"Instrumental median gain: {gate['instrumental_median_gain_db']} dB.",
                  f"Recorded stem-sum mismatch: {payload['median_stem_sum_nrmse_pct']:.4f}% NRMSE.",
                  "", "## Projection precision", "",
                  "The approved FP32 gate checks every bin against 1e-5 + 4 × 2^-24 × (|X| + |V| + |I|).",
                  "Quiet bins are below 0.1% of the chunk/channel mixture peak, including zero bins. High-energy bins are at least 10% of that peak. The wider arithmetic below is diagnostic.",
                  "", "| Measurement | Worst across 15 tracks |", "| --- | ---: |"])
    for name, value in payload["projection_precision_maxima"].items():
        lines.append(f"| {name} | {value:.8g} |")
    lines.extend(["", "## Gate failures", "", *[f"- {reason}" for reason in gate["failures"]], ""])
    (output / f"{tag}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Exp07 four-way oracle gate")
    parser.add_argument("--checkpoint", type=Path, default=Exp07Paths().backbone)
    parser.add_argument("--val-split", action="store_true", required=True)
    parser.add_argument("--overlap", type=float, default=0.5, choices=(0.5,))
    parser.add_argument("--output-dir", type=Path, default=Exp07Paths().reports / "oracle")
    parser.add_argument("--tag", default="exp07_oracle_15tracks")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite {args.output_dir}")
    if Path(args.tag).name != args.tag or not args.tag.startswith("exp07_oracle_"):
        parser.error("The report tag must start with exp07_oracle_ and contain no directory separators")
    _, tracks = create_train_validation_split()
    if len(tracks) != 15:
        raise ValueError(f"Expected 15 validation tracks, got {len(tracks)}")
    model, metadata, device = load_separation_model(args.checkpoint)
    if metadata.get("exp_name") != "exp04_asymmetric_leakage":
        raise ValueError("An Exp04 asymmetric-leakage checkpoint is required")
    records = []
    for index, track in enumerate(tracks, 1):
        row = _track_result(track, model, device, args.overlap, batch_size=8)
        records.append(row)
        sdrs = [f"{row['methods'][name]['vocals_median_sdr_db']:.3f}" for name in METHODS]
        print(f"[{index}/15] {track.name}: vocal SDR {' / '.join(sdrs)} dB", flush=True)
    gate = evaluate_oracle_gate(records)
    payload = {
        "decision": "GO" if gate.passed else "STOP", "gate": asdict(gate),
        "thresholds": asdict(OracleThresholds()), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": str(device), "overlap_ratio": args.overlap, "chunk_samples": CHUNK_SAMPLES, "batch_size": 8,
        "overlap_add": "Unfloored positive Hann weight sum for all four estimates",
        "projection_precision_maxima": {
            key: max(row["projection_precision"][key] for row in records if row["projection_precision"][key] is not None)
            for key in records[0]["projection_precision"]
            if any(row["projection_precision"][key] is not None for row in records)
        },
        "summary": {method: {key: _median([row["methods"][method][key] for row in records])
                             for key in records[0]["methods"][method]} for method in METHODS},
        "median_stem_sum_nrmse_pct": _median([row["stem_sum_nrmse_pct"] for row in records]),
        "median_stem_sum_sdr_db": _median([row["stem_sum_sdr_db"] for row in records]),
        "tracks": records,
    }
    _write_reports(args.output_dir, args.tag, payload)
    print(f"{payload['decision']}: report written to {args.output_dir}", flush=True)
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
