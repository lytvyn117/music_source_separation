import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

from config import SAMPLE_RATE, SEGMENT_LENGTH_SECONDS
from src.audio.stft import compute_magnitude, compute_stft
from src.data.splits import create_train_validation_split
from src.evaluation.mask_diagnostics import (
    DEFAULT_FREQUENCY_BANDS,
    mask_diagnostics,
    spectral_band_metrics,
    waveform_artifact_metrics,
)
from src.inference.separate_track import separate_waveform
from src.models.unet_v2 import UNetV2


def load_transition_models(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[UNetV2, UNetV2, Dict]:
    """Load identical Exp04 weights under softmax and sigmoid activations."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    config = checkpoint.get("config", {})
    source_mask_mode = config.get("mask_mode", "softmax")
    if source_mask_mode != "softmax":
        raise ValueError(
            f"Transition probe requires a softmax source checkpoint, got {source_mask_mode!r}"
        )

    common = {
        "input_channels": 2,
        "base_channels": config.get("base_channels", 16),
        "pad_mode": config.get("pad_mode", "exact"),
        "log_input": config.get("log_input", False),
    }
    softmax_model = UNetV2(**common, mask_mode="softmax").to(device)
    sigmoid_model = UNetV2(**common, mask_mode="sigmoid").to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    softmax_model.load_state_dict(state_dict, strict=True)
    sigmoid_model.load_state_dict(state_dict, strict=True)
    softmax_model.eval()
    sigmoid_model.eval()

    weights_identical = all(
        torch.equal(value, sigmoid_model.state_dict()[name])
        for name, value in softmax_model.state_dict().items()
    )
    if not weights_identical:
        raise RuntimeError("Softmax and sigmoid transition models do not share identical weights")

    metadata = {
        "checkpoint": str(checkpoint_path),
        "source_mask_mode": source_mask_mode,
        "probe_mask_mode": "sigmoid",
        "weights_identical": True,
        "base_channels": common["base_channels"],
        "pad_mode": common["pad_mode"],
        "log_input": common["log_input"],
    }
    return softmax_model, sigmoid_model, metadata


def _weighted_mean(records: Sequence[Dict], key: str, weight_key: str) -> float:
    weighted = sum(float(record[key]) * int(record[weight_key]) for record in records)
    total = sum(int(record[weight_key]) for record in records)
    return weighted / max(total, 1)


def _merge_mask_diagnostics(records: Sequence[Dict]) -> Dict:
    if not records:
        return {"finite": True, "nonfinite_values": 0, "active_bins": 0, "regions": {}}
    nonfinite = sum(int(record.get("nonfinite_values", 0)) for record in records)
    if nonfinite:
        return {"finite": False, "nonfinite_values": nonfinite}

    active_records = [record for record in records if record.get("active_bins", 0) > 0]
    active_bins = sum(int(record["active_bins"]) for record in active_records)
    if not active_records:
        return {"finite": True, "nonfinite_values": 0, "active_bins": 0, "regions": {}}

    merged: Dict = {
        "finite": True,
        "nonfinite_values": 0,
        "active_bins": active_bins,
    }
    for key in (
        "mask_sum_mean",
        "mask_sum_std",
        "mask_product_mean",
        "both_masks_gt_0_5_fraction",
        "vocal_mask_irm_mae",
        "instrumental_mask_irm_mae",
    ):
        merged[key] = _weighted_mean(active_records, key, "active_bins")

    for source_key in ("vocal_mask_distribution", "instrumental_mask_distribution"):
        merged[source_key] = {
            statistic: _weighted_mean(
                [
                    {
                        statistic: record[source_key][statistic],
                        "active_bins": record["active_bins"],
                    }
                    for record in active_records
                ],
                statistic,
                "active_bins",
            )
            for statistic in ("mean", "std", "min", "p10", "p50", "p90", "max")
        }

    merged["regions"] = {}
    for region in ("vocal_dominant", "instrumental_dominant", "overlap"):
        region_rows = []
        for record in active_records:
            row = record["regions"].get(region, {})
            if row.get("count", 0) > 0:
                region_rows.append(row)
        count = sum(int(row["count"]) for row in region_rows)
        if count == 0:
            merged["regions"][region] = {"count": 0}
            continue
        merged["regions"][region] = {"count": count}
        for key in (
            "vocal_mask_irm_mae",
            "instrumental_mask_irm_mae",
            "vocal_prediction_nmae",
            "instrumental_prediction_nmae",
        ):
            merged["regions"][region][key] = _weighted_mean(
                region_rows, key, "count"
            )
    return merged


@torch.no_grad()
def track_mask_diagnostics(
    model: torch.nn.Module,
    mixture: np.ndarray,
    vocals_reference: np.ndarray,
    instrumental_reference: np.ndarray,
    device: torch.device,
    sample_rate: int,
    batch_size: int,
) -> Dict:
    chunk_samples = int(round(SEGMENT_LENGTH_SECONDS * sample_rate))
    total_samples = mixture.shape[0]
    starts = list(range(0, total_samples, chunk_samples))
    records: List[Dict] = []

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        mix_chunks = []
        vocal_chunks = []
        inst_chunks = []
        for start in batch_starts:
            end = min(total_samples, start + chunk_samples)
            pad = chunk_samples - (end - start)
            pad_spec = ((0, pad), (0, 0))
            mix_chunks.append(np.pad(mixture[start:end], pad_spec).T)
            vocal_chunks.append(np.pad(vocals_reference[start:end], pad_spec).T)
            inst_chunks.append(np.pad(instrumental_reference[start:end], pad_spec).T)

        mix = torch.from_numpy(np.stack(mix_chunks).astype(np.float32)).to(device)
        vocals = torch.from_numpy(np.stack(vocal_chunks).astype(np.float32)).to(device)
        instrumental = torch.from_numpy(np.stack(inst_chunks).astype(np.float32)).to(device)
        mix_magnitude = compute_magnitude(compute_stft(mix))
        vocal_magnitude = compute_magnitude(compute_stft(vocals))
        inst_magnitude = compute_magnitude(compute_stft(instrumental))
        masks = model(mix_magnitude)
        records.append(mask_diagnostics(masks, vocal_magnitude, inst_magnitude))

    return _merge_mask_diagnostics(records)


@torch.no_grad()
def _track_spectral_bands(
    vocals_reference: np.ndarray,
    instrumental_reference: np.ndarray,
    vocals_estimate: np.ndarray,
    instrumental_estimate: np.ndarray,
    device: torch.device,
    sample_rate: int,
) -> Dict[str, Dict[str, float]]:
    n_fft = 2048
    chunk_samples = int(round(SEGMENT_LENGTH_SECONDS * sample_rate))
    total_samples = vocals_reference.shape[0]
    accumulators = {
        name: {
            "vocals_reference_energy": 0.0,
            "vocals_error_energy": 0.0,
            "vocal_to_instrumental_projection": 0.0,
            "frequency_bins": 0,
        }
        for name, _, _ in DEFAULT_FREQUENCY_BANDS
    }

    arrays = (
        vocals_reference,
        instrumental_reference,
        vocals_estimate,
        instrumental_estimate,
    )
    for start in range(0, total_samples, chunk_samples):
        end = min(total_samples, start + chunk_samples)
        tensors = [
            torch.from_numpy(array[start:end].T.copy()).unsqueeze(0).to(device)
            for array in arrays
        ]
        stfts = [compute_stft(tensor) for tensor in tensors]
        chunk_stats = spectral_band_metrics(
            *stfts,
            n_fft=n_fft,
            sample_rate=sample_rate,
        )
        for band, stats in chunk_stats.items():
            for key in (
                "vocals_reference_energy",
                "vocals_error_energy",
                "vocal_to_instrumental_projection",
            ):
                accumulators[band][key] += float(stats[key])
            accumulators[band]["frequency_bins"] = int(stats["frequency_bins"])

    output: Dict[str, Dict[str, float]] = {}
    eps = 1e-12
    for band, sums in accumulators.items():
        reference_energy = sums["vocals_reference_energy"]
        error_energy = sums["vocals_error_energy"]
        coefficient = sums["vocal_to_instrumental_projection"] / max(
            reference_energy, eps
        )
        output[band] = {
            **sums,
            "vocals_sdr_db": float(
                10.0 * np.log10((reference_energy + eps) / (error_energy + eps))
            ),
            "vocal_to_instrumental_bleed_coefficient": float(coefficient),
            "vocal_to_instrumental_bleed_energy_pct": float(
                coefficient * coefficient * 100.0
            ),
        }
    return output


def _median(records: Sequence[Dict], path: Sequence[str]) -> float:
    values = []
    for record in records:
        value = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _model_summary(records: Sequence[Dict], model_key: str) -> Dict:
    summary = {
        "tracks": len(records),
        "mask_sum_mean_median": _median(records, (model_key, "mask", "mask_sum_mean")),
        "mask_sum_std_median": _median(records, (model_key, "mask", "mask_sum_std")),
        "mask_product_mean_median": _median(
            records, (model_key, "mask", "mask_product_mean")
        ),
        "both_masks_gt_0_5_fraction_median": _median(
            records, (model_key, "mask", "both_masks_gt_0_5_fraction")
        ),
        "vocal_mask_irm_mae_median": _median(
            records, (model_key, "mask", "vocal_mask_irm_mae")
        ),
        "instrumental_mask_irm_mae_median": _median(
            records, (model_key, "mask", "instrumental_mask_irm_mae")
        ),
        "vocals_gain_db_median": _median(
            records, (model_key, "waveform", "vocals_gain_db")
        ),
        "instrumental_gain_db_median": _median(
            records, (model_key, "waveform", "instrumental_gain_db")
        ),
        "combined_gain_db_median": _median(
            records, (model_key, "waveform", "combined_gain_db")
        ),
        "mixture_reconstruction_residual_ratio_median": _median(
            records,
            (model_key, "waveform", "mixture_reconstruction_residual_ratio"),
        ),
        "source_waveform_cosine_median": _median(
            records, (model_key, "waveform", "source_waveform_cosine")
        ),
        "all_finite": all(
            record[model_key]["mask"]["finite"]
            and record[model_key]["waveform"]["finite"]
            for record in records
        ),
        "frequency_bands": {},
        "regions": {},
    }
    for source_key in ("vocal_mask_distribution", "instrumental_mask_distribution"):
        summary[f"{source_key}_median"] = {
            statistic: _median(
                records,
                (model_key, "mask", source_key, statistic),
            )
            for statistic in ("mean", "std", "min", "p10", "p50", "p90", "max")
        }
    for region in ("vocal_dominant", "instrumental_dominant", "overlap"):
        summary["regions"][region] = {
            "total_bins": int(
                sum(record[model_key]["mask"]["regions"][region]["count"] for record in records)
            ),
            **{
                f"{metric}_median": _median(
                    records,
                    (model_key, "mask", "regions", region, metric),
                )
                for metric in (
                    "vocal_mask_irm_mae",
                    "instrumental_mask_irm_mae",
                    "vocal_prediction_nmae",
                    "instrumental_prediction_nmae",
                )
            },
        }
    for band, _, _ in DEFAULT_FREQUENCY_BANDS:
        summary["frequency_bands"][band] = {
            "vocals_sdr_db_median": _median(
                records, (model_key, "bands", band, "vocals_sdr_db")
            ),
            "vocal_to_instrumental_bleed_coefficient_median": _median(
                records,
                (
                    model_key,
                    "bands",
                    band,
                    "vocal_to_instrumental_bleed_coefficient",
                ),
            ),
            "vocal_to_instrumental_bleed_energy_pct_median": _median(
                records,
                (
                    model_key,
                    "bands",
                    band,
                    "vocal_to_instrumental_bleed_energy_pct",
                ),
            ),
        }
    return summary


def _write_reports(output_dir: Path, payload: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "exp04_sigmoid_transition_diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    transition_path = output_dir / "exp04_sigmoid_transition_by_track.csv"
    with transition_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "track_name",
            "model",
            "mask_sum_mean",
            "mask_product_mean",
            "both_masks_gt_0_5_fraction",
            "vocal_mask_irm_mae",
            "instrumental_mask_irm_mae",
            "vocals_gain_db",
            "instrumental_gain_db",
            "combined_gain_db",
            "mixture_reconstruction_residual_ratio",
            "source_waveform_cosine",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in payload["tracks"]:
            for model_key in ("softmax", "sigmoid"):
                writer.writerow(
                    {
                        "track_name": record["track_name"],
                        "model": model_key,
                        **{
                            key: record[model_key]["mask"][key]
                            for key in (
                                "mask_sum_mean",
                                "mask_product_mean",
                                "both_masks_gt_0_5_fraction",
                                "vocal_mask_irm_mae",
                                "instrumental_mask_irm_mae",
                            )
                        },
                        **{
                            key: record[model_key]["waveform"][key]
                            for key in (
                                "vocals_gain_db",
                                "instrumental_gain_db",
                                "combined_gain_db",
                                "mixture_reconstruction_residual_ratio",
                                "source_waveform_cosine",
                            )
                        },
                    }
                )

    band_path = output_dir / "exp04_frequency_band_metrics.csv"
    with band_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "track_name",
            "model",
            "band",
            "vocals_sdr_db",
            "vocal_to_instrumental_bleed_coefficient",
            "vocal_to_instrumental_bleed_energy_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in payload["tracks"]:
            for model_key in ("softmax", "sigmoid"):
                for band, stats in record[model_key]["bands"].items():
                    writer.writerow(
                        {
                            "track_name": record["track_name"],
                            "model": model_key,
                            "band": band,
                            **{key: stats[key] for key in fields[3:]},
                        }
                    )

    markdown = [
        "# Exp04 Frequency and Sigmoid Transition Diagnostics",
        "",
        f"Checkpoint: `{payload['metadata']['checkpoint']}`",
        "",
        "The sigmoid probe uses the exact Exp04 parameters and changes only the mask activation.",
        "",
        "| Model | Median mask sum | Median mask product | Both masks > 0.5 | Median reconstruction residual | All finite |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model_key in ("softmax", "sigmoid"):
        summary = payload["summary"][model_key]
        markdown.append(
            f"| {model_key} | {summary['mask_sum_mean_median']:.6f} | "
            f"{summary['mask_product_mean_median']:.6f} | "
            f"{summary['both_masks_gt_0_5_fraction_median']:.6f} | "
            f"{summary['mixture_reconstruction_residual_ratio_median']:.6f} | "
            f"{summary['all_finite']} |"
        )
    markdown.extend(
        [
            "",
            "## Exp04 softmax frequency bands",
            "",
            "| Band | Median vocal spectral SDR (dB) | Median vocal bleed coefficient | Median bleed energy (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for band, stats in payload["summary"]["softmax"]["frequency_bands"].items():
        markdown.append(
            f"| {band} | {stats['vocals_sdr_db_median']:.4f} | "
            f"{stats['vocal_to_instrumental_bleed_coefficient_median']:.6f} | "
            f"{stats['vocal_to_instrumental_bleed_energy_pct_median']:.6f} |"
        )
    markdown.extend(
        [
            "",
            "## Exp04 softmax time-frequency regions",
            "",
            "Prediction NMAE is normalized by total mixture magnitude within each region.",
            "",
            "| Region | Vocal mask vs IRM MAE | Instrumental mask vs IRM MAE | Vocal prediction NMAE | Instrumental prediction NMAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for region, stats in payload["summary"]["softmax"]["regions"].items():
        markdown.append(
            f"| {region} | {stats['vocal_mask_irm_mae_median']:.6f} | "
            f"{stats['instrumental_mask_irm_mae_median']:.6f} | "
            f"{stats['vocal_prediction_nmae_median']:.6f} | "
            f"{stats['instrumental_prediction_nmae_median']:.6f} |"
        )
    (output_dir / "exp04_diagnostic_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def run_analysis(
    checkpoint_path: Path,
    output_dir: Path,
    overlap_ratio: float = 0.5,
    batch_size: int = 8,
    max_tracks: Optional[int] = None,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax_model, sigmoid_model, metadata = load_transition_models(
        checkpoint_path,
        device,
    )
    _, validation_files = create_train_validation_split()
    track_dirs = validation_files[:max_tracks] if max_tracks is not None else validation_files
    records = []

    for index, track_dir in enumerate(track_dirs, start=1):
        mixture, sample_rate = sf.read(
            track_dir / "mixture.wav", dtype="float32", always_2d=True
        )
        vocals_reference, _ = sf.read(
            track_dir / "vocals.wav", dtype="float32", always_2d=True
        )
        instrumental_reference, _ = sf.read(
            track_dir / "instrumental.wav", dtype="float32", always_2d=True
        )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate for {track_dir}: {sample_rate}")

        record = {"track_name": track_dir.name}
        for model_key, model in (("softmax", softmax_model), ("sigmoid", sigmoid_model)):
            estimates = separate_waveform(
                model,
                mixture,
                device=device,
                sample_rate=sample_rate,
                overlap_ratio=overlap_ratio,
                batch_size=batch_size,
            )
            record[model_key] = {
                "mask": track_mask_diagnostics(
                    model,
                    mixture,
                    vocals_reference,
                    instrumental_reference,
                    device,
                    sample_rate,
                    batch_size,
                ),
                "waveform": waveform_artifact_metrics(
                    mixture,
                    estimates["vocals"],
                    estimates["instrumental"],
                ),
                "bands": _track_spectral_bands(
                    vocals_reference,
                    instrumental_reference,
                    estimates["vocals"],
                    estimates["instrumental"],
                    device,
                    sample_rate,
                ),
            }
        records.append(record)
        print(f"[{index:2d}/{len(track_dirs):2d}] {track_dir.name}", flush=True)

    payload = {
        "metadata": {
            **metadata,
            "device": str(device),
            "num_tracks": len(records),
            "overlap_ratio": overlap_ratio,
            "batch_size": batch_size,
            "frequency_bands": [list(band) for band in DEFAULT_FREQUENCY_BANDS],
            "quantile_aggregation": "active-bin-weighted mean of per-batch quantiles",
        },
        "summary": {
            "softmax": _model_summary(records, "softmax"),
            "sigmoid": _model_summary(records, "sigmoid"),
        },
        "tracks": records,
    }
    _write_reports(Path(output_dir), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Exp04 frequency errors and the softmax-to-sigmoid transition."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/experiments/exp04_asymmetric_leakage/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/exp06"))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()
    run_analysis(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        overlap_ratio=args.overlap,
        batch_size=args.batch_size,
        max_tracks=args.max_tracks,
    )


if __name__ == "__main__":
    main()
