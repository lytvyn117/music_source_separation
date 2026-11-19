import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf

from src.data.splits import create_train_validation_split
from src.evaluation.analyze_exp06 import track_mask_diagnostics
from src.evaluation.mask_diagnostics import waveform_artifact_metrics
from src.inference.separate_track import load_separation_model, separate_waveform


COMPARISON_METRICS = (
    "vocals_global_sdr_db",
    "vocals_median_sdr_db",
    "vocals_global_si_sdr_db",
    "vocals_median_si_sdr_db",
    "vocals_energy_ratio_db",
    "vocals_mae",
    "inst_global_sdr_db",
    "inst_median_sdr_db",
    "inst_global_si_sdr_db",
    "inst_median_si_sdr_db",
    "inst_energy_ratio_db",
    "inst_mae",
    "vocal_to_inst_bleed_coeff",
    "vocal_to_inst_bleed_energy_pct",
    "inst_to_vocal_bleed_coeff",
    "inst_to_vocal_bleed_energy_pct",
)


def _number(value):
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_csv_records(path: Path) -> List[Dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: _number(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def join_metric_records(
    exp04_records: Sequence[Mapping],
    exp06_records: Sequence[Mapping],
) -> List[Dict]:
    exp04 = {str(row["track_name"]): row for row in exp04_records}
    exp06 = {str(row["track_name"]): row for row in exp06_records}
    if set(exp04) != set(exp06):
        only04 = sorted(set(exp04) - set(exp06))
        only06 = sorted(set(exp06) - set(exp04))
        raise ValueError(
            f"Experiment track sets differ: only Exp04={only04}, only Exp06={only06}"
        )

    joined = []
    for track_name in exp04:
        old = exp04[track_name]
        new = exp06[track_name]
        common_metrics = [
            metric for metric in COMPARISON_METRICS if metric in old and metric in new
        ]
        row: Dict = {"track_name": track_name}
        for metric in common_metrics:
            old_value = float(old[metric])
            new_value = float(new[metric])
            row[f"exp04_{metric}"] = old_value
            row[f"exp06_{metric}"] = new_value
            row[f"delta_{metric}"] = new_value - old_value
        joined.append(row)
    return joined


def summarize_deltas(
    rows: Sequence[Mapping],
    metric: str,
    higher_is_better: bool = True,
    tie_tolerance: float = 1e-12,
) -> Dict[str, float]:
    values = np.asarray(
        [float(row[f"delta_{metric}"]) for row in rows],
        dtype=np.float64,
    )
    scored = values if higher_is_better else -values
    return {
        "wins": int(np.sum(scored > tie_tolerance)),
        "regressions": int(np.sum(scored < -tie_tolerance)),
        "ties": int(np.sum(np.abs(scored) <= tie_tolerance)),
        "median_delta": float(np.median(values)),
        "mean_delta": float(np.mean(values)),
        "min_delta": float(np.min(values)),
        "max_delta": float(np.max(values)),
    }


def select_challenging_tracks(rows: Sequence[Mapping], count: int = 6) -> List[Dict]:
    if count < 1:
        return []
    rows = [dict(row) for row in rows]
    selected: List[Dict] = []
    used = set()

    def take(candidates: Sequence[Mapping], reason: str, limit: int) -> None:
        for candidate in candidates:
            name = str(candidate["track_name"])
            if name in used:
                continue
            selected.append({"track_name": name, "reason": reason})
            used.add(name)
            if sum(item["reason"] == reason for item in selected) >= limit:
                break

    overlap = sorted(rows, key=lambda row: (-float(row["overlap_score"]), row["track_name"]))
    improvement = sorted(
        rows,
        key=lambda row: (-float(row["delta_vocals_median_sdr_db"]), row["track_name"]),
    )
    regression = sorted(
        rows,
        key=lambda row: (float(row["delta_vocals_median_sdr_db"]), row["track_name"]),
    )
    take(overlap, "overlap", min(2, count))
    take(improvement, "improvement", min(2, max(0, count - len(selected))))
    take(regression, "regression", min(2, max(0, count - len(selected))))

    if len(selected) < count:
        fallback = sorted(
            rows,
            key=lambda row: (
                -abs(float(row["delta_vocals_median_sdr_db"])),
                -float(row["overlap_score"]),
                row["track_name"],
            ),
        )
        for candidate in fallback:
            if len(selected) >= count:
                break
            name = str(candidate["track_name"])
            if name not in used:
                selected.append({"track_name": name, "reason": "challenge"})
                used.add(name)
    return selected[:count]


def select_excerpt_start(
    vocals_reference: np.ndarray,
    sample_rate: int,
    duration_seconds: int = 20,
    step_seconds: int = 5,
) -> int:
    vocals = np.asarray(vocals_reference, dtype=np.float64)
    window = min(len(vocals), int(duration_seconds * sample_rate))
    if len(vocals) <= window:
        return 0
    step = max(1, int(step_seconds * sample_rate))
    starts = list(range(0, len(vocals) - window + 1, step))
    if starts[-1] != len(vocals) - window:
        starts.append(len(vocals) - window)
    return max(
        starts,
        key=lambda start: float(np.mean(vocals[start : start + window] ** 2)),
    )


def export_audio_bundle(
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    sample_rate: int,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, values in arrays.items():
        path = output_dir / f"{name}.wav"
        sf.write(path, np.asarray(values, dtype=np.float32), sample_rate, subtype="FLOAT")
        paths[name] = path
    return paths


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _attach_overlap_scores(rows: List[Dict], diagnostic_path: Path) -> None:
    diagnostics = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))
    scores = {
        record["track_name"]: record["softmax"]["mask"]["regions"]["overlap"][
            "vocal_mask_irm_mae"
        ]
        for record in diagnostics["tracks"]
    }
    for row in rows:
        row["overlap_score"] = float(scores[row["track_name"]])


def _attach_artifact_metrics(
    rows: List[Dict],
    track_dirs: Mapping[str, Path],
    exp04_checkpoint: Path,
    exp06_checkpoint: Path,
    overlap_ratio: float,
    batch_size: int,
) -> None:
    exp04_model, _, device = load_separation_model(exp04_checkpoint)
    exp06_model, exp06_metadata, _ = load_separation_model(exp06_checkpoint, device=device)
    if exp06_metadata.get("config", {}).get("mask_mode") != "sigmoid":
        raise ValueError("Exp06 checkpoint did not restore mask_mode='sigmoid'")

    for index, row in enumerate(rows, start=1):
        track_dir = track_dirs[row["track_name"]]
        mixture, sample_rate = sf.read(
            track_dir / "mixture.wav", dtype="float32", always_2d=True
        )
        vocals_reference, _ = sf.read(
            track_dir / "vocals.wav", dtype="float32", always_2d=True
        )
        instrumental_reference, _ = sf.read(
            track_dir / "instrumental.wav", dtype="float32", always_2d=True
        )
        for prefix, model in (("exp04", exp04_model), ("exp06", exp06_model)):
            estimates = separate_waveform(
                model,
                mixture,
                device=device,
                sample_rate=sample_rate,
                overlap_ratio=overlap_ratio,
                batch_size=batch_size,
            )
            waveform = waveform_artifact_metrics(
                mixture,
                estimates["vocals"],
                estimates["instrumental"],
            )
            masks = track_mask_diagnostics(
                model,
                mixture,
                vocals_reference,
                instrumental_reference,
                device,
                sample_rate,
                batch_size,
            )
            for key in (
                "vocals_gain_db",
                "instrumental_gain_db",
                "combined_gain_db",
                "mixture_reconstruction_residual_ratio",
                "mixture_reconstruction_residual_db",
                "source_waveform_cosine",
            ):
                row[f"{prefix}_{key}"] = float(waveform[key])
            for key in (
                "mask_sum_mean",
                "mask_sum_std",
                "mask_product_mean",
                "both_masks_gt_0_5_fraction",
                "vocal_mask_irm_mae",
                "instrumental_mask_irm_mae",
            ):
                row[f"{prefix}_{key}"] = float(masks[key])

        for key in (
            "vocals_gain_db",
            "instrumental_gain_db",
            "combined_gain_db",
            "mixture_reconstruction_residual_ratio",
            "mixture_reconstruction_residual_db",
            "source_waveform_cosine",
            "mask_sum_mean",
            "mask_sum_std",
            "mask_product_mean",
            "both_masks_gt_0_5_fraction",
            "vocal_mask_irm_mae",
            "instrumental_mask_irm_mae",
        ):
            row[f"delta_{key}"] = row[f"exp06_{key}"] - row[f"exp04_{key}"]
        print(f"[artifacts {index:2d}/{len(rows):2d}] {row['track_name']}", flush=True)


def _export_selected_audio(
    selected: Sequence[Mapping],
    rows_by_name: Mapping[str, Mapping],
    track_dirs: Mapping[str, Path],
    exp04_checkpoint: Path,
    exp06_checkpoint: Path,
    output_dir: Path,
    overlap_ratio: float,
    batch_size: int,
) -> Dict:
    exp04_model, _, device = load_separation_model(exp04_checkpoint)
    exp06_model, _, _ = load_separation_model(exp06_checkpoint, device=device)
    manifest = {
        "exp04_checkpoint": str(exp04_checkpoint),
        "exp06_checkpoint": str(exp06_checkpoint),
        "duration_seconds": 20,
        "independent_normalization": False,
        "tracks": [],
    }

    for item in selected:
        name = item["track_name"]
        track_dir = track_dirs[name]
        mixture, sample_rate = sf.read(
            track_dir / "mixture.wav", dtype="float32", always_2d=True
        )
        vocals_reference, _ = sf.read(
            track_dir / "vocals.wav", dtype="float32", always_2d=True
        )
        instrumental_reference, _ = sf.read(
            track_dir / "instrumental.wav", dtype="float32", always_2d=True
        )
        exp04 = separate_waveform(
            exp04_model,
            mixture,
            device=device,
            sample_rate=sample_rate,
            overlap_ratio=overlap_ratio,
            batch_size=batch_size,
        )
        exp06 = separate_waveform(
            exp06_model,
            mixture,
            device=device,
            sample_rate=sample_rate,
            overlap_ratio=overlap_ratio,
            batch_size=batch_size,
        )
        start = select_excerpt_start(vocals_reference, sample_rate)
        end = min(len(mixture), start + 20 * sample_rate)
        directory = Path(output_dir) / f"{item['reason']}__{_safe_name(name)}"
        paths = export_audio_bundle(
            directory,
            {
                "mixture": mixture[start:end],
                "ground_truth_vocals": vocals_reference[start:end],
                "ground_truth_instrumental": instrumental_reference[start:end],
                "exp04_vocals": exp04["vocals"][start:end],
                "exp04_instrumental": exp04["instrumental"][start:end],
                "exp06_vocals": exp06["vocals"][start:end],
                "exp06_instrumental": exp06["instrumental"][start:end],
            },
            sample_rate,
        )
        metadata = {
            **dict(item),
            "directory": str(directory.resolve()),
            "start_seconds": start / sample_rate,
            "duration_seconds": (end - start) / sample_rate,
            "metrics": dict(rows_by_name[name]),
            "files": {key: path.name for key, path in paths.items()},
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        manifest["tracks"].append(metadata)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run_comparison(
    exp04_metrics_path: Path,
    exp06_metrics_path: Path,
    diagnostic_path: Path,
    exp04_checkpoint: Path,
    exp06_checkpoint: Path,
    output_csv: Path,
    output_json: Path,
    audio_output_dir: Path,
    overlap_ratio: float = 0.5,
    batch_size: int = 8,
) -> Dict:
    rows = join_metric_records(
        load_csv_records(exp04_metrics_path),
        load_csv_records(exp06_metrics_path),
    )
    _attach_overlap_scores(rows, diagnostic_path)
    _, validation_files = create_train_validation_split()
    track_dirs = {path.name: path for path in validation_files}
    if set(track_dirs) != {row["track_name"] for row in rows}:
        raise ValueError("Evaluation rows do not match the unchanged validation split")

    _attach_artifact_metrics(
        rows,
        track_dirs,
        exp04_checkpoint,
        exp06_checkpoint,
        overlap_ratio,
        batch_size,
    )
    selected = select_challenging_tracks(rows, count=6)
    manifest = _export_selected_audio(
        selected,
        {row["track_name"]: row for row in rows},
        track_dirs,
        exp04_checkpoint,
        exp06_checkpoint,
        audio_output_dir,
        overlap_ratio,
        batch_size,
    )

    summary = {}
    higher_is_better = {
        "vocals_median_sdr_db": True,
        "inst_median_sdr_db": True,
        "vocals_median_si_sdr_db": True,
        "vocal_to_inst_bleed_coeff": False,
        "vocal_to_inst_bleed_energy_pct": False,
        "mixture_reconstruction_residual_ratio": False,
        "mask_product_mean": False,
        "both_masks_gt_0_5_fraction": False,
    }
    for metric, direction in higher_is_better.items():
        summary[metric] = {
            "exp04_median": float(
                np.median([row[f"exp04_{metric}"] for row in rows])
            ),
            "exp06_median": float(
                np.median([row[f"exp06_{metric}"] for row in rows])
            ),
            **summarize_deltas(rows, metric, higher_is_better=direction),
        }

    payload = {
        "exp04_checkpoint": str(exp04_checkpoint),
        "exp06_checkpoint": str(exp06_checkpoint),
        "num_tracks": len(rows),
        "overlap_ratio": overlap_ratio,
        "summary": summary,
        "selected_audio": manifest,
        "tracks": rows,
    }
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Exp06 and Exp04 on matched tracks.")
    parser.add_argument("--exp04-metrics", type=Path, required=True)
    parser.add_argument("--exp06-metrics", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--exp04-checkpoint", type=Path, required=True)
    parser.add_argument("--exp06-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/tables/exp06_vs_exp04_per_track.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/exp06/exp06_vs_exp04_summary.json"),
    )
    parser.add_argument(
        "--audio-output-dir",
        type=Path,
        default=Path("reports/audio/exp06_vs_exp04_20s"),
    )
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_comparison(
        exp04_metrics_path=args.exp04_metrics,
        exp06_metrics_path=args.exp06_metrics,
        diagnostic_path=args.diagnostics,
        exp04_checkpoint=args.exp04_checkpoint,
        exp06_checkpoint=args.exp06_checkpoint,
        output_csv=args.output_csv,
        output_json=args.output_json,
        audio_output_dir=args.audio_output_dir,
        overlap_ratio=args.overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
