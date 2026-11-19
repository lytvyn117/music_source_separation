import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_ROOT = Path(r"C:\Users\YOUR_USER\music_source_separation")
STAGING_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(STAGING_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
import src
if str(STAGING_ROOT / "src") not in src.__path__:
    src.__path__.append(str(STAGING_ROOT / "src"))

import numpy as np
import soundfile as sf
import torch

from config import (
    CHECKPOINTS_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
    SEGMENT_LENGTH_SECONDS,
)
from src.audio.separation import reconstruct_source
from src.audio.stft import compute_magnitude, compute_stft
from src.evaluation.metrics import (
    compute_leakage_diagnostics,
    evaluate_stem_pair,
)
from src.inference.separate_track import (
    load_separation_model,
    separate_waveform,
)


@torch.no_grad()
def compute_oracle_irm_separation(
    mixture: np.ndarray,
    vocals_ref: np.ndarray,
    inst_ref: np.ndarray,
    device: torch.device,
    sample_rate: int = SAMPLE_RATE,
    segment_length_seconds: float = SEGMENT_LENGTH_SECONDS,
) -> Dict[str, np.ndarray]:
    """
    Compute the Oracle Ideal Ratio Mask (IRM) upper bound using ground-truth magnitudes
    and mixture phase: M_v = |V| / (|V| + |I| + eps), M_i = 1 - M_v.
    """
    chunk_samples = int(round(segment_length_seconds * sample_rate))
    total_samples = mixture.shape[0]
    voc_out = np.zeros_like(mixture, dtype=np.float32)
    inst_out = np.zeros_like(mixture, dtype=np.float32)

    for start in range(0, total_samples, chunk_samples):
        end = min(total_samples, start + chunk_samples)
        length = end - start
        mix_c = torch.from_numpy(mixture[start:end].T.copy()).unsqueeze(0).to(device)
        voc_c = torch.from_numpy(vocals_ref[start:end].T.copy()).unsqueeze(0).to(device)
        inst_c = torch.from_numpy(inst_ref[start:end].T.copy()).unsqueeze(0).to(device)

        X = compute_stft(mix_c)
        V_mag = compute_magnitude(compute_stft(voc_c))
        I_mag = compute_magnitude(compute_stft(inst_c))

        irm_v = V_mag / (V_mag + I_mag + 1e-8)
        irm_i = 1.0 - irm_v

        v_rec = reconstruct_source(X, irm_v, length=length)[0].T.cpu().numpy()
        i_rec = reconstruct_source(X, irm_i, length=length)[0].T.cpu().numpy()

        voc_out[start:end] = v_rec
        inst_out[start:end] = i_rec

    return {"vocals": voc_out, "instrumental": inst_out}


def evaluate_track(
    track_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    segment_length_seconds: float = SEGMENT_LENGTH_SECONDS,
    overlap_ratio: float = 0.5,
    batch_size: int = 8,
    max_duration_seconds: Optional[float] = None,
    include_oracle_irm: bool = False,
    export_audio_dir: Optional[Path] = None,
) -> Dict:
    """
    Evaluate model separation on a single processed MUSDB18 track directory.
    """
    t0 = time.perf_counter()

    info = sf.info(track_dir / "mixture.wav")
    sr = info.samplerate
    frames_to_read = (
        int(round(max_duration_seconds * sr))
        if max_duration_seconds is not None
        else -1
    )

    mixture, _ = sf.read(
        track_dir / "mixture.wav",
        frames=frames_to_read,
        dtype="float32",
        always_2d=True,
    )
    vocals_ref, _ = sf.read(
        track_dir / "vocals.wav",
        frames=frames_to_read,
        dtype="float32",
        always_2d=True,
    )
    inst_ref, _ = sf.read(
        track_dir / "instrumental.wav",
        frames=frames_to_read,
        dtype="float32",
        always_2d=True,
    )

    preds = separate_waveform(
        model=model,
        mixture=mixture,
        device=device,
        sample_rate=sr,
        segment_length_seconds=segment_length_seconds,
        overlap_ratio=overlap_ratio,
        batch_size=batch_size,
    )

    vocals_metrics = evaluate_stem_pair(
        reference=vocals_ref,
        estimate=preds["vocals"],
        mixture=mixture,
        sample_rate=sr,
    )
    inst_metrics = evaluate_stem_pair(
        reference=inst_ref,
        estimate=preds["instrumental"],
        mixture=mixture,
        sample_rate=sr,
    )
    leakage = compute_leakage_diagnostics(
        vocals_ref=vocals_ref,
        inst_ref=inst_ref,
        vocals_est=preds["vocals"],
        inst_est=preds["instrumental"],
    )

    elapsed = time.perf_counter() - t0
    duration_sec = mixture.shape[0] / sr

    record = {
        "track_name": track_dir.name,
        "duration_sec": round(duration_sec, 2),
        "inference_sec": round(elapsed, 3),
        "realtime_factor": round(duration_sec / max(elapsed, 1e-6), 1),
        # Vocals
        "vocals_global_sdr_db": vocals_metrics["global_sdr_db"],
        "vocals_median_sdr_db": vocals_metrics["median_sdr_db"],
        "vocals_global_si_sdr_db": vocals_metrics["global_si_sdr_db"],
        "vocals_median_si_sdr_db": vocals_metrics["median_si_sdr_db"],
        "vocals_sdr_imp_db": vocals_metrics["sdr_improvement_db"],
        "vocals_energy_ratio_db": vocals_metrics["energy_ratio_db"],
        "vocals_mae": vocals_metrics["waveform_mae"],
        # Instrumental
        "inst_global_sdr_db": inst_metrics["global_sdr_db"],
        "inst_median_sdr_db": inst_metrics["median_sdr_db"],
        "inst_global_si_sdr_db": inst_metrics["global_si_sdr_db"],
        "inst_median_si_sdr_db": inst_metrics["median_si_sdr_db"],
        "inst_sdr_imp_db": inst_metrics["sdr_improvement_db"],
        "inst_energy_ratio_db": inst_metrics["energy_ratio_db"],
        "inst_mae": inst_metrics["waveform_mae"],
        # Leakage
        "vocal_to_inst_bleed_coeff": leakage["vocal_to_inst_bleed_coeff"],
        "vocal_to_inst_bleed_energy_pct": leakage["vocal_to_inst_bleed_energy_pct"],
        "inst_to_vocal_bleed_coeff": leakage["inst_to_vocal_bleed_coeff"],
        "inst_to_vocal_bleed_energy_pct": leakage["inst_to_vocal_bleed_energy_pct"],
    }

    if include_oracle_irm:
        irm_preds = compute_oracle_irm_separation(
            mixture=mixture,
            vocals_ref=vocals_ref,
            inst_ref=inst_ref,
            device=device,
            sample_rate=sr,
            segment_length_seconds=segment_length_seconds,
        )
        irm_voc = evaluate_stem_pair(vocals_ref, irm_preds["vocals"], mixture, sr)
        irm_inst = evaluate_stem_pair(inst_ref, irm_preds["instrumental"], mixture, sr)
        record["oracle_irm_vocals_median_sdr_db"] = irm_voc["median_sdr_db"]
        record["oracle_irm_inst_median_sdr_db"] = irm_inst["median_sdr_db"]

    if export_audio_dir is not None:
        track_out = export_audio_dir / track_dir.name
        track_out.mkdir(parents=True, exist_ok=True)
        # Save a representative 15-second excerpt from an active vocal section (or start if shorter)
        excerpt_len = min(mixture.shape[0], 15 * sr)
        # Find 15s window with highest vocal RMS energy so the excerpt clearly contains singing
        best_start = 0
        if mixture.shape[0] > excerpt_len:
            step = 5 * sr
            best_rms = -1.0
            for s in range(0, mixture.shape[0] - excerpt_len, step):
                rms = float(np.sqrt(np.mean(vocals_ref[s : s + excerpt_len] ** 2)))
                if rms > best_rms:
                    best_rms = rms
                    best_start = s
        s, e = best_start, best_start + excerpt_len
        sf.write(track_out / "mixture_excerpt.wav", mixture[s:e], sr, subtype="FLOAT")
        sf.write(track_out / "vocals_predicted.wav", preds["vocals"][s:e], sr, subtype="FLOAT")
        sf.write(track_out / "vocals_ground_truth.wav", vocals_ref[s:e], sr, subtype="FLOAT")
        sf.write(track_out / "instrumental_predicted.wav", preds["instrumental"][s:e], sr, subtype="FLOAT")
        sf.write(track_out / "instrumental_ground_truth.wav", inst_ref[s:e], sr, subtype="FLOAT")
        record["exported_excerpt_start_sec"] = round(s / sr, 2)

    return record


def summarize_records(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate mean, median, std, min, and max across evaluated tracks for all numeric metrics.
    """
    if not records:
        return {}

    numeric_keys = [
        k
        for k, v in records[0].items()
        if isinstance(v, (int, float)) and k not in ("exported_excerpt_start_sec",)
    ]

    summary = {}
    for k in numeric_keys:
        vals = np.array([r[k] for r in records if np.isfinite(r[k])], dtype=np.float64)
        if len(vals) == 0:
            continue
        summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
    return summary


def run_evaluation(
    split_dir: Path = PROCESSED_DIR / "test",
    checkpoint_path: Path = CHECKPOINTS_DIR / "best_model.pt",
    output_dir: Path = Path("reports"),
    audio_export_dir: Optional[Path] = None,
    max_tracks: Optional[int] = None,
    max_duration_seconds: Optional[float] = None,
    overlap_ratio: float = 0.5,
    batch_size: int = 8,
    include_oracle_irm: bool = False,
    export_audio_count: int = 3,
    tag: str = "baseline_test",
    use_val_split: bool = False,
) -> Dict:
    """
    Run reproducible evaluation across MUSDB18 tracks, saving CSV, JSON, and exported audio examples.
    """
    model, metadata, device = load_separation_model(checkpoint_path=checkpoint_path)

    if use_val_split:
        from src.data.splits import create_train_validation_split

        _, val_files = create_train_validation_split()
        track_dirs = val_files
        source_label = "15-song validation split"
    else:
        track_dirs = sorted(p for p in Path(split_dir).iterdir() if p.is_dir())
        source_label = str(split_dir)

    if max_tracks is not None:
        track_dirs = track_dirs[:max_tracks]

    print(
        f"Evaluating {len(track_dirs)} tracks from {source_label} on {device} "
        f"(overlap={overlap_ratio}, batch_size={batch_size})...",
        flush=True,
    )

    records = []
    for idx, track_dir in enumerate(track_dirs, start=1):
        export_dir = (
            audio_export_dir
            if (audio_export_dir is not None and idx <= export_audio_count)
            else None
        )
        rec = evaluate_track(
            track_dir=track_dir,
            model=model,
            device=device,
            overlap_ratio=overlap_ratio,
            batch_size=batch_size,
            max_duration_seconds=max_duration_seconds,
            include_oracle_irm=include_oracle_irm,
            export_audio_dir=export_dir,
        )
        records.append(rec)
        print(
            f"[{idx:2d}/{len(track_dirs):2d}] {rec['track_name'][:32]:32s} | "
            f"Voc MedSDR: {rec['vocals_median_sdr_db']:6.2f} dB (Glob: {rec['vocals_global_sdr_db']:6.2f}) | "
            f"Inst MedSDR: {rec['inst_median_sdr_db']:6.2f} dB (Glob: {rec['inst_global_sdr_db']:6.2f}) | "
            f"Bleed V->I: {rec['vocal_to_inst_bleed_coeff']*100:5.1f}% | {rec['inference_sec']:.2f}s",
            flush=True,
        )

    summary = summarize_records(records)

    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / f"{tag}_metrics.csv"
    json_path = tables_dir / f"{tag}_summary.json"

    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    payload = {
        "tag": tag,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "split_dir": str(split_dir),
        "num_tracks": len(records),
        "overlap_ratio": overlap_ratio,
        "summary": summary,
        "tracks": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved per-track CSV to:  {csv_path}", flush=True)
    print(f"Saved JSON summary to:   {json_path}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net source separation model on MUSDB18."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROCESSED_DIR / "test",
        help="Path to processed split directory (default: data/processed/test).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "best_model.pt",
        help="Path to model checkpoint (default: models/checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for CSV/JSON reports.",
    )
    parser.add_argument(
        "--audio-export-dir",
        type=Path,
        default=None,
        help="Optional directory to export separated audio excerpts.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit number of evaluated tracks (e.g., 2 for a quick smoke test).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional max duration in seconds per track.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Chunk overlap ratio in [0.0, 0.9] (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="GPU chunk batch size (default: 8).",
    )
    parser.add_argument(
        "--oracle-irm",
        action="store_true",
        help="Also compute Oracle Ideal Ratio Mask (IRM) upper-bound SDR.",
    )
    parser.add_argument(
        "--val-split",
        action="store_true",
        help="Evaluate on the 15-song training validation split rather than the test set.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline_test",
        help="Filename prefix for saved CSV and JSON reports.",
    )

    args = parser.parse_args()
    run_evaluation(
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        audio_export_dir=args.audio_export_dir,
        max_tracks=args.max_tracks,
        max_duration_seconds=args.max_duration,
        overlap_ratio=args.overlap,
        batch_size=args.batch_size,
        include_oracle_irm=args.oracle_irm,
        tag=args.tag,
        use_val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
