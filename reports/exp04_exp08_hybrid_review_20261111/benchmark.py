"""Compare the two frozen inference paths on three development-validation songs."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/exp04_exp08_hybrid_20261111"
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

from src.experiments.exp04_exp08_hybrid.inference import (
    HybridSeparator, load_exp04, read_stereo, separate_exp04,
)
from src.experiments.exp08_subband.data import PROCESSED_DIR
from src.experiments.exp08_subband.train import configure_determinism


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gain(reference, estimate):
    signal = np.einsum("ct,ct->", estimate, estimate, dtype=np.float64)
    target = np.einsum("ct,ct->", reference, reference, dtype=np.float64)
    return float(10 * np.log10((signal + 1e-8) / (target + 1e-8)))


def validate(track, mixture, estimates, arm, original):
    if estimates.shape != (2, *mixture.shape) or not np.isfinite(estimates).all():
        raise ValueError("Full-song sample count or stereo layout changed")
    sample = slice(track["start_sample"], track["start_sample"] + track["samples"])
    differences, gains = [], []
    for stem, estimate in zip(("vocals", "instrumental"), estimates, strict=True):
        clip = read_stereo(SOURCE / "listen" / track["clip"] / f"{arm}_{stem}.wav")
        error = float(np.abs(estimate[:, sample] - clip).max())
        differences.append(error)
        ref = read_stereo(PROCESSED_DIR / "train" / track["track"] / f"{stem}.wav")
        gains.append(gain(ref, estimate))
    if max(differences) > 2e-6 or max(abs(a-b) for a, b in zip(gains, original["rms_gain_db"])) > 1e-4:
        raise ValueError("Wrapper differs from the frozen development outputs")
    consistency = float(np.abs(estimates.sum(0) - mixture).max())
    if arm == "hybrid" and consistency > 2e-5 * max(1., float(np.abs(mixture).max())):
        raise FloatingPointError("Hybrid reconstruction exceeded its tolerance")
    return {"full_song_samples": mixture.shape[1], "stereo": True, "sample_rate": 44100,
            "saved_excerpt_peak_differences": differences, "rms_gain_db": gains,
            "mixture_consistency_peak": consistency,
            "output_peaks": np.max(np.abs(estimates), axis=(1, 2)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("exp04", "hybrid"))
    args = parser.parse_args()
    output = HERE / f"benchmark_{args.arm}.json"
    if output.exists():
        raise FileExistsError(output)
    configure_determinism(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VRAM benchmark")
    results = {r["track"]: r for r in read_json(SOURCE / "results.json")["tracks"]}
    clips = read_json(SOURCE / "listening_manifest.json")
    clips.sort(key=lambda r: (results[r["track"]]["samples"], r["track"]))
    clips = [clips[i] for i in (0, (len(clips)-1)//2, len(clips)-1)]
    load_started = time.perf_counter()
    if args.arm == "exp04":
        model = load_exp04("cuda")

        def infer(wave):
            row = separate_exp04(model, wave, "cuda")
            return np.stack((row["vocals"], row["instrumental"]))
    else:
        model = HybridSeparator("cuda")
        infer = model.separate
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started

    warm = read_stereo(SOURCE / "listen" / clips[0]["clip"] / "mixture.wav")
    infer(warm[:, :441000])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for clip in clips:
        waveform = read_stereo(PROCESSED_DIR / "train" / clip["track"] / "mixture.wav")
        timings = []
        for _ in range(2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            estimates = infer(waveform)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t)
        checks = validate(clip, waveform, estimates, args.arm, results[clip["track"]]["scores"][args.arm])
        rows.append({"track": clip["track"], "audio_seconds": waveform.shape[1] / 44100,
                     "timed_seconds": timings, **checks})
        print(args.arm, clip["track"], timings, flush=True)
    times = [sum(row["timed_seconds"][repeat] for row in rows) for repeat in range(2)]
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"arm": args.arm, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "model_load_seconds": load_s, "timed_total_seconds": times,
                   "audio_seconds": sum(row["audio_seconds"] for row in rows),
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                   "tracks": rows}, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
