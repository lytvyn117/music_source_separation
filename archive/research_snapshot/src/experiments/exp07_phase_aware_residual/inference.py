import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config import SAMPLE_RATE
from src.audio.stft import compute_istft, compute_stft

from .chunking import OverlapAdd, chunk_batch, chunk_starts
from .complex_ops import make_vocal_ablation_stfts, waveform_nrmse_pct
from .model import PhaseAwareResidualSeparator
from .oracle_analysis import CHUNK_SAMPLES


@dataclass(frozen=True)
class WaveformPair:
    vocals: np.ndarray
    instrumental: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    variants: dict[str, WaveformPair]
    chunk_starts: list[int]

    def __getitem__(self, name):
        return self.variants[name]


@dataclass(frozen=True)
class ConsistencyReport:
    global_nrmse_pct: float
    global_max_abs: float
    max_seam_window_abs: float


@torch.inference_mode()
def separate_waveform(model, mixture, overlap_ratio=0.5, batch_size=4, include_ablations=False):
    mix = np.asarray(mixture, dtype=np.float32)
    if mix.ndim != 2 or 2 not in mix.shape or not np.isfinite(mix).all() or batch_size < 1:
        raise ValueError("Finite stereo audio and a positive batch size are required")
    samples_first = mix.shape[-1] == 2
    if samples_first:
        mix = mix.T
    length = mix.shape[-1]
    starts = chunk_starts(length, CHUNK_SAMPLES, overlap_ratio)
    names = ("full", "magnitude_only", "phase_only") if include_ablations else ("full",)
    accumulators = {name: OverlapAdd(4, length, CHUNK_SAMPLES, len(starts) > 1 and overlap_ratio > 0)
                    for name in names}
    device = next(model.parameters()).device
    model.eval()
    for offset in range(0, len(starts), batch_size):
        subset = starts[offset:offset + batch_size]
        batch = torch.from_numpy(chunk_batch(mix, subset, CHUNK_SAMPLES)).to(device)
        with torch.autocast(device.type, enabled=False):
            x = compute_stft(batch)
            prediction = model.forward_stft(x)
            variants = (make_vocal_ablation_stfts(prediction.initial_vocal_stft, prediction.vocals_stft)
                        if include_ablations else {"full": prediction.vocals_stft})
            for name, vocal in variants.items():
                waves = compute_istft(torch.cat((vocal, x - vocal), dim=1), length=CHUNK_SAMPLES)
                for wave, start in zip(waves.cpu().numpy(), subset):
                    accumulators[name].add(wave, start)
    results = {}
    for name, accumulator in accumulators.items():
        vocals, instrumental = np.split(accumulator.finish(), 2, axis=0)
        results[name] = WaveformPair(vocals.T, instrumental.T) if samples_first else WaveformPair(vocals, instrumental)
    return InferenceResult(results, starts)


def measure_full_song_consistency(mixture, result, starts, seam_window=4096):
    x = np.asarray(mixture)
    if result.vocals.shape != x.shape or result.instrumental.shape != x.shape:
        raise ValueError("Both output shapes must match the original stereo waveform")
    summed = result.vocals + result.instrumental
    residual = np.abs(summed - x)
    peak = float(residual.max())
    if x.shape[-1] == 2:
        residual = residual.T
    left, right = seam_window // 2, (seam_window + 1) // 2
    seam = max((float(residual[:, max(0, s - left):s + right].max()) for s in starts[1:]), default=0.0)
    return ConsistencyReport(waveform_nrmse_pct(summed, x), peak, seam)


def main():
    parser = argparse.ArgumentParser(description="Separate a stereo WAV with an Exp07 checkpoint")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    waveform, rate = sf.read(args.input, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or waveform.shape[1] != 2:
        parser.error("Expected 44.1 kHz stereo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != "exp07_phase_aware_residual":
        parser.error("Expected a local Exp07 checkpoint")
    model = PhaseAwareResidualSeparator.from_exp04(checkpoint["exp04_path"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pair = separate_waveform(model, waveform)["full"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sf.write(args.output_dir / "vocals.wav", pair.vocals, rate, subtype="FLOAT")
    sf.write(args.output_dir / "instrumental.wav", pair.instrumental, rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
