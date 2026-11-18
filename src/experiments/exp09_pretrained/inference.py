"""Isolated full-song inference for Sony Research's released BS-RoFormer variant 1."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend

from reports.exp09_zero_shot_preparation.windows_probe import CHUNK, OVERLAP, load_sony
from src.experiments.exp04_exp08_hybrid.inference import read_stereo, write_pair


class SonySeparator:
    """Return FP32 stereo estimates in vocal, instrumental order."""

    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(device)
        self.window = torch.bartlett_window(2 * OVERLAP, device=self.device)

    @classmethod
    def from_release(cls):
        # The verified loader imports a small subset of the pinned Sony source.
        # Restore the caller's import path and module names after construction.
        module_names = ("utils", "utils.torch_common", "sony_bsroformer", "sony_siso")
        original_modules = {name: sys.modules.get(name) for name in module_names}
        original_path = sys.path[:]
        try:
            model, source, _, _, _, _ = load_sony()
        finally:
            sys.path[:] = original_path
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        # Keep the release's scaled dot-product attention path and pin the
        # backend proven to work for FP32 on this Windows CUDA installation.
        for module in model.modules():
            if isinstance(module, source.Attend):
                module.cuda_config = [SDPBackend.EFFICIENT_ATTENTION]
        return cls(model, "cuda")

    @torch.inference_mode()
    def separate(self, mixture):
        waveform = np.asarray(mixture, dtype=np.float32)
        if waveform.ndim != 2 or waveform.shape[0] != 2 or not waveform.shape[1]:
            raise ValueError("Expected nonempty stereo audio shaped [2, samples]")
        if not np.isfinite(waveform).all():
            raise ValueError("Expected finite audio samples")
        length = waveform.shape[1]
        # Match Sony's chunk count, including its padded final chunk.
        chunks = (max(length - OVERLAP, 0) + OVERLAP - 1) // OVERLAP + 1
        vocals = np.zeros_like(waveform)
        with torch.autocast(self.device.type, enabled=False):
            for index in range(chunks):
                offset = index * OVERLAP
                segment = np.ascontiguousarray(waveform[:, offset:offset + CHUNK])
                samples = segment.shape[1]
                batch = torch.from_numpy(segment).to(self.device)
                batch = F.pad(batch, (0, CHUNK - samples)).unsqueeze(0)
                estimate = self.model.inference(batch)[0, 0]
                if index:
                    estimate[:, :OVERLAP].mul_(self.window[:OVERLAP])
                if index + 1 < chunks:
                    estimate[:, -OVERLAP:].mul_(self.window[OVERLAP:])
                vocals[:, offset:offset + samples] += estimate[:, :samples].cpu().numpy()
        instrumental = waveform - vocals
        if not np.isfinite(vocals).all() or not np.isfinite(instrumental).all():
            raise FloatingPointError("Sony inference produced nonfinite output")
        tolerance = 2e-5 * max(1., float(np.abs(waveform).max()))
        if np.max(np.abs(vocals + instrumental - waveform)) > tolerance:
            raise FloatingPointError("Sony waveform reconstruction exceeds tolerance")
        return np.stack((vocals, instrumental))

    def separate_file(self, input_path, output_path):
        if Path(output_path).exists():
            raise FileExistsError(output_path)
        write_pair(output_path, self.separate(read_stereo(input_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="44.1 kHz stereo mixture")
    parser.add_argument("output", type=Path, help="New directory for two float WAV stems")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    SonySeparator.from_release().separate_file(args.input, args.output)


if __name__ == "__main__":
    main()

