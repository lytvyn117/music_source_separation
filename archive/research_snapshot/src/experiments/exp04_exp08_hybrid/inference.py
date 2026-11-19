"""Frozen full-song Exp04 and Exp08 blend for 44.1 kHz stereo."""
import argparse
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import soundfile as sf
import torch

from src.experiments.exp08_subband.data import BASELINE, BASELINE_SHA256, sha256
from src.experiments.exp08_subband.inference import separate as separate_exp08
from src.experiments.exp08_subband.model import SubbandSeparator
from src.experiments.exp08_subband.train import configure_determinism
from src.inference.separate_track import load_separation_model, separate_waveform

RATE = 44100
EXP08_CHECKPOINT = BASELINE.parents[1] / "exp08_subband/full_20261111_seed42/best.pt"
EXP08_SHA256 = "195ec1db7ed67776be0397bcfccb312031207505ba6c5ce8a70f1633dfbb4ba3"


def _stereo(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != 2 or wave.shape[1] == 0:
        raise ValueError("Expected a nonempty 44.1 kHz stereo waveform shaped [2, samples]")
    if not np.isfinite(wave).all():
        raise ValueError("Stereo waveform must contain finite samples")
    return wave


def read_stereo(path):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE:
        raise ValueError("Expected 44.1 kHz stereo audio")
    return _stereo(wave.T.copy())


def fixed_blend(mixture, vocals04, vocals08):
    mixture = _stereo(mixture)
    first, second = _stereo(vocals04), _stereo(vocals08)
    if first.shape != mixture.shape or second.shape != mixture.shape:
        raise ValueError("Expected estimates aligned to the original stereo sample grid")
    vocal = .5 * first + .5 * second
    instrumental = mixture - vocal
    tolerance = 2e-5 * max(1., float(np.abs(mixture).max()))
    if (not np.isfinite(instrumental).all()
            or np.max(np.abs(vocal + instrumental - mixture)) > tolerance):
        raise FloatingPointError("Hybrid mixture consistency failed")
    return np.stack((vocal, instrumental))


def load_exp04(device):
    if sha256(BASELINE) != BASELINE_SHA256:
        raise ValueError("Exp04 checkpoint fingerprint changed")
    return load_separation_model(BASELINE, device)[0]


def separate_exp04(model, mixture, device):
    return separate_waveform(model, mixture, device=device, segment_length_seconds=6.,
                             overlap_ratio=.5, batch_size=8)


class HybridSeparator:
    def __init__(self, device):
        self.device = torch.device(device)
        self.exp04 = load_exp04(self.device)
        if sha256(EXP08_CHECKPOINT) != EXP08_SHA256:
            raise ValueError("Exp08 checkpoint fingerprint changed")
        # This pinned local checkpoint includes NumPy and Python RNG metadata.
        state = torch.load(EXP08_CHECKPOINT, map_location="cpu", weights_only=False)
        if state["epoch"] != 53:
            raise ValueError("Expected the selected Exp08 epoch-53 checkpoint")
        self.exp08 = SubbandSeparator()
        self.exp08.load_state_dict(state["model"], strict=True)
        self.exp08.to(self.device).eval()

    @torch.inference_mode()
    def separate(self, mixture):
        mixture = _stereo(mixture)
        first = separate_exp04(self.exp04, mixture, self.device)["vocals"]
        second = separate_exp08(self.exp08, mixture, self.device, chunk_samples=441000, batch_size=2)[0]
        return fixed_blend(mixture, first, second)


def write_pair(output, estimates):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    vocal, instrumental = estimates
    vocal, instrumental = _stereo(vocal), _stereo(instrumental)
    if vocal.shape != instrumental.shape:
        raise ValueError("Expected aligned vocal and instrumental outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    files = staging / "vocals.wav", staging / "instrumental.wav"
    try:
        for path, wave in zip(files, (vocal, instrumental), strict=True):
            sf.write(path, wave.T, RATE, subtype="FLOAT")
        staging.rename(output)
    finally:
        if staging.exists():
            for path in files:
                path.unlink(missing_ok=True)
            staging.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="New directory for two float WAV stems")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    waveform = read_stereo(args.input)
    configure_determinism(42)
    model = HybridSeparator(args.device)
    write_pair(args.output, model.separate(waveform))


if __name__ == "__main__":
    main()
