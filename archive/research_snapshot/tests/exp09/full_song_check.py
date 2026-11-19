"""Measure the Sony adapter on one full-length synthetic stereo fixture."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import soundfile as sf
import torch

from src.experiments.exp04_exp08_hybrid.inference import read_stereo, write_pair
from src.experiments.exp09_pretrained.inference import SonySeparator
from reports.exp09_zero_shot_preparation.windows_probe import CHUNK


RATE = 44100


def fixture():
    time = np.arange(240 * RATE + 127, dtype=np.float64) / RATE
    envelope = .6 + .4 * np.sin(2 * np.pi * .071 * time)
    left = .22 * np.sin(2 * np.pi * 211 * time) + .038 * np.cos(2 * np.pi * 883 * time)
    right = .048 * np.cos(2 * np.pi * 617 * time) + .018 * np.sin(2 * np.pi * 1217 * time)
    mixture = (envelope * np.stack((left, right))).astype(np.float32)
    mixture[:, 34 * RATE:35 * RATE] = 0
    mixture[0, 82 * RATE + 11] = .57
    mixture[1, 149 * RATE + 19] = -.31
    return mixture


def run():
    mixture = fixture()
    separator = SonySeparator.from_release()
    separator.separate(mixture[:, :2 * RATE])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(2):
        started = perf_counter()
        estimates = separator.separate(mixture)
        torch.cuda.synchronize()
        timings.append(perf_counter() - started)
        if len(timings) == 1:
            first = estimates
        else:
            np.testing.assert_array_equal(estimates, first)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    assert estimates.shape == (2, *mixture.shape) and np.isfinite(estimates).all()
    residual = float(np.max(np.abs(estimates.sum(0) - mixture)))
    assert residual <= 2e-5
    short = mixture[:, :RATE + 173]
    gain_outputs = []
    for scale in (1., .25):
        scaled = np.ascontiguousarray(short * scale)
        chunk = torch.nn.functional.pad(torch.from_numpy(scaled).cuda(), (0, CHUNK - scaled.shape[1]))
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            native = separator.model.inference(chunk.unsqueeze(0))[0, 0, :, :scaled.shape[1]].cpu().numpy()
        estimate = separator.separate(scaled)
        np.testing.assert_array_equal(estimate[0], native)
        np.testing.assert_array_equal(estimate[1], scaled - native)
        gain_outputs.append(estimate)
    gain_error = float(np.max(np.abs(gain_outputs[1] - gain_outputs[0] * .25)))
    side_energy = [float(np.mean(np.square((stem[0] - stem[1]) / 2, dtype=np.float64))) for stem in estimates]
    assert all(value > 0 for value in side_energy)

    with TemporaryDirectory(dir=Path.cwd()) as folder:
        root = Path(folder)
        audio = root / "mixture.wav"
        sf.write(audio, mixture.T, RATE, subtype="FLOAT")
        np.testing.assert_array_equal(read_stereo(audio), mixture, err_msg="Input FLOAT WAV round trip")
        direct = root / "direct"
        write_pair(direct, estimates)
        for name, expected in zip(("vocals", "instrumental"), estimates, strict=True):
            np.testing.assert_array_equal(read_stereo(direct / f"{name}.wav"), expected,
                                          err_msg=f"Output FLOAT WAV round trip, {name}")

        output = root / "cli"
        subprocess.run([sys.executable, "-B", "-m", "src.experiments.exp09_pretrained.inference",
                        str(audio), str(output)], check=True)
        replay = {}
        for name, expected in zip(("vocals", "instrumental"), estimates, strict=True):
            path = output / f"{name}.wav"
            assert sf.info(path).subtype == "FLOAT", sf.info(path)
            actual = read_stereo(path)
            peak_error = float(np.max(np.abs(actual - expected)))
            replay[name] = peak_error
            assert peak_error <= 2e-6, f"CLI {name} peak error {peak_error}"

    return {
        "scope": "full-length synthetic stereo, no separation quality scores",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "sample_rate": RATE, "samples_per_channel": mixture.shape[1],
        "duration_seconds": mixture.shape[1] / RATE, "timings_seconds": timings,
        "median_realtime_factor": (mixture.shape[1] / RATE) / float(np.median(timings)),
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "same_process_exact_repeat": True, "float_wav_roundtrip_exact": True,
        "fresh_cli_peak_difference": replay,
        "mixture_reconstruction_peak": residual, "native_gain_parity_at_1x_and_0_25x": True,
        "native_quarter_scale_equivariance_peak_difference": gain_error,
        "side_mean_squared": side_energy, "output_peaks": [float(np.abs(stem).max()) for stem in estimates],
        "timed_scope": "adapter and GPU transfers, excluding model load, file I/O and scoring",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

