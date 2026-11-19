"""Generate synthetic inputs and run the documented CUDA inference command."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
RATE = 44100
CHECKPOINT_SHA256 = "bcad32866b37f98a19e1add45ae28c59f2740f88870fb0e6cad9a5fa021f1bef"


def fixture(frames):
    time = np.arange(frames) / RATE
    left = .12 * np.sin(2 * np.pi * 197 * time) + .08 * np.sin(2 * np.pi * 347 * time)
    right = .10 * np.cos(2 * np.pi * 197 * time) - .06 * np.sin(2 * np.pi * 449 * time)
    mix = np.column_stack((left, right)).astype(np.float32)
    for index in (0, min(frames - 1, 88200), frames - 1):
        mix[index] += [.19, -.29]
    return mix


def launch(source, output, checkpoint):
    return subprocess.run(
        [sys.executable, "-B", str(HERE / "infer.py"), "--input-audio", str(source),
         "--output-dir", str(output), "--checkpoint", str(checkpoint)],
        capture_output=True, text=True,
    )


def check_case(folder, checkpoint, frames, chunks):
    source = folder / f"input {frames}.wav"
    output = folder / f"stems {frames}"
    mix = fixture(frames)
    sf.write(source, mix, RATE, subtype="FLOAT")
    command = launch(source, output, checkpoint)
    assert command.returncode == 0, command.stdout + command.stderr

    manifest = json.loads((output / "inference_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert manifest["weights"] == "raw" and manifest["training_step"] == 24000
    assert manifest["parameters"] == 11662596
    assert manifest["chunk_samples"] == 176400 and manifest["inference_hop_samples"] == 88200
    assert manifest["chunks"] == chunks

    stems = []
    for name in ("vocals", "instrumental"):
        path = output / f"{name}.wav"
        info = sf.info(path)
        assert (info.samplerate, info.channels, info.frames, info.subtype) == (RATE, 2, frames, "FLOAT")
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        assert np.isfinite(audio).all()
        stems.append(audio)
    assert np.any(stems[0]) and np.any(stems[1])
    residual = float(np.abs(stems[0] + stems[1] - mix).max())
    assert residual == manifest["mixture_max_abs"] <= manifest["mixture_tolerance"]
    print(f"{frames} frames, {chunks} chunks, mixture max residual {residual:.9g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    with TemporaryDirectory(prefix="exp10-inference-") as path:
        folder = Path(path)
        check_case(folder, checkpoint, 275217, 3)
        check_case(folder, checkpoint, 3, 1)
        source, output = folder / "mono.wav", folder / "rejected"
        sf.write(source, fixture(32)[:, 0], RATE, subtype="FLOAT")
        bad = launch(source, output, checkpoint)
        assert bad.returncode != 0 and "exactly two channels" in bad.stderr
        assert not output.exists()
        print("Mono input rejected before export.")


if __name__ == "__main__":
    main()
