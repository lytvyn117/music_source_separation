"""Verify the PowerShell demo with original synthetic audio in a temporary directory."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
RATE = 44100
SHA = "bcad32866b37f98a19e1add45ae28c59f2740f88870fb0e6cad9a5fa021f1bef"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fixture(frames):
    t = np.arange(frames) / RATE
    left = .12 * np.sin(2 * np.pi * 197 * t) + .08 * np.sin(2 * np.pi * 347 * t)
    right = .10 * np.cos(2 * np.pi * 197 * t) - .06 * np.sin(2 * np.pi * 449 * t)
    mix = np.column_stack((left, right)).astype(np.float32)
    for index in (0, min(frames - 1, 88200), frames - 1):
        mix[index] += [.19, -.29]
    return mix


def launch(checkpoint, source, output):
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(HERE / "run_demo.ps1"), "-InputAudio", str(source), "-OutputDir", str(output),
         "-Checkpoint", str(checkpoint), "-Python", sys.executable],
        capture_output=True, text=True,
    )


def success(checkpoint, root, name, frames):
    mix = fixture(frames)
    source, output = root / f"{name} input.wav", root / f"{name} outputs"
    sf.write(source, mix, RATE, subtype="FLOAT")
    original = digest(source)
    process = launch(checkpoint, source, output)
    assert process.returncode == 0, process.stdout + process.stderr
    assert digest(source) == original
    manifest = json.loads((output / "demo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == SHA and manifest["weights"] == "raw"
    assert manifest["training_step"] == 24000 and manifest["inference_hop_samples"] == 88200
    assert manifest["inference_seconds"] > 0 and manifest["peak_gpu_allocated_mib"] > 0
    stems = []
    for filename in ("vocals.wav", "instrumental.wav"):
        path = output / filename
        assert sf.info(path).subtype == "FLOAT"
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        assert audio.shape == mix.shape and sample_rate == RATE and np.isfinite(audio).all()
        stems.append(audio)
    residual = float(np.abs(stems[0] + stems[1] - mix).max())
    assert residual <= 2e-5 * max(1.0, float(np.abs(mix).max()))
    assert residual == manifest["mixture_max_abs"]
    return manifest, stems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional new JSON evidence file")
    parser.add_argument("--repo-root", type=Path, help="Compare waveforms with the original Exp14 predictor")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    assert digest(checkpoint) == SHA
    with TemporaryDirectory(prefix="exp10-demo-") as directory:
        root = Path(directory)
        frames = 275217
        multi, stems = success(checkpoint, root, "multi chunk", frames)
        short, _ = success(checkpoint, root, "three frames", 3)
        assert multi["chunks"] == 3 and short["chunks"] == 1
        rejection = {}
        for name, audio, rate, expected in (
            ("mono", fixture(25)[:, 0], RATE, "exactly two channels"),
            ("wrong rate", fixture(25), 48000, "44100 Hz"),
            ("silent", np.zeros((25, 2)), RATE, "silent input"),
            ("nonfinite", np.full((25, 2), np.nan), RATE, "non-finite"),
            ("unsupported", None, RATE, "unsupported or unreadable"),
            ("missing", None, RATE, "input file does not exist"),
        ):
            source, output = root / f"{name}.wav", root / f"{name} result"
            if audio is not None:
                sf.write(source, audio, rate, subtype="FLOAT")
            if name == "unsupported":
                source.write_text("this file contains text", encoding="utf-8")
            result = launch(checkpoint, source, output)
            assert result.returncode != 0 and expected.lower() in result.stderr.lower(), (
                name, result.stdout, result.stderr)
            assert not output.exists()
            rejection[name] = expected
        archived_deltas = None
        if args.repo_root is not None:
            from reproduce_development import load_evaluator

            evaluator = load_evaluator(args.repo_root.resolve())
            torch = evaluator.torch
            run = evaluator.RUN
            cfg = evaluator.read_json(evaluator.ROOT / "config.json")
            provenance = evaluator.read_json(run / "run.json")
            device, _ = evaluator.setup(cfg, provenance["runtime"])
            with checkpoint.open("rb") as handle:
                weights = torch.load(handle, map_location="cpu", weights_only=True)["raw_model"]
            model = evaluator.Separator(checkpoint_blocks=True).to(device).eval().requires_grad_(False)
            model.load_state_dict(weights, strict=True)
            with evaluator.sdpa_kernel(evaluator.BACKENDS), torch.autocast("cuda", enabled=False):
                ref_v, ref_i, _ = evaluator.predict(
                    model, np.ascontiguousarray(fixture(frames).T), 2, device)
            archived_deltas = [float(np.abs(a - b.T).max()) for a, b in zip(stems, (ref_v, ref_i))]
            assert archived_deltas == [0.0, 0.0], archived_deltas
    assert digest(checkpoint) == SHA
    report = {"fixture": "original synthetic unequal stereo tones and impulses",
              "multi_chunk": multi, "three_frames": short, "rejected_inputs": rejection,
              "archived_predictor_stem_max_abs_deltas": archived_deltas}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
