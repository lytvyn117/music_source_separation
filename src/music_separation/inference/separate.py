"""Separate native stereo audio with an Exp10 checkpoint."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from music_separation.audio.overlap_add import OverlapAdd, chunk_batch, chunk_starts
from music_separation.models.bs_roformer import RATE, SEGMENT, Separator, seed_all

CHECKPOINT_SHA = "bcad32866b37f98a19e1add45ae28c59f2740f88870fb0e6cad9a5fa021f1bef"
TENSOR_SHA = {
    "raw": "91174de2ea346d53a0c44eee787d9d7eed7b7e2351662de75564680e74737c15",
    "ema": "72b7c89269a529286a31ec8926da2a909ed6bb522d4b14fcdbe95c70b2d22e61",
}
IDENTITY = "6a808d7c04d8cfd66027048ab62ef0b847c47e503bcc0444ef3e02ae79a565d7"
STEP = 24000
from music_separation.paths import INFERENCE_CONFIG
from music_separation.settings import load_inference_settings

DEFAULT_SETTINGS = load_inference_settings(INFERENCE_CONFIG)
HOP_SAMPLES = DEFAULT_SETTINGS.hop_samples
MIB = 1024**2
FORMATS = {"WAV", "WAVEX", "FLAC"}
BACKENDS = [getattr(SDPBackend, name) for name in DEFAULT_SETTINGS.attention_backends]


def read_input(path):
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    if path.suffix.lower() not in {".wav", ".flac"}:
        raise ValueError("Unsupported or unreadable format. Supply stereo WAV or FLAC.")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        try:
            with sf.SoundFile(handle) as stream:
                if stream.format not in FORMATS:
                    raise ValueError("Unsupported or unreadable format. Supply stereo WAV or FLAC.")
                if stream.channels != 2:
                    raise ValueError(f"Expected exactly two channels. Found {stream.channels}.")
                if stream.samplerate != RATE:
                    raise ValueError(f"Expected {RATE} Hz. Found {stream.samplerate} Hz.")
                audio = stream.read(dtype="float32", always_2d=True)
        except sf.LibsndfileError as exc:
            raise ValueError(f"Unsupported or unreadable audio: {exc}") from exc
    if not np.isfinite(audio).all():
        raise ValueError("Input has non-finite samples.")
    if not len(audio) or not np.any(audio):
        raise ValueError("Empty or silent input. Select a non-silent excerpt.")
    return np.ascontiguousarray(audio.T), digest


def load_model(path, device, weights="raw", expected_sha=CHECKPOINT_SHA, cpu_group_chunk=8):
    if not path.is_file():
        raise ValueError(f"Checkpoint file does not exist: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected_sha.lower():
            raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}. Expected {expected_sha}.")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    identity, step = payload["identity"], payload["step"]
    frozen = digest == CHECKPOINT_SHA
    if frozen and (identity != IDENTITY or step != STEP):
        raise ValueError("Checkpoint run identity or step differs from the freeze.")
    state = payload["raw_model"] if weights == "raw" else payload["ema"]["weights"]
    weight_digest = hashlib.sha256()
    for key, tensor in state.items():
        if tensor.dtype != torch.float32 or not bool(tensor.isfinite().all()):
            raise ValueError(f"Invalid {weights} tensor: {key}")
        weight_digest.update(key.encode())
        weight_digest.update(tensor.numpy().tobytes())
    tensor_sha = weight_digest.hexdigest()
    if frozen and tensor_sha != TENSOR_SHA[weights]:
        raise ValueError(f"{weights} tensor digest differs from the freeze.")
    model = Separator(checkpoint_blocks=True, group_chunk=cpu_group_chunk if device.type == "cpu" else 0)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device), {
        "checkpoint_sha256": digest, "weight_tensor_sha256": tensor_sha,
        "run_identity": identity, "training_step": step, "weights": weights,
    }


def inference_device(settings=DEFAULT_SETTINGS, requested="cuda"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda")
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
    seed_all(settings.seed, deterministic=True)
    torch.set_num_threads(settings.threads)
    device = torch.device(selected, 0) if selected == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


@torch.inference_mode()
def separate(model, mix, device, settings=DEFAULT_SETTINGS):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    starts = chunk_starts(mix.shape[1], SEGMENT, 1 - settings.hop_samples / SEGMENT)
    accumulator = OverlapAdd(2, mix.shape[1], SEGMENT, len(starts) > 1)
    backends = [SDPBackend.MATH] if device.type == "cpu" else [getattr(SDPBackend, name) for name in settings.attention_backends]
    with sdpa_kernel(backends), torch.autocast(device.type, enabled=False):
        for index, start in enumerate(starts, 1):
            wave = torch.from_numpy(chunk_batch(mix, [start], SEGMENT)).to(device)
            vocal, backing = model(wave)
            if not bool((torch.isfinite(vocal) & torch.isfinite(backing)).all()):
                raise ValueError(f"Non-finite model prediction at sample {start}.")
            accumulator.add(vocal[0].cpu().numpy(), start)
            print(f"Processed window {index}/{len(starts)}", flush=True)
    vocals = accumulator.finish()
    instrumental = mix - vocals
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measurements = {
        "chunks": len(starts),
        "inference_seconds": perf_counter() - started,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB if device.type == "cuda" else None,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB if device.type == "cuda" else None,
    }
    return vocals, instrumental, measurements


def verify_audio(mix, vocals, instrumental):
    if any(stem.shape != mix.shape or not np.isfinite(stem).all()
           for stem in (vocals, instrumental)):
        raise ValueError("Unexpected reconstructed shape or non-finite output.")
    residual = float(np.abs(vocals + instrumental - mix).max())
    tolerance = 2e-5 * max(1., float(np.abs(mix).max()))
    if residual > tolerance:
        raise ValueError(f"Mixture residual {residual:.8g} exceeds {tolerance:.8g}.")
    return residual, tolerance


def export(output, vocals, instrumental, manifest):
    # Stage both stems and their manifest together, then publish the new directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".exp10-inference-", dir=output.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        for name, stem in (("vocals.wav", vocals), ("instrumental.wav", instrumental)):
            path = stage / name
            sf.write(path, stem.T, RATE, format="WAV", subtype="FLOAT")
            decoded, rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != RATE or not np.array_equal(decoded, stem.T):
                raise ValueError(f"Export read-back failed for {name}.")
        (stage / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--config", type=Path, default=INFERENCE_CONFIG)
    parser.add_argument("--input-audio", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA)
    args = parser.parse_args()
    source, output, checkpoint = (p.resolve() for p in (args.input_audio, args.output_dir, args.checkpoint))
    try:
        if output.exists():
            raise ValueError(f"Output directory already exists. Choose a new path: {output}")
        settings = load_inference_settings(args.config)
        mix, input_sha = read_input(source)
        device = inference_device(settings, args.device)
        print(f"Using {device} (FP32, {settings.threads} CPU threads)", flush=True)
        model, provenance = load_model(checkpoint, device, args.weights, args.checkpoint_sha256, settings.cpu_group_chunk)
        vocals, instrumental, timings = separate(model, mix, device, settings)
        residual, tolerance = verify_audio(mix, vocals, instrumental)
        peaks = {name: float(np.abs(stem).max())
                 for name, stem in (("input", mix), ("vocals", vocals), ("instrumental", instrumental))}
        warnings = [f"{name} exceeds full scale. Playback or integer conversion can clip."
                    for name, peak in peaks.items() if peak > 1.0]
        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_sha256": input_sha, **provenance,
            "parameters": sum(param.numel() for param in model.parameters()),
            "sample_rate_hz": RATE, "channels": 2, "frames": mix.shape[1],
            "sample_order": "as supplied, without channel mixing or reordering",
            "chunk_samples": SEGMENT, "inference_hop_samples": settings.hop_samples,
            "inference_settings": settings.to_dict(),
            "precision": "FP32", "torch": str(torch.__version__),
            "device": str(device), "requested_device": args.device,
            "effective_attention_backends": ["MATH"] if device.type == "cpu" else list(settings.attention_backends),
            "attention_group_chunk": model.blocks[0].group_chunk,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            **timings,
            "mixture_max_abs": residual, "mixture_tolerance": tolerance,
            "peaks_full_scale": peaks, "warnings": warnings,
            "export": "32-bit IEEE float WAV, no gain change, bit-exact audio read-back",
        }
        export(output, vocals, instrumental, manifest)
        for warning in warnings:
            print(f"Warning: {warning}")
        print(f"Wrote vocals.wav, instrumental.wav and inference_manifest.json to {output}")
        print(f"Inference {timings['inference_seconds']:.3f} s.")
        if device.type == "cuda":
            print(f"Peak GPU allocated {timings['peak_gpu_allocated_mib']:.2f} MiB, "
                  f"reserved {timings['peak_gpu_reserved_mib']:.2f} MiB.")
    except torch.cuda.OutOfMemoryError:
        parser.exit(1, "Inference failed: CUDA out of memory. Free GPU memory and rerun with a fresh output path.\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Inference failed: {exc}\n")


if __name__ == "__main__":
    main()
