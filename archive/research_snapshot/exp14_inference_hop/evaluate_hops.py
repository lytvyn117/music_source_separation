"""Read-only Exp10 inference on the frozen development cache."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / "exp10_scratch"
RUN = ROOT / "runs" / "exp10_seed909"
sys.path.insert(0, str(ROOT))

from checkpoint import file_hash, read_json, write_json
from data import load_roster, open_references, verify_cache
from chunking import OverlapAdd, chunk_batch, chunk_starts
from metrics import compute_framewise_metrics
from scratch_model import RATE, SEGMENT, Separator, seed_all
from validation import song_metrics

BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
CHECKPOINT_SHA = "bcad32866b37f98a19e1add45ae28c59f2740f88870fb0e6cad9a5fa021f1bef"
SCHEDULE = [("raw", 2), ("raw", 1), ("ema", 1), ("ema", 2)]
MIB = 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def starts_for(length, hop_seconds):
    return chunk_starts(length, SEGMENT, 1 - hop_seconds * RATE / SEGMENT)


def verify_overlap_add():
    rng = np.random.default_rng(42)
    # Cover the short-song padding branch and a long song with a tail-aligned chunk.
    for length in (SEGMENT - 7, SEGMENT + 6 * RATE + 17):
        x = rng.uniform(-1, 1, (2, length)).astype(np.float32)
        windows = []
        for hop in (2, 1):
            starts = starts_for(length, hop)
            require(starts[0] == 0 and starts[-1] == max(0, length - SEGMENT), "Chunk coverage")
            require(all(b - a == hop * RATE for a, b in zip(starts[:-2], starts[1:-1])),
                    "Unexpected inference stride")
            accumulator = OverlapAdd(2, length, SEGMENT, len(starts) > 1)
            windows.append(accumulator.window)
            for start in starts:
                chunk = chunk_batch(x, [start], SEGMENT)[0]
                require(chunk.shape == (2, SEGMENT), "Unexpected chunk size")
                if length < SEGMENT:
                    require(np.array_equal(chunk[:, :length], x) and not chunk[:, length:].any(),
                            "Unexpected short-song padding")
                accumulator.add(chunk, start)
            np.testing.assert_allclose(accumulator.finish(), x, atol=4e-7, rtol=0)
        np.testing.assert_array_equal(*windows)


def checkpoint_weights(index, provenance, config, archived):
    selected = index["best_overall"]
    path = RUN / "best_overall.pt"
    require(selected["step"] == 24000 and selected["identity"] == provenance["identity"]
            and index["best_qualified"] is None
            and index["qualification_status"] == "no_candidate_qualified"
            and selected["weights"] == "raw" and selected["sha256"] == CHECKPOINT_SHA,
            "The selected checkpoint differs from the frozen Exp14 design")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        require(digest == selected["sha256"] and path.stat().st_size == selected["bytes"],
                "Checkpoint does not match the Exp10 index")
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
    require(checkpoint["identity"] == selected["identity"]
            and checkpoint["step"] == selected["step"] == checkpoint["ema"]["updates"]
            and checkpoint["ema"]["decay"] == config["ema_decay"]
            and checkpoint["history"][-1] == archived, "Checkpoint metadata or validation mismatch")
    weights = {"raw": checkpoint["raw_model"], "ema": checkpoint["ema"]["weights"]}
    model = Separator(checkpoint_blocks=True)
    digests = {}
    for name, state in weights.items():
        digest = hashlib.sha256()
        for key, tensor in state.items():
            require(tensor.dtype == torch.float32 and bool(tensor.isfinite().all()),
                    f"Invalid {name} tensor {key}")
            digest.update(key.encode())
            digest.update(tensor.numpy().tobytes())
        model.load_state_dict(state, strict=True)
        digests[name] = digest.hexdigest()
    require(digests["raw"] != digests["ema"], "Raw and EMA unexpectedly have identical weights")
    return weights, digests


def setup(config, expected):
    seed_all(config["exp09_settings"]["seed"], deterministic=True)
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device("cuda", expected["device_index"])
    torch.cuda.set_device(device)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    runtime = {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "driver": sorted(set(driver)), "device_index": torch.cuda.current_device(),
        "numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
        "threads": torch.get_num_threads(), "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "attention_backends": [backend.name for backend in BACKENDS],
    }
    require(runtime == expected, f"Runtime differs from archived Exp10: {runtime}")
    return device, runtime


def preflight():
    provenance = read_json(RUN / "run.json")
    index = read_json(RUN / "checkpoint_index.json")
    archived = read_json(RUN / "validation_history.json")[-1]
    for relative, digest in provenance["files_sha256"].items():
        require(file_hash(ROOT / relative) == digest, f"Exp10 source changed: {relative}")
    config = read_json(ROOT / "config.json")
    settings = config["validation"]
    require((settings["precision"], settings["segment_frames"], settings["hop_frames"],
             settings["sample_rate"], settings["window"]) ==
            ("FP32", SEGMENT, 2 * RATE, RATE, "positive Hann"), "Unexpected baseline settings")
    torch.set_num_threads(4)
    weights, digests = checkpoint_weights(index, provenance, config, archived)
    roster = load_roster()
    development = roster["development"]
    panel_path = "dev_eval_20261114/frozen_panel.json"
    protocol_path = "split_external_final.json"
    metadata_hashes = {}
    for relative in (panel_path, protocol_path):
        metadata_hashes[relative] = file_hash(ROOT / "preserved_exp09" / relative)
        require(metadata_hashes[relative] == provenance["source_evidence_sha256"][relative],
                f"Frozen metadata changed: {relative}")
    panel = read_json(ROOT / "preserved_exp09" / panel_path)
    protocol = read_json(ROOT / "preserved_exp09" / protocol_path)
    require(roster["holdout"] == "external and sealed" and protocol["external_final_required"]
            and not protocol["test_artist_groups"], "Unexpected final-panel metadata")
    require(list(development) == panel["admitted"], "Development manifest order differs")
    for name in weights:
        require(list(development) == [row["uuid"] for row in archived["validation"][name]["songs"]],
                f"Archived {name} song order differs")
    for key, row in development.items():
        reference = panel["songs"][key]
        require(row["artist_key"] == reference["artist_key"]
                and row["reference_hashes"] == reference["reference_hashes"]
                and reference["frames_min"] <= row["frames"] <= reference["frames_max"]
                and row["frames"] > SEGMENT and row["native_start_frame"] == 0,
                f"Frozen identity, span or reference mismatch for {key}")
    cache = verify_cache(development)
    require(cache == provenance["development_cache"], "Cache differs from the archived run")
    verify_overlap_add()
    device, runtime = setup(config, provenance["runtime"])
    free, total = torch.cuda.mem_get_info(device)
    record = {
        "timestamp_utc": utc_now(), "checkpoint": {
            **index["best_overall"], "path": str(RUN / "best_overall.pt"),
            "weight_tensor_sha256": digests, "qualification_status": index["qualification_status"],
        },
        "metadata_sha256": metadata_hashes, "exp10_files_sha256": provenance["files_sha256"],
        "exp14_design_sha256": file_hash(OUT / "EXP14_DESIGN.md"),
        "runner_sha256": file_hash(Path(__file__)), "cache": cache,
        "development": [{"uuid": k, "title": panel["songs"][k]["title"],
                         "artist": panel["songs"][k]["artist"], "artist_key": row["artist_key"],
                         "frames": row["frames"], "reference_hashes": row["reference_hashes"]}
                        for k, row in development.items()],
        "artist_groups": len({row["artist_key"] for row in development.values()}),
        "final_holdout_access": "External and sealed. Only admitted development UUIDs are readable by this runner.",
        "validation": settings, "arm_order_weights_hop_seconds": SCHEDULE, "runtime": runtime,
        "gpu_memory_preflight_bytes": {"free": free, "total": total},
        "disk_free_bytes": shutil.disk_usage(OUT).free,
        "synthetic_overlap_add": "pass",
    }
    return record, weights, development, archived["validation"], device


@torch.inference_mode()
def predict(model, mix, hop, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    starts = starts_for(mix.shape[1], hop)
    accumulator = OverlapAdd(2, mix.shape[1], SEGMENT, len(starts) > 1)
    for start in starts:
        wave = torch.from_numpy(chunk_batch(mix, [start], SEGMENT))
        pred, _ = model(wave.to(device))
        accumulator.add(pred[0].cpu().numpy(), start)
    pred_v = accumulator.finish()
    pred_i = mix - pred_v
    torch.cuda.synchronize(device)
    measurements = {"chunks": len(starts), "inference_seconds": perf_counter() - started,
                    "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
                    "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB}
    return pred_v, pred_i, measurements


def evaluate_arm(model, name, hop, development, archived, device):
    started = utc_now()
    torch.cuda.empty_cache()
    rows = []
    parity = {}
    with sdpa_kernel(BACKENDS), torch.autocast(device.type, enabled=False), torch.inference_mode():
        vocals, _ = model(torch.zeros((1, 2, SEGMENT), dtype=torch.float32, device=device))
        require(bool(vocals.isfinite().all()), "Nonfinite warmup")
        del vocals, _
        torch.cuda.synchronize(device)
        for key, old in zip(development, archived, strict=True):
            with open_references(development, key) as refs:
                vocals, inst = (np.ascontiguousarray(ref.T) for ref in refs)
            mix = vocals + inst
            pred_v, pred_i, timings = predict(model, mix, hop, device)
            scores = song_metrics(mix, vocals, inst, pred_v, pred_i)
            vocal_si_sdr = compute_framewise_metrics(vocals, pred_v)["median_si_sdr_db"]
            row = {"uuid": key, "artist_key": development[key]["artist_key"], **timings, **scores,
                   "vocal_si_sdr_db": vocal_si_sdr,
                   "abs_vocal_bleed_pct": 100 * scores["abs_vocal_bleed_coeff"],
                   "mixture_tolerance": 2e-5 * max(1., float(np.abs(mix).max()))}
            require(all(value is not None and math.isfinite(value) for k, value in row.items()
                        if k not in {"uuid", "artist_key"}), f"Invalid {name} {hop}s metric for {key}")
            if hop == 2:
                for field, value in old.items():
                    if isinstance(value, float):
                        error = abs(row[field] - value)
                        parity[field] = max(parity.get(field, 0.), error)
                        require(error <= 1e-6, f"Baseline parity failed for {name} {key} {field}: {error}")
                    else:
                        require(row[field] == value, f"Baseline parity failed for {name} {key} {field}")
            rows.append(row)
            print(f"{name.upper()} {hop}s {len(rows):2}/{len(development)} {key} "
                  f"{timings['chunks']} chunks {timings['inference_seconds']:.3f}s", flush=True)
            del vocals, inst, mix, pred_v, pred_i
    return {"weights": name, "hop_seconds": hop, "started_utc": started, "ended_utc": utc_now(),
            "baseline_parity_max_abs": parity if hop == 2 else None, "songs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    args = parser.parse_args()
    destinations = {(name, hop): OUT / f"arm_{name}_{hop}s.json" for name, hop in SCHEDULE}
    if args.mode == "run":
        require(not any(path.exists() for path in destinations.values()), "Arm outputs already exist")
    record, weights, development, archived, device = preflight()
    write_json(OUT / f"{args.mode}_provenance.json", record)
    print(f"{args.mode}: SHA-256 verified, both states finite and strict-loadable, "
          f"{len(development)} songs / {record['artist_groups']} artists, "
          f"{record['gpu_memory_preflight_bytes']['free'] / MIB:.0f} MiB free GPU, "
          f"{record['disk_free_bytes'] / 1024**3:.2f} GiB free disk", flush=True)
    if args.mode == "preflight":
        return
    model = Separator(checkpoint_blocks=True).to(device).eval().requires_grad_(False)
    for name, hop in SCHEDULE:
        model.load_state_dict(weights[name], strict=True)
        print(f"Starting {name.upper()} {hop}s", flush=True)
        arm = evaluate_arm(model, name, hop, development, archived[name]["songs"], device)
        write_json(destinations[name, hop], arm)
        print(f"Saved {destinations[name, hop].name}", flush=True)


if __name__ == "__main__":
    main()
