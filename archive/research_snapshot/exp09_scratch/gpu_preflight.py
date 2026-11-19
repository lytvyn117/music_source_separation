"""Bounded real-data probes. Selects training UUIDs only."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import hashlib
import json
import platform
import statistics
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from audit import write_json
from loader import Shard
from scratch_model import SEGMENT, Separator, crop_for, seed_all
from scratch_train import ExactRun, calibrate, model_digest

GIB = 1024**3
ROOT = Path(__file__).resolve().parent
SEED = 909
OVERFIT_STEPS = 80
BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def device_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {"free_gib": free / GIB, "total_gib": total / GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}


def batch(reader, lengths, index, count=1):
    mixtures, vocals = [], []
    for j in range(count):
        key, start = crop_for(lengths, SEGMENT, SEED, index + j)
        x, v, _ = reader[key](start, SEGMENT)
        mixtures.append(torch.from_numpy(x.T.copy()))
        vocals.append(torch.from_numpy(v.T.copy()))
    return torch.stack(mixtures).cuda(), torch.stack(vocals).cuda()


def profile(reader, lengths, mode, weights, count, accumulation, deterministic=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = device_snapshot()
    if before["free_gib"] < 2.5:
        return {"status": "insufficient free VRAM", "before": before}
    seed_all(SEED, deterministic=deterministic)
    try:
        run = ExactRun(Separator(checkpoint_blocks=mode).cuda(), weights, SEED, amp=True)
        samples = []
        with sdpa_kernel(BACKENDS):
            for j in range(7):
                t = perf_counter()
                x, v = batch(reader, lengths, j * count, count)
                torch.cuda.synchronize()
                loaded = perf_counter()
                metrics = run.step(x, v, accumulation=accumulation)
                torch.cuda.synchronize()
                if j >= 2:
                    done = perf_counter()
                    samples.append({"data_seconds": loaded - t, "step_seconds": done - loaded,
                                    "end_to_end_seconds": done - t})
        return {"status": "pass", "before": before, "memory": device_snapshot(),
                "samples": samples, "last_loss": metrics["loss"],
                "median_seconds": statistics.median(s["end_to_end_seconds"] for s in samples),
                "microbatch": count // accumulation, "accumulation": accumulation,
                "checkpoint_blocks": mode, "deterministic": deterministic}
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "CUDA OOM", "before": before, "memory": device_snapshot(),
                "error": str(exc), "microbatch": count // accumulation,
                "accumulation": accumulation, "checkpoint_blocks": mode, "deterministic": deterministic}


@torch.no_grad()
def scores(model, x, target):
    with sdpa_kernel(BACKENDS), torch.autocast("cuda", dtype=torch.bfloat16):
        v, i, mask = model(x, return_mask=True)
    def sdr(pred, ref):
        return float(10 * torch.log10(ref.square().sum() / (pred - ref).square().sum().clamp_min(1e-20)))
    return {"vocal_sdr_db": sdr(v, target), "instrumental_sdr_db": sdr(i, x - target),
            "mask_max": float(mask.abs().max()), "mixture_max_abs": float((v + i - x).abs().max()),
            "output_peak": float(torch.maximum(v.abs().max(), i.abs().max())),
            "wave_dtype": str(v.dtype), "mask_dtype": str(mask.dtype)}


def batch4_resume(reader, lengths, weights, identity, out):
    seed_all(SEED, deterministic=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    data_id = f"{identity}:batch4:seed{SEED}"
    run = ExactRun(Separator(checkpoint_blocks=True).cuda(), weights, SEED, amp=True)
    with TemporaryDirectory(dir=out) as tmp, sdpa_kernel(BACKENDS):
        path = Path(tmp) / "batch4.pt"
        run.step(*batch(reader, lengths, run.index * 4, 4))
        run.save(path, data_id)
        x, y = batch(reader, lengths, run.index * 4, 4)
        expected_step = run.step(x, y)
        expected = model_digest(run.model)
        expected_lr = run.scheduler.get_last_lr()
        del run
        replay = ExactRun(Separator(checkpoint_blocks=True).cuda(), weights, SEED, amp=True)
        replay.restore(path, data_id)
        actual = batch(reader, lengths, replay.index * 4, 4)
        matches = {"samples": all(torch.equal(a, b) for a, b in zip((x, y), actual)),
                   "metrics": replay.step(*actual) == expected_step,
                   "weights": model_digest(replay.model) == expected,
                   "scheduler": replay.scheduler.get_last_lr() == expected_lr}
    return {"matches": matches, "memory": device_snapshot()}


def overfit(reader, lengths, weights, identity, out):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seed_all(SEED, deterministic=True)
    run = ExactRun(Separator(checkpoint_blocks=True).cuda(), weights, SEED, amp=True, total=24000)
    initial = model_digest(run.model)
    x, y = batch(reader, lengths, 8)
    for crop_index in range(8, 80):
        if crop_index > 8:
            x, y = batch(reader, lengths, crop_index)
        if float(y.square().mean().sqrt()) >= .018:
            break
    else:
        raise ValueError("Could not choose an active vocal crop")
    torch.cuda.synchronize()
    points = {0: scores(run.model, x, y)}
    data_id = f"{identity}:seed{SEED}:fixed{crop_index}"
    started = perf_counter()
    trace = []
    for j in range(OVERFIT_STEPS):
        with sdpa_kernel(BACKENDS):
            trace.append(run.step(x, y, gradient_audit=(j == 0)))
        if run.index in (1, 10, 20, 40, OVERFIT_STEPS):
            points[run.index] = scores(run.model, x, y)
        if run.index in (10, 40):
            print("fixed crop", run.index, trace[-1]["loss"], points[run.index], flush=True)
    memory = device_snapshot()
    torch.cuda.synchronize()
    wall = perf_counter() - started
    with TemporaryDirectory(dir=out) as tmp:
        path = Path(tmp) / "resume.pt"
        run.save(path, data_id)
        bytes_saved = path.stat().st_size
        next_key = crop_for(lengths, SEGMENT, SEED, run.index)
        replay_x, replay_y = batch(reader, lengths, run.index)
        rng = torch.rand(8, device="cuda")
        with sdpa_kernel(BACKENDS):
            uninterrupted = run.step(replay_x, replay_y)
        expected = model_digest(run.model)
        scheduler = run.scheduler.state_dict()
        expected_moments = [[state[k].detach().cpu().clone() for k in ("exp_avg", "exp_avg_sq", "step")]
                            for state in run.optimizer.state.values()]
        del run
        replay = ExactRun(Separator(checkpoint_blocks=True).cuda(), weights, SEED, amp=True, total=24000)
        replay.restore(path, data_id)
        rng_equal = torch.equal(rng, torch.rand(8, device="cuda"))
        resumed_key = crop_for(lengths, SEGMENT, SEED, replay.index)
        resumed_x, resumed_y = batch(reader, lengths, replay.index)
        with sdpa_kernel(BACKENDS):
            resumed = replay.step(resumed_x, resumed_y)
        actual_moments = [[state[k].detach().cpu() for k in ("exp_avg", "exp_avg_sq", "step")]
                          for state in replay.optimizer.state.values()]
        matches = {"rng": rng_equal, "sample": next_key == resumed_key,
                   "metrics": uninterrupted == resumed, "weights": model_digest(replay.model) == expected,
                   "scheduler": replay.scheduler.state_dict() == scheduler,
                   "optimizer": len(actual_moments) == len(expected_moments) and
                   all(torch.equal(a, b) for ra, rb in zip(actual_moments, expected_moments)
                       for a, b in zip(ra, rb))}
    return {"initial_sha256": initial, "crop": crop_for(lengths, SEGMENT, SEED, crop_index),
            "crop_index": crop_index, "fixed_steps": OVERFIT_STEPS, "points": points,
            "trace": trace, "wall_seconds": wall, "memory": memory, "resume": matches,
            "checkpoint_bytes": bytes_saved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--effective-batch4-only", action="store_true")
    args = parser.parse_args()
    seed_all(SEED)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required")
    manifest = ROOT / "manifest.json"
    split = ROOT / "split_external_final.json"
    identity = hashlib.sha256(manifest.read_bytes() + split.read_bytes()).hexdigest()
    ids = {r["uuid"] for r in json.loads(split.read_text(encoding="utf-8"))["assignments"]
           if r["role"] == "train_clean"}
    known = [s["uuid"] for s in json.loads((ROOT / "audio_probe.json").read_text(encoding="utf-8"))["verified"]
             if s["length_policy"] == "strict"]
    songs = [s for s in json.loads(manifest.read_text(encoding="utf-8"))
             if s["uuid"] in ids.intersection(known)]
    if len(songs) != 3:
        raise ValueError("Missing training-only probe recordings")
    out = ROOT / "gpu_preflight.json"
    result = {"torch": torch.__version__, "platform": platform.platform(),
              "device": torch.cuda.get_device_name(), "native_bf16": True,
              "before": device_snapshot(), "songs": [s["uuid"] for s in songs], "data_identity": identity}
    with ZipFile(args.archive) as archive, Shard(ROOT, 4 * GIB) as shard, ExitStack() as stack:
        for song in songs:
            shard.stage(archive, song)
        lengths = {k: frames for k, (_, frames) in shard.songs.items()}
        reader = {k: stack.enter_context(shard.reader(k)) for k in lengths}
        if args.effective_batch4_only:
            result = json.loads(out.read_text(encoding="utf-8"))
            if result["data_identity"] != identity or not all(result["overfit"]["resume"].values()):
                raise ValueError("Need completed GPU preflight for this dataset")
            for name, count, accumulation in (("batch2_accum2", 4, 2), ("batch4", 4, 1)):
                previous = result.get(name)
                if previous is not None:
                    result.setdefault("effective_batch4_previous_profiles", {}).setdefault(name, []).append(previous)
                result[name] = profile(reader, lengths, True, result["weights"], count,
                                       accumulation, deterministic=True)
                print(name, result[name], flush=True)
                write_json(out, result)
            result["batch4_resume"] = batch4_resume(reader, lengths, result["weights"], identity, ROOT)
            print("batch4 resume", result["batch4_resume"], flush=True)
            write_json(out, result)
            return
        model = Separator().cuda()
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with sdpa_kernel(BACKENDS), torch.autocast("cuda", dtype=torch.bfloat16):
            weights, evidence = calibrate(model, (batch(reader, lengths, n) for n in range(4)))
        del model
        result["weights"], result["calibration"] = weights, evidence
        print("calibration", weights, flush=True)
        for label, mode, count, accumulate in (
            ("baseline", False, 1, 1), ("checkpoint", True, 1, 1),
            ("checkpoint_batch2", True, 2, 1), ("checkpoint_accum2", True, 2, 2),
        ):
            result[label] = profile(reader, lengths, mode, weights, count, accumulate)
            print(label, result[label], flush=True)
            write_json(out, result)
        result["overfit"] = overfit(reader, lengths, weights, identity, ROOT)
        write_json(out, result)
        print("resume", result["overfit"]["resume"], flush=True)


if __name__ == "__main__":
    main()
