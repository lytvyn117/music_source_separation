"""Matched, bounded train-only comparison with a fixed artist-disjoint probe panel."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import statistics
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from audit import write_json
from gpu_preflight import BACKENDS, device_snapshot
from loader import Shard
from loss_candidates import WEIGHTS, get_loss, projection_terms
from scratch_model import SEGMENT, Separator, seed_all
from scratch_train import ExactRun, model_digest

ROOT = Path(__file__).resolve().parent
SEED = 909
UPDATES = 160
BATCH = 4


def select(rows):
    groups = {}
    for row in sorted(rows, key=lambda r: r["uuid"]):
        groups.setdefault(row["artist_key"], []).append(row)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    # One track per artist, reproducible without looking at probe metrics.
    picked = [random.Random(f"{SEED}:{k}").choice(groups[k]) for k in keys[:18]]
    return picked[:12], picked[12:]


def windows(reader, frames, key):
    rng = random.Random(f"{SEED}:windows:{key}")
    candidates = []
    for _ in range(64):
        start = rng.randrange(frames - SEGMENT + 1)
        _, v, i = reader(start, SEGMENT)
        if min(np.sqrt(np.mean(v**2)), np.sqrt(np.mean(i**2))) > .005:
            candidates.append(start)
    if len(candidates) < 4:
        raise ValueError(f"Insufficient active vocal and backing crops: {key}")
    return candidates


def tensors(readers, schedule):
    x, y = [], []
    for key, start in schedule:
        mix, vocal, _ = readers[key](start, SEGMENT)
        x.append(mix.T)
        y.append(vocal.T)
    return tuple(torch.from_numpy(np.stack(source)).cuda() for source in (x, y))


def schedule_for(starts, index):
    rng = random.Random(f"{SEED}:batch:{index}")
    keys = rng.sample(sorted(starts), BATCH)
    return [(k, rng.choice(starts[k])) for k in keys]


@torch.no_grad()
def evaluate(model, readers, panel):
    model.eval()
    rows = []
    for key, starts in panel.items():
        x, y = tensors(readers, [(key, start) for start in starts])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v, i, mask = model(x, return_mask=True)
        error = (v - y).square().sum(dim=(1, 2)).clamp_min(1e-12)
        voc = 10 * torch.log10(y.square().sum(dim=(1, 2)) / error)
        inst = 10 * torch.log10((x - y).square().sum(dim=(1, 2)) / error)
        projection, active = projection_terms(v, y)
        common_mse = ((v - y).square().mean(dim=(1, 2)) /
                      x.square().mean(dim=(1, 2)).clamp_min(1e-8))
        terms = {}
        for name, weights in WEIGHTS.items():
            parts = get_loss(name)(model, x, v, y)
            terms[name] = {"total": sum(weights[k] * float(part) for k, part in parts.items()),
                           "components": {k: float(part) for k, part in parts.items()}}
        rows.append({"uuid": key, "vocal_sdr_db": float(voc.mean()),
                     "instrumental_sdr_db": float(inst.mean()), "normalized_wave_mse": float(common_mse.mean()),
                     "abs_projection": float(projection[active].abs().mean()),
                     "mixture_residual": float((v + i - x).abs().max()),
                     "mask_max": float(mask.abs().max()), "terms": terms})
    return {"songs": rows, "mean": {
        k: statistics.mean(row[k] for row in rows)
        for k in ("vocal_sdr_db", "instrumental_sdr_db", "normalized_wave_mse", "abs_projection")},
        "objective_means": {name: statistics.mean(r["terms"][name]["total"] for r in rows) for name in WEIGHTS}}


def run_arm(name, readers, train, panel, identity):
    seed_all(SEED, deterministic=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run = ExactRun(Separator(checkpoint_blocks=True).cuda(), WEIGHTS[name], SEED, True,
                   objective_fn=get_loss(name), objective_name=name)
    digest = model_digest(run.model)
    points, timings = {}, []
    seen = set()
    with sdpa_kernel(BACKENDS):
        points[0] = evaluate(run.model, readers, panel)
        for j in range(UPDATES):
            t = perf_counter()
            address = schedule_for(train, j)
            seen.update(key for key, _ in address)
            x, y = tensors(readers, address)
            torch.cuda.synchronize()
            loaded = perf_counter()
            metrics = run.step(x, y)
            torch.cuda.synchronize()
            done = perf_counter()
            timings.append([loaded - t, done - loaded, metrics])
            if run.index in (40, 80, UPDATES):
                points[run.index] = evaluate(run.model, readers, panel)
                print(name, run.index, points[run.index]["mean"], flush=True)
        memory = device_snapshot()
        with TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "checkpoint.pt"
            run.save(path, identity)
            address = schedule_for(train, run.index)
            x, y = tensors(readers, address)
            rng = torch.rand(4, device="cuda")
            expected = run.step(x, y)
            expected_model = model_digest(run.model)
            expected_scheduler = run.scheduler.state_dict()
            moments = [[state[k].detach().cpu().clone() for k in ("step", "exp_avg", "exp_avg_sq")]
                       for state in run.optimizer.state.values()]
            del run
            replay = ExactRun(Separator(checkpoint_blocks=True).cuda(), WEIGHTS[name], SEED, True,
                              objective_fn=get_loss(name), objective_name=name)
            replay.restore(path, identity)
            assert replay.index == UPDATES
            torch.testing.assert_close(torch.rand(4, device="cuda"), rng, rtol=0, atol=0)
            assert schedule_for(train, replay.index) == address
            assert expected == replay.step(x, y)
            assert expected_model == model_digest(replay.model)
            assert replay.scheduler.state_dict() == expected_scheduler
            assert len(moments) == len(replay.optimizer.state)
            assert all(torch.equal(old, state[k].detach().cpu())
                       for saved, state in zip(moments, replay.optimizer.state.values())
                       for old, k in zip(saved, ("step", "exp_avg", "exp_avg_sq")))
    assert len(seen)==len(train) and not seen.intersection(panel)
    return {"initial_sha256": digest, "evaluation": points, "memory": memory, "steps": timings,
            "median_elapsed_step": statistics.median(a + b for a, b, _ in timings[8:]),
            "median_data_step": statistics.median(a for a, _, _ in timings[8:]),
            "seen": sorted(seen), "exact_resume": True, "weights": WEIGHTS[name]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--finish-mse", action="store_true")
    args = parser.parse_args()
    seed_all(SEED, deterministic=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native BF16 required")
    ids = json.loads((ROOT / "training_eligibility.json").read_text(encoding="utf-8"))["strict_train"]
    rows = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    train, panel = select([r for r in rows if r["uuid"] in ids])
    result = {"train": [{k: r[k] for k in ("uuid", "artist", "title")} for r in train],
              "probe_only": [{k: r[k] for k in ("uuid", "artist", "title")} for r in panel],
              "batch": BATCH, "updates_per_arm": UPDATES}
    with ZipFile(args.archive) as archive, Shard(ROOT, 4 * 1024**3) as shard, ExitStack() as stack:
        start = perf_counter()
        for row in train + panel:
            shard.stage(archive, row)
        result["stage_seconds"] = perf_counter() - start
        result["stage_peak_payload_bytes"] = shard.peak_reserved_bytes
        readers = {k: stack.enter_context(shard.reader(k)) for k in shard.songs}
        starts = {k: windows(readers[k], frames, k) for k, (_, frames) in shard.songs.items()}
        training = {r["uuid"]: starts[r["uuid"]] for r in train}
        scoring = {r["uuid"]: starts[r["uuid"]][:4] for r in panel}
        result["train_windows"], result["probe_windows"] = training, scoring
        identity = hashlib.sha256(json.dumps({
            "train": training, "score": scoring, "seed": SEED,
            "manifest": hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()
        }, sort_keys=True).encode()).hexdigest()
        arms = list(WEIGHTS)
        if args.finish_mse:
            previous = json.loads((ROOT / "multisong_probe.json").read_text(encoding="utf-8"))
            for key in ("train", "probe_only", "train_windows", "probe_windows", "batch", "updates_per_arm"):
                if result[key] != previous[key]:
                    raise ValueError(f"Matched probe identity differs: {key}")
            result["preflight_l1"] = previous["preflight_l1"]
            arms = ["normalized_mse_bleed"]
        for arm in arms:
            result[arm] = run_arm(arm, readers, training, scoring, identity)
            write_json(ROOT / "multisong_probe.json", result)
    assert result["preflight_l1"]["initial_sha256"] == result["normalized_mse_bleed"]["initial_sha256"]


if __name__ == "__main__":
    main()
