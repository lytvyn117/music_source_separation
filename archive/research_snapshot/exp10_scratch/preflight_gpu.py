"""Bounded real-data GPU preflight. At most 18 optimizer steps including replay."""
import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from checkpoint import BEST_ALIASES, Store, file_hash, read_json, write_json
from data import ARCHIVE, ROOT, verify_cache
from scratch_train import model_digest
from train_exp10 import BACKENDS, BatchSource, Experiment, bind, memory, provenance, setup
from validation import METRICS


def equal(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError(f"Tensor differs at {path}")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Missing keys at {path}")
        for key in a:
            equal(a[key], b[key], f"{path}.{key}")
    elif a != b:
        raise AssertionError(f"Value differs at {path}")


def cpu_snapshot(state):
    if isinstance(state, dict):
        return {k: cpu_snapshot(v) for k, v in state.items()}
    if isinstance(state, (tuple, list)):
        return type(state)(cpu_snapshot(value) for value in state)
    return state.detach().cpu().clone() if isinstance(state, torch.Tensor) else state


def require(condition, problem):
    if not condition:
        raise RuntimeError(problem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, choices=list("abcdefgh"))
    args = parser.parse_args()
    started = perf_counter()
    config, roster = bind()
    settings = config["exp09_settings"]
    refs = verify_cache(roster["development"])
    dev = roster["development"]
    one = [next(iter(dev))]
    runtime = setup(settings["seed"])
    run_info = provenance(config, refs, runtime, "preflight")
    out = ROOT / "preflight" / args.tag
    out.mkdir(exist_ok=False, parents=True)
    device = torch.device("cuda")
    durations = []
    baseline = {}
    timing = []
    identity = run_info["identity"]
    log_path = ROOT / "preserved_exp09" / "runs" / "exp09_mse_seed909" / "steps.jsonl"
    write_json(out / "run.json", {**run_info, "preflight_gpu_sha256": file_hash(ROOT / "preflight_gpu.py"),
                                 "exp09_steps_sha256": file_hash(log_path)})

    def update(run, data, compare_exp09):
        group, addresses, batch = data.batch(run.step)
        torch.cuda.synchronize()
        start = perf_counter()
        metrics = run.update(batch)
        torch.cuda.synchronize()
        durations.append(perf_counter() - start)
        if compare_exp09:
            old = baseline[run.step]
            require(old["shard"] == group and old["crops"] == [list(a) for a in addresses],
                    "Exp09 crop addresses differ")
            require(all(np.isclose(metrics[k], old[k], rtol=2e-6, atol=1e-7) for k in metrics),
                    f"Exp09 objective or gradient differs at step {run.step}: {metrics}")
        return metrics

    with log_path.open(encoding="utf-8") as source:
        for _ in range(12):
            row = json.loads(next(source))
            baseline[row["step"]] = row

    with Store(out, identity, 6, 12) as store, sdpa_kernel(BACKENDS):
        run = Experiment(config, identity, device)
        start = perf_counter()
        store.commit(run.state())
        timing.append(perf_counter() - start)
        with BatchSource(ARCHIVE, out, roster, settings["seed"], limit=12) as data:
            for _ in range(6):
                update(run, data, True)
            raw_sha = model_digest(run.raw.model)
            rng_before = run.state()["rng"]
            early = run.validate(dev, one)
            require(model_digest(run.raw.model) == raw_sha, "Validation changed raw weights")
            equal(run.state()["rng"], rng_before, "validation_rng")
            run.record(early)
            start = perf_counter()
            store.commit(run.state())
            timing.append(perf_counter() - start)
            for _ in range(6):
                update(run, data, True)
            expected = cpu_snapshot(run.state())
            summary = run.validate(dev)
            equal(expected, cpu_snapshot(run.state()), "full_validation_state")
            for arm in summary.values():
                require([r["uuid"] for r in arm["songs"]] == list(dev)
                        and len({r["artist_key"] for r in arm["songs"]}) == 7,
                        "Incomplete validation coverage")
                require(set(arm["artist_macro"]) == set(METRICS), "Incomplete metric set")
            require(summary["raw"]["songs"] != summary["ema"]["songs"],
                    "Paired validation produced identical scores with different weights")
            staging = data.staging
        first_peak = memory()
        del run
        # Restart from numbered step 6 even with missing aliases and a damaged derived history.
        (out / "latest.pt").unlink()
        (out / "best_overall.pt").unlink()
        (out / "validation_history.json").write_text("interrupted")
        setup(settings["seed"])
        run = Experiment(config, identity, device)
        run.restore(store.recover())
        with BatchSource(ARCHIVE, out, roster, settings["seed"], limit=12) as data:
            for _ in range(6):
                update(run, data, True)
            actual = cpu_snapshot(run.state())
            equal(expected, actual)
        repeat = run.validate(dev, one)
        for name, arm in repeat.items():
            equal(arm["songs"][0], summary[name]["songs"][0], f"validation_replay.{name}")
        run.record(summary)
        start = perf_counter()
        store.commit(run.state())
        timing.append(perf_counter() - start)
        recovered = store.recover()
        equal(cpu_snapshot(run.state()), recovered)
        index = read_json(out / "checkpoint_index.json")
        require(file_hash(out / "latest.pt") == index["latest"]["sha256"],
                "Latest alias hash differs")
        for name in BEST_ALIASES:
            alias = out / f"{name}.pt"
            require(alias.exists() == (index[name] is not None), f"Incorrect {name} alias presence")
            if index[name] is not None:
                require(file_hash(alias) == index[name]["sha256"], f"{name} alias hash differs")
        replay_peak = memory()
    require(max(first_peak["peak_reserved_gib"], replay_peak["peak_reserved_gib"])
            <= config["memory_guard_gib"], "Memory guard exceeded")
    result = {
        "full_training_started": False, "exp09_steps_matched": 12, "updates_executed": len(durations),
        "exact_resume": "raw model, optimizer, scheduler, EMA, CPU/CUDA/NumPy/Python RNG and history",
        "deterministic_repeat_song": one[0], "training_staging": staging,
        "validation_songs": len(summary["raw"]["songs"]),
        "validation_chunks_per_weight_set": sum(r["chunks"] for r in summary["raw"]["songs"]),
        "full_panel_seconds": {name: arm["seconds"] for name, arm in summary.items()},
        "repeat_song_seconds": {name: arm["seconds"] for name, arm in repeat.items()},
        "qualification_status": index["qualification_status"],
        "best_overall": index["best_overall"], "best_qualified": index["best_qualified"],
        "training_update_seconds": durations, "median_update_seconds": median(durations[2:12]),
        "numbered_checkpoint_bytes": index["latest"]["bytes"], "checkpoint_commit_seconds": timing,
        "gpu_first_pass": first_peak, "gpu_replay": replay_peak,
        "run_provenance_sha256": file_hash(out / "run.json"),
        "wall_seconds": perf_counter() - started,
    }
    write_json(out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
