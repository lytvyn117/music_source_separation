"""Run the frozen Exp10 experiment after explicit full-run authorization."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from checkpoint import (BEST_ALIASES, EMA, Store, atomic_file, capture_rng, file_hash,
                        qualification_status, read_json, restore_rng, write_json)
from data import ARCHIVE, CACHE, ROOT, RestrictedArchive, Shard, load_roster, make_cache, verify_cache
from selection import decision
from validation import evaluate
from loss_candidates import WEIGHTS, get_loss
from scratch_model import SEGMENT, Separator, seed_all
from scratch_train import ExactRun, model_digest
from shard_plan import sample_position

BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
CONFIG = ROOT / "config.json"
GIB = 1024**3


def bind():
    config = read_json(CONFIG)
    settings = config["exp09_settings"]
    roster = load_roster()
    if (settings["loss"] != WEIGHTS[settings["objective"]] or settings["microbatch"] != 4
            or settings["accumulation"] != 1 or settings["updates"] != 24000
            or settings["optimizer"] != {"name": "AdamW", "lr": .0003, "weight_decay": .01,
                                         "clip_grad_l2": 1.0}
            or config["scheduler"] != {"name": "CosineAnnealingLR", "eta_min": .00003,
                                       "T_max": 24000}
            or not settings["activation_checkpointing"] or settings["seed"] != 909
            or settings["checkpoint_every"] != 1000 or config["validation"]["interval"] != 1000
            or config["validation"] != {"precision": "FP32", "interval": 1000,
                                        "segment_frames": SEGMENT, "hop_frames": SEGMENT // 2,
                                        "window": "positive Hann", "activity_dbfs": -45,
                                        "side_activity_dbfs": -50, "sample_rate": 44100,
                                        "songs": 16, "artists": 7}
            or config["ema_decay"] != .999):
        raise ValueError("Config differs from the controlled experiment")
    return config, roster


def setup(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_all(seed, deterministic=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This experiment requires native CUDA BF16")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()), "driver": sorted(set(driver)),
        "device_index": torch.cuda.current_device(),
        "numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
        "threads": torch.get_num_threads(), "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "attention_backends": [backend.name for backend in BACKENDS],
    }


def provenance(config, references, runtime, mode):
    inputs = [CONFIG, ROOT / "roster.json", ROOT / "train_exp10.py", ROOT / "checkpoint.py",
              ROOT / "selection.py", ROOT / "validation.py", ROOT / "data.py"]
    inputs.extend(sorted((ROOT / "exp09_core").glob("*.py")))
    payload = {
        "schema": 2, "mode": mode, "initialization": "random initialization from seed 909",
        "validation_weights": ["raw", "ema"], "stopping_weights": "ema",
        "initial_model_sha256": config["initial_model_sha256"],
        "files_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in inputs},
        "exp09_identity": config["exp09_identity"],
        "source_evidence_sha256": config["frozen_inputs_sha256"],
        "development_cache": references, "runtime": runtime,
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"identity": identity, **payload}


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"free_gib": free / GIB, "total_gib": total / GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}


class BatchSource:
    """Keep the exact Exp09 shard rotation and random access indexed by update."""
    def __init__(self, archive, out, roster, seed, limit=None):
        self.archive, self.out, self.seed, self.limit = archive, out, seed, limit
        self.roster, self.plan = roster["train"], roster["shard_plan"]
        self.frames = {k: (n, None) for k, n in self.plan["frames_by_uuid"].items()}
        self.previous = None
        self.staging = []
        self.stack = ExitStack()

    def __enter__(self):
        self.archive = self.stack.enter_context(RestrictedArchive(self.archive, self.roster))
        self.active = self.stack.enter_context(ExitStack())
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def address(self, update):
        plan = self.plan
        return sample_position(plan["shards"], self.frames, SEGMENT, self.seed, update, 4,
                               plan["updates_per_shard"])

    def batch(self, update):
        group, addresses = self.address(update)
        if group != self.previous:
            start = perf_counter()
            self.active.close()
            shard = self.active.enter_context(Shard(self.out, self.plan["budget_bytes"]))
            keys = self.plan["shards"][group]
            if self.limit is not None:
                keys = sorted({key for step in range(self.limit)
                               for key, _ in self.address(step)[1]})
            for key in keys:
                staged = shard.stage(self.archive, self.roster[key])
                if staged["frames"] != self.frames[key][0]:
                    raise ValueError(f"Training reference length changed: {key}")
            self.readers = {k: self.active.enter_context(shard.reader(k)) for k in keys}
            self.previous = group
            self.staging.append({"group": group, "seconds": perf_counter() - start, "songs": len(keys),
                                 "peak_bytes": shard.peak_reserved_bytes})
        mixtures, targets = [], []
        for key, offset in addresses:
            mix, vocal, _ = self.readers[key](offset, SEGMENT)
            mixtures.append(mix.T)
            targets.append(vocal.T)
        return group, addresses, tuple(torch.from_numpy(np.stack(rows)).cuda()
                                       for rows in (mixtures, targets))


class Experiment:
    def __init__(self, config, identity, device):
        self.config, self.identity, self.device = config, identity, device
        settings = config["exp09_settings"]
        model = Separator(checkpoint_blocks=True).to(device)
        if model_digest(model) != config["initial_model_sha256"]:
            raise RuntimeError("Scratch initialization differs from Exp09")
        self.raw = ExactRun(model, settings["loss"], settings["seed"], device.type == "cuda",
                            lr=settings["optimizer"]["lr"], total=settings["updates"],
                            objective_fn=get_loss(settings["objective"]),
                            objective_name=settings["objective"])
        self.ema = EMA(model, config["ema_decay"])
        self.history = []

    @property
    def step(self):
        return self.raw.index

    def update(self, batch):
        metrics = self.raw.step(*batch)
        self.ema.update(self.raw.model)
        return metrics

    def validate(self, development, ids=None):
        rng = capture_rng(self.device)
        ids = None if ids is None else tuple(ids)
        summary = {}
        try:
            eval_model = deepcopy(self.raw.model).eval().requires_grad_(False)
            with sdpa_kernel(BACKENDS), torch.autocast(self.device.type, enabled=False):
                for weights in ("raw", "ema"):
                    restore_rng(rng, self.device)
                    if weights == "ema":
                        eval_model.load_state_dict(self.ema.weights, strict=True)
                    started = perf_counter()
                    summary[weights] = {**evaluate(eval_model, development, self.device, ids),
                                        "seconds": perf_counter() - started}
            return summary
        finally:
            restore_rng(rng, self.device)

    def record(self, summary):
        scores = {name: arm["artist_macro"] for name, arm in summary.items()}
        new = decision(self.history, self.step, scores, self.config["selection"])
        self.history.append({"step": self.step, "validation": summary, "selection": new})

    def state(self):
        return {
            "identity": self.identity, "step": self.step, "raw_model": self.raw.model.state_dict(),
            "optimizer": self.raw.optimizer.state_dict(), "scheduler": self.raw.scheduler.state_dict(),
            "ema": self.ema.state_dict(), "rng": capture_rng(self.device), "history": self.history,
        }

    def restore(self, state):
        step = state["step"]
        if (state["identity"] != self.identity or step != state["ema"]["updates"]
                or step != state["scheduler"]["last_epoch"]):
            raise ValueError("Checkpoint state does not match this experiment/update")
        self.raw.model.load_state_dict(state["raw_model"], strict=True)
        self.raw.optimizer.load_state_dict(state["optimizer"])
        self.raw.scheduler.load_state_dict(state["scheduler"])
        self.ema.load_state_dict(state["ema"])
        self.raw.index = step
        self.history = state["history"]
        restore_rng(state["rng"], self.device)


def replay_log(path, step):
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row["step"] > step:
            break
        retained.append(line)
    payload = "".join(line + "\n" for line in retained).encode()
    atomic_file(path, lambda handle: handle.write(payload))


def open_state(store, record, run, resume):
    run_file = store.root / "run.json"
    if resume:
        if run_file.is_file() and read_json(run_file) != record:
            raise ValueError("Run source, configuration, reference data or environment changed")
        if store.published():
            if not run_file.is_file():
                raise ValueError("Committed checkpoint has no run provenance")
            return store.recover()
        aliases = ("latest.pt", "steps.jsonl", *(f"{name}.pt" for name in BEST_ALIASES))
        if any((store.root / name).exists() for name in aliases):
            raise ValueError("Run outputs exist without a numbered checkpoint")
    write_json(run_file, record)
    store.commit(run.state())
    return None


def train(out, resume):
    config, roster = bind()
    settings = config["exp09_settings"]
    runtime = setup(settings["seed"])
    refs = verify_cache(roster["development"])
    record = provenance(config, refs, runtime, "full")
    if not resume:
        out.mkdir(exist_ok=False, parents=True)
    interval, maximum = settings["checkpoint_every"], settings["updates"]
    with Store(out, record["identity"], interval, maximum) as store, sdpa_kernel(BACKENDS):
        run = Experiment(config, store.identity, torch.device("cuda"))
        state = open_state(store, record, run, resume)
        if state is not None:
            run.restore(state)
            del state
        log_path = out / "steps.jsonl"
        replay_log(log_path, run.step)
        with BatchSource(ARCHIVE, out, roster, settings["seed"]) as data:
            while run.step < maximum and not (run.history and run.history[-1]["selection"]["stop"]):
                group, addresses, batch = data.batch(run.step)
                started = perf_counter()
                metrics = run.update(batch)
                free = memory()
                if free["peak_reserved_gib"] > config["memory_guard_gib"]:
                    raise RuntimeError("CUDA memory guard exceeded")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"step": run.step, "seconds": perf_counter() - started,
                                          "crops": addresses, "shard": group, **metrics, **free},
                                         allow_nan=False) + "\n")
                del batch
                if run.step % interval == 0:
                    run.record(run.validate(roster["development"]))
                    if memory()["peak_reserved_gib"] > config["memory_guard_gib"]:
                        raise RuntimeError("CUDA memory guard exceeded during validation")
                    store.commit(run.state())
                    row = run.history[-1]
                    scores = {name: arm["artist_macro"] for name, arm in row["validation"].items()}
                    print(f"Update {run.step} {scores} {row['selection']}", flush=True)
                elif run.step % 100 == 0:
                    print(f"Update {run.step} loss={metrics['loss']:.5f}", flush=True)
        reason = "max_updates" if run.step == maximum else "early_stop"
        selection = run.history[-1]["selection"]
        status = qualification_status(selection)
        write_json(out / "completion.json", {"step": run.step, "reason": reason,
                                              "selection": selection, "qualification_status": status})
        if status == "no_candidate_qualified":
            print("No development candidate met all guardrails. See best_overall.pt.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--authorize-full-training", action="store_true")
    modes.add_argument("--prepare-development-cache", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_development_cache:
        if args.out or args.resume:
            parser.error("Cache preparation takes neither an output override nor a resume flag")
        development = load_roster()["development"]
        if not CACHE.is_dir():
            make_cache(ARCHIVE, development)
        verify_cache(development)
    else:
        if args.out is None:
            parser.error("Specify --out for an Exp10 run")
        if args.out.resolve().parent != (ROOT / "runs").resolve():
            parser.error("Full-run outputs must be direct children of the Exp10 runs directory")
        train(args.out.resolve(), args.resume)


if __name__ == "__main__":
    main()
