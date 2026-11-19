"""Operational wrapper for the approved Exp08 fit and evaluation APIs."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.experiments.exp08_subband import evaluate, train
from src.experiments.exp08_subband.checkpointing import Checkpoints
from src.experiments.exp08_subband.data import (
    RecordedClips, bind_run, reserve_run, sha256,
)
from src.experiments.exp08_subband.loss import LOSS_WEIGHTS
from src.experiments.exp08_subband.model import SubbandSeparator

RUN_ID = "full_20261111_seed42"
REPORT = ROOT / "reports/exp08_subband" / RUN_ID
LOG = ROOT / "outputs/logs/exp08_subband" / RUN_ID
MODELS = ROOT / "models/experiments/exp08_subband" / RUN_ID
SMOKE = ROOT / "reports/exp08_subband/smoke_20261111_verified"
GIB = 2**30


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def smi(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def gpu_read():
    line = smi("--query-gpu=temperature.gpu,memory.free,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits").strip().splitlines()[0]
    return dict(zip(("temperature_c", "free_mib", "total_mib", "gpu_util_pct"),
                    (int(value.strip()) for value in line.split(",")), strict=True))


def checkpoint_hashes():
    checkpoints = {}
    for p in sorted((ROOT / "models").rglob("*")):
        if p.is_file() and MODELS not in p.parents:
            checkpoints[p.relative_to(ROOT).as_posix()] = sha256(p)
    return checkpoints


def verify_protected():
    path = SMOKE.parent / "smoke_20261111/protected_before.json"
    hashes = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in hashes.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Protected file fingerprint changed: {name}")
    for name in ("smoke_20261111", "smoke_20261111_verified"):
        directory = MODELS.parent / name
        record = json.loads((directory / "commit.json").read_text(encoding="utf-8"))
        for entry in (record["best"], record["latest"], *record["snapshots"].values()):
            if sha256(directory / entry["file"]) != entry["sha256"]:
                raise ValueError(f"Smoke checkpoint fingerprint changed: {name}/{entry['file']}")
    return len(hashes)


def alias(entry, name):
    source = MODELS / entry["file"]
    target = MODELS / name
    temporary = MODELS / f".{name}.tmp"
    shutil.copyfile(source, temporary)
    if sha256(temporary) != entry["sha256"]:
        temporary.unlink()
        raise ValueError(f"Alias fingerprint mismatch: {name}")
    temporary.replace(target)


def enrich_score(original):
    def score(track, references, estimates, mixture):
        row = original(track, references, estimates, mixture)
        ref = np.einsum("sct,sct->s", references, references, dtype=np.float64)
        est = np.einsum("sct,sct->s", estimates, estimates, dtype=np.float64)
        row["rms_gain_db"] = (10 * np.log10((est + 1e-8) / (ref + 1e-8))).tolist()
        row["bleed_energy_pct"] = 100 * row["bleed"]**2
        row["output_peak"] = np.max(np.abs(estimates), axis=(1, 2)).tolist()
        return row
    return score


@torch.inference_mode()
def diagnostics(model, dataset):
    model.eval()
    rows = []
    for batch in train.make_loader(dataset, train.TrainConfig(), workers=0, training=False):
        audio = batch["audio"]
        device_audio = audio[:, 0].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            estimates = model(device_audio).cpu().numpy()
        for index, estimate in enumerate(estimates):
            clip = audio[index].numpy()
            rows.append(evaluate.score_song(batch["track"][index], clip[1:], estimate, clip[0]))
    metrics = {}
    for key in ("vocal_sdr", "instrumental_sdr", "vocal_si_sdr", "bleed", "bleed_energy_pct"):
        values = [row[key] for row in rows if math.isfinite(row[key])]
        metrics[key] = float(np.median(values)) if values else None
    return {"medians": metrics, "clips": evaluate._json_finite(rows)}


class MonitoredCheckpoints(Checkpoints):
    def __init__(self, fingerprint, model, diagnostic_data):
        super().__init__(MODELS, fingerprint)
        self.model = model
        self.diagnostic_data = diagnostic_data
        self.time_start = None

    def save(self, state):
        entry = super().save(state)
        record = self._record()
        epoch = entry["epoch"]
        if epoch == 0:
            loaded = self.load("latest")
            if loaded["epoch"] != 0 or loaded["optimizer"]["state"]:
                raise RuntimeError("Initial checkpoint is not a verified fresh state")
            alias(entry, "initial.pt")
            write_json(REPORT / "initial_verified.json", {"timestamp": stamp(), **entry})
        alias(record["latest"], "latest.pt")
        if record["best"]["epoch"] == epoch:
            alias(record["best"], "best.pt")
        if epoch > 0 and epoch % 10 == 0:
            alias(entry, f"snapshot_epoch_{epoch:03d}.pt")
        summary = diagnostics(self.model, self.diagnostic_data)
        metrics = {"epoch": epoch, "timestamp": stamp(), "history": state["history"][-1],
                   "selection": state["selection"], "diagnostics": summary,
                   "gpu": gpu_read(),
                   "elapsed_training_s": time.perf_counter() - self.time_start
                   if self.time_start else 0.}
        append_json(LOG / "epochs.jsonl", metrics)
        print(f"epoch={epoch:02d} latest={record['latest']['file']} best={record['best']['epoch']} "
              f"train={metrics['history'].get('train_total_loss')} "
              f"val={metrics['history']['val_total_loss']:.6f} "
              f"metrics={summary['medians']} gpu={metrics['gpu']}", flush=True)
        return entry


def install_monitor(store, max_temp):
    update = train.optimizer_update
    count = 0
    peak_alloc = peak_res = 0
    min_free = float("inf")
    peak_temp = 0
    total_clipped = 0
    last_gpu = None
    started = None

    def checked(model, optimizer, audio, config):
        nonlocal count, peak_alloc, peak_res, min_free, peak_temp, total_clipped, last_gpu, started
        if not (MODELS / "initial.pt").is_file():
            raise RuntimeError("Initial checkpoint must exist before the first update")
        if started is None:
            started = time.perf_counter()
            store.time_start = started
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * GIB or torch.cuda.max_memory_reserved() > 8 * GIB:
            raise FloatingPointError("Approved GPU memory limit failed before update")
        if count % 16 == 0:
            last_gpu = gpu_read()
            peak_temp = max(peak_temp, last_gpu["temperature_c"])
            append_json(LOG / "gpu.jsonl", {"update": count, "timestamp": stamp(), **last_gpu})
            if last_gpu["temperature_c"] >= max_temp:
                raise FloatingPointError("GPU reached its reported maximum operating temperature")
        lr = optimizer.param_groups[0]["lr"]
        row = update(model, optimizer, audio, config)
        count += 1
        total_clipped += row["clipped"]
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        free = torch.cuda.mem_get_info()[0]
        peak_alloc, peak_res, min_free = max(peak_alloc, allocated), max(peak_res, reserved), min(min_free, free)
        append_json(LOG / "steps.jsonl", {"update": count, "lr": lr, **row})
        if reserved > 8 * GIB or free < 2 * GIB:
            raise FloatingPointError("Approved GPU memory limit failed during update")
        return row

    def totals():
        return {"updates": count, "clipped_updates": total_clipped,
                "peak_allocated_gib": peak_alloc / GIB, "peak_reserved_gib": peak_res / GIB,
                "minimum_free_gib": min_free / GIB if math.isfinite(min_free) else None,
                "peak_sampled_temperature_c": peak_temp,
                "training_seconds": time.perf_counter() - started if started else 0.}

    train.optimizer_update = checked
    return totals


def main():
    if MODELS.exists() or (REPORT / "start.json").exists():
        raise FileExistsError("This Exp08 run ID has already been used")
    if not LOG.is_dir():
        raise ValueError("The launch log directory must already be reserved")
    free = shutil.disk_usage(ROOT).free
    if free < 10 * GIB:
        raise RuntimeError("Less than 10 GiB of disk space is free")
    initial_gpu = gpu_read()
    thermals = smi("-q", "-d", "TEMPERATURE")
    match = re.search(r"GPU Max Operating Temp\s*:\s*(\d+) C", thermals)
    if not match:
        raise RuntimeError("Could not read GPU operating limit")
    max_temp = int(match[1])
    if initial_gpu["free_mib"] < 2048 or initial_gpu["temperature_c"] >= max_temp:
        raise RuntimeError("Initial GPU memory or temperature check failed")
    protected = verify_protected()
    before = checkpoint_hashes()
    write_json(REPORT / "protected_models_before.json", before)
    config = train.TrainConfig()
    smoke_config = json.loads((SMOKE / "run_started.json").read_text(encoding="utf-8"))["config"]
    if asdict(config) != smoke_config:
        raise ValueError("Configuration differs from the verified smoke run")
    if json.loads((SMOKE / "result.json").read_text(encoding="utf-8"))["decision"] != "GO":
        raise ValueError("Verified preflight status is not GO")
    manifest = REPORT / "manifest.json"
    shutil.copyfile(SMOKE / "manifest.json", manifest)
    fingerprint = {"config": asdict(config), "manifest_sha256": sha256(manifest)}
    _, val_tracks, run_manifest = bind_run(manifest, fingerprint, asdict(config))
    fixed = [run_manifest["validation_clips"][index * 10 + 5] for index in range(15)]
    if len({row["track"] for row in fixed}) != 15:
        raise ValueError("Diagnostic panel must cover all 15 validation songs")
    write_json(REPORT / "diagnostic_panel.json", fixed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Verified CUDA BF16 execution is unavailable")
    train.configure_determinism(config.seed)
    model = SubbandSeparator().cuda()
    if sum(p.numel() for p in model.parameters()) != 653650:
        raise ValueError("Parameter count differs from the approved model")
    reserve_run(MODELS)
    write_json(REPORT / "start.json", {"timestamp": stamp(), "pid": os.getpid(),
               "config": asdict(config), "loss_weights": LOSS_WEIGHTS, "gpu": initial_gpu,
               "gpu_max_operating_temperature_c": max_temp, "disk_free_gib": free / GIB,
               "protected_files_verified": protected, "protected_model_files": len(before),
               "manifest_sha256": fingerprint["manifest_sha256"],
               "entrypoint": "train.fit(..., max_epoch=config.epochs, resume=False)",
               "warmup_updates": 256, "cosine_floor": 1e-5, "grad_clip": 1.0,
               "early_stopping_patience": 12, "selection_min_delta": 1e-4,
               "optimizer": "AdamW", "betas": [.9, .999], "weight_decay": 1e-4,
               "segment_seconds": 10, "deterministic_cuda": True})
    print(f"Verified preflight inputs. Starting one run {RUN_ID}, PID {os.getpid()}.", flush=True)
    evaluate.score_song = enrich_score(evaluate.score_song)
    store = MonitoredCheckpoints(fingerprint, model, RecordedClips(val_tracks, fixed))
    totals = install_monitor(store, max_temp)
    reason = "completed"
    detail = None
    try:
        train.fit(model, store, manifest, config, LOG / "failures.jsonl",
                  max_epoch=config.epochs, resume=False)
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
        reason, detail = "safety_stop", str(error)
        traceback.print_exc()
    except Exception as error:
        reason, detail = "execution_error", str(error)
        traceback.print_exc()
    state = store.load("latest") if store.commit.exists() else None
    record = store._record() if state else None
    if reason == "completed" and state["epoch"] < config.epochs:
        reason = "early_stopping"
    write_json(REPORT / "training_result.json",
               {"timestamp": stamp(), "reason": reason, "detail": detail,
                "completed_epochs": state["epoch"] if state else 0, "checkpoints": record,
                "history": state["history"] if state else [], **totals()})
    print(f"Training ended at epoch {state['epoch'] if state else 0}: {reason}.", flush=True)
    if state is not None:
        if sha256(MODELS / "best.pt") != record["best"]["sha256"]:
            raise RuntimeError("Best checkpoint alias does not match the committed selection")
        store.model = None
        del model
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < 2 * GIB:
            raise RuntimeError("Evaluation is waiting for at least 2 GiB of GPU headroom")
        print("Evaluating the selected checkpoint on the 15 frozen validation songs.", flush=True)
        evaluate.evaluate_selected(store, manifest, REPORT / "evaluation", "cuda")
    after = checkpoint_hashes()
    if after != before:
        raise RuntimeError("Protected checkpoint inventory changed")
    write_json(REPORT / "protected_models_after.json", after)
    write_json(REPORT / "finished.json", {"timestamp": stamp(), "reason": reason,
               "protected_files_verified": verify_protected(), "protected_model_files": len(after)})
    print("Run artifacts and evaluation are complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        if not (REPORT / "wrapper_error.json").exists():
            write_json(REPORT / "wrapper_error.json", {"timestamp": stamp(), "error": str(error)})
        raise
