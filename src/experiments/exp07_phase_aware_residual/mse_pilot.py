"""Single-run entry point for the authorized deterministic Exp07 MSE pilot."""

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

from config import VALIDATION_SAMPLES
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split

from .checkpointing import CheckpointManager, sha256
from .config import EXPERIMENT, Exp07Config, Exp07Paths
from .deterministic_mse import configure_determinism, loader, restore_state, save_state
from .diagnostics import compute_gradient_diagnostics, compute_residual_diagnostics
from .losses import COMPONENTS, LossWeights, make_projected_targets
from .model import PhaseAwareResidualSeparator
from .mse_evaluation import evaluate
from .mse_probe_loss import MSEProbeLoss
from .preflight import check_environment, hash_protected_checkpoints, validate_oracle_report
from .trainer import _average, _require_stability, configure_optimizer, to_device


CONFIG = replace(Exp07Config(), lr=3e-4)
WEIGHTS = LossWeights(1., .667601, .070409, .008311)


class PilotSafetyStop(RuntimeError):
    pass


class TimeLimit(RuntimeError):
    pass


def enforce_safety(stats, reserved):
    if not all(math.isfinite(v) for v in stats.values()):
        raise PilotSafetyStop("Nonfinite loss, gradient, or reconstruction diagnostic")
    if reserved > CONFIG.max_reserved_gib:
        raise PilotSafetyStop(f"Reserved VRAM {reserved:.3f} GiB exceeds ceiling")
    if stats["clipped_sample_pct"] > .1:
        raise PilotSafetyStop(f"Output clipping {stats['clipped_sample_pct']:.5f}% exceeds 0.1%")
    if stats["complex_violations"] or stats.get("nonfinite_gradients", 0) or stats.get("nonfinite_count", 0):
        raise PilotSafetyStop("Invalid gradients or mixture reconstruction")
    for key in ("active_amplification_change_db", "high_amplification_change_db"):
        if stats[key] > 3.:
            raise PilotSafetyStop(f"{key} {stats[key]:.4f} exceeds 3 dB")


def stop_reason(patience, elapsed, expected_epoch):
    if patience >= CONFIG.patience:
        return "early_stopping"
    if elapsed + expected_epoch >= CONFIG.max_hours * 3600:
        return "runtime_budget"
    return None


@torch.no_grad()
def amplification(output):
    mixture = output.mixture_stft.abs()
    peak = mixture.amax(dim=(-2, -1), keepdim=True)
    scale = mixture.flatten()[::17].clamp_min(1e-12)
    changes = {}
    for label, fraction in (("active", .001), ("high", .1)):
        active = ((mixture > 0) & (mixture >= fraction * peak)).flatten()[::17]
        values = [0.]
        if bool(active.any()):
            for index, source in enumerate((output.vocals_stft, output.instrumental_stft)):
                before = torch.quantile(output.masks[:, index].flatten()[::17][active], .99)
                after = torch.quantile((source.abs().flatten()[::17] / scale)[active], .99)
                values.append(float(20 * torch.log10(after.clamp_min(1e-12) / before.clamp_min(1e-12))))
        changes[label + "_amplification_change_db"] = max(values)
    return changes


def verify_launch(paths, mode, resume=False):
    results_root = paths.reports / "mse_pilot"
    environment = check_environment(paths, CONFIG, check_collision=not resume)
    if results_root.exists() and not resume:
        raise FileExistsError(f"Pilot report collision: {results_root}")
    if resume and ((results_root / "training_status.json").exists() or (results_root / "evaluation").exists()):
        raise FileExistsError("Pilot has a terminal status or evaluation; resume is disabled")
    preflight = json.loads((paths.reports / "complex_mse_deterministic/result.json").read_text(encoding="utf-8"))
    reference = json.loads((paths.reports / "preflight.json").read_text(encoding="utf-8"))
    protected = hash_protected_checkpoints(paths.root)
    if (preflight.get("passed") is not True or not preflight.get("resume", {}).get("exact")
            or preflight["configuration"] != asdict(CONFIG) or preflight["weights"] != asdict(WEIGHTS)
            or preflight["settings"] != mode or len(protected) != 36
            or protected != reference["protected_checkpoints"]):
        raise ValueError("Deterministic preflight, configuration, or protected checkpoint manifest mismatch")
    validate_oracle_report(paths.reports / "oracle_scale_aware/exp07_oracle_scale_aware_summary.json",
                           sha256(paths.backbone))
    gpu_apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True, timeout=15).stdout
    conflicts = [line for line in gpu_apps.splitlines() if "python" in line.lower()
                 and line.split(",", 1)[0].strip() not in {str(os.getpid()), str(os.getppid())}]
    if conflicts:
        raise RuntimeError(f"GPU already has a Python compute process: {conflicts}")
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < preflight["timing"]["peak_memory"]["reserved_gib"] + .5:
        raise RuntimeError("Insufficient free VRAM for the measured deterministic workload")
    return {"environment": environment | {"free_vram_gib": free / 2**30, "total_vram_gib": total / 2**30},
            "protected_checkpoints": protected, "expected_epoch_seconds": sum(preflight["timing"][k] for k in
                ("train_seconds", "validation_seconds", "checkpoint_seconds")), "settings": mode,
            "configuration": asdict(CONFIG), "weights": asdict(WEIGHTS)}


def _write(path, payload):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temp, path)


def _output(model, criterion, batch, optimizer=None):
    mix, *sources = to_device(batch, "cuda")
    targets = make_projected_targets(mix, *sources)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    prediction = model.forward_stft(targets.mixture_stft, mix.shape[-1])
    loss = criterion(prediction, targets, WEIGHTS)
    if not torch.isfinite(loss.total):
        raise PilotSafetyStop("Nonfinite objective")
    stats = {name: float(getattr(loss, name).detach()) for name in (*COMPONENTS, "total")}
    stats.update(compute_residual_diagnostics(prediction, mix, targets))
    stats.update(amplification(prediction))
    if optimizer is not None:
        loss.total.backward()
        stats.update(compute_gradient_diagnostics(model, 1.0))
        torch.nn.utils.clip_grad_norm_(model.refinement.parameters(), 1.0, error_if_nonfinite=True)
    _require_stability(stats)
    enforce_safety(stats, torch.cuda.max_memory_reserved() / 2**30)
    if optimizer is not None:
        optimizer.step()
    return stats


def train(model, paths, launch, *, resume=False):
    rng = torch.Generator().manual_seed(CONFIG.seed)
    criterion = MSEProbeLoss()
    optimizer, scheduler = configure_optimizer(model, CONFIG)
    train_files, val_files = create_train_validation_split()
    train_loader = loader(AugmentedMUSDBDataset(track_files=train_files, samples_per_epoch=2000, augment=True), rng)
    val_loader = loader(AugmentedMUSDBDataset(track_files=val_files, samples_per_epoch=VALIDATION_SAMPLES,
                                            random_segments=False), rng)
    manager = CheckpointManager(paths.checkpoints)
    config = {k: launch[k] for k in ("settings", "configuration", "weights")} | {"loss": "normalized_complex_mse",
        "head_bf16": True, "num_workers": 0, "exp04_sha256": sha256(paths.backbone)}
    fields = {"config": config, "experiment": EXPERIMENT, "exp04_path": str(paths.backbone), "epoch": 0,
              "best_epoch": 0, "global_step": 0, "patience_counter": 0, "history": [], "elapsed_seconds": 0.}
    start = time.perf_counter()
    spent = 0.
    stage = {"epoch": 0, "batch": 0, "phase": "initial_validation"}
    frozen = {key: tensor.detach().cpu().clone() for key, tensor in model.backbone.state_dict().items()}

    def elapsed():
        return spent + time.perf_counter() - start

    def deadline():
        if elapsed() >= CONFIG.max_hours * 3600:
            raise TimeLimit("Training reached the three-hour limit")

    @torch.inference_mode()
    def validate():
        model.eval()
        stage["phase"] = "validation"
        values = []
        for step, batch in enumerate(val_loader, 1):
            stage["batch"] = step
            deadline()
            values.append(_output(model, criterion, batch))
        return _average(values)

    reason, message = "maximum_epochs", None
    try:
        if resume:
            state = manager.load_latest(config)
            restore_state(state, model, optimizer, scheduler, rng)
            fields.update({key: state[key] for key in (*fields, "validation_objective")})
            spent = fields["elapsed_seconds"]
            # latest is the epoch commit point. Reconcile an interrupted promotion/snapshot.
            manager.maybe_save_best(state, fields["validation_objective"])
            snapshot = paths.checkpoints / f"checkpoint_epoch_{fields['epoch']}.pt"
            if fields["epoch"] > 0 and fields["epoch"] % CONFIG.snapshot_every == 0 and not snapshot.exists():
                manager.save_snapshot(state)
            del state
        else:
            fields["validation_objective"] = validate()["total"]
            fields["elapsed_seconds"] = elapsed()
            manager.initialize(save_state(model, optimizer, scheduler, rng, **fields))
        print(f"Initial/resumed weighted validation objective {fields['validation_objective']:.7f}", flush=True)
        for epoch in range(fields["epoch"] + 1, CONFIG.epochs + 1):
            duration = max([launch["expected_epoch_seconds"]] + [row["seconds"] for row in fields["history"][-3:]])
            stop = stop_reason(fields["patience_counter"], elapsed(), duration)
            if stop:
                reason = stop
                break
            if shutil.disk_usage(paths.root).free / 2**30 < CONFIG.min_free_disk_gib:
                raise PilotSafetyStop("Available disk dropped below 15 GiB")
            stage.update(epoch=epoch, phase="training")
            epoch_start = time.perf_counter()
            model.train()
            rows = []
            lr = optimizer.param_groups[0]["lr"]
            for step, batch in enumerate(train_loader, 1):
                stage["batch"] = step
                deadline()
                rows.append(_output(model, criterion, batch, optimizer))
                if step % 100 == 0:
                    print(f"Epoch {epoch:02d}/30 step {step}/500 loss {rows[-1]['total']:.6f}", flush=True)
            validation = validate()
            deadline()
            if model.backbone.training or any(not torch.equal(t.cpu(), frozen[k])
                                              for k, t in model.backbone.state_dict().items()):
                raise PilotSafetyStop("Frozen Exp04 weights or buffers changed")
            scheduler.step()
            current = validation["total"]
            better = current < manager.best_value
            fields["history"].append({"epoch": epoch, "lr": lr, "seconds": time.perf_counter() - epoch_start,
                "training": _average(rows), "validation": validation, "max_gradient_norm":
                max(row["total_gradient_norm"] for row in rows), "gradient_clip_events":
                sum(row["gradient_clip_event"] for row in rows)})
            fields.update(epoch=epoch, global_step=fields["global_step"] + len(rows), validation_objective=current,
                          best_epoch=epoch if better else fields["best_epoch"],
                          patience_counter=0 if better else fields["patience_counter"] + 1, elapsed_seconds=elapsed())
            state = save_state(model, optimizer, scheduler, rng, **fields)
            manager.save_latest(state)
            manager.maybe_save_best(state, current)
            if epoch % CONFIG.snapshot_every == 0:
                manager.save_snapshot(state)
            _write(paths.logs / "history.json", fields["history"])
            print(f"Epoch {epoch:02d} complete: val={current:.7f}, best={fields['best_epoch']}, "
                  f"patience={fields['patience_counter']}, elapsed={elapsed()/60:.1f} min", flush=True)
            if fields["patience_counter"] >= CONFIG.patience:
                reason = "early_stopping"
                break
    except TimeLimit as exc:
        reason, message = "runtime_budget", str(exc)
    except (FloatingPointError, RuntimeError, OSError) as exc:
        reason, message = "safety_stop", f"{type(exc).__name__}: {exc}"
    return {"reason": reason, "message": message, "last_stage": stage, "elapsed_seconds": elapsed(),
            "latest_completed_epoch": fields["epoch"], "best_epoch": fields["best_epoch"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--launch-approved-pilot", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    flags = configure_determinism()
    paths = Exp07Paths()
    launch = verify_launch(paths, flags, args.resume)
    print(json.dumps({"prelaunch": launch["environment"], "protected_checkpoints": 36,
                      "configuration": launch["configuration"], "weights": launch["weights"]}), flush=True)
    if args.check:
        return 0
    root = paths.reports / "mse_pilot"
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
        paths.logs.mkdir(parents=True, exist_ok=False)
        _write(root / "launch.json", launch)
    elif launch["protected_checkpoints"] != json.loads((root / "launch.json").read_text(
            encoding="utf-8"))["protected_checkpoints"]:
        raise ValueError("Protected checkpoint manifest differs from original launch")
    torch.cuda.set_per_process_memory_fraction(min(1., CONFIG.max_reserved_gib * 2**30 /
                                                    torch.cuda.get_device_properties(0).total_memory))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = PhaseAwareResidualSeparator.from_exp04(paths.backbone, "cuda")
    model.refinement.to(memory_format=torch.channels_last)
    model.use_bf16 = True
    status = train(model, paths, launch, resume=args.resume)
    status["protected_checkpoints_unchanged"] = (hash_protected_checkpoints(paths.root) ==
                                                 launch["protected_checkpoints"])
    _write(root / "training_status.json", status)
    print(f"Training stopped: {status}", flush=True)
    if not status["protected_checkpoints_unchanged"]:
        raise ValueError("Protected checkpoint changed; evaluation cancelled")
    best_path = paths.checkpoints / "best.pt"
    if not best_path.is_file():
        return 2
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    torch.cuda.empty_cache()
    decision = evaluate(model, root / "evaluation", {"epoch": best["epoch"], "sha256": sha256(best_path),
                        "validation_objective": best["validation_objective"], "stop_reason": status["reason"]})
    del model
    torch.cuda.empty_cache()
    integrity = hash_protected_checkpoints(paths.root) == launch["protected_checkpoints"]
    _write(root / "completion.json", {"evaluation": decision, "protected_checkpoints_unchanged": integrity})
    print(f"Evaluation finished. Protected checkpoints unchanged: {integrity}", flush=True)
    return 0 if integrity and status["reason"] != "safety_stop" else 2


if __name__ == "__main__":
    raise SystemExit(main())
