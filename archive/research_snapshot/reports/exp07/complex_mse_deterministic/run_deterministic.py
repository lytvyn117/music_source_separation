"""Disposable deterministic resume and timing. Never launches the pilot."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split
from src.experiments.exp07_phase_aware_residual.checkpointing import CheckpointManager, sha256
from src.experiments.exp07_phase_aware_residual.deterministic_mse import configure_determinism, loader, restore_state, save_state
from src.experiments.exp07_phase_aware_residual.losses import LossWeights, make_projected_targets
from src.experiments.exp07_phase_aware_residual.mse_probe_loss import MSEProbeLoss
from src.experiments.exp07_phase_aware_residual.preflight import check_environment, hash_protected_checkpoints, validate_oracle_report
from src.experiments.exp07_phase_aware_residual.trainer import configure_optimizer, to_device, training_step

ROOT = Path("reports/exp07/complex_mse_deterministic")
WEIGHTS = LossWeights(1., .667601, .070409, .008311)
spec = importlib.util.spec_from_file_location("disposable_mse_probe", ROOT.parent / "complex_mse_probe/run_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.ROOT = ROOT


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return {"path": path, "reason": "type", "a": str(type(a)), "b": str(type(b))}
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            result = {"path": path, "reason": "tensor", "a_shape": list(a.shape), "b_shape": list(b.shape)}
            if a.shape == b.shape and a.dtype == b.dtype:
                result["max_abs"] = float((a.double() - b.double()).abs().max())
            return result
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            return {"path": path, "reason": "keys"}
        for key in a:
            result = first_difference(a[key], b[key], f"{path}/{key}")
            if result:
                return result
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return {"path": path, "reason": "length", "a": len(a), "b": len(b)}
        for index, (x, y) in enumerate(zip(a, b)):
            result = first_difference(x, y, f"{path}/{index}")
            if result:
                return result
    elif a != b:
        return {"path": path, "reason": "value", "a": a, "b": b}
    return None


def batch_order(batch):
    return [{"track": name, "start_s": float(start),
             "waveform_sha256": {key: hashlib.sha256(batch[key][j].numpy().tobytes()).hexdigest()
                                 for key in ("mixture", "vocals", "instrumental")}}
            for j, (name, start) in enumerate(zip(batch["track_name"], batch["start"]))]


@torch.no_grad()
def validation(model, criterion, batches):
    model.eval()
    values = []
    order = []
    for batch in batches:
        order.extend(batch_order(batch))
        mix, *refs = to_device(batch, "cuda")
        targets = make_projected_targets(mix, *refs)
        output = model.forward_stft(targets.mixture_stft, mix.shape[-1])
        values.append(float(criterion(output, targets, WEIGHTS).total))
    model.train()
    return sum(values) / len(values), order


def resume_worker(which, work, settings):
    model = probe.new_model()
    frozen = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    optimizer, scheduler = configure_optimizer(model, probe.CONFIG)
    criterion = MSEProbeLoss()
    generator = torch.Generator().manual_seed(42)
    train, val = create_train_validation_split()
    train_loader = loader(AugmentedMUSDBDataset(track_files=train, samples_per_epoch=16, augment=True), generator)
    val_loader = loader(AugmentedMUSDBDataset(track_files=val, samples_per_epoch=8, random_segments=False), generator)
    manager = CheckpointManager(work / ("uninterrupted" if which == "a" else "interrupted"))
    config = {"mode": settings, "weights": asdict(WEIGHTS), "optimizer": asdict(probe.CONFIG), "workers": 0}
    fields = {"config": config, "epoch": 0, "global_step": 0, "history": [], "best_epoch": 0, "patience_counter": 0}
    if which == "b_resume":
        checkpoint = manager.load_latest(config)
        restore_state(checkpoint, model, optimizer, scheduler, generator)
        fields.update({key: checkpoint[key] for key in fields})
    else:
        fields["validation_objective"], _ = validation(model, criterion, val_loader)
        manager.initialize(save_state(model, optimizer, scheduler, generator, **fields))
    trace = []
    final_epoch = 1 if which == "b_stop" else 3
    for epoch in range(fields["epoch"] + 1, final_epoch + 1):
        if fields["patience_counter"] >= probe.CONFIG.patience:
            break
        for batch in train_loader:
            fields["global_step"] += 1
            mix, *refs = to_device(batch, "cuda")
            targets = make_projected_targets(mix, *refs)
            stats = training_step(model, optimizer, criterion, WEIGHTS, mix, targets)
            trace.append({"epoch": epoch, "global_step": fields["global_step"], "sample_order": batch_order(batch),
                          "losses": {k: stats[k] for k in ("complex", "waveform_l1", "mr_stft", "leakage", "total")},
                          "gradients": cpu_tree({n: p.grad for n, p in model.refinement.named_parameters()}),
                          "parameters": cpu_tree(model.refinement.state_dict()),
                          "optimizer": cpu_tree(optimizer.state_dict()),
                          "scheduler": scheduler.state_dict()})
        val_value, order = validation(model, criterion, val_loader)
        scheduler.step()
        improved = val_value < manager.best_value
        fields.update(epoch=epoch, validation_objective=val_value,
                      best_epoch=epoch if improved else fields["best_epoch"],
                      patience_counter=0 if improved else fields["patience_counter"] + 1)
        fields["history"].append({"epoch": epoch, "value": val_value, "sample_order": order})
        state = save_state(model, optimizer, scheduler, generator, **fields)
        manager.maybe_save_best(state, val_value)
        manager.save_latest(state)
        trace.append({"boundary": True, "state": copy_fields(fields),
                      "scheduler": scheduler.state_dict(), "best_value": manager.best_value})
    if any(not torch.equal(v.cpu(), frozen[k]) for k, v in model.backbone.state_dict().items()):
        raise RuntimeError("Frozen Exp04 weights/buffers changed")
    torch.save(trace, work / f"{which}.pt")
    print(f"{which}: PID {os.getpid()}, epoch {fields['epoch']}, step {fields['global_step']}", flush=True)


def copy_fields(fields):
    # Freeze the historical list at each boundary for exact trace comparison.
    import copy
    return copy.deepcopy(fields)


def orchestrate(settings):
    prior_path = ROOT.parent / "complex_mse_formal/result.json"
    old_files = {p: sha256(p) for p in ROOT.parent.rglob("*") if p.is_file() and ROOT not in p.parents
                 and "__pycache__" not in p.parts}
    protected = hash_protected_checkpoints(probe.PATHS.root)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT.parent / "preflight.json").read_text(encoding="utf-8"))["protected_checkpoints"]
    result = {"settings": settings, "configuration": asdict(probe.CONFIG), "weights": asdict(WEIGHTS), "failures": []}
    try:
        if not (len(protected) == 36 and protected == manifest):
            raise RuntimeError("Protected checkpoint mismatch")
        result["environment"] = check_environment(probe.PATHS, probe.CONFIG)
        result["oracle"] = validate_oracle_report(ROOT.parent / "oracle_scale_aware/exp07_oracle_scale_aware_summary.json",
                                                  sha256(probe.PATHS.backbone))
        if not (prior["learnability"]["passed"] and not prior["risk_analysis"]["failures"]):
            raise RuntimeError("Earlier MSE quality gates did not pass")
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "42"
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            work = Path(folder)
            children = {}
            for name in ("a", "b_stop", "b_resume"):
                result["resume_active_stage"] = name
                child = subprocess.Popen([sys.executable, "-B", "-u", __file__, "--worker", name,
                                          "--work", str(work)], env=env)
                children[name] = child.pid
                if child.wait():
                    raise RuntimeError(f"Deterministic worker {name} failed; see subprocess traceback")
            a, stop, resumed = [torch.load(work / f"{name}.pt", weights_only=False)
                                for name in ("a", "b_stop", "b_resume")]
            first = first_difference(a, stop + resumed, "trajectory")
            result.pop("resume_active_stage")
            result["resume"] = {"exact": first is None, "first_difference": first, "pids": children,
                                "epochs": 3, "batches_per_epoch": 4, "updates_after_resume": 8,
                                "worker_count": 0, "samples_per_update": 4}
            if first:
                result["failures"].append("Deterministic CUDA resume diverged")
        # Retiming runs separately and never makes a failed resume pass.
        stats_max = {}
        original_step = probe.training_step
        def measured_step(*args, **kwargs):
            row = original_step(*args, **kwargs)
            for key in ("total_gradient_norm", "gradient_clip_event", "nonfinite_count", "nonfinite_gradients",
                        "complex_violations", "complex_max_error_to_bound", "waveform_max_abs",
                        "waveform_nrmse_pct", "clipped_sample_pct", "spectral_amplification_vs_mixture_p99"):
                stats_max[key] = max(stats_max.get(key, 0), row[key])
            return row
        probe.training_step = measured_step
        torch.cuda.reset_peak_memory_stats()
        result["timing"] = probe.time_disposable_epoch(probe.new_model(), MSEProbeLoss(), WEIGHTS)
        result["timing_stability_maxima"] = stats_max
        if result["timing"]["peak_memory"]["reserved_gib"] > 10.5:
            result["failures"].append("Reserved VRAM exceeded 10.5 GiB")
        if result["timing"]["projected_30_epoch_hours"] > 3:
            result["failures"].append("Projected runtime exceeded three hours")
    except (RuntimeError, ValueError, OSError) as error:
        result["failures"].append(str(error))
    finally:
        result["protected_checkpoints_unchanged"] = protected == hash_protected_checkpoints(probe.PATHS.root)
        result["previous_reports_unchanged"] = all(p.is_file() and sha256(p) == h for p, h in old_files.items())
        result["passed"] = not result["failures"] and result["protected_checkpoints_unchanged"] and result["previous_reports_unchanged"]
        probe.write_json(ROOT / "result.json", result)
    print("PASS" if result["passed"] else f"STOP: {result['failures']}", flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("a", "b_stop", "b_resume"))
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if not args.worker and (ROOT / "result.json").exists():
        raise FileExistsError("Deterministic preflight result exists")
    settings = configure_determinism()
    if args.worker:
        torch.cuda.set_per_process_memory_fraction(min(1., 10.5 * 2**30 /
                                                         torch.cuda.get_device_properties(0).total_memory))
        resume_worker(args.worker, args.work, settings)
    else:
        torch.cuda.set_per_process_memory_fraction(min(1., 10.5 * 2**30 /
                                                         torch.cuda.get_device_properties(0).total_memory))
        raise SystemExit(orchestrate(settings))
