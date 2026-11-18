import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import VALIDATION_SAMPLES
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split

from .checkpointing import CheckpointManager, capture_rng_state, restore_rng_state, sha256
from .config import EXPERIMENT, Exp07Config, Exp07Paths, NumericalTolerances
from .diagnostics import compute_gradient_diagnostics, compute_residual_diagnostics
from .losses import COMPONENTS, LossWeights, PhaseAwareLoss, make_projected_targets
from .model import PhaseAwareResidualSeparator


@dataclass(frozen=True)
class EpochMetrics:
    steps: int
    metrics: dict


def configure_optimizer(model, config):
    optimizer = torch.optim.AdamW(model.refinement.parameters(), lr=config.lr)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)


def to_device(batch, device):
    if isinstance(batch, dict):
        batch = tuple(batch[key] for key in ("mixture", "vocals", "instrumental"))
    return tuple(t.to(device, dtype=torch.float32) for t in batch)


def training_step(model, optimizer, criterion, weights, mixture, targets, *, masks=None, diagnostics=True):
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_stft(targets.mixture_stft, mixture.shape[-1], masks=masks)
    loss = criterion(output, targets, weights)
    if not torch.isfinite(loss.total):
        raise FloatingPointError("Non-finite training objective")
    loss.total.backward()
    gradient = compute_gradient_diagnostics(model, 1.0) if diagnostics else {}
    torch.nn.utils.clip_grad_norm_(model.refinement.parameters(), 1.0, error_if_nonfinite=True)
    if diagnostics:
        gradient.update(compute_residual_diagnostics(output, mixture, targets))
        _require_stability(gradient)
    optimizer.step()
    return {key: float(getattr(loss, key).detach()) for key in (*COMPONENTS, "total")} | gradient


def _require_stability(stats):
    if any(not math.isfinite(v) for v in stats.values()) or stats["nonfinite_count"]:
        raise FloatingPointError("Non-finite Exp07 reconstruction")
    tolerance = NumericalTolerances()
    if (stats["complex_violations"] or stats["waveform_max_abs"] > tolerance.waveform_max_abs
            or stats["waveform_nrmse_pct"] > tolerance.waveform_nrmse_pct):
        raise FloatingPointError("Exp07 mixture-additivity tolerance exceeded")


def _average(rows):
    if not rows:
        raise ValueError("Expected at least one batch")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_epoch(model, batches, optimizer, weights):
    model.train()
    criterion = PhaseAwareLoss()
    device = next(model.parameters()).device
    rows = []
    for batch in batches:
        mixture, vocals, instrumental = to_device(batch, device)
        targets = make_projected_targets(mixture, vocals, instrumental)
        rows.append(training_step(model, optimizer, criterion, weights, mixture, targets))
    return EpochMetrics(len(rows), _average(rows))


@torch.inference_mode()
def validate_epoch(model, batches, weights):
    model.eval()
    criterion = PhaseAwareLoss()
    rows = []
    for batch in batches:
        mixture, vocals, instrumental = to_device(batch, next(model.parameters()).device)
        targets = make_projected_targets(mixture, vocals, instrumental)
        output = model.forward_stft(targets.mixture_stft, mixture.shape[-1])
        loss = criterion(output, targets, weights)
        row = {key: float(getattr(loss, key)) for key in COMPONENTS}
        row["validation_objective"] = float(loss.total)
        stats = compute_residual_diagnostics(output, mixture, targets)
        _require_stability(stats)
        rows.append(row | stats)
    return EpochMetrics(len(rows), _average(rows))


def training_state(model, optimizer, scheduler, generator, **fields):
    return dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=None,
                rng_state=capture_rng_state(generator), **fields)


def restore_training_state(state, model, optimizer, scheduler, generator):
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng_state(state["rng_state"], generator)


def run_pilot(config, paths, preflight, resume=False):
    if preflight.get("passed") is not True or preflight.get("config") != asdict(config):
        raise ValueError("A passing preflight for the exact pilot configuration is required")
    from .preflight import check_environment, hash_protected_checkpoints

    check_environment(paths, config, check_collision=not resume)
    if hash_protected_checkpoints(paths.root) != preflight.get("protected_checkpoints"):
        raise ValueError("Protected checkpoint manifest changed")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, config.max_reserved_gib * 2**30 / total_memory))
    model = PhaseAwareResidualSeparator.from_exp04(paths.backbone, "cuda")
    model.refinement.to(memory_format=torch.channels_last)
    model.use_bf16 = preflight["benchmark"]["head_bf16"]
    weights = LossWeights(**preflight["calibration"]["weights"])
    optimizer, scheduler = configure_optimizer(model, config)
    train, validation = create_train_validation_split()
    train_loader = DataLoader(AugmentedMUSDBDataset(track_files=train, samples_per_epoch=config.samples_per_epoch,
                                                   augment=True), batch_size=config.batch_size,
                              num_workers=0, generator=generator)
    validation_loader = DataLoader(AugmentedMUSDBDataset(track_files=validation, samples_per_epoch=VALIDATION_SAMPLES,
                                                        random_segments=False), batch_size=config.batch_size,
                                   num_workers=0, generator=generator)
    manager = CheckpointManager(paths.checkpoints)
    run_config = dict(pilot=asdict(config), weights=asdict(weights), head_bf16=model.use_bf16,
                      exp04_sha256=sha256(paths.backbone))
    fields = dict(experiment=EXPERIMENT, config=run_config, exp04_path=str(paths.backbone),
                  exp04_sha256=run_config["exp04_sha256"], global_step=0, epoch=0, history=[],
                  best_epoch=0, patience_counter=0, elapsed_seconds=0.0,
                  deterministic_cuda=torch.are_deterministic_algorithms_enabled(),
                  selection_metric="validation_objective")
    if resume:
        state = manager.load_latest(run_config)
        restore_training_state(state, model, optimizer, scheduler, generator)
        fields.update({key: state[key] for key in fields})
    else:
        fields["validation_objective"] = validate_epoch(model, validation_loader, weights).metrics["validation_objective"]
        manager.initialize(training_state(model, optimizer, scheduler, generator, **fields))
    paths.logs.mkdir(parents=True, exist_ok=resume)
    prior = fields["elapsed_seconds"]
    start = time.perf_counter()
    for epoch in range(fields["epoch"] + 1, config.epochs + 1):
        projected_epoch = preflight["benchmark"]["projected_epoch_seconds"]
        elapsed = prior + time.perf_counter() - start
        if fields["patience_counter"] >= config.patience or elapsed + projected_epoch > config.max_hours * 3600:
            break
        training = train_epoch(model, train_loader, optimizer, weights)
        validation = validate_epoch(model, validation_loader, weights)
        scheduler.step()
        objective = validation.metrics["validation_objective"]
        improved = objective < manager.best_value
        fields["history"].append(dict(epoch=epoch, training=training.metrics, validation=validation.metrics))
        fields.update(epoch=epoch, global_step=fields["global_step"] + training.steps,
                      validation_objective=objective, best_epoch=epoch if improved else fields["best_epoch"],
                      patience_counter=0 if improved else fields["patience_counter"] + 1,
                      elapsed_seconds=prior + time.perf_counter() - start)
        state = training_state(model, optimizer, scheduler, generator, **fields)
        manager.maybe_save_best(state, objective)
        manager.save_latest(state)
        if epoch % config.snapshot_every == 0:
            manager.save_snapshot(state)
        (paths.logs / "history.json").write_text(json.dumps(fields["history"], indent=2), encoding="utf-8")
    if hash_protected_checkpoints(paths.root) != preflight["protected_checkpoints"]:
        raise ValueError("Protected checkpoint manifest changed during pilot")


def main():
    parser = argparse.ArgumentParser(description="Run the separately approved Exp07 pilot")
    parser.add_argument("--launch-approved-pilot", action="store_true", required=True)
    parser.add_argument("--require-preflight", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_pilot(Exp07Config(), Exp07Paths(), json.loads(args.require_preflight.read_text(encoding="utf-8")), args.resume)


if __name__ == "__main__":
    main()
