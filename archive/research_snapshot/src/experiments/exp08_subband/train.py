"""Deterministic Exp08 optimization, validation, and epoch-boundary resume."""
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.audio.stft import compute_stft
from src.training.experiment_selection import SelectionState, selection_value, should_stop, update_selection

from .checkpointing import restore_rng, rng_state
from .data import RecordedClips, TrainClips, bind_run
from .loss import LOSS_WEIGHTS, active_vocal_frames, composite_loss, project_vocals
from .model import MASK_LIMIT, synthesize


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    lr: float = 3e-4
    microbatch: int = 2
    accumulation: int = 4
    samples_per_epoch: int = 2048
    epochs: int = 60
    bf16: bool = True

    @property
    def steps_per_epoch(self):
        return self.samples_per_epoch // (self.microbatch * self.accumulation)


def configure_determinism(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)


def make_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(.9, .999), weight_decay=1e-4)


def make_scheduler(optimizer, config):
    def factor(step):
        if step < 256:
            return (step + 1) / 256
        fraction = min(1., (step - 255) / max(1, config.steps_per_epoch * config.epochs - 255))
        floor = 1e-5 / config.lr
        return floor + .5 * (1 - floor) * (1 + math.cos(math.pi * fraction))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_loader(dataset, config, workers=2, training=True):
    batch_size = config.microbatch * config.accumulation if training else config.microbatch
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, generator=generator,
                      pin_memory=torch.cuda.is_available(), drop_last=training)


@torch.no_grad()
def guard(mixture, spectra, waveform, predicted, *, check=True):
    mixture_abs = mixture.abs()
    vocal_abs, inst_abs = (s.abs() for s in spectra.unbind(1))
    roundoff = (4 * 2**-24) * (mixture_abs + vocal_abs + inst_abs)
    complex_error = ((spectra.sum(1) - mixture).abs() / (1e-5 + roundoff)).amax()
    amplification = (torch.maximum(vocal_abs, inst_abs) - MASK_LIMIT * mixture_abs - roundoff).amax()
    waveform_error = (predicted.sum(1) - waveform).abs().amax()
    waveform_tolerance = 2e-5 * waveform.abs().amax().clamp_min(1.)
    values = torch.stack((complex_error, amplification, waveform_error / waveform_tolerance,
                          waveform_error))
    if not check:
        return values
    _check_guard(values.tolist())
    return values


def _check_guard(values):
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Nonfinite spectrum or waveform reconstruction")
    complex_error, amplification, waveform_ratio, _ = values
    if amplification > 0:
        raise FloatingPointError("Spectral amplification exceeded the mask bound")
    if max(complex_error, waveform_ratio) > 1:
        raise FloatingPointError("Mixture reconstruction tolerance exceeded")


def _forward(model, audio, projected_wave, config, effective_active=None):
    mixture_wave = audio[:, 0]
    batch, stems, channels, length = audio.shape
    stft = compute_stft(audio.reshape(batch * stems, channels, length))
    mix, vocal, inst = stft.reshape(batch, stems, channels, *stft.shape[-2:]).unbind(1)
    with torch.autocast(mix.device.type, dtype=torch.bfloat16,
                        enabled=config.bf16 and mix.device.type == "cuda"):
        spectra, mask = model.separate_spectrum(mix)
    est_vocal = synthesize(spectra[:, 0], length)
    with torch.no_grad():
        est_inst = synthesize(spectra[:, 1], length)
        predicted = torch.stack((est_vocal.detach(), est_inst), 1)
        checks = guard(mix, spectra, mixture_wave, predicted, check=False)
    total, terms = composite_loss(
        mix, spectra[:, 0], spectra[:, 1], project_vocals(mix, vocal, inst),
        mixture_wave, est_vocal, projected_wave, effective_active=effective_active,
        accumulation=config.accumulation if effective_active is not None else 1,
    )
    with torch.no_grad():
        reference_power = audio[:, 1:].square().mean((2, 3))
        ratio = predicted.square().mean((2, 3)).clamp_min(1e-12) / reference_power.clamp_min(1e-12)
        gain = 10 * ratio.log10().abs()
        diagnostics = torch.cat((
            gain.amax(0),
            (predicted.abs() > 1).float().mean((0, 2, 3)),
            predicted.mean(dim=-1).abs().amax((0, 2)),
            (.5 * (predicted[:, :, 0] - predicted[:, :, 1])).square().mean((0, 2)).sqrt(),
            torch.stack((((mask.real - .5).abs() >= .65 * .99).float().mean(),
                         (mask.imag.abs() >= .25 * .99).float().mean())),
            gain.masked_fill(reference_power < .005**2, 0).amax(0),
            predicted.abs().amax((0, 2, 3)),
        ))
    return total, terms, checks, diagnostics


def _norm(tensors):
    return torch.linalg.vector_norm(torch.stack([value.detach().norm() for value in tensors]))


def optimizer_update(model, optimizer, audio, config, detailed=False):
    expected = config.microbatch * config.accumulation
    if audio.shape[0] != expected:
        raise ValueError(f"Expected {expected} clips for one optimizer update")
    model.train()
    params = tuple(model.parameters())
    device = params[0].device
    audio = audio.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    projected_wave = project_vocals(*audio.unbind(1))
    active = active_vocal_frames(projected_wave).sum()
    sums, checks, diagnostics = [], [], []
    components = {}
    for batch, target in zip(audio.split(config.microbatch), projected_wave.split(config.microbatch)):
        total, terms, safety, signal = _forward(model, batch, target, config, active)
        if detailed:
            for name, term in terms.items():
                grads = torch.autograd.grad(term * (LOSS_WEIGHTS[name] / config.accumulation),
                                            params, retain_graph=True)
                if name in components:
                    for running, grad in zip(components[name], grads, strict=True):
                        running.add_(grad.detach())
                else:
                    components[name] = [grad.detach() for grad in grads]
        (total / config.accumulation).backward()
        sums.append(torch.stack((total.detach(), *(term.detach() for term in terms.values()))))
        checks.append(safety)
        diagnostics.append(signal)
    losses = torch.stack(sums).mean(0)
    limits = torch.stack(checks).amax(0)
    signal = torch.stack(diagnostics).amax(0)
    values = torch.cat((losses, limits, signal)).tolist()
    loss_values = values[:len(losses)]
    _check_guard(values[len(losses):len(losses)+4])
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Nonfinite training inputs, loss, or signal diagnostics")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite accumulated gradient norm")
    # Measure the actual AdamW displacement on every timed update.
    with torch.no_grad():
        before = torch.nn.utils.parameters_to_vector(params)
        branch_norms = {}
        if detailed:
            factor = min(1., 1. / (grad_norm + 1e-6))
            for name, module in model.named_children():
                branch_norms[name] = float(_norm(p.grad for p in module.parameters())) / factor
        optimizer.step()
        after = torch.nn.utils.parameters_to_vector(params)
        update_norm = float((after - before).norm())
        parameter_norm = float(before.norm())
    if not math.isfinite(update_norm):
        raise FloatingPointError("Nonfinite AdamW parameter update")
    return {"loss": dict(zip(("total", *LOSS_WEIGHTS), loss_values)),
            "preclip_norm": grad_norm, "clipped": grad_norm > 1.,
            "update_norm": update_norm, "relative_update": update_norm / parameter_norm,
            "guard": values[len(losses):len(losses)+4],
            "signal_max_by_microbatch": values[len(losses)+4:],
            "branch_gradient_norms": branch_norms,
            "component_gradient_norms": {key: float(_norm(grads)) for key, grads in components.items()}}


@torch.inference_mode()
def validate(model, batches, config):
    model.eval()
    totals = defaultdict(float)
    samples = active_frames = 0
    device = next(model.parameters()).device
    for batch in batches:
        audio = batch["audio"].to(device, non_blocking=True)
        target = project_vocals(*audio.unbind(1))
        count = int(active_vocal_frames(target).sum())
        _, terms, limits, _ = _forward(model, audio, target, config)
        _check_guard(limits.tolist())
        for name, term in terms.items():
            totals[name] += float(term) * (count if name == "vocal_leakage" else len(audio))
        samples += len(audio)
        active_frames += count
    if samples == 0:
        raise ValueError("Validation manifest is empty")
    means = {name: value / (max(1, active_frames) if name == "vocal_leakage" else samples)
             for name, value in totals.items()}
    means["val_total_loss"] = sum(LOSS_WEIGHTS[name] * value for name, value in means.items())
    selection_value(means)
    return means


def save_epoch(store, model, optimizer, scheduler, epoch, selection, history, config):
    state = {"epoch": epoch, "selection": asdict(selection), "history": list(history),
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "rng": rng_state(),
             "sampler": {"epoch": epoch + 1, "index": 0, "seed": config.seed}}
    return store.save(state)


def resume_epoch(store, model, optimizer, scheduler, config):
    if store.fingerprint.get("config") != asdict(config):
        raise ValueError("Checkpoint training configuration changed")
    state = store.load()
    expected = {"epoch": state["epoch"] + 1, "index": 0, "seed": config.seed}
    if state["sampler"] != expected:
        raise ValueError("Checkpoint sampler is not at the expected epoch boundary")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state["epoch"], SelectionState(**state["selection"]), list(state["history"])


def fit(model, store, manifest_path, config, failure_path, *, max_epoch, resume=False):
    train_tracks, val_tracks, manifest = bind_run(manifest_path, store.fingerprint, asdict(config))
    val_data = RecordedClips(val_tracks, manifest["validation_clips"])
    val_batches = make_loader(val_data, config, training=False)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    if resume:
        last_epoch, selection, history = resume_epoch(store, model, optimizer, scheduler, config)
    else:
        validation = validate(model, val_batches, config)
        last_epoch, selection, history = 0, SelectionState(selection_value(validation)), [validation]
        save_epoch(store, model, optimizer, scheduler, 0, selection, history, config)
    for epoch in range(last_epoch + 1, min(max_epoch, config.epochs) + 1):
        if should_stop(selection, 12):
            break
        train_data = TrainClips(train_tracks, epoch, samples=config.samples_per_epoch, seed=config.seed)
        running_loss = 0.
        signal_max = guard_max = None
        clipped_updates = 0
        preclip_max = 0.
        for step, batch in enumerate(make_loader(train_data, config), 1):
            try:
                stats = optimizer_update(model, optimizer, batch["audio"], config)
            except FloatingPointError as error:
                record_failure(failure_path, epoch, step, batch, error)
                raise
            running_loss += stats["loss"]["total"]
            signals = np.asarray(stats["signal_max_by_microbatch"])
            guards = np.asarray(stats["guard"])
            signal_max = signals if signal_max is None else np.maximum(signal_max, signals)
            guard_max = guards if guard_max is None else np.maximum(guard_max, guards)
            clipped_updates += int(stats["clipped"])
            preclip_max = max(preclip_max, stats["preclip_norm"])
            scheduler.step()
        validation = validate(model, val_batches, config)
        selection, _ = update_selection(selection, selection_value(validation), epoch, 1e-4)
        history.append({"epoch": epoch, "train_total_loss": running_loss / step, **validation,
                        "signal_max": signal_max.tolist(), "guard_max": guard_max.tolist(),
                        "clipped_updates": clipped_updates, "preclip_max": preclip_max})
        save_epoch(store, model, optimizer, scheduler, epoch, selection, history, config)
    return selection, history


def record_failure(path, epoch, step, batch, error):
    metadata = {key: value.tolist() if torch.is_tensor(value) else value
                for key, value in batch.items() if key != "audio"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"epoch": epoch, "step": step, "batch": metadata,
                                 "error": str(error)}) + "\n")
