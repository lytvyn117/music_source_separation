"""Disposable Exp07 diagnostics. Run from the repo root with the project venv."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.audio.stft import compute_istft
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split
from src.experiments.exp07_phase_aware_residual.complex_ops import safe_complex_rms
from src.experiments.exp07_phase_aware_residual.complex_ops import complex_to_channels, channels_to_complex
from src.experiments.exp07_phase_aware_residual.config import Exp07Config, Exp07Paths
from src.experiments.exp07_phase_aware_residual.diagnostics import compute_gradient_diagnostics, compute_residual_diagnostics
from src.experiments.exp07_phase_aware_residual.losses import COMPONENTS, LossWeights, PhaseAwareLoss, make_projected_targets
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator
from src.experiments.exp07_phase_aware_residual.preflight import _fixed_batch, hash_protected_checkpoints
from src.experiments.exp07_phase_aware_residual.trainer import configure_optimizer, to_device, validate_epoch


ROOT = Path("reports/exp07")
CONFIG = Exp07Config()
SEED = 42


def seed():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def model_new():
    seed()
    model = PhaseAwareResidualSeparator.from_exp04(Exp07Paths().backbone, "cuda")
    model.refinement.to(memory_format=torch.channels_last)
    model.use_bf16 = True
    return model.train()


def rms(x):
    return float(x.abs().square().mean().sqrt())


def quantiles(x):
    return np.percentile(x.detach().float().cpu().flatten().numpy(), [0, 50, 90, 95, 99, 99.9, 100]).tolist()


def real_vector(t):
    return (torch.view_as_real(t) if t.is_complex() else t).float().flatten()


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def sdr(pred, target):
    return float(10 * torch.log10(target.square().sum() / (target - pred).square().sum().clamp_min(1e-12)))


def gradient_alignment(model, x, masks, targets, criterion, weights, ideal, length):
    pred = model.forward_stft(x, length, masks=masks)
    result = criterion(pred, targets, weights)
    output_gradients, parameter_gradients = {}, {}
    for name in COMPONENTS:
        gradients = torch.autograd.grad(getattr(result, name), (pred.residual_stft, *model.refinement.final.parameters()),
                                        retain_graph=True)
        weight = getattr(weights, name)
        output_gradients[name] = real_vector(gradients[0]) * weight
        parameter_gradients[name] = torch.cat([real_vector(g) for g in gradients[1:]]) * weight
    report = {}
    for name, vectors in (("output", output_gradients), ("final_parameter", parameter_gradients)):
        combined = sum(vectors.values())
        report[name] = {"weighted_norm": {k: float(v.norm()) for k, v in vectors.items()},
                        "cosine": {k: {l: cosine(v, w) for l, w in vectors.items()} for k, v in vectors.items()},
                        "combined_over_sum_norm": float(combined.norm() / sum(v.norm() for v in vectors.values())),
                        "combined_dot_complex_over_complex_norm2":
                        float(torch.dot(combined, vectors["complex"]) / vectors["complex"].square().sum())}
    descent = real_vector(ideal - pred.residual_stft.detach())
    report["alignment_with_ideal_remaining_correction"] = {
        k: cosine(-v, descent) for k, v in output_gradients.items()
    }
    report["residual_output_gradient_abs_quantiles"] = {
        k: quantiles(v.abs()) for k, v in output_gradients.items()
    }
    return report


def one_segment(manifest, weights, steps, experiments):
    mixture, vocal, instrumental = _fixed_batch([manifest])
    targets = make_projected_targets(mixture, vocal, instrumental)
    x = targets.mixture_stft
    scale = safe_complex_rms(x)
    criterion = PhaseAwareLoss()
    result = {}
    for name, objective, lr, gain, bf16 in experiments:
        model = model_new()
        model.use_bf16 = bf16
        masks = model.backbone_masks(x)
        v0 = x * masks[:, 0]
        ideal = targets.vocals_stft - v0
        length = mixture.shape[-1]
        params = dict(model.refinement.named_parameters())
        originals = {k: v.detach().clone() for k, v in params.items()}
        optimizer, _ = configure_optimizer(model, replace(CONFIG, lr=lr))
        if gain != 1.0:
            # Change only output parameterization, with the same zero initialization.
            model.refinement.final.register_forward_hook(lambda module, args, out: out.float() * gain)

        if not result:
            feature_stats = []
            hook = model.refinement.stem.register_forward_pre_hook(
                lambda mod, args: feature_stats.append({"shape": list(args[0].shape),
                    "rms_by_channel": args[0].float().square().mean(dim=(0, 2, 3)).sqrt().tolist()}))
            with torch.no_grad():
                model.forward_stft(x, masks=masks)
            hook.remove()
            energy = ideal.abs().square().flatten()
            top = energy.topk(max(1, energy.numel() // 100)).values
            result["target"] = {"shape": list(x.shape), "mixture_rms_per_channel": scale.flatten().tolist(),
                                "ideal_residual_rms": rms(ideal), "initial_vocal_rms": rms(v0),
                                "ideal_vs_initial_vocal_rms": rms(ideal) / rms(v0),
                                "ideal_abs_quantiles": quantiles(ideal.abs()),
                                "ideal_normalized_abs_quantiles": quantiles((ideal / scale).abs()),
                                "top_1pct_bins_energy_share": float(top.sum() / energy.sum()),
                                "features": feature_stats[0]}
        trace, layers, alignments = [], {}, {}
        selected = {0, 1, 2, 10, 25, 50, 100, steps}
        for step in range(steps + 1):
            if step in selected:
                with torch.no_grad():
                    pred = model.forward_stft(x, length, masks=masks)
                    values = criterion(pred, targets, weights)
                    trace.append({"step": step, **{k: float(getattr(values, k)) for k in (*COMPONENTS, "total")},
                                  "vocal_sdr_projected": sdr(pred.vocals_waveform, targets.vocals_waveform),
                                  "vocal_sdr_original": sdr(pred.vocals_waveform, vocal),
                                  "predicted_vs_ideal_rms": rms(pred.residual_stft) / rms(ideal),
                                  "relative_residual_rmse": rms(pred.residual_stft - ideal) / rms(ideal),
                                  "correction_cosine": cosine(real_vector(pred.residual_stft), real_vector(ideal)),
                                  "peak": float(pred.vocals_waveform.abs().max())})
                del pred, values
            if objective in ("complex_l1", "composite") and gain == 1.0 and lr == CONFIG.lr and bf16 and step in (0, 2, steps):
                alignments[str(step)] = gradient_alignment(model, x, masks, targets, criterion, weights, ideal, length)
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_stft(x, length if objective == "composite" else None, masks=masks)
            difference = torch.view_as_real((pred.residual_stft - ideal) / scale)
            if objective == "composite":
                loss = criterion(pred, targets, weights).total
            elif objective == "complex_l2":
                loss = difference.square().mean()
            else:
                loss = sum(torch.view_as_real((getattr(pred, key) - getattr(targets, key)) / scale).abs().mean()
                           for key in ("vocals_stft", "instrumental_stft")) * 0.5
            loss.backward()
            capture = step + 1 in selected
            if capture:
                before = {k: v.detach().clone() for k, v in params.items()}
                gradient = {k: float(v.grad.norm()) for k, v in params.items()}
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if capture:
                layers[str(step + 1)] = {k: {"gradient_l2": gradient[k],
                    "parameter_l2": float(before[k].norm()), "update_l2": float((v.detach() - before[k]).norm()),
                    "relative_update": float((v.detach() - before[k]).norm() / before[k].norm()) if before[k].norm() else None,
                    "displacement_from_initial_l2": float((v.detach() - originals[k]).norm())}
                    for k, v in params.items()}
            del pred, difference, loss
        assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())
        result[name] = {"objective": objective, "lr": lr, "output_gain": gain, "bf16": bf16, "trace": trace,
                        "per_parameter": layers, "gradient_alignment": alignments}
        print(name, trace[-1], flush=True)
        del model, optimizer, params
        torch.cuda.empty_cache()
    return result


def profile(weights, steps):
    model = model_new()
    opt, _ = configure_optimizer(model, CONFIG)
    loss_fn = PhaseAwareLoss()
    train, validation = create_train_validation_split()
    loader = iter(DataLoader(AugmentedMUSDBDataset(track_files=train, samples_per_epoch=4 * (steps + 4),
                                                  augment=True), batch_size=4, num_workers=0))
    rows = []
    for iteration in range(steps + 4):
        wall = time.perf_counter()
        raw = next(loader)
        loaded = time.perf_counter()
        mix, *stems = to_device(raw, "cuda")
        moved = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        events[0].record()
        targets = make_projected_targets(mix, *stems)
        events[1].record()
        masks = model.backbone_masks(targets.mixture_stft)
        events[2].record()
        pred = model.forward_stft(targets.mixture_stft, masks=masks)
        events[3].record()
        waves = compute_istft(torch.cat((pred.vocals_stft, pred.instrumental_stft), dim=1),
                             length=mix.shape[-1]).split(2, dim=1)
        pred = replace(pred, vocals_waveform=waves[0], instrumental_waveform=waves[1])
        events[4].record()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(pred, targets, weights)
        events[5].record()
        if not torch.isfinite(loss.total):
            raise FloatingPointError("nonfinite")
        loss.total.backward()
        events[6].record()
        pre_sync = time.perf_counter()
        events[6].synchronize()
        synced = time.perf_counter()
        compute_gradient_diagnostics(model, 1.0)
        torch.nn.utils.clip_grad_norm_(model.refinement.parameters(), 1.0, error_if_nonfinite=True)
        compute_residual_diagnostics(pred, mix, targets)
        opt.step()
        torch.cuda.synchronize()
        done = time.perf_counter()
        names = ("targets_stft_istft", "backbone", "head", "prediction_istft", "loss", "backward")
        if iteration >= 4:
            row = {name + "_gpu_ms": a.elapsed_time(b) for name, a, b in zip(names, events, events[1:])}
            rows.append(row | {"iteration": iteration, "total_wall_ms": (done-wall)*1000,
                               "data_loading_ms": (loaded-wall)*1000, "device_transfer_ms": (moved-loaded)*1000,
                               "backward_sync_wait_ms": (synced-pre_sync)*1000, "diagnostics_optimizer_ms": (done-synced)*1000,
                               "enqueue_through_backward_ms": (pre_sync-moved)*1000})
        del pred, targets, loss, waves, masks
    start = time.perf_counter()
    val = DataLoader(AugmentedMUSDBDataset(track_files=validation, samples_per_epoch=8, random_segments=False),
                     batch_size=4, num_workers=0)
    validate_epoch(model, val, weights)
    torch.cuda.synchronize()
    validation_ms = (time.perf_counter() - start)*1000
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        start = time.perf_counter()
        path = Path(tmp) / "diagnostic.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, path)
        save_ms, size = (time.perf_counter() - start)*1000, path.stat().st_size
    return {"rows": rows, "validation_two_batches_ms": validation_ms, "temporary_checkpoint_ms": save_ms,
            "temporary_checkpoint_bytes": size, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


@torch.no_grad()
def readout_ceiling(model, targets, masks):
    """In-sample least-squares readouts; no model changes or new parameters installed."""
    cache = []
    hook = model.refinement.final.register_forward_pre_hook(lambda m, a: cache.append(a[0].float()))
    try:
        output = model.forward_stft(targets.mixture_stft, masks=masks)
    finally:
        hook.remove()
    h = cache[0][0].permute(1, 2, 0).contiguous()
    ideal = targets.vocals_stft - output.initial_vocal_stft
    scale = safe_complex_rms(targets.mixture_stft)
    y = complex_to_channels(ideal / scale)[0].permute(1, 2, 0).contiguous()
    initial_sdr = sdr(compute_istft(output.initial_vocal_stft, length=targets.vocals_waveform.shape[-1]),
                      targets.vocals_waveform)
    scores = {}
    for name, groups in (("shared_existing_readout", 1), ("frequency_grouped_offline_control", 32)):
        predictions = []
        for band in np.array_split(np.arange(h.shape[0]), groups):
            start, end = int(band[0]), int(band[-1]) + 1
            a, b = h[start:end].flatten(0, 1), y[start:end].flatten(0, 1)
            am, bm = a.mean(0), b.mean(0)
            ac = a - am
            gram = ac.T @ ac
            ridge = gram.trace() / gram.shape[0] * 1e-6
            coeff = torch.linalg.solve(gram + torch.eye(gram.shape[0], device=gram.device) * ridge,
                                       ac.T @ (b - bm))
            predictions.append(((a - am) @ coeff + bm).reshape(end - start, h.shape[1], 4))
        residual = channels_to_complex(torch.cat(predictions).permute(2, 0, 1)[None]) * scale
        wav = compute_istft(output.initial_vocal_stft + residual, length=targets.vocals_waveform.shape[-1])
        scores[name] = {"relative_residual_rmse": rms(residual - ideal) / rms(ideal),
                        "vocal_sdr_gain_db": sdr(wav, targets.vocals_waveform) - initial_sdr}
    return scores


def capacity_control(manifest, steps):
    mix, v, i = _fixed_batch([manifest])
    targets = make_projected_targets(mix, v, i)
    model = model_new()
    masks = model.backbone_masks(targets.mixture_stft)
    before = readout_ceiling(model, targets, masks)
    opt, _ = configure_optimizer(model, replace(CONFIG, lr=3e-4))
    scale = safe_complex_rms(targets.mixture_stft)
    trace = []
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model.forward_stft(targets.mixture_stft, masks=masks)
        delta = out.vocals_stft - targets.vocals_stft
        objective = torch.view_as_real(delta / scale).square().mean()
        if step in (0, 25, 50, 100, steps):
            with torch.no_grad():
                trace.append({"step": step, "complex_l2": float(objective),
                              "relative_residual_rmse": rms(delta) / rms(targets.vocals_stft - out.initial_vocal_stft),
                              "vocal_sdr_db": sdr(compute_istft(out.vocals_stft, length=mix.shape[-1]),
                                                  targets.vocals_waveform)})
        if step < steps:
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.refinement.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
    return {"initial_frozen_features": before, "after_training_frozen_features": readout_ceiling(model, targets, masks),
            "trace": trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "profile", "capacity", "controls"))
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 300:
        parser.error("require a new report path and 1–300 diagnostic steps")
    torch.cuda.set_per_process_memory_fraction(min(1, CONFIG.max_reserved_gib * 2**30 /
                                                     torch.cuda.get_device_properties(0).total_memory))
    before = hash_protected_checkpoints(Exp07Paths().root)
    preflight = json.loads((ROOT / "preflight.json").read_text())
    weights = LossWeights(**preflight["calibration"]["weights"])
    settings = {"seed": SEED, "steps": args.steps, "weights": asdict(weights), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "cuda_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "preflight_checkpoint_hashes_match": before == preflight["protected_checkpoints"]}
    if args.mode in ("probe", "controls"):
        manifest = preflight["fixed_segments"][0]
        settings.update(segment=manifest, samples=264600, batch_size=1, scheduler="none", augment=False,
                        gradient_clip=1.0, optimizer="AdamW defaults except lr")
        experiments = [
            ("complex_l1", "complex_l1", 1e-4, 1.0, True),
            ("composite", "composite", 1e-4, 1.0, True),
            ("complex_l1_lr3e4", "complex_l1", 3e-4, 1.0, True),
            ("complex_l1_gain01", "complex_l1", 1e-4, 0.1, True),
            ("complex_l2", "complex_l2", 1e-4, 1.0, True),
        ]
        if args.mode == "controls":
            experiments = [
                ("complex_l1_fp32", "complex_l1", 1e-4, 1.0, False),
                ("complex_l2_lr3e4", "complex_l2", 3e-4, 1.0, True),
            ]
        measurements = one_segment(manifest, weights, args.steps, experiments)
    elif args.mode == "capacity":
        manifest = preflight["fixed_segments"][0]
        settings.update(segment=manifest, samples=264600, batch_size=1, scheduler="none", augment=False,
                        gradient_clip=1.0, optimizer="AdamW", lr=3e-4, head_bf16=True, loss="normalized complex L2")
        measurements = capacity_control(manifest, args.steps)
    else:
        settings.update(batch_size=4, warmup=4, head_bf16=True, data_loader_workers=0, augment=True)
        measurements = profile(weights, args.steps)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"settings": settings, "measurements": measurements,
                   "protected_checkpoints_unchanged": before == hash_protected_checkpoints(Exp07Paths().root)},
                  handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
