"""Isolated Exp07 MSE experiment. Run from the repository root."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import VALIDATION_SAMPLES
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split
from src.evaluation.metrics import compute_leakage_diagnostics, compute_sdr
from src.experiments.exp07_phase_aware_residual.checkpointing import CheckpointManager
from src.experiments.exp07_phase_aware_residual.complex_ops import safe_complex_rms
from src.experiments.exp07_phase_aware_residual.config import Exp07Config, Exp07Paths
from src.experiments.exp07_phase_aware_residual.diagnostics import compute_residual_diagnostics
from src.experiments.exp07_phase_aware_residual.losses import COMPONENTS, LossWeights, make_projected_targets
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator
from src.experiments.exp07_phase_aware_residual.mse_probe_loss import MSEProbeLoss
from src.experiments.exp07_phase_aware_residual.preflight import _fixed_batch, check_environment, hash_protected_checkpoints
from src.experiments.exp07_phase_aware_residual.trainer import configure_optimizer, to_device, training_state, training_step


ROOT = Path("reports/exp07/complex_mse_probe")
PATHS = Exp07Paths()
CONFIG = replace(Exp07Config(), lr=3e-4)
THRESHOLDS = {"mse_reduction_pct": 15, "total_reduction_pct": 10, "vocal_sdr_gain_db": 1, "vocal_sdr_wins": 3}
RATIOS = {"waveform_l1": 0.25, "mr_stft": 0.25, "leakage": 0.10}


def new_model():
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    model = PhaseAwareResidualSeparator.from_exp04(PATHS.backbone, "cuda")
    model.refinement.to(memory_format=torch.channels_last)
    model.use_bf16 = True
    return model.train()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def peak_memory():
    return {name + "_gib": getattr(torch.cuda, "max_memory_" + name)() / 2**30
            for name in ("allocated", "reserved")}


def calibrate(model, criterion, segments):
    params = tuple(model.refinement.final.parameters())

    def measure(weights):
        rows = []
        for start in (0, 4):
            mix, *refs = _fixed_batch(segments[start:start + 4])
            targets = make_projected_targets(mix, *refs)
            prediction = model.forward_stft(targets.mixture_stft, mix.shape[-1])
            losses = criterion(prediction, targets, weights)
            row = {}
            for index, key in enumerate(COMPONENTS):
                grads = torch.autograd.grad(getattr(losses, key) * getattr(weights, key), params,
                                            retain_graph=index < len(COMPONENTS) - 1)
                row[key] = float(sum(g.detach().square().sum() for g in grads).sqrt())
            rows.append(row)
            del prediction, losses, targets, mix, refs
        return {key: float(np.mean([row[key] for row in rows])) for key in COMPONENTS}

    raw = measure(LossWeights())
    if any(not math.isfinite(value) or value <= 0 for value in raw.values()):
        raise FloatingPointError(f"Invalid calibration norms {raw}")
    weights = LossWeights(**{key: min(10., max(1e-4, raw["complex"] * ratio / raw[key]))
                             for key, ratio in RATIOS.items()})
    weighted = measure(weights)
    share = weighted["complex"] / sum(weighted.values())
    if not 0.5 <= share <= 0.8:
        raise FloatingPointError(f"Measured weighted complex gradient share outside gate: {share}")
    return weights, {"weights": asdict(weights), "raw_gradient_norms": raw,
                     "remeasured_weighted_gradient_norms": weighted, "complex_share": share}


@torch.no_grad()
def score(model, criterion, weights, mix, references, targets, masks):
    output = model.forward_stft(targets.mixture_stft, mix.shape[-1], masks=masks)
    losses = criterion(output, targets, weights)
    ideal = targets.vocals_stft - output.initial_vocal_stft
    correction = output.residual_stft
    scale = safe_complex_rms(targets.mixture_stft, eps=1e-5)
    axes = (1, 2, 3)
    normalized_error = ((correction - ideal) / scale).abs().square().mean(dim=axes).sqrt()
    normalized_ideal = (ideal / scale).abs().square().mean(dim=axes).sqrt()
    pred_ratio = (correction.abs().square().mean(dim=axes) / ideal.abs().square().mean(dim=axes)).sqrt()
    rows = []
    for index, (v, i, v_ref, i_ref, v_proj, i_proj) in enumerate(zip(
        output.vocals_waveform.cpu().numpy(), output.instrumental_waveform.cpu().numpy(),
        references[0].cpu().numpy(), references[1].cpu().numpy(),
        targets.vocals_waveform.cpu().numpy(), targets.instrumental_waveform.cpu().numpy(),
    )):
        rows.append({"vocal_sdr_projected": compute_sdr(v_proj, v), "instrumental_sdr_projected": compute_sdr(i_proj, i),
                     "vocal_sdr_original": compute_sdr(v_ref, v), "instrumental_sdr_original": compute_sdr(i_ref, i),
                     "normalized_residual_rmse": float(normalized_error[index]),
                     "remaining_normalized_ideal_rmse_fraction": float(normalized_error[index] / normalized_ideal[index]),
                     "predicted_to_ideal_rms": float(pred_ratio[index]),
                     **compute_leakage_diagnostics(v_ref, i_ref, v, i)})
    return {"loss": {key: float(getattr(losses, key)) for key in (*COMPONENTS, "total")}, "segments": rows,
            "stability": compute_residual_diagnostics(output, mix, targets)}


def learnability(model, criterion, weights, segments):
    mix, *refs = _fixed_batch(segments)
    targets = make_projected_targets(mix, *refs)
    masks = model.backbone_masks(targets.mixture_stft)
    optimizer, _ = configure_optimizer(model, CONFIG)
    params = dict(model.refinement.named_parameters())
    initial_params = {k: v.detach().clone() for k, v in params.items()}
    before_backbone = {k: v.cpu().clone() for k, v in model.backbone.state_dict().items()}
    initial = score(model, criterion, weights, mix, refs, targets, masks)
    layers = {}
    max_fields = ("waveform_peak", "clipped_sample_pct", "total_gradient_norm", "nonfinite_gradients",
                  "nonfinite_count", "complex_violations", "complex_max_error_to_bound",
                  "spectral_amplification_vs_v0_max", "spectral_amplification_vs_v0_p99",
                  "spectral_amplification_vs_mixture_max", "spectral_amplification_vs_mixture_p99")
    maxima = dict.fromkeys(max_fields, 0.)
    clipped_updates = 0
    with (ROOT / "updates.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, 301):
            capture = step in (1, 2, 25, 100, 200, 300)
            if capture:
                before = {k: v.detach().clone() for k, v in params.items()}
            stats = training_step(model, optimizer, criterion, weights, mix, targets, masks=masks)
            if capture:
                layers[step] = {
                    name: {"gradient_l2_after_clip": float(p.grad.norm()),
                           "update_l2": float((p.detach() - before[name]).norm()),
                           "before_parameter_l2": float(before[name].norm()),
                           "displacement_l2": float((p.detach() - initial_params[name]).norm())}
                    for name, p in params.items()
                }
            for key in maxima:
                maxima[key] = max(maxima[key], stats[key])
            clipped_updates += stats["gradient_clip_event"]
            log.write(json.dumps({"step": step, **stats}, allow_nan=False) + "\n")
            if step % 25 == 0:
                print(f"{step}/300: MSE {stats['complex']:.6f}, objective {stats['total']:.6f}", flush=True)
    final = score(model, criterion, weights, mix, refs, targets, masks)
    mse_reduction = 100 * (1 - final["loss"]["complex"] / initial["loss"]["complex"])
    total_reduction = 100 * (1 - final["loss"]["total"] / initial["loss"]["total"])
    gains = [b["vocal_sdr_projected"] - a["vocal_sdr_projected"]
             for a, b in zip(initial["segments"], final["segments"])]
    unchanged = all(torch.equal(v.cpu(), before_backbone[k]) for k, v in model.backbone.state_dict().items())
    wins = sum(gain >= THRESHOLDS["vocal_sdr_gain_db"] for gain in gains)
    failures = []
    if mse_reduction < THRESHOLDS["mse_reduction_pct"]:
        failures.append("MSE reduction below 15%")
    if total_reduction < THRESHOLDS["total_reduction_pct"]:
        failures.append("Weighted objective reduction below 10%")
    if wins < THRESHOLDS["vocal_sdr_wins"]:
        failures.append("Fewer than 3/4 segments gained 1 dB vocal SDR")
    if not unchanged or final["stability"]["correction_energy_ratio"] <= 0:
        failures.append("Frozen-backbone or nonzero-correction check failed")
    if max(maxima["clipped_sample_pct"], final["stability"]["clipped_sample_pct"]) > 0.1:
        failures.append("More than 0.1% of source samples exceeded full scale")
    amp = "spectral_amplification_vs_mixture_p99"
    if final["stability"][amp] > initial["stability"][amp] * 10**(3 / 20):
        failures.append("Final active-bin p99 amplification exceeded the initial value by over 3 dB")
    return {"passed": not failures, "failures": failures, "initial": initial, "final": final,
            "mse_reduction_pct": mse_reduction, "total_reduction_pct": total_reduction,
            "normalized_rmse_reduction_pct": 100 * (1 - math.sqrt(1 - mse_reduction / 100)),
            "projected_vocal_sdr_gains_db": gains, "wins": wins, "frozen_backbone_verified": unchanged,
            "per_parameter": layers, "training_maxima": maxima, "gradient_clipped_steps": clipped_updates}


def time_disposable_epoch(model, criterion, weights):
    train, val = create_train_validation_split()
    rng = torch.Generator().manual_seed(CONFIG.seed)
    optimizer, scheduler = configure_optimizer(model, CONFIG)
    train_loader = DataLoader(AugmentedMUSDBDataset(track_files=train, samples_per_epoch=2000, augment=True),
                              batch_size=4, num_workers=0, generator=rng)
    val_loader = DataLoader(AugmentedMUSDBDataset(track_files=val, samples_per_epoch=VALIDATION_SAMPLES,
                                                random_segments=False), batch_size=4, num_workers=0, generator=rng)
    start = time.perf_counter()
    prior = start
    times = []
    for batch in train_loader:
        mix, *refs = to_device(batch, "cuda")
        targets = make_projected_targets(mix, *refs)
        training_step(model, optimizer, criterion, weights, mix, targets)
        torch.cuda.synchronize()
        now = time.perf_counter()
        times.append(now - prior)
        prior = now
        if len(times) % 50 == 0:
            print(f"Timing epoch {len(times)}/500 steps", flush=True)
    train_seconds = time.perf_counter() - start
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in val_loader:
            mix, *refs = to_device(batch, "cuda")
            targets = make_projected_targets(mix, *refs)
            out = model.forward_stft(targets.mixture_stft, mix.shape[-1])
            if not torch.isfinite(criterion(out, targets, weights).total):
                raise FloatingPointError("Nonfinite timing validation")
            compute_residual_diagnostics(out, mix, targets)
    torch.cuda.synchronize()
    val_seconds = time.perf_counter() - start
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        manager = CheckpointManager(Path(folder))
        state = training_state(model, optimizer, scheduler, rng, epoch=1, validation_objective=1.)
        start = time.perf_counter()
        manager.initialize(state)
        checkpoint_seconds = time.perf_counter() - start
    epoch_seconds = train_seconds + val_seconds + checkpoint_seconds
    return {"steps": len(times), "median_step_ms": float(np.median(times) * 1000),
            "p95_step_ms": float(np.percentile(times, 95) * 1000), "mean_step_ms": float(np.mean(times) * 1000),
            "train_seconds": train_seconds, "validation_seconds": val_seconds, "checkpoint_seconds": checkpoint_seconds,
            "projected_30_epoch_hours": (30 * epoch_seconds + val_seconds) / 3600, "step_seconds": times,
            "peak_memory": peak_memory()}


def main():
    output = ROOT / "result.json"
    if output.exists() or (ROOT / "updates.jsonl").exists():
        raise FileExistsError("Probe output collision")
    prior = json.loads(Path("reports/exp07/preflight.json").read_text(encoding="utf-8"))
    original = hash_protected_checkpoints(PATHS.root)
    assert len(original) == 36 and original == prior["protected_checkpoints"]
    report = {"configuration": asdict(CONFIG), "thresholds": THRESHOLDS, "steps": 300,
              "augmentation_in_probe": False, "scheduler_in_probe": None,
              "segments": prior["fixed_segments"], "failures": []}
    try:
        report["environment"] = check_environment(PATHS, CONFIG)
        torch.cuda.set_per_process_memory_fraction(min(1., 10.5 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        criterion = MSEProbeLoss()
        model = new_model()
        weights, report["calibration"] = calibrate(model, criterion, prior["fixed_segments"])
        print(f"Calibration {report['calibration']}", flush=True)
        report["calibration_peak_memory"] = peak_memory()
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = new_model()
        report["learnability"] = learnability(model, criterion, weights, prior["fixed_segments"][:4])
        report["probe_peak_memory"] = peak_memory()
        report["failures"].extend(report["learnability"]["failures"])
        del model
        if report["learnability"]["passed"]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            report["runtime"] = time_disposable_epoch(new_model(), criterion, weights)
            if report["runtime"]["projected_30_epoch_hours"] > 3:
                report["failures"].append("Projected runtime exceeded three hours")
        else:
            report["runtime"] = {"status": "not run after failed learnability gate"}
    except (FloatingPointError, RuntimeError) as error:
        report["failures"].append(str(error))
        if torch.cuda.is_initialized():
            report["peak_before_error"] = peak_memory()
    finally:
        report["protected_checkpoints_unchanged"] = original == hash_protected_checkpoints(PATHS.root)
        report["passed"] = not report["failures"] and report["protected_checkpoints_unchanged"]
        write_json(output, report)
    print("PASS" if report["passed"] else f"STOP: {report['failures']}", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
