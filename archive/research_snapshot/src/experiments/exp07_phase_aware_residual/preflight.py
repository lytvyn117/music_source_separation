import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from config import SAMPLE_RATE, VALIDATION_SAMPLES
from src.audio.stft import compute_istft
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split

from .checkpointing import CheckpointManager, sha256
from .complex_ops import complex_additivity, waveform_nrmse_pct
from .config import EXPERIMENT, Exp07Config, Exp07Paths, NumericalTolerances, OracleThresholds
from .inference import measure_full_song_consistency, separate_waveform
from .losses import PhaseAwareLoss, calibrate_loss_weights, make_projected_targets
from .model import PhaseAwareResidualSeparator
from .oracle_analysis import CHUNK_SAMPLES, evaluate_oracle_gate
from .trainer import (configure_optimizer, restore_training_state, to_device, training_state,
                      training_step, validate_epoch)


class PreflightFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise PreflightFailure(message)


def hash_protected_checkpoints(root):
    experiments = root / "models/experiments"
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(experiments.glob("exp0[1-6]_*/**/*.pt"))}


def check_environment(paths, config, *, check_collision=True):
    free = shutil.disk_usage(paths.root).free / 2**30
    require(free >= config.min_free_disk_gib, f"Only {free:.2f} GiB free disk")
    if check_collision:
        for path in (paths.checkpoints, paths.logs):
            require(not path.exists() or not any(path.iterdir()), f"Pilot output collision: {path}")
    if os.name == "nt":
        query = (f"Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {{ "
                 f"$_.ProcessId -notin @({os.getpid()}, {os.getppid()}) -and $_.Name -match '^python(w)?\\.exe$' -and "
                 "$_.CommandLine -match 'src[.]training|train_experiment|src[.]evaluation|oracle_analysis|"
                 "exp07_phase_aware_residual[.](trainer|preflight)' } | Select-Object -ExpandProperty ProcessId")
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", query],
                                    capture_output=True, text=True, timeout=30, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            raise PreflightFailure("Windows training-process inspection failed") from exc
        require(not result.stdout.strip(), f"Conflicting training/evaluation process: {result.stdout.strip()}")
    else:
        raise PreflightFailure("GPU process conflict checks are implemented for the approved Windows host")
    require(torch.cuda.is_available(), "CUDA is required for the approved preflight")
    gpu = torch.cuda.get_device_name()
    require("3080 Ti" in gpu, f"Expected RTX 3080 Ti, detected {gpu}")
    return {"free_disk_gib": free, "gpu": gpu}


def validate_oracle_report(path, expected_hash):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        decision = evaluate_oracle_gate(report["tracks"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PreflightFailure("Oracle report is absent or invalid") from exc
    require(report.get("decision") == "GO" and decision.passed
            and report.get("thresholds") == asdict(OracleThresholds())
            and report.get("checkpoint_sha256") == expected_hash, "Oracle report is stale or failed the approved gate")
    return asdict(decision)


def select_fixed_active_segments(train_files, count=4):
    manifest = []
    for track in train_files:
        with sf.SoundFile(track / "vocals.wav") as stream:
            require(stream.channels == 2 and stream.samplerate == SAMPLE_RATE, f"Invalid stereo track {track.name}")
            end = stream.frames - CHUNK_SAMPLES
            if end < 0:
                continue
            starts = sorted(set(range(0, end + 1, CHUNK_SAMPLES // 2)) | {end})
            scores = []
            for start in starts:
                stream.seek(start)
                wave = stream.read(CHUNK_SAMPLES, dtype="float32", always_2d=True)
                scores.append(float(np.sqrt(np.mean(wave**2))))
        selected = int(np.argmax(scores))
        if scores[selected] >= 0.005:
            manifest.append({"track": str(track), "start_sample": starts[selected], "vocal_rms": scores[selected]})
        if len(manifest) == count:
            return manifest
    raise PreflightFailure(f"Could not find {count} fixed active-vocal training tracks")


def _fixed_batch(segments):
    result = []
    for source in ("mixture", "vocals", "instrumental"):
        waves = []
        for segment in segments:
            with sf.SoundFile(Path(segment["track"]) / f"{source}.wav") as stream:
                stream.seek(segment["start_sample"])
                wave = stream.read(CHUNK_SAMPLES, dtype="float32", always_2d=True)
                require(wave.shape == (CHUNK_SAMPLES, 2) and stream.samplerate == SAMPLE_RATE,
                        f"Invalid fixed {source} segment")
                waves.append(wave.T)
        result.append(torch.from_numpy(np.stack(waves)).cuda())
    return tuple(result)


def benchmark_cuda(step, *, batch_size=4, warmup=10, measured=30):
    require(batch_size == 4, "The approved benchmark requires batch size 4")
    elapsed = []
    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        warmup_alloc = torch.cuda.max_memory_allocated()
        warmup_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(measured):
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
    except torch.cuda.OutOfMemoryError as exc:
        raise PreflightFailure("CUDA allocation failed at approved batch size 4 within the VRAM cap") from exc
    return {"median_step_ms": float(np.median(elapsed) * 1000),
            "p95_step_ms": float(np.percentile(elapsed, 95) * 1000),
            "mean_step_ms": float(np.mean(elapsed) * 1000),
            "peak_allocated_gib": max(warmup_alloc, torch.cuda.max_memory_allocated()) / 2**30,
            "peak_reserved_gib": max(warmup_reserved, torch.cuda.max_memory_reserved()) / 2**30,
            "warmup_steps": warmup, "measured_steps": measured, "batch_size": batch_size}


def evaluate_learnability(total_reduction, complex_reduction, sdr_gains, correction, finite):
    return (finite and math.isfinite(correction) and correction > 0 and len(sdr_gains) == 4
            and all(math.isfinite(value) for value in (*sdr_gains, total_reduction, complex_reduction))
            and total_reduction >= 20 and complex_reduction >= 10
            and sum(gain >= 1 for gain in sdr_gains) >= 3)


def run_overfit_gate(model, batch, weights, config, steps, log_path):
    require(1 <= steps <= 300, "Learnability probe is limited to 300 updates")
    model.train()
    optimizer, _ = configure_optimizer(model, config)
    criterion = PhaseAwareLoss()
    mix, *sources = batch
    targets = make_projected_targets(mix, *sources)
    masks = model.backbone_masks(targets.mixture_stft)

    @torch.no_grad()
    def score():
        output = model.forward_stft(targets.mixture_stft, mix.shape[-1], masks=masks)
        values = criterion(output, targets, weights)
        reference = targets.vocals_waveform
        axes = (1, 2)
        errors = (reference - output.vocals_waveform).square().sum(axes)
        sdr = 10 * torch.log10(reference.square().sum(axes).clamp_min(1e-12) / errors.clamp_min(1e-12))
        return {"total": float(values.total), "complex": float(values.complex),
                "sdr_db": sdr.tolist(), "correction_rms": float(output.residual_stft.abs().square().mean().sqrt())}

    initial = score()
    first = second = None
    with log_path.open("x", encoding="utf-8") as log:
        for step in range(1, steps + 1):
            stats = training_step(model, optimizer, criterion, weights, mix, targets, masks=masks)
            log.write(json.dumps({"step": step, **stats}, allow_nan=False) + "\n")
            if step == 1:
                first = stats
            if step == 2:
                second = stats
            if step % 25 == 0:
                print(f"Learnability {step}/{steps}: objective {stats['total']:.6f}", flush=True)
    final = score()
    total = 100 * (1 - final["total"] / initial["total"])
    complex_loss = 100 * (1 - final["complex"] / initial["complex"])
    gains = [after - before for before, after in zip(initial["sdr_db"], final["sdr_db"])]
    return {"passed": evaluate_learnability(total, complex_loss, gains, final["correction_rms"], True),
            "initial": initial, "final": final, "total_reduction_pct": total,
            "complex_reduction_pct": complex_loss, "projected_vocal_sdr_gains_db": gains,
            "steps": steps, "first_step": first, "second_step": second}


@torch.no_grad()
def _verify_initial(model, sample):
    output = model(sample)
    errors = {}
    for index, source in enumerate(("vocals", "instrumental")):
        direct = compute_istft(output.mixture_stft * output.masks[:, index], length=sample.shape[-1])
        actual = getattr(output, f"{source}_waveform")
        errors[source] = {"max_abs": float((actual - direct).abs().max()),
                          "nrmse_pct": waveform_nrmse_pct(actual, direct)}
    err, bound = complex_additivity(output.mixture_stft, output.vocals_stft, output.instrumental_stft)
    tolerance = NumericalTolerances()
    require(torch.count_nonzero(output.residual_stft).item() == 0 and bool((err <= bound).all())
            and all(v["max_abs"] <= tolerance.waveform_max_abs and v["nrmse_pct"] <= tolerance.waveform_nrmse_pct
                    for v in errors.values()), "Exp04 initialization compatibility failed")
    return errors | {"complex_max_error_to_bound": float((err / bound).max())}


def verify_resume(model, batch, weights, config, work):
    optimizer, scheduler = configure_optimizer(model, config)
    generator = torch.Generator().manual_seed(config.seed)
    fields = {"config": asdict(config), "epoch": 0, "validation_objective": 1.0}
    manager = CheckpointManager(work)
    manager.initialize(training_state(model, optimizer, scheduler, generator, **fields))
    initial_sha = sha256(work / "initial.pt")
    mix, *stems = [wave[:1, :, :20003] for wave in batch]
    targets = make_projected_targets(mix, *stems)
    criterion = PhaseAwareLoss()

    def update():
        # Consume the actual RNG families, as an epoch-boundary loader can do.
        random.random()
        np.random.random()
        torch.rand(1, generator=generator)
        torch.rand(1, device=mix.device)
        stats = training_step(model, optimizer, criterion, weights, mix, targets)
        scheduler.step()
        return stats

    update()
    fields.update(epoch=config.snapshot_every, validation_objective=0.5, global_step=1, history=[0.5])
    state = training_state(model, optimizer, scheduler, generator, **fields)
    require(manager.maybe_save_best(state, 0.5), "Best checkpoint was not updated")
    manager.save_latest(state)
    manager.save_snapshot(state)
    uninterrupted = update()
    expected = {name: p.detach().clone() for name, p in model.refinement.named_parameters()}
    restored = manager.load_latest(fields["config"])
    restore_training_state(restored, model, optimizer, scheduler, generator)
    resumed = update()
    delta = max(float((expected[name] - p.detach()).abs().max()) for name, p in model.refinement.named_parameters())
    objective_delta = abs(uninterrupted["total"] - resumed["total"])
    require(sha256(work / "initial.pt") == initial_sha and restored["history"] == [0.5]
            and (work / f"checkpoint_epoch_{config.snapshot_every}.pt").is_file()
            and delta <= 1e-6 and objective_delta <= 1e-6, "Checkpoint/resume verification failed")
    return {"passed": True, "max_parameter_delta": delta, "objective_delta": objective_delta,
            "initial_sha256": initial_sha, "best_epoch": restored["epoch"],
            "snapshot_epoch": config.snapshot_every, "bitwise_production_resume_claimed": False}


def run_preflight(oracle_report, output, *, overfit_steps=300, paths=Exp07Paths(), config=Exp07Config()):
    require(not output.exists() and not output.with_suffix(".md").exists(), "Preflight report collision")
    result = {"passed": False, "config": asdict(config), "failures": []}
    protected = None
    try:
        result["environment"] = check_environment(paths, config)
        protected = hash_protected_checkpoints(paths.root)
        require(len(protected) == 36, f"Expected 36 protected checkpoints, found {len(protected)}")
        result["protected_checkpoints"] = protected
        result["oracle"] = validate_oracle_report(oracle_report, sha256(paths.backbone))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, config.max_reserved_gib * 2**30 / gpu_total))
        train, validation = create_train_validation_split()
        manifest = select_fixed_active_segments(train, count=8)
        result["fixed_segments"] = manifest
        log_dir = paths.root / "outputs/logs/exp07_phase_aware_residual_preflight"
        log_dir.mkdir(parents=True, exist_ok=True)
        def new_model():
            torch.manual_seed(config.seed)
            fresh = PhaseAwareResidualSeparator.from_exp04(paths.backbone, "cuda")
            fresh.refinement.to(memory_format=torch.channels_last)
            fresh.use_bf16 = torch.cuda.is_bf16_supported()
            return fresh

        model = new_model()
        result["head_parameters"] = sum(p.numel() for p in model.refinement.parameters())
        fixed = _fixed_batch(manifest[:4])
        result["initialization"] = _verify_initial(model, fixed[0][:1])
        before = {name: tensor.cpu().clone() for name, tensor in model.backbone.state_dict().items()}
        calibration = calibrate_loss_weights(model, (fixed, _fixed_batch(manifest[4:])))
        result["calibration"] = asdict(calibration)
        require(0.50 <= calibration.complex_share <= 0.80, "Calibration complex gradient share outside 50–80%")
        weights = calibration.weights
        print(f"Calibrated loss weights: {asdict(weights)}", flush=True)
        track = min(validation, key=lambda item: sf.info(item / "mixture.wav").frames)
        mixture, rate = sf.read(track / "mixture.wav", dtype="float32", always_2d=True)
        require(rate == SAMPLE_RATE, "Full-song verification track must be 44.1 kHz")
        inference = separate_waveform(model, mixture, batch_size=1, include_ablations=True)
        result["full_song"] = {"track": track.name, "samples": len(mixture), "variants": {}}
        for name, pair in inference.variants.items():
            report = measure_full_song_consistency(mixture, pair, inference.chunk_starts)
            result["full_song"]["variants"][name] = asdict(report)
            tolerance = NumericalTolerances()
            require(report.global_nrmse_pct <= tolerance.waveform_nrmse_pct
                    and max(report.global_max_abs, report.max_seam_window_abs) <= tolerance.waveform_max_abs,
                    f"Full-song additivity failed for {name}")
        with tempfile.TemporaryDirectory(dir=log_dir) as directory:
            result["checkpoint_resume"] = verify_resume(model, fixed, weights, config, Path(directory))
        del model, inference
        torch.cuda.empty_cache()
        model = new_model().train()
        optimizer, _ = configure_optimizer(model, config)
        criterion = PhaseAwareLoss()
        train_loader = iter(DataLoader(AugmentedMUSDBDataset(track_files=train, samples_per_epoch=160, augment=True),
                                       batch_size=4, num_workers=0))

        def step():
            mix, *sources = to_device(next(train_loader), "cuda")
            training_step(model, optimizer, criterion, weights, mix, make_projected_targets(mix, *sources))

        result["benchmark"] = benchmark_cuda(step)
        torch.cuda.synchronize()
        start = time.perf_counter()
        validation_loader = DataLoader(AugmentedMUSDBDataset(track_files=validation, samples_per_epoch=16,
                                                            random_segments=False), batch_size=4, num_workers=0)
        validate_epoch(model, validation_loader, weights)
        torch.cuda.synchronize()
        validation_per_batch = (time.perf_counter() - start) / 4
        bench = result["benchmark"]
        training_seconds = bench["mean_step_ms"] / 1000 * (config.samples_per_epoch // config.batch_size)
        validation_seconds = validation_per_batch * math.ceil(VALIDATION_SAMPLES / config.batch_size)
        bench.update(head_bf16=model.use_bf16, projected_epoch_seconds=training_seconds + validation_seconds,
                     projected_30_epoch_hours=(config.epochs * (training_seconds + validation_seconds)
                                               + validation_seconds) / 3600)
        require(bench["peak_reserved_gib"] <= config.max_reserved_gib, "Reserved VRAM exceeded 10.5 GiB")
        print(f"CUDA benchmark: {bench}", flush=True)
        if bench["projected_30_epoch_hours"] > config.max_hours:
            result["failures"].append("Projected 30-epoch runtime exceeds three hours")
        del model, optimizer
        torch.cuda.empty_cache()
        model = new_model()
        result["learnability"] = run_overfit_gate(model, fixed, weights, config, overfit_steps,
                                                 log_dir / f"{output.stem}_overfit.jsonl")
        first, second = result["learnability"]["first_step"], result["learnability"]["second_step"]
        require(first["final_gradient_norm"] > 0 and first["upstream_gradient_norm"] == 0
                and second is not None and second["upstream_gradient_norm"] > 0, "Two-step gradient flow failed")
        require(all(torch.equal(value.cpu(), before[name]) for name, value in model.backbone.state_dict().items()),
                "Frozen Exp04 backbone changed")
        result["frozen_backbone_verified"] = True
        require(result["learnability"]["passed"], "Fixed-segment learnability threshold failed")
    except (PreflightFailure, FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
        result["failures"].append(str(exc))
        if torch.cuda.is_available():
            result["cuda_peak_before_stop"] = {"allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                                "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    finally:
        if protected is not None:
            result["protected_checkpoints_unchanged"] = hash_protected_checkpoints(paths.root) == protected
            if not result["protected_checkpoints_unchanged"]:
                result["failures"].append("A protected checkpoint changed")
        result["passed"] = not result["failures"] and result.get("frozen_backbone_verified") is True
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as report:
            json.dump(result, report, indent=2, allow_nan=False)
        with output.with_suffix(".md").open("x", encoding="utf-8") as report:
            status = "PASS" if result["passed"] else "STOP"
            report.write(f"# Exp07 preflight\n\n**{status}**\n\n")
            report.write("\n".join(f"- {failure}" for failure in result["failures"]))
            report.write("\n\nSee the paired JSON for per-stage measurements.\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run blocking Exp07 CUDA preflight without starting the pilot")
    parser.add_argument("--checkpoint", type=Path, default=Exp07Paths().backbone)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(4,), default=4)
    parser.add_argument("--overfit-steps", type=int, choices=range(2, 301), default=300, metavar="2..300")
    args = parser.parse_args()
    if args.checkpoint.resolve() != Exp07Paths().backbone.resolve():
        parser.error("Only the approved Exp04 checkpoint may be used")
    result = run_preflight(args.oracle_report, args.output, overfit_steps=args.overfit_steps)
    print("PASS" if result["passed"] else f"STOP: {result['failures']}", flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
