"""Formal MSE preflight, isolated from the approved L1 trainer and pilot."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import soundfile as sf
import torch

from config import HOP_LENGTH, N_FFT, SAMPLE_RATE
from src.audio.stft import compute_istft, compute_stft
from src.data.splits import create_train_validation_split
from src.experiments.exp07_phase_aware_residual.checkpointing import CheckpointManager, sha256
from src.experiments.exp07_phase_aware_residual.complex_ops import complex_additivity, safe_complex_rms, waveform_nrmse_pct
from src.experiments.exp07_phase_aware_residual.config import NumericalTolerances
from src.experiments.exp07_phase_aware_residual.inference import measure_full_song_consistency, separate_waveform
from src.experiments.exp07_phase_aware_residual.losses import COMPONENTS, LossWeights, make_projected_targets
from src.experiments.exp07_phase_aware_residual.mse_probe_loss import MSEProbeLoss
from src.experiments.exp07_phase_aware_residual.preflight import (
    _fixed_batch, _verify_initial, check_environment, hash_protected_checkpoints, require, validate_oracle_report,
)
from src.experiments.exp07_phase_aware_residual.trainer import (
    configure_optimizer, restore_training_state, training_state, training_step,
)

ROOT = Path("reports/exp07/complex_mse_formal")
OLD = ROOT.parent / "complex_mse_probe"
WEIGHTS = LossWeights(complex=1., waveform_l1=0.667601, mr_stft=0.070409, leakage=0.008311)
spec = importlib.util.spec_from_file_location("exp07_isolated_probe", OLD / "run_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.ROOT = ROOT
TOL = NumericalTolerances()


def projected_check(batch):
    mix, *refs = batch
    target = make_projected_targets(mix, *refs)
    error, bound = complex_additivity(target.mixture_stft, target.vocals_stft, target.instrumental_stft)
    waveform = target.vocals_waveform + target.instrumental_waveform
    result = {"complex_ratio": float((error / bound).max()), "waveform_peak_error": float((mix - waveform).abs().max()),
              "waveform_nrmse_pct": waveform_nrmse_pct(waveform, mix)}
    require(result["complex_ratio"] <= 1 and result["waveform_peak_error"] <= TOL.waveform_max_abs
            and result["waveform_nrmse_pct"] <= TOL.waveform_nrmse_pct, "Projected targets failed additivity")
    return result


def approved_gradient_share(model, criterion, segments):
    accumulated = []
    params = tuple(model.refinement.final.parameters())
    for start in (0, 4):
        mix, *refs = _fixed_batch(segments[start:start + 4])
        targets = make_projected_targets(mix, *refs)
        out = model.forward_stft(targets.mixture_stft, mix.shape[-1])
        loss = criterion(out, targets, WEIGHTS)
        row = {}
        for index, key in enumerate(COMPONENTS):
            gradients = torch.autograd.grad(getattr(loss, key) * getattr(WEIGHTS, key), params,
                                            retain_graph=index < len(COMPONENTS) - 1)
            row[key] = float(sum(g.square().sum() for g in gradients).sqrt())
        accumulated.append(row)
        del out, loss, targets, mix, refs
    means = {key: float(np.mean([row[key] for row in accumulated])) for key in COMPONENTS}
    share = means["complex"] / sum(means.values())
    require(0.5 <= share <= 0.8, "Approved MSE weights are not gradient-dominant")
    return {"weighted_gradient_norms": means, "complex_share": share}


def _energy(x):
    return float(x.abs().square().sum())


@torch.no_grad()
def analyze_risks(model, batch, segments):
    mix, *refs = batch
    target = make_projected_targets(mix, *refs)
    output = model.forward_stft(target.mixture_stft, mix.shape[-1])
    length = mix.shape[-1]
    x, v0 = output.mixture_stft, output.initial_vocal_stft
    v, i = output.vocals_stft, output.instrumental_stft
    baseline_wav = compute_istft(torch.cat((v0, x * output.masks[:, 1]), dim=1), length=length)
    delta = output.residual_stft
    consistent_delta = compute_stft(compute_istft(delta, length=length))
    scale = safe_complex_rms(x, eps=1e-5)
    rows, failures = [], []
    audio_root = ROOT / "audio"
    audio_root.mkdir(exist_ok=False)
    for j, segment in enumerate(segments):
        track = Path(segment["track"]).stem
        ideal = (target.vocals_stft[j] - v0[j]) / scale[j]
        predicted = delta[j] / scale[j]
        waveform_visible = consistent_delta[j] / scale[j]
        magnitude = x[j].abs()
        local = magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        gain_groups = (("quiet", local < .001), ("low", (local >= .001) & (local < .01)),
                       ("mid", (local >= .01) & (local < .1)), ("high", local >= .1))
        ideal_energy = _energy(ideal)
        row = {"track": track, "decomposition": {
            "ideal_normalized_energy": ideal_energy, "prediction_normalized_energy": _energy(predicted),
            "twice_alignment": float(2 * (predicted * ideal.conj()).real.sum()),
            "remaining_rmse_fraction": (_energy(ideal - predicted) / ideal_energy)**.5,
            "analysis_resynthesized_remaining_rmse_fraction": (_energy(ideal - waveform_visible) / ideal_energy)**.5,
            "analysis_resynthesis_difference_energy": _energy(predicted - waveform_visible),
        }, "sources": {}}
        for name, source, base, reference in (
            ("vocals", v[j], v0[j], target.vocals_stft[j]),
            ("instrumental", i[j], x[j] * output.masks[j, 1], target.instrumental_stft[j]),
        ):
            base_ratio = base.abs() / magnitude.clamp_min(1e-12)
            ratio = source.abs() / magnitude.clamp_min(1e-12)
            post_error = (source - reference).abs().square()
            base_error = (base - reference).abs().square()
            groups = {}
            for label, mask in gain_groups:
                if not bool(mask.any()):
                    continue
                r = ratio[mask]
                before_p99 = float(torch.quantile(base_ratio[mask], .99))
                after_p99 = float(torch.quantile(r, .99))
                groups[label] = {"bins": int(mask.sum()), "max_ratio": float(r.max()),
                                 "p99_before": before_p99, "p99_after": after_p99,
                                 "p99_change_db": float(20 * np.log10(max(after_p99, 1e-12) / max(before_p99, 1e-12))),
                                 "mixture_stft_energy": _energy(x[j][mask]),
                                 "post_error_energy": float(post_error[mask].sum()),
                                 "initial_error_energy": float(base_error[mask].sum())}
            active = local >= .001
            high = local >= .1
            high_p99_change = groups["high"]["p99_change_db"]
            if high_p99_change > 3:
                failures.append(f"{track} {name}: high-energy p99 rose over 3 dB")
            outs = []
            for label, mask in (("all_active", active), ("high_energy", high)):
                values, indices = ratio.masked_fill(~mask, -1).flatten().topk(5)
                for value, index in zip(values.tolist(), indices.tolist()):
                    c, f, t = np.unravel_index(index, ratio.shape)
                    ix = (int(c), int(f), int(t))
                    outs.append({"region": label, "ratio": value, "channel": int(c),
                                 "frequency_hz": f * SAMPLE_RATE / N_FFT,
                                 "segment_time_s": t * HOP_LENGTH / SAMPLE_RATE,
                                 "song_time_s": (segment["start_sample"] + t * HOP_LENGTH) / SAMPLE_RATE,
                                 "mixture_fraction_of_channel_peak": float(local[ix]),
                                 "mixture_bin_energy": float(magnitude[ix].square()),
                                 "estimate_bin_energy": float(source[ix].abs().square()),
                                 "reference_bin_energy": float(reference[ix].abs().square()),
                                 "initial_error_energy": float(base_error[ix]),
                                 "post_error_energy": float(post_error[ix]),
                                 "fraction_of_total_source_error": float(post_error[ix] / post_error.sum())})
            total_post = float(post_error.sum())
            loud = active & (ratio >= 2)
            row["sources"][name] = {"groups": groups, "outliers": outs,
                                     "active_over_2x_count": int(loud.sum()),
                                     "active_over_2x_fraction_of_total_error": float(post_error[loud].sum()) / total_post,
                                     "high_peak_review_flag": groups["high"]["max_ratio"] > 2}
        directory = audio_root / f"{j + 1:02d}"
        directory.mkdir()
        waves = {"mixture": mix[j], "reference_vocals": refs[0][j], "reference_instrumental": refs[1][j],
                 "exp04_vocals": baseline_wav[j, :2], "exp04_instrumental": baseline_wav[j, 2:],
                 "mse_vocals": output.vocals_waveform[j], "mse_instrumental": output.instrumental_waveform[j]}
        row["waveform"] = {}
        for name, wave in waves.items():
            sf.write(directory / f"{name}.wav", wave.T.cpu().numpy(), SAMPLE_RATE, subtype="FLOAT")
            row["waveform"][name] = {"peak": float(wave.abs().max()), "rms": float(wave.square().mean().sqrt()),
                                     "clipped_samples": int((wave.abs() > 1).sum())}
        row["audio_dir"] = directory.as_posix()
        rows.append(row)
    return {"segments": rows, "failures": failures, "listening_judgment": "pending independent listening"}


def resume_probe(model, batch, criterion):
    model.train()
    optimizer, scheduler = configure_optimizer(model, probe.CONFIG)
    rng = torch.Generator().manual_seed(42)
    config = {"loss": "normalized_complex_mse", "weights": asdict(WEIGHTS), "optimizer": asdict(probe.CONFIG)}
    mix, *refs = [wave[:1, :, :20003] for wave in batch]
    targets = make_projected_targets(mix, *refs)

    def update():
        draws = (random.random(), np.random.rand(), torch.rand(1).item(),
                 torch.rand(1, generator=rng).item(), torch.rand(1, device="cuda").item())
        stats = training_step(model, optimizer, criterion, WEIGHTS, mix, targets)
        scheduler.step()
        return draws, stats["total"]

    @torch.no_grad()
    def validation():
        output = model.forward_stft(targets.mixture_stft, mix.shape[-1])
        return float(criterion(output, targets, WEIGHTS).total)

    def equals(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equals(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equals(x, y) for x, y in zip(a, b))
        return a == b

    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        directory = Path(temp)
        manager = CheckpointManager(directory)
        initial_loss = validation()
        fields = dict(config=config, epoch=0, validation_objective=initial_loss, history=[])
        manager.initialize(training_state(model, optimizer, scheduler, rng, **fields))
        initial_sha = sha256(directory / "initial.pt")
        update()
        current = validation()
        fields.update(epoch=10, validation_objective=current, history=[current])
        state = training_state(model, optimizer, scheduler, rng, **fields)
        selected = manager.maybe_save_best(state, current)
        require(selected == (current < initial_loss), "Wrong best selection using actual MSE objective")
        require(not manager.maybe_save_best(state, manager.best_value + .01), "Worse objective replaced best")
        best = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
        require(best["epoch"] == (10 if selected else 0), "Incorrect saved best checkpoint")
        manager.save_latest(state)
        manager.save_snapshot(state)
        require((directory / "checkpoint_epoch_10.pt").is_file(), "Snapshot missing")
        direct = update()
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        import copy
        expected_opt = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        saved = manager.load_latest(config)
        restore_training_state(saved, model, optimizer, scheduler, rng)
        resumed = update()
        checks = {"parameters_and_buffers_exact": equals(expected, model.state_dict()),
                  "optimizer_exact": equals(expected_opt, optimizer.state_dict()),
                  "scheduler_exact": equals(expected_scheduler, scheduler.state_dict()),
                  "rng_draws_exact": direct[0] == resumed[0], "objective_exact": direct[1] == resumed[1],
                  "initial_immutable": initial_sha == sha256(directory / "initial.pt"),
                  "saved_history_exact": saved["history"] == [current]}
        require(all(checks.values()), f"Resume mismatch: {checks}")
        return checks | {"selection_objective_initial": initial_loss, "selection_objective_after_update": current,
                         "best_epoch": best["epoch"], "snapshot_epoch": 10}


def full_song_check(model):
    _, val = create_train_validation_split()
    track = min(val, key=lambda p: sf.info(p / "mixture.wav").frames)
    mix, rate = sf.read(track / "mixture.wav", dtype="float32", always_2d=True)
    require(rate == SAMPLE_RATE, "Invalid full-song sample rate")
    result = separate_waveform(model, mix, batch_size=1, include_ablations=True)
    rows = {}
    for name, pair in result.variants.items():
        score = measure_full_song_consistency(mix, pair, result.chunk_starts)
        require(pair.vocals.shape == mix.shape == pair.instrumental.shape
                and score.global_nrmse_pct <= TOL.waveform_nrmse_pct
                and max(score.global_max_abs, score.max_seam_window_abs) <= TOL.waveform_max_abs,
                f"Full song failed: {name}")
        rows[name] = asdict(score)
    return {"track": track.name, "samples": len(mix), "variants": rows}


def main():
    result_path = ROOT / "result.json"
    if result_path.exists() or (ROOT / "updates.jsonl").exists():
        raise FileExistsError("Formal output collision")
    paths, config = probe.PATHS, probe.CONFIG
    protected = hash_protected_checkpoints(paths.root)
    old_files = {p: sha256(p) for p in ROOT.parent.rglob("*") if p.is_file()
                 and ROOT not in p.parents and "__pycache__" not in p.parts}
    preflight = json.loads((ROOT.parent / "preflight.json").read_text(encoding="utf-8"))
    result = {"configuration": asdict(config), "weights": asdict(WEIGHTS), "failures": [], "memory": {}}
    try:
        require(len(protected) == 36 and protected == preflight["protected_checkpoints"], "Protected manifest mismatch")
        result["environment"] = check_environment(paths, config)
        torch.cuda.set_per_process_memory_fraction(min(1., config.max_reserved_gib * 2**30 /
                                                         torch.cuda.get_device_properties(0).total_memory))
        oracle_path = ROOT.parent / "oracle_scale_aware/exp07_oracle_scale_aware_summary.json"
        result["oracle"] = validate_oracle_report(oracle_path, sha256(paths.backbone))
        segments = preflight["fixed_segments"]
        criterion = MSEProbeLoss()
        model = probe.new_model()
        batch = _fixed_batch(segments[:4])
        result["projected_targets"] = projected_check(batch)
        result["initialization"] = _verify_initial(model, batch[0][:1])
        torch.cuda.reset_peak_memory_stats()
        _, result["recalibration"] = probe.calibrate(model, criterion, segments)
        require(all(abs(result["recalibration"]["weights"][key] - getattr(WEIGHTS, key)) <= 5.1e-7
                    for key in COMPONENTS), "Recalibration does not round to approved values")
        result["approved_weight_gradient_check"] = approved_gradient_share(model, criterion, segments)
        result["memory"]["calibration"] = probe.peak_memory()
        print(f"Approved-weight gradient share: {result['approved_weight_gradient_check']['complex_share']:.5f}", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result["learnability"] = probe.learnability(model, criterion, WEIGHTS, segments[:4])
        result["memory"]["learnability"] = probe.peak_memory()
        result["failures"].extend(result["learnability"]["failures"])
        first, second = result["learnability"]["per_parameter"][1], result["learnability"]["per_parameter"][2]
        require(first["final.weight"]["gradient_l2_after_clip"] > 0
                and first["stem.0.weight"]["gradient_l2_after_clip"] == 0
                and second["stem.0.weight"]["gradient_l2_after_clip"] > 0,
                "Expected two-step gradient flow failed")
        result["gradient_propagation_verified"] = True
        result["risk_analysis"] = analyze_risks(model, batch, segments[:4])
        result["failures"].extend(result["risk_analysis"]["failures"])
        result["full_song"] = full_song_check(model)
        result["checkpoint_resume"] = resume_probe(model, batch, criterion)
        del model, batch
        if not result["failures"]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result["runtime"] = probe.time_disposable_epoch(probe.new_model(), criterion, WEIGHTS)
            if result["runtime"]["projected_30_epoch_hours"] > config.max_hours:
                result["failures"].append("Projected runtime exceeds three hours")
            result["memory"]["timing_epoch"] = probe.peak_memory()
        else:
            result["runtime"] = {"status": "not run after blocking gate"}
        if any(stage["reserved_gib"] > config.max_reserved_gib for stage in result["memory"].values()):
            result["failures"].append("Reserved VRAM exceeded 10.5 GiB")
    except (RuntimeError, ValueError, OSError) as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        result["protected_checkpoints_unchanged"] = protected == hash_protected_checkpoints(paths.root)
        result["previous_reports_unchanged"] = all(path.exists() and sha256(path) == checksum
                                                    for path, checksum in old_files.items())
        result["passed"] = not result["failures"] and result["protected_checkpoints_unchanged"] and result["previous_reports_unchanged"]
        probe.write_json(result_path, result)
    print("PASS" if result["passed"] else f"STOP: {result['failures']}", flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
