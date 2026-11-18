"""Bounded, train-only Exp08 readiness checks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch

from src.audio.stft import compute_stft
from src.inference.separate_track import load_separation_model, separate_waveform
from src.training.experiment_selection import SelectionState, update_selection

from .checkpointing import Checkpoints, rng_state
from .data import (
    BASELINE, FEASIBILITY, PROJECT_ROOT, RecordedClips, TrainClips, fingerprint_manifest,
    frozen_partition, reserve_run, sha256,
)
from .evaluate import score_song
from .inference import separate
from .loss import project_vocals
from .model import SubbandSeparator, synthesize
from .train import (
    TrainConfig, configure_determinism, make_loader, make_optimizer, make_scheduler,
    optimizer_update, record_failure, resume_epoch, save_epoch, validate,
)

GIB = 2**30
PROFILE_UPDATES = 100
WARMUP_UPDATES = 8


def _write(path, content):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)


def _distribution(values):
    return {"median": median(values), "p95": float(np.percentile(values, 95)), "max": max(values)}


def profile_summary(stats, clips_per_update, allocated, reserved, headroom):
    norms = [row["preclip_norm"] for row in stats]
    seconds = [row["seconds"] for row in stats]
    elapsed = sum(seconds)
    throughput = len(stats) * clips_per_update / elapsed
    clipped = sum(row["clipped"] for row in stats)
    guards = np.asarray([row["guard"] for row in stats])
    return {
        "updates": len(stats), "clips": len(stats) * clips_per_update, "seconds": elapsed,
        "clips_per_sec": throughput, "median_step_s": median(seconds),
        "p95_step_s": float(np.percentile(seconds, 95)),
        "peak_allocated_gib": allocated, "peak_reserved_gib": reserved,
        "minimum_sampled_headroom_gib": headroom, "clipped_updates": clipped,
        "preclip_norm": _distribution(norms),
        "update_norm": _distribution([row["update_norm"] for row in stats]),
        "relative_update": _distribution([row["relative_update"] for row in stats]),
        "guard_max": np.max(guards, axis=0).tolist(),
        "signal_max": np.max([row["signal_max_by_microbatch"] for row in stats], axis=0).tolist(),
        "gates": {"updates": len(stats) >= 100, "throughput": throughput >= 8,
                  "reserved": reserved <= 8, "headroom": headroom >= 2,
                  "gradient_clipping": clipped <= len(stats) * .5,
                  "finite_reconstruction": bool(np.isfinite(guards).all()
                                                and (guards[:, :3] <= [1., 0., 1.]).all())},
    }


def profile(tracks, config, logs):
    model = SubbandSeparator().cuda()
    optimizer = make_optimizer(model, config)
    clips_per_update = config.microbatch * config.accumulation
    dataset = TrainClips(tracks, epoch=1, samples=clips_per_update * (PROFILE_UPDATES + WARMUP_UPDATES))
    batches = iter(make_loader(dataset, config))
    stats, min_free = [], float("inf")
    with (logs / "profile_steps.jsonl").open("x", encoding="utf-8") as handle:
        for index in range(PROFILE_UPDATES + WARMUP_UPDATES):
            if index == WARMUP_UPDATES:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = next(batches)
            try:
                row = optimizer_update(model, optimizer, batch["audio"], config)
            except FloatingPointError as error:
                record_failure(logs / "failures.jsonl", 1, index, batch, error)
                raise
            torch.cuda.synchronize()
            free = torch.cuda.mem_get_info()[0] / GIB
            handle.write(json.dumps({"step": index + 1, **row}, allow_nan=False) + "\n")
            handle.flush()
            row["seconds"] = time.perf_counter() - started
            if index >= WARMUP_UPDATES:
                min_free = min(min_free, free)
                stats.append(row)
            if (index + 1) % 20 == 0:
                print(f"Profile progress {index+1}/{PROFILE_UPDATES+WARMUP_UPDATES} updates", flush=True)
    summary = profile_summary(stats, clips_per_update, torch.cuda.max_memory_allocated() / GIB,
                              torch.cuda.max_memory_reserved() / GIB, min_free)
    _write(logs / "timed_steps.json", stats)
    return summary


def _panel_records():
    evidence = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
    records = [{key: row[key] for key in ("track", "start_sample")} for row in evidence["clips"]]
    assert [row["track"] for row in records[:10]] == evidence["fit_tracks"]
    return records[:8], records[10:]


def bounded_oracle(mixture, projected_vocal):
    """Project the ideal ratio onto the closure of the frozen mask rectangle."""
    ratio = projected_vocal / torch.where(mixture == 0, 1, mixture)
    imag = ratio.imag.clamp(-.25, .25).clone()
    imag[..., [0, -1], :] = 0
    return torch.complex(ratio.real.clamp(-.15, 1.15), imag) * mixture


@torch.inference_mode()
def replay_oracle(tracks):
    evidence = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
    records = [{key: row[key] for key in ("track", "start_sample")} for row in evidence["clips"]]
    dataset = RecordedClips(tracks, records)
    exp04, _, _ = load_separation_model(BASELINE, "cuda")
    rows = []
    for clip in dataset:
        wave = clip["audio"]
        mix, v, i = compute_stft(wave.cuda()).unbind(0)
        vocals = bounded_oracle(mix, project_vocals(mix, v, i))
        bounded = synthesize(torch.stack((vocals, mix - vocals)), wave.shape[-1]).cpu().numpy()
        baseline = separate_waveform(exp04, wave[0].numpy(), device="cuda",
                                     segment_length_seconds=6., overlap_ratio=.5)
        before = score_song(clip["track"], wave[1:].numpy(),
                            np.stack((baseline["vocals"], baseline["instrumental"])), wave[0].numpy())
        after = score_song(clip["track"], wave[1:].numpy(), bounded, wave[0].numpy())
        row = {"track": clip["track"], "start_sample": clip["start_sample"]}
        for label, scores in (("exp04", before), ("bounded", after)):
            row[label] = {name: scores[name] if np.isfinite(scores[name]) else None
                          for name in ("vocal_sdr", "instrumental_sdr", "bleed")}
        rows.append(row)
    paired = {}
    for metric in ("vocal_sdr", "instrumental_sdr", "bleed"):
        transform = abs if metric == "bleed" else float
        paired[metric] = median([transform(row["bounded"][metric]) - transform(row["exp04"][metric])
                                 for row in rows if row["exp04"][metric] is not None
                                 and row["bounded"][metric] is not None])
    return {"paired_delta": paired, "rows": rows,
            "passed": paired["vocal_sdr"] >= 1 and paired["instrumental_sdr"] >= .5 and paired["bleed"] < 0}


def _panel_measure(model, dataset, config):
    loss = validate(model, make_loader(dataset, config, workers=0, training=False), config)
    rows = []
    for clip in dataset:
        mixture, vocal, instrumental = clip["audio"].numpy()
        estimate = separate(model, mixture, "cuda", batch_size=1)
        rows.append(score_song(clip["track"], np.stack((vocal, instrumental)), estimate, mixture))
    measures = {key: median([row[key] for row in rows if np.isfinite(row[key])])
                for key in ("vocal_sdr", "instrumental_sdr", "vocal_si_sdr", "bleed")}
    measures["abs_bleed"] = median([abs(row["bleed"]) for row in rows])
    return {"loss": loss, "metrics": measures}


def overfit(tracks, config):
    fit_records, held_records = _panel_records()
    fitting = RecordedClips(tracks, fit_records)
    held = RecordedClips(tracks, held_records)
    configure_determinism(config.seed)
    model = SubbandSeparator().cuda()
    optimizer = make_optimizer(model, config)
    before = {"fit": _panel_measure(model, fitting, config), "held": _panel_measure(model, held, config)}
    audio = torch.stack([row["audio"] for row in fitting])
    stats = [optimizer_update(model, optimizer, audio, config, detailed=step in (0, 31))
             for step in range(32)]
    after = {"fit": _panel_measure(model, fitting, config), "held": _panel_measure(model, held, config)}
    gates = {
        "fitting_loss_decreased": after["fit"]["loss"]["val_total_loss"] < before["fit"]["loss"]["val_total_loss"],
        "fitting_vocal_sdr_improved": after["fit"]["metrics"]["vocal_sdr"] > before["fit"]["metrics"]["vocal_sdr"],
        "fitting_absolute_bleed_decreased": after["fit"]["metrics"]["abs_bleed"] < before["fit"]["metrics"]["abs_bleed"],
        "held_absolute_bleed_decreased": after["held"]["metrics"]["abs_bleed"] < before["held"]["metrics"]["abs_bleed"],
        "major_branch_gradients": all(
            bool(row["branch_gradient_norms"]) and all(value > 0 and np.isfinite(value)
                for value in row["branch_gradient_norms"].values()) for row in (stats[0], stats[-1])),
    }
    return {"before": before, "after": after, "steps": stats, "gates": gates,
            "fit_records": fit_records, "held_records": held_records}


def _exact(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_exact(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b))
    return a == b


def exact_resume(tracks, config, store):
    """Run two one-update epoch fixtures with the full model and frozen schedule."""
    configure_determinism(config.seed)
    model = SubbandSeparator().cuda()
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    held = RecordedClips(tracks, _panel_records()[1])
    val_batches = make_loader(held, config, workers=0, training=False)
    zero = validate(model, val_batches, config)
    selection, history = SelectionState(zero["val_total_loss"]), [zero]
    save_epoch(store, model, optimizer, scheduler, 0, selection, history, config)

    def batch_for(epoch):
        size = config.microbatch * config.accumulation
        dataset = TrainClips(tracks, epoch, samples=size, seed=config.seed)
        return next(iter(make_loader(dataset, config, workers=0)))

    first = optimizer_update(model, optimizer, batch_for(1)["audio"], config)
    scheduler.step()
    validation = validate(model, val_batches, config)
    selection, improved = update_selection(selection, validation["val_total_loss"], 1, 1e-4)
    history.append(validation)
    save_epoch(store, model, optimizer, scheduler, 1, selection, history, config)
    selected = store.load("best")["epoch"]

    next_batch = batch_for(2)
    expected_step = optimizer_update(model, optimizer, next_batch["audio"], config)
    scheduler.step()
    expected_rng = rng_state()

    restored = SubbandSeparator().cuda()
    restored_optimizer = make_optimizer(restored, config)
    restored_scheduler = make_scheduler(restored_optimizer, config)
    epoch, restored_selection, restored_history = resume_epoch(
        store, restored, restored_optimizer, restored_scheduler, config)
    resumed_batch = batch_for(epoch + 1)
    resumed_step = optimizer_update(restored, restored_optimizer, resumed_batch["audio"], config)
    restored_scheduler.step()
    checks = {
        "next_batch": _exact(next_batch, resumed_batch), "step_metrics": _exact(expected_step, resumed_step),
        "model": _exact(model.state_dict(), restored.state_dict()),
        "optimizer": _exact(optimizer.state_dict(), restored_optimizer.state_dict()),
        "scheduler": _exact(scheduler.state_dict(), restored_scheduler.state_dict()),
        "rng": _exact(expected_rng, rng_state()),
        "history": _exact(history, restored_history), "selection": selection == restored_selection,
        "selected_checkpoint": selected == selection.best_epoch, "latest_epoch": epoch == 1,
    }
    return {"checks": checks, "epoch0_val": zero, "epoch1_val": validation,
            "epoch1_step": first, "selected_epoch": selected, "improved": improved,
            "continuation_preclip_norm": resumed_step["preclip_norm"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").isalnum():
        parser.error("Run ID must contain ASCII letters, digits, and underscores")
    config = TrainConfig()
    configure_determinism(config.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The approved BF16 CUDA mode is unavailable")
    reports = PROJECT_ROOT / "reports/exp08_subband" / args.run_id
    # The protected fingerprint can be recorded before this entry point is invoked.
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "run_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"config": asdict(config), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name()}, handle, indent=2)
    checkpoints = reserve_run(PROJECT_ROOT / "models/experiments/exp08_subband" / args.run_id)
    logs = reserve_run(PROJECT_ROOT / "outputs/logs/exp08_subband" / args.run_id)
    train, val = frozen_partition()
    print("Hashing the frozen 85/15 audio and source manifests", flush=True)
    manifest = fingerprint_manifest(train, val)
    _write(reports / "manifest.json", manifest)
    store = Checkpoints(checkpoints, {"config": asdict(config),
                         "manifest_sha256": sha256(reports / "manifest.json")})
    print("Profiling the frozen real-data workload", flush=True)
    measured = profile(train, config, logs)
    _write(reports / "profile.json", measured)
    print(json.dumps(measured), flush=True)
    torch.cuda.empty_cache()

    print("Running the 32-update train-only overfit probe", flush=True)
    fit = overfit(train, config)
    _write(reports / "overfit.json", fit)
    torch.cuda.empty_cache()
    print("Verifying CUDA checkpoint selection and exact continuation", flush=True)
    resume = exact_resume(train, config, store)
    _write(reports / "resume.json", resume)
    print("Replaying the bounded oracle on the frozen train-only panel", flush=True)
    oracle = replay_oracle(train)
    _write(reports / "oracle_replay.json", oracle)
    result = {"gates": measured["gates"] | fit["gates"] | {"resume": all(resume["checks"].values()),
                                                         "bounded_oracle": oracle["passed"]},
              "estimated_training_hours_excluding_validation": 60 * config.samples_per_epoch
                                                                  / measured["clips_per_sec"] / 3600}
    result["decision"] = "GO" if all(result["gates"].values()) else "NO-GO"
    _write(reports / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
