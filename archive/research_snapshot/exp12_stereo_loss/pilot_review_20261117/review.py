"""Read-only verification and instrumented 2-second development replay of Exp12."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from checkpoint import Store, file_hash, read_json
from data import ARCHIVE, open_references, verify_cache
from evaluate_pilot import ARMS, BASELINE, METRICS, ROUNDING, SIDE, report, summarize
from scratch_model import SEGMENT, Separator
from scratch_train import model_digest
from train_exp12 import BACKENDS, bind, setup
from validation import EPS, artist_macro, song_metrics

RUN = ROOT / "runs" / "exp12_pilot_seed909"
EXP14 = ROOT.parent / "exp14_inference_hop"
EXP11 = ROOT.parent / "exp11_scratch" / "runs" / "exp11_pilot_seed909" / "matched_step6000.json"
GIB = 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(name, payload):
    with (OUT / name).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def error_stats(vocals, instrumental, pred_v, pred_i):
    totals = np.zeros((2, 4), dtype=np.float64)
    max_error_sum = 0.0
    for begin in range(0, vocals.shape[1], 262144):
        end = begin + 262144
        vocal_error = pred_v[:, begin:end].astype(np.float64) - vocals[:, begin:end]
        inst_error = pred_i[:, begin:end].astype(np.float64) - instrumental[:, begin:end]
        max_error_sum = max(max_error_sum, float(np.max(np.abs(vocal_error + inst_error))))
        for sums, error in zip(totals, (vocal_error, inst_error), strict=True):
            left, right = error
            mid, side = (left + right) * 0.5, (left - right) * 0.5
            sums += [np.einsum("i,i->", x, x) for x in (left, right, mid, side)]
    return {
        **{
            label: dict(zip(("left", "right", "mid", "side"),
                            np.sqrt(values / vocals.shape[1]).tolist(), strict=True))
            for label, values in zip(("vocal_error_rms", "instrumental_error_rms"), totals, strict=True)
        },
        "max_abs_sum_of_stem_errors": max_error_sum,
    }


def synthetic():
    pattern = np.tile(np.array([1, -1, 1, -1], dtype=np.float32), 11025)
    vocals = np.stack((3 * pattern, pattern)) / 16
    instrumental = np.stack((-pattern, 3 * pattern)) / 8
    mixture = vocals + instrumental
    old_error = np.stack((3 * pattern, -pattern)) / 64
    new_error = np.stack((3 * pattern, pattern)) / 64
    result = {}
    for label, error in (("a", old_error), ("b", new_error)):
        pred_v = vocals + error
        pred_i = mixture - pred_v
        row = song_metrics(mixture, vocals, instrumental, pred_v, pred_i)
        diagnostics = error_stats(vocals, instrumental, pred_v, pred_i)
        require(diagnostics["max_abs_sum_of_stem_errors"] == row["mixture_max_abs"] == 0,
                "Synthetic estimate and reference complement failed")
        require(diagnostics["vocal_error_rms"] == diagnostics["instrumental_error_rms"],
                "Synthetic complementary RMS mismatch")
        result[label] = {**{key: row[key] for key in SIDE}, **diagnostics}
    delta = {key: result["b"][key] - result["a"][key] for key in SIDE}
    expected = 10 * math.log10((len(pattern) * (1 / 32)**2 + EPS)
                               / (len(pattern) * (1 / 64)**2 + EPS))
    require(all(abs(value - expected) < 1e-10 for value in delta.values()),
            "Synthetic Side delta does not match the shared error-power formula")
    return {**result, "delta_db": delta, "predicted_delta_db": expected}


def historical_side(exp12):
    with (EXP14 / "per_artist_paired.csv").open(encoding="utf-8", newline="") as handle:
        exp14_artists = list(csv.DictReader(handle))
    with (EXP14 / "per_song_paired.csv").open(encoding="utf-8", newline="") as handle:
        exp14_songs = list(csv.DictReader(handle))
    archived11 = read_json(EXP11)
    differences = {}
    for arm in ARMS:
        archived14 = {
            hop: {row["uuid"]: row for row in read_json(EXP14 / f"arm_{arm}_{hop}s.json")["songs"]}
            for hop in (1, 2)
        }
        songs14 = [r for r in exp14_songs if r["weights"] == arm]
        artists14 = [r for r in exp14_artists if r["weights"] == arm]
        require(len(songs14) == len(archived14[1]) == len(archived14[2]) == 16
                and len(artists14) == 7, "Wrong Exp14 comparison panel")
        groups = defaultdict(list)
        for row in songs14:
            key = row["uuid"]
            value = {}
            for metric in SIDE:
                delta = archived14[1][key][metric] - archived14[2][key][metric]
                require(float(row["delta_" + metric]) == delta, "Exp14 song CSV mismatch")
                value[metric] = delta
            groups[row["artist_key"]].append(value)
        for row in artists14:
            for metric in SIDE:
                average = np.mean([group[metric] for group in groups[row["artist_key"]]])
                require(abs(float(row["delta_" + metric]) - average) < 1e-12,
                        "Exp14 group CSV mismatch")
        differences[arm] = {
            "exp14_max_song_delta_difference_db": max(
                abs(float(row["delta_" + SIDE[0]]) - float(row["delta_" + SIDE[1]]))
                for row in songs14
            ),
            "exp14_max_artist_delta_difference_db": max(
                abs(float(row["delta_" + SIDE[0]]) - float(row["delta_" + SIDE[1]]))
                for row in artists14
            ),
            "exp11_max_artist_delta_difference_db": max(
                abs(row[SIDE[0]] - row[SIDE[1]])
                for row in archived11["comparison"][arm]["artists_delta"].values()
            ),
            "exp12_max_artist_delta_difference_db": max(
                abs(row[SIDE[0]]["delta"] - row[SIDE[1]]["delta"])
                for row in exp12["arms"][arm]["artists"].values()
            ),
        }
    return differences


def training_evidence(config, record, state, development):
    steps = [json.loads(line) for line in (RUN / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    require(all(math.isfinite(row[field]) and row[field] >= 0 for row in steps
                for field in ("seconds", "peak_allocated_gib", "free_gib")), "Invalid telemetry")
    index = read_json(RUN / "checkpoint_index.json")
    store = Store(RUN, record["identity"], 1000, 6000)
    require(store.published() == [store.bundle(step) for step in range(0, 6001, 1000)],
            "Unexpected checkpoint boundaries")
    checkpoints = {}
    lr_base = config["exp09_settings"]["optimizer"]["lr"]
    eta_min = config["scheduler"]["eta_min"]
    for step in range(0, 6001, 1000):
        path, meta = store.checked_path(store.bundle(step))
        boundary = state if step == 6000 else torch.load(path, map_location="cpu", weights_only=True)
        store.validate_header(boundary, step)
        scheduler = boundary["scheduler"]
        expected_lr = eta_min + (lr_base - eta_min) * (1 + math.cos(math.pi * step / 24000)) / 2
        require(scheduler["last_epoch"] == boundary["ema"]["updates"] == step
                and scheduler["T_max"] == 24000
                and boundary["ema"]["decay"] == config["ema_decay"]
                and math.isclose(scheduler["eta_min"], eta_min, abs_tol=1e-14), "Schedule or EMA mismatch")
        groups = boundary["optimizer"]["param_groups"]
        require(len(groups) == 1 and groups[0]["weight_decay"] == .01
                and groups[0]["foreach"] is False and groups[0]["fused"] is False, "AdamW options mismatch")
        require(math.isclose(groups[0]["lr"], expected_lr, abs_tol=1e-14)
                and math.isclose(scheduler["_last_lr"][0], expected_lr, abs_tol=1e-14), "LR mismatch")
        optimizer_state = boundary["optimizer"]["state"]
        require((not step and not optimizer_state) or
                (len(optimizer_state) == len(groups[0]["params"])
                 and all(int(entry["step"]) == step for entry in optimizer_state.values())),
                "AdamW per-parameter update count mismatch")
        if step == 0:
            model = Separator(checkpoint_blocks=True)
            model.load_state_dict(boundary["raw_model"], strict=True)
            require(model_digest(model) == config["initial_model_sha256"]
                    and sum(p.numel() for p in model.parameters()) == 11662596,
                    "Initialization checkpoint differs from Exp10")
        checkpoints[str(step)] = {**meta, "lr": groups[0]["lr"], "ema_updates": step}
    require(checkpoints["6000"]["sha256"] == index["latest"]["sha256"]
            == file_hash(RUN / "best_overall.pt"), "Final checkpoint alias mismatch")
    require(not (RUN / "best_qualified.pt").exists() and index["best_qualified"] is None,
            "Unexpected qualified alias")
    for boundary in state["history"]:
        for validation in boundary["validation"].values():
            rows, _, _ = summarize(validation, development)
            for row in rows.values():
                require(row["mixture_max_abs"] <= 2e-5 and all(
                    math.isfinite(value) for key, value in row.items() if key not in ("uuid", "artist_key")
                ), "Invalid intermediate validation")
    start = (RUN / "run.json").stat().st_mtime
    end = (RUN / "completion.json").stat().st_mtime
    return {
        "checkpoints": checkpoints,
        "first_run_record_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "completion_record_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "file_timestamp_elapsed_seconds": end - start,
        "update_compute_seconds": math.fsum(row["seconds"] for row in steps),
        "validation_total_seconds": math.fsum(
            val["seconds"] for boundary in state["history"] for val in boundary["validation"].values()
        ),
        "step6000_validation_seconds_including_reference_io_and_metrics": {
            arm: state["history"][-1]["validation"][arm]["seconds"] for arm in ARMS
        },
        "peak_logged_allocated_gib": max(row["peak_allocated_gib"] for row in steps),
        "peak_logged_reserved_gib": max(row["peak_reserved_gib"] for row in steps),
        "minimum_logged_free_gib": min(row["free_gib"] for row in steps),
        "validation_boundaries": [entry["step"] for entry in state["history"]],
    }


def audit():
    config, roster = bind()
    record = read_json(RUN / "run.json")
    matched = report(RUN)
    require(matched == read_json(RUN / "matched_step6000.json"), "Matched archive differs from fresh replay")
    exp10 = ROOT.parent / "exp10_scratch"
    require(config == read_json(exp10 / "config.json"), "Exp10 config changed")
    unchanged = {
        name: file_hash(ROOT / name)
        for name in ("roster.json", "checkpoint.py", "validation.py", "selection.py", *(
            f"exp09_core/{path.name}" for path in sorted((ROOT / "exp09_core").glob("*.py"))
            if path.name != "loss_candidates.py"
        ))
    }
    require(all(file_hash(exp10 / name) == sha for name, sha in unchanged.items()),
            "A controlled model, roster, optimizer, sampling, scoring or checkpoint file changed")
    require(record["development_cache"] == verify_cache(roster["development"]),
            "Development cache drifted")
    baseline_index = read_json(BASELINE / "checkpoint_index.json")
    protected = {
        BASELINE / "validation_history.json": matched["exp10_baseline_history_sha256"],
        BASELINE / "steps.jsonl": matched["exp10_baseline_step_log_sha256"],
        BASELINE / "latest.pt": baseline_index["latest"]["sha256"],
        BASELINE / "best_overall.pt": baseline_index["best_overall"]["sha256"],
        **{EXP14 / path: sha for path, sha in read_json(EXP14 / "EXP14_RESULTS.json")["arm_files_sha256"].items()},
    }
    require(all(file_hash(path) == sha for path, sha in protected.items()), "Protected artifact hash mismatch")
    identical_exp14 = historical_side(matched)
    fixture = synthetic()
    print("Matched evaluation replay, development cache, protected hashes and Side SDR checks verified",
          flush=True)
    state = torch.load(RUN / "latest.pt", map_location="cpu", weights_only=True)
    training = training_evidence(config, record, state, roster["development"])
    save("audit.json", {
        "matched": matched, "run": record, "config": config,
        "unchanged_exp10_file_sha256": unchanged,
        "protected_artifact_sha256": {p.relative_to(ROOT.parent).as_posix(): h for p, h in protected.items()},
        "archive_bytes": ARCHIVE.stat().st_size,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "training_shard_budget_bytes": roster["shard_plan"]["budget_bytes"],
        "checked_roster_size": {role: len(roster[role]) for role in ("train", "development")},
        "artist_group_count": len({row["artist_key"] for row in roster["development"].values()}),
        "synthetic": fixture, "archive_side_differences": identical_exp14, "training": training,
        "evidence_sha256": {name: file_hash(RUN / name) for name in
                            ("run.json", "completion.json", "checkpoint_index.json", "steps.jsonl",
                             "validation_history.json", "matched_step6000.json")},
    })
    print("All seven serialized boundaries, optimizer counts, scheduler and validation records verified",
          flush=True)


def profile():
    sys.path.append(str(EXP14))
    from evaluate_hops import predict

    verified = read_json(OUT / "audit.json")
    config, roster = bind()
    development = roster["development"]
    runtime = setup(config["exp09_settings"]["seed"])
    require(runtime == verified["run"]["runtime"], "Replay runtime differs from the pilot")
    path = RUN / "latest.pt"
    require(file_hash(path) == verified["matched"]["exp12_checkpoint"]["sha256"], "Replay checkpoint changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    model = Separator(checkpoint_blocks=True).to(device).eval().requires_grad_(False)
    results = {}
    for label in ARMS:
        weights = state["raw_model"] if label == "raw" else state["ema"]["weights"]
        model.load_state_dict(weights, strict=True)
        require(all(bool(tensor.isfinite().all()) for tensor in weights.values()), "Nonfinite model weight")
        reference = {r["uuid"]: r for r in state["history"][-1]["validation"][label]["songs"]}
        rows = []
        max_metric_difference = 0.0
        torch.cuda.empty_cache()
        started = perf_counter()
        with sdpa_kernel(BACKENDS), torch.autocast(device.type, enabled=False), torch.inference_mode():
            for key in development:
                with open_references(development, key) as refs:
                    v, i = (np.ascontiguousarray(ref.T) for ref in refs)
                mix = v + i
                pred_v, pred_i, telemetry = predict(model, mix, 2, device)
                scores = song_metrics(mix, v, i, pred_v, pred_i)
                old = reference[key]
                require(telemetry["chunks"] == old["chunks"], "Chunk count differs from archived pilot")
                require(abs(scores["mixture_max_abs"] - old["mixture_max_abs"]) <= ROUNDING,
                        "Replayed mixture residual differs from archived pilot")
                max_metric_difference = max(max_metric_difference, *(abs(scores[k] - old[k]) for k in METRICS))
                require(max_metric_difference <= 1e-6, "Replay score differs from archived pilot")
                rows.append({"uuid": key, "artist_key": old["artist_key"], **scores, **telemetry,
                             **error_stats(v, i, pred_v, pred_i)})
                print(label, len(rows), "/", len(development), f"{telemetry['inference_seconds']:.2f} s", flush=True)
        results[label] = {
            "songs": rows, "artist_macro": artist_macro(rows),
            "inference_seconds": math.fsum(row["inference_seconds"] for row in rows),
            "wall_seconds_including_io_scoring_and_exploratory_diagnostics": perf_counter() - started,
            "peak_gpu_allocated_mib": max(row["peak_gpu_allocated_mib"] for row in rows),
            "peak_gpu_reserved_mib": max(row["peak_gpu_reserved_mib"] for row in rows),
            "chunks": sum(row["chunks"] for row in rows),
            "max_archived_score_difference": max_metric_difference,
        }
    require(file_hash(path) == verified["matched"]["exp12_checkpoint"]["sha256"], "Checkpoint changed during replay")
    save("profile.json", {
        "runtime": runtime, "hop_seconds": 2, "segment_frames": SEGMENT,
        "checkpoint_sha256": file_hash(path), "arms": results,
    })


if __name__ == "__main__":
    if sys.argv[1:] == ["audit"]:
        audit()
    elif sys.argv[1:] == ["profile"]:
        profile()
    else:
        raise SystemExit("Usage: review.py audit | profile")
