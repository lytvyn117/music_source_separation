"""Compare a completed Exp12 pilot with committed Exp10 step-6000 validation."""

import argparse
import json
import math
from pathlib import Path

import torch

from checkpoint import Store, file_hash, read_json
from train_exp12 import (
    FAILURE_LOG,
    PILOT_UPDATES,
    ROOT,
    bind,
    provenance,
)
from validation import METRICS, artist_macro

BASELINE = ROOT.parent / "exp10_scratch" / "runs" / "exp10_seed909"
BASELINE_HISTORY_SHA256 = "cc997976ed2bf37a847656843d8c504292b4f3a6486067062c19add4c0ffc86d"
BASELINE_STEP_LOG_SHA256 = "6f9e10f2bc452623086b2c4c67156e87ab8784487ed1a1b40d25bb712fba372e"
BASELINE_VALIDATION_STEP = 6000
ARMS = ("raw", "ema")
RESIDUAL = "mixture_max_abs"
BLEED = "abs_vocal_bleed_coeff"
SIDE = ("v_side_sdr_db", "i_side_sdr_db")
ROUNDING = 2.0**-23


def check_updates(run, config):
    settings = config["exp09_settings"]
    metrics = (*settings["loss"], "loss", "gradient_l2", "peak_reserved_gib")
    with (run / "steps.jsonl").open(encoding="utf-8") as stream:
        pilot = [json.loads(line) for line in stream]
    if len(pilot) != PILOT_UPDATES:
        raise ValueError("Expected exactly 6000 logged pilot updates")
    baseline_steps = BASELINE / "steps.jsonl"
    if file_hash(baseline_steps) != BASELINE_STEP_LOG_SHA256:
        raise ValueError("Exp10 update archive changed")
    with baseline_steps.open(encoding="utf-8") as stream:
        for step, new in enumerate(pilot, 1):
            old = json.loads(next(stream))
            if (
                new["step"] != step
                or old["step"] != step
                or new["shard"] != old["shard"]
                or new["crops"] != old["crops"]
            ):
                raise ValueError(f"Baseline crop parity failed at update {step}")
            if (
                any(not math.isfinite(new[k]) or new[k] < 0 for k in metrics)
                or new["peak_reserved_gib"] > config["memory_guard_gib"]
            ):
                raise ValueError(f"Invalid numerical diagnostics at update {step}")
    return {
        "finite_matched_updates": len(pilot),
        "maximum_preclip_gradient_l2": max(r["gradient_l2"] for r in pilot),
        "maximum_logged_reserved_gib": max(r["peak_reserved_gib"] for r in pilot),
    }


def summarize(arm, development):
    songs = {row["uuid"]: row for row in arm["songs"]}
    if (
        len(songs) != len(arm["songs"])
        or songs.keys() != development.keys()
        or any(row["artist_key"] != development[key]["artist_key"] for key, row in songs.items())
    ):
        raise ValueError("Validation songs or artists differ from the frozen roster")
    scores = artist_macro(list(songs.values()))
    reported = arm["artist_macro"]
    if any(not math.isclose(scores[k], reported[k], rel_tol=0.0, abs_tol=1e-10) for k in METRICS):
        raise ValueError("Stored artist macro disagrees with its songs")
    if any(not math.isfinite(r[RESIDUAL]) or r[RESIDUAL] < 0 for r in songs.values()):
        raise ValueError("Invalid song mixture-consistency residual")
    groups = {r["artist_key"] for r in songs.values()}
    artists = {
        artist: artist_macro([r for r in songs.values() if r["artist_key"] == artist])
        for artist in sorted(groups)
    }
    return songs, artists, scores


def compare(new, old, development):
    new_songs, new_artists, new_macro = summarize(new, development)
    old_songs, old_artists, old_macro = summarize(old, development)

    def paired(a, b, metrics):
        return {k: {"exp12": a[k], "exp10": b[k], "delta": a[k] - b[k]} for k in metrics}

    overall = paired(new_macro, old_macro, METRICS)
    songs = {
        key: {
            "artist_key": row["artist_key"],
            **paired(row, old_songs[key], (*METRICS, RESIDUAL)),
        }
        for key, row in new_songs.items()
    }
    artists = {
        group: paired(row, old_artists[group], METRICS) for group, row in new_artists.items()
    }
    joint_side_gains = [
        artist for artist, row in artists.items() if all(row[k]["delta"] > 0 for k in SIDE)
    ]
    worst_new = max(r[RESIDUAL] for r in new_songs.values())
    worst_old = max(r[RESIDUAL] for r in old_songs.values())
    gates = {
        "vocal_sdr": overall["vocal_sdr_db"]["delta"] >= -0.10,
        "instrumental_sdr": overall["instrumental_sdr_db"]["delta"] >= -0.10,
        "vocal_side_sdr": overall[SIDE[0]]["delta"] >= 0.20,
        "instrumental_side_sdr": overall[SIDE[1]]["delta"] >= 0.20,
        "absolute_vocal_bleed": overall[BLEED]["delta"] <= 0.01,
        "artist_spread": len(joint_side_gains) >= 4,
        "mixture_consistency": (
            worst_new - worst_old <= ROUNDING
            and all(r[RESIDUAL]["delta"] <= ROUNDING for r in songs.values())
        ),
    }
    return {
        "artist_macro": overall,
        "artists": artists,
        "songs": songs,
        "artists_with_both_side_gains": joint_side_gains,
        "panel_max_mixture_residual": paired(
            {RESIDUAL: worst_new}, {RESIDUAL: worst_old}, (RESIDUAL,)
        ),
        "screening_gates": gates,
        "passes_screening": all(gates.values()),
        "vocal_si_sdr_paired_delta": None,
    }


def report(run):
    config, roster = bind()
    record = read_json(run / "run.json")
    if record != provenance(config, record["development_cache"], record["runtime"], "pilot"):
        raise ValueError("Pilot provenance differs from the current frozen code or configuration")
    done = read_json(run / "completion.json")
    if (done["step"], done["reason"]) != (PILOT_UPDATES, "pilot_boundary"):
        raise ValueError("The pilot did not publish a step-6000 completion")
    metadata = read_json(run / "checkpoint_index.json")["latest"]
    path = run / "latest.pt"
    if (
        metadata["step"] != PILOT_UPDATES
        or metadata["identity"] != record["identity"]
        or path.stat().st_size != metadata["bytes"]
        or file_hash(path) != metadata["sha256"]
    ):
        raise ValueError("The step-6000 alias differs from the recorded checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    store = Store(run, record["identity"], config["validation"]["interval"], PILOT_UPDATES)
    store.validate_header(state, PILOT_UPDATES)
    if (
        state["history"] != read_json(run / "validation_history.json")
        or done["selection"] != state["history"][-1]["selection"]
        or state["ema"]["updates"] != PILOT_UPDATES
        or state["scheduler"]["last_epoch"] != PILOT_UPDATES
        or state["scheduler"]["T_max"] != config["scheduler"]["T_max"]
    ):
        raise ValueError("The saved history or optimizer schedule does not match pilot completion")
    new = state["history"][-1]["validation"]
    del state
    archived = BASELINE / "validation_history.json"
    if file_hash(archived) != BASELINE_HISTORY_SHA256:
        raise ValueError("Exp10 validation archive changed")
    old = next(
        r["validation"] for r in read_json(archived) if r["step"] == BASELINE_VALIDATION_STEP
    )
    checks = check_updates(run, config)
    failure_path = run / FAILURE_LOG
    checks["first_compute_failure"] = read_json(failure_path) if failure_path.exists() else None
    checks["clean_compute_history"] = not failure_path.exists()
    arms = {arm: compare(new[arm], old[arm], roster["development"]) for arm in ARMS}
    return {
        "step": PILOT_UPDATES,
        "exp12_checkpoint": metadata,
        "exp10_baseline_history_sha256": BASELINE_HISTORY_SHA256,
        "exp10_baseline_step_log_sha256": BASELINE_STEP_LOG_SHA256,
        "stability_and_sampler": checks,
        "arms": arms,
        "both_arms_pass_screening": (
            checks["clean_compute_history"] and all(a["passes_screening"] for a in arms.values())
        ),
        "review": "Review the paired artist and song distributions before a decision.",
        "bleed_units": "fraction (multiply deltas by 100 for percentage points)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if (
        run.parent != (ROOT / "runs").resolve()
        or output.parent != run
        or output.suffix != ".json"
        or output.exists()
    ):
        parser.error("Use a completed Exp12 run and a new JSON output within that run")
    result = report(run)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


if __name__ == "__main__":
    main()
