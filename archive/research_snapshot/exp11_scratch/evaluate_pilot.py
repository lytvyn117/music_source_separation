"""Evaluate the two committed step-6000 weights with the frozen development pipeline."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from checkpoint import Store, file_hash, read_json
from data import ROOT, verify_cache
from metrics import compute_framewise_metrics
from scratch_model import Separator
from train_exp11 import BACKENDS, PILOT_UPDATES, bind, provenance, setup
import validation

EXP10_RUN = ROOT.parent / "exp10_scratch" / "runs" / "exp10_seed909"
EXTRA = ("vocal_si_sdr_db", "vocal_bleed_energy_pct", "vocal_retention", "mixture_max_abs")
METRICS = (*validation.METRICS, *EXTRA)
TOLERANCES = {
    "vocal_sdr_db": .10, "instrumental_sdr_db": .10, "v_side_sdr_db": .15,
    "i_side_sdr_db": .15, "abs_vocal_bleed_coeff": .01,
}
BASE_SONG_METRICS = validation.song_metrics


def additional_metrics(mix, vocals, inst, pred_v, pred_i):
    row = BASE_SONG_METRICS(mix, vocals, inst, pred_v, pred_i)
    si_sdr = compute_framewise_metrics(vocals, pred_v, silence_threshold_db=-45)["median_si_sdr_db"]
    retention = float(np.einsum("ij,ij->", vocals, pred_v, dtype=np.float64)
                      / (validation.energy(vocals) + validation.EPS))
    return {**row, "vocal_si_sdr_db": si_sdr,
            "vocal_bleed_energy_pct": 100 * row["signed_vocal_bleed_coeff"] ** 2,
            "vocal_retention": retention}


def aggregate(songs):
    groups = defaultdict(list)
    for row in songs:
        groups[row["artist_key"]].append(row)
    if len(songs) != 16 or len(groups) != 7 or len({r["uuid"] for r in songs}) != len(songs):
        raise ValueError("Expected the 16 unique frozen tracks in seven artist groups")

    def mean_or_missing(values):
        if any(v is None for v in values):
            return None
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite development metric")
        return float(np.mean(values))

    artists = {a: {k: mean_or_missing([r[k] for r in rs]) for k in METRICS}
               for a, rs in sorted(groups.items())}
    return {
        "songs": songs, "artists": artists,
        "artist_macro": {k: mean_or_missing([r[k] for r in artists.values()]) for k in METRICS},
        "panel_max_mixture_residual": max(r["mixture_max_abs"] for r in songs),
    }


def checkpoint_at(run, provenance_record, maximum, interval):
    store = Store(run, provenance_record["identity"], interval, maximum)
    path, metadata = store.checked_path(store.bundle(PILOT_UPDATES))
    state = torch.load(path, map_location="cpu", weights_only=True)
    store.validate_header(state, PILOT_UPDATES)
    return state, metadata


def measure(state, roster):
    device = torch.device("cuda")
    model = Separator(checkpoint_blocks=True).to(device).eval().requires_grad_(False)
    result = {}
    # The training evaluator stays byte-identical; enrich only offline song reporting.
    with (
        patch.object(validation, "song_metrics", additional_metrics),
        sdpa_kernel(BACKENDS),
        torch.autocast(device.type, enabled=False),
    ):
        for w, weights in (("raw", state["raw_model"]), ("ema", state["ema"]["weights"])):
            model.load_state_dict(weights, strict=True)
            measured = validation.evaluate(model, roster, device)["songs"]
            original = state["history"][-1]["validation"][w]["songs"]
            original_by_id = {r["uuid"]: r for r in original}
            for row in measured:
                for k in validation.METRICS:
                    if not math.isclose(row[k], original_by_id[row["uuid"]][k], rel_tol=0, abs_tol=1e-6):
                        raise ValueError(f"Re-scoring differs from the committed validation: {w} {row['uuid']} {k}")
            result[w] = aggregate(measured)
    return result


def archived_baseline(run):
    history = read_json(run / "validation_history.json")
    snapshot = next(row for row in history if row["step"] == PILOT_UPDATES)["validation"]
    return {
        w: aggregate([
            {**r, "vocal_si_sdr_db": None,
             "vocal_bleed_energy_pct": 100 * r["signed_vocal_bleed_coeff"] ** 2,
             "vocal_retention": 1. - r["signed_vocal_bleed_coeff"]}
            for r in arm["songs"]
        ]) for w, arm in snapshot.items()
    }


def compare(exp11, baseline):
    def delta(new, old):
        return {k: new[k] - old[k] if new[k] is not None and old[k] is not None else None
                for k in METRICS}

    old_songs = {r["uuid"]: r for r in baseline["songs"]}
    if (set(old_songs) != {r["uuid"] for r in exp11["songs"]}
            or baseline["artists"].keys() != exp11["artists"].keys()
            or any(r["artist_key"] != old_songs[r["uuid"]]["artist_key"] for r in exp11["songs"])):
        raise ValueError("Step 6000 weights were not scored on matching songs and artists")
    macro_gap = delta(exp11["artist_macro"], baseline["artist_macro"])
    gates = {k: (macro_gap[k] <= margin if k == "abs_vocal_bleed_coeff" else macro_gap[k] >= -margin)
             for k, margin in TOLERANCES.items()}
    residual_diff = exp11["panel_max_mixture_residual"] - baseline["panel_max_mixture_residual"]
    gates["mixture_residual"] = residual_diff <= 0
    return {
        "artist_macro_delta": macro_gap, "preliminary_nonregression": gates,
        "panel_max_mixture_residual_delta": residual_diff,
        "artists_delta": {a: delta(r, baseline["artists"][a]) for a, r in exp11["artists"].items()},
        "songs_delta": [{"uuid": r["uuid"], "artist_key": r["artist_key"], **delta(r, old_songs[r["uuid"]])}
                        for r in exp11["songs"]],
    }


def pilot_diagnostics(run, settings):
    amplitudes = defaultdict(list)
    terms = (*settings["loss"], "loss", "gradient_l2", "peak_reserved_gib")
    with (run / "steps.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if [row["step"] for row in rows] != list(range(1, PILOT_UPDATES + 1)):
        raise ValueError("The pilot update log is incomplete or out of order")
    if any(not math.isfinite(row[k]) for row in rows for k in terms):
        raise ValueError("The pilot update log has nonfinite diagnostics")
    for row in rows:
        crops = row["crops"]
        if len(crops) != settings["microbatch"]:
            raise ValueError("Incomplete pilot crop log")
        for crop in crops:
            levels = tuple(crop[k] for k in ("peak", "rms", "fraction_over_full_scale"))
            if not all(math.isfinite(x) for x in levels):
                raise ValueError("Nonfinite pilot crop amplitudes")
            amplitudes["remixed" if crop["remixed"] else "original"].append(levels)
    report = {}
    for branch, levels in amplitudes.items():
        a = np.asarray(levels)
        report[branch] = {
            "crops": len(a), "crops_over_full_scale": int((a[:, 0] > 1).sum()),
            "quantiles": {metric: dict(zip(("p05", "p50", "p95", "p99", "max"),
                                         np.quantile(a[:, i], (.05, .50, .95, .99, 1)).tolist()))
                          for i, metric in enumerate(("peak", "rms", "fraction_over_full_scale"))},
        }
    return {"finite_logged_updates": len(rows), "maximum_preclip_gradient_l2": max(r["gradient_l2"] for r in rows),
            "maximum_logged_reserved_GiB": max(r["peak_reserved_gib"] for r in rows), "amplitudes": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if run.parent != (ROOT / "runs").resolve() or output.parent != run or output.suffix != ".json":
        parser.error("Use an Exp11 run and a new JSON file within that run")
    if output.exists():
        parser.error("The evaluation output must be new")
    config, roster = bind()
    settings = config["exp09_settings"]
    runtime = setup(settings["seed"])
    refs = verify_cache(roster["development"])
    record = provenance(config, refs, runtime, "pilot")
    if read_json(run / "run.json") != record:
        raise ValueError("Pilot run provenance differs from the current runtime or inputs")
    interval = settings["checkpoint_every"]
    state, pilot_meta = checkpoint_at(run, record, PILOT_UPDATES, interval)
    diagnostics = pilot_diagnostics(run, settings)
    print("Scoring Exp11 Raw and EMA at update 6000", flush=True)
    exp11 = measure(state, roster["development"])
    del state
    baseline_record = read_json(EXP10_RUN / "run.json")
    try:
        baseline_state, baseline_meta = checkpoint_at(EXP10_RUN, baseline_record, settings["updates"], interval)
    except (FileNotFoundError, PermissionError) as exc:
        baseline = archived_baseline(EXP10_RUN)
        baseline_evidence = {"mode": "archived_step6000_song_metrics",
                             "log_sha256": file_hash(EXP10_RUN / "validation_history.json"),
                             "detail": str(exc), "vocal_si_sdr_db": "unavailable",
                             "vocal_retention": "1 minus signed bleed, up to float32 rounding"}
    else:
        print("Scoring matched Exp10 Raw and EMA at update 6000", flush=True)
        baseline = measure(baseline_state, roster["development"])
        baseline_evidence = {"mode": "checkpoint", "checkpoint": baseline_meta}
    result = {
        "step": PILOT_UPDATES, "exp11_checkpoint": pilot_meta, "exp10_evidence": baseline_evidence,
        "pilot_diagnostics": diagnostics,
        "exp11": exp11, "exp10": baseline,
        "comparison": {w: compare(exp11[w], baseline[w]) for w in ("raw", "ema")},
        "judgment": "Review stability, all artist effects and meaningful positive signal before a full-run decision.",
    }
    # Reserve the new output name atomically before publishing the JSON payload.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
