"""Build the 240-row metadata manifest and audit the frozen Exp09/Exp10 records."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from checkpoint import file_hash, read_json, write_json
from data import ROOT, load_roster
from scratch_model import RATE
from selection import decision

EXP10 = ROOT.parent / "exp10_scratch"
EXP09 = ROOT.parent.parent / "music_source_separation_exp09_scratch"
RUN10 = EXP10 / "runs" / "exp10_seed909"
RUN09 = EXP09 / "runs" / "exp09_mse_seed909"
COMPONENTS = ("complex_mse", "logmag_l1", "wave_l1", "side_l1", "leakage", "loss", "gradient_l2")


def manifest():
    source = {r["uuid"]: r for r in read_json(EXP09 / "manifest.json")}
    assignments = {r["uuid"]: r for r in read_json(EXP09 / "split_external_final.json")["assignments"]}
    eligibility = read_json(EXP09 / "training_eligibility.json")
    pcm = read_json(EXP09 / "eligibility_complete.json")
    panel = read_json(EXP09 / "dev_eval_20261114" / "frozen_panel.json")
    prior = read_json(EXP10 / "exp11_audit" / "audit.json")
    tails = {row["uuid"]: row for row in read_json(EXP09 / "tail_recovery.json")["songs"]}
    candidates = {r["uuid"]: r for r in prior["remaining"] if r["disposition"] == "technical_common_span_pass"}
    roster = load_roster()
    train = set(roster["train"])
    admitted = set(roster["development"])
    dev = set(panel["songs"])
    reserve = {k for k, row in assignments.items() if row["role"] == "held_out_artist_reserve"}
    assert len(source) == 240 and source.keys() == assignments.keys()
    assert train == set(eligibility["strict_train"]) and admitted == set(panel["admitted"])
    assert dev == {k for k, row in assignments.items() if row["role"] == "validation"}
    assert set(pcm["rows"]) | dev | reserve == source.keys()
    assert set(pcm["rows"]).isdisjoint(dev | reserve)
    assert set(candidates) == set(tails) and len(candidates) == 7
    hashes = [{t["pcm_sha256"] for key in keys for t in rows[key]["tracks"] if t.get("pcm_sha256")}
              for keys, rows in ((train, pcm["rows"]), (dev, panel["songs"]))]
    member_signatures = [{(t["bytes"], t["crc32"]) for key in keys for t in source[key]["tracks"]}
                         for keys in (train, dev, reserve)]
    artist_keys = [{source[k]["artist_key"] for k in keys} for keys in (train, dev, reserve)]
    duplicates = panel["candidates"]
    integrity = {
        "train_dev_uuid_intersection": sorted(train & dev),
        "train_reserve_uuid_intersection": sorted(train & reserve),
        "train_dev_artist_intersection": sorted(artist_keys[0] & artist_keys[1]),
        "train_reserve_artist_intersection": sorted(artist_keys[0] & artist_keys[2]),
        "train_dev_exact_pcm_stem_hash_matches": len(hashes[0] & hashes[1]),
        "train_dev_wav_size_crc_matches": len(member_signatures[0] & member_signatures[1]),
        "train_reserve_wav_size_crc_matches": len(member_signatures[0] & member_signatures[2]),
        "archived_development_acoustic_duplicate_candidates": duplicates,
        "archived_screen_counts": {k: len(v) if isinstance(v, (list, dict)) else v
                                   for k, v in panel["screen"].items()},
        "sealed_external_final_metadata": "Roster unavailable in active split evidence",
        "sealed_external_final_pcm": "Not accessed",
        "exact_duplicates_excluded": pcm["exact_duplicate_groups"],
    }
    if (any(integrity[k] for k in ("train_dev_uuid_intersection", "train_reserve_uuid_intersection",
                                  "train_dev_artist_intersection", "train_reserve_artist_intersection",
                                  "train_dev_exact_pcm_stem_hash_matches",
                                  "train_dev_wav_size_crc_matches", "train_reserve_wav_size_crc_matches"))
            or duplicates):
        raise RuntimeError(f"Potential split contamination: {integrity}")
    rows = []
    current_audio = ROOT / "audio_audit.json"
    observed = {}
    if current_audio.is_file():
        for row in read_json(current_audio)["observations"]:
            key = row["crop"]["vocal_uuid"]
            entry = observed.setdefault(key, {"crops": 0, "max_abs_fp32": 0.})
            entry["crops"] += 1
            entry["max_abs_fp32"] = max(entry["max_abs_fp32"], row["max_abs"])
    for k, song in sorted(source.items()):
        metadata = assignments[k]
        audio = pcm["rows"].get(k, panel["songs"].get(k))
        tracks = audio["tracks"] if audio else []
        starts = sorted({t["bwf_start"] for t in tracks if t["bwf_start"] is not None})
        low, high = (audio["frames_min"], audio["frames_max"]) if audio else (None, None)
        split = ("train" if k in train else "development" if k in admitted else
                 "development_excluded" if k in dev else "artist_reserve" if k in reserve else "unassigned")
        status = (eligibility["rows"][k]["status"] if k in eligibility["rows"] else
                  "development_role" if k in dev else "protected_artist_reserve")
        bleed_counts = Counter({name: sum(t["has_bleed"] is value for t in song["tracks"])
                                for name, value in (("reported", True), ("unknown", None), ("clear", False))})
        alignment = ("no_audio_evidence" if not audio else "conflicting_native_starts" if len(starts) > 1
                     else "missing_BWF" if audio["missing_bwf"] else "equal_starts_and_lengths" if low == high
                     else "equal_starts_unequal_lengths")
        rows.append({
            "track_uuid": k, "artist_identifier": song["artist_key"], "artist": song["artist"],
            "title": song["title"], "genre": song["genre"],
            "current_split": split, "assignment_role": metadata["role"],
            "training_eligible": k in train, "exclusion_reason": None if k in train else status,
            "development_exclusion_reasons": audio.get("reasons") if k in dev else None,
            "duration_seconds": low / RATE if low is not None and low == high else None,
            "duration_seconds_range": [low / RATE, high / RATE] if audio else None,
            "sample_rate_hz": RATE if audio else None, "channel_count": 2 if audio else None,
            "vocal_target_available": any(t["available"] and t["stem"] == "vocals" for t in song["tracks"]),
            "instrumental_target_available": any(t["available"] and t["stem"] != "vocals" for t in song["tracks"]),
            "target_rms_audit": audio["rms"] if audio else None,
            "alignment": {"status": alignment, "bwf_starts": starts,
                          "common_span_candidate": k in candidates},
            "stem_sum": {
                "training_domain": "float32 vocal + instrumental" if k in train else None,
                "approved_training_crop_probe": observed.get(k),
                "recorded_mixed_sum_residual_max": audio.get("mixed_sum_residual") if audio else None,
                "common_span_reconstruction_max": tails.get(k, {}).get("reconstruction_max_abs"),
                "raw_mix_checked": False, "audited_source_sum_peak": audio.get("peak") if audio else None,
                "audited_source_sum_over_full_scale": audio.get("overs") if audio else None,
            },
            "bleed": {"track_counts": dict(bleed_counts), "primary_exclusion": status in
                      {"review_unknown_bleed", "review_reported_bleed"}},
            "member_count": len(song["tracks"]), "metadata_flags": song["flags"],
            "audio_evidence": "archived_PCM_check" if audio else "metadata_only",
        })
    grouped = Counter(row["exclusion_reason"] or "eligible_strict" for row in rows)
    return {
        "recording_count": len(rows), "counts_by_eligibility": dict(grouped),
        "counts_by_current_split": dict(Counter(r["current_split"] for r in rows)),
        "train_artists": len(artist_keys[0]), "development_assigned_artists": len(artist_keys[1]),
        "development_scored_artists": len({source[k]["artist_key"] for k in admitted}),
        "training_hours": sum(r["duration_seconds"] for r in rows if r["training_eligible"]) / 3600,
        "integrity": integrity, "common_span_candidates": list(candidates.values()), "recordings": rows,
        "inputs_sha256": {str(path): file_hash(path) for path in (
            EXP09 / "manifest.json", EXP09 / "split_external_final.json", EXP09 / "training_eligibility.json",
            EXP09 / "eligibility_complete.json", EXP09 / "tail_recovery.json",
            EXP09 / "dev_eval_20261114" / "frozen_panel.json", EXP10 / "exp11_audit" / "audit.json",
            EXP10 / "roster.json")},
    }


def trajectory():
    config = read_json(EXP10 / "config.json")
    history = read_json(RUN10 / "validation_history.json")
    for index, row in enumerate(history):
        scores = {w: arm["artist_macro"] for w, arm in row["validation"].items()}
        if decision(history[:index], row["step"], scores, config["selection"]) != row["selection"]:
            raise ValueError(f"Archived qualification disagrees with policy at update {row['step']}")
    misses = {w: Counter(k for row in history for k in row["selection"]["candidates"][w]["failed_guardrails"])
              for w in ("raw", "ema")}
    with (RUN10 / "steps.jsonl").open(encoding="utf-8") as stream:
        log = [json.loads(line) for line in stream]
    with (RUN09 / "steps.jsonl").open(encoding="utf-8") as stream:
        prior = [json.loads(line) for line in stream]
    tracked = dict.fromkeys((*COMPONENTS, "crops", "shard"), 0)
    common_keys = tracked.keys() & log[0].keys() & prior[0].keys()
    for a, b in zip(log, prior, strict=True):
        for k in common_keys:
            tracked[k] += a[k] != b[k]
    def quantify(values):
        return dict(zip(("p05", "p50", "p95"), (float(q) for q in np.quantile(values, (.05, .5, .95)))))
    components = {}
    for name in COMPONENTS:
        vals = [row[name] for row in log]
        weight = config["exp09_settings"]["loss"].get(name)
        components[name] = {"coefficient": weight, "all_updates": quantify(vals),
                            "last_1000_updates": quantify(vals[-1000:]),
                            "weighted_all_updates": quantify(np.array(vals) * weight) if weight is not None else None}
    by_name = {k: quantify([a["validation"][k]["seconds"] for a in history]) for k in ("raw", "ema")}
    exp09_eval = read_json(EXP09 / "dev_eval_20261114" / "results.json")["aggregate"]["exp09"]
    return {
        "update_count": len(log), "validated_updates": [x["step"] for x in history],
        "loss_scales": components, "Exp09_Exp10_nonidentical_steps": {k: tracked[k] for k in sorted(common_keys)},
        "guardrail_limits": config["selection"]["guardrails"],
        "guardrail_misses": {w: dict(v) for w, v in misses.items()},
        "total_guardrail_misses": dict(misses["raw"] + misses["ema"]),
        "pilot_baseline": next(x for x in history if x["step"] == 6000),
        "exp09_development_aggregate": exp09_eval,
        "final_validation": history[-1],
        "completion": read_json(RUN10 / "completion.json"),
        "resources": {"update_seconds": quantify([row["seconds"] for row in log]),
                      "validation_seconds": by_name,
                      "peak_allocated_GiB": max(row["peak_allocated_gib"] for row in log),
                      "peak_reserved_GiB": max(row["peak_reserved_gib"] for row in log),
                      "training_loop_seconds": sum(row["seconds"] for row in log),
                      "step6000_training_loop_seconds": sum(row["seconds"] for row in log[:6000]),
                      "step6000_validation_seconds": sum(row["validation"][k]["seconds"]
                                                        for row in history[:6] for k in ("raw", "ema")),
                      "validation_seconds_sum": sum(row["validation"][k]["seconds"]
                                                    for row in history for k in ("raw", "ema"))},
    }


def preservation():
    source10 = read_json(RUN10 / "run.json")
    config = read_json(EXP10 / "config.json")
    inputs = [(EXP10 / name, digest) for name, digest in source10["files_sha256"].items()]
    raw_evidence = {**config["exp09_settings"]["data_sha256"], **source10["source_evidence_sha256"]}
    inputs += [(EXP09 / name, digest) for name, digest in raw_evidence.items()]
    inputs += [(EXP10 / "development_cache" / f"{uuid}.npy", row["sha256"])
               for uuid, row in source10["development_cache"].items()]
    aliases = read_json(RUN10 / "checkpoint_index.json")
    inputs += [(RUN10 / f"{name}.pt", aliases[name]["sha256"]) for name in ("latest", "best_overall")]
    for path, expected in inputs:
        if file_hash(path) != expected:
            raise ValueError(f"Baseline differs from its frozen SHA-256: {path}")
    return {
        "checks_against_frozen_provenance": {str(path): expected for path, expected in inputs},
        "exp10_run_artifact_sha256": {name: file_hash(RUN10 / name) for name in
                                    ("run.json", "completion.json", "checkpoint_index.json",
                                     "steps.jsonl", "validation_history.json")},
        "exp09_step_log_sha256": file_hash(RUN09 / "steps.jsonl"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    write_json(args.output_dir / "moisesdb_eligibility_manifest.json", manifest())
    write_json(args.output_dir / "baseline_audit.json", trajectory())
    write_json(args.output_dir / "baseline_preservation.json", preservation())


if __name__ == "__main__":
    main()
