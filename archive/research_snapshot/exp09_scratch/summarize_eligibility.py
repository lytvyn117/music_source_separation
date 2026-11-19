"""Build an auditable, conservative training allowlist from the sealed-split scan."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from audit import write_json

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def by_hash(row):
    grouped = defaultdict(list)
    for track in row["tracks"]:
        grouped[track["pcm_sha256"]].append(track["path"])
    return grouped


def main():
    audit = load("eligibility_complete.json")
    if audit["status"] != "complete" or audit.get("timestamp_scan") != "before and after PCM data":
        raise ValueError("Finish the training-artist scan before finalizing eligibility")
    manifest = {s["uuid"]: s for s in load("manifest.json")}
    split_path = ROOT / "split_external_final.json"
    split = load(split_path.name)
    assignments = {r["uuid"]: r for r in split["assignments"]}
    rows = audit["rows"]
    sealed = {k for k, r in assignments.items() if r["role"] in
              {"validation", "test", "held_out_artist_reserve"}}
    if set(rows) != set(assignments) - sealed:
        raise ValueError("Training scan and proposed artist groups disagree")
    recordings = defaultdict(list)
    for key, row in rows.items():
        if "tracks" in row:
            recordings[tuple(sorted(t["pcm_sha256"] for t in row["tracks"]))].append(key)
    duplicates = [group for group in recordings.values() if len(group) > 1]
    shared_pcm = defaultdict(set)
    for key, row in rows.items():
        for track in row.get("tracks", []):
            shared_pcm[track["pcm_sha256"]].add(key)
    shared_pairs = Counter()
    for keys in shared_pcm.values():
        for pair in itertools.combinations(sorted(keys), 2):
            shared_pairs[pair] += 1
    for group in duplicates:
        for key in group:
            rows[key]["status"] = "exclude_duplicate_recording"
    suspected = [p for p in audit["acoustic_pairs"]
                 if sum(hit["correlation"] >= 0.70 for hit in p["measurements"]) >= 2]
    for pair in suspected:
        for key in (pair["a"], pair["b"]):
            if rows[key]["status"] == "eligible_strict":
                rows[key]["status"] = "review_acoustic_duplicate"
    clean = [row for row in rows.values() if row["status"] == "eligible_strict"]
    status_counts = Counter(row["status"] for row in rows.values())
    hours = defaultdict(float)
    for row in rows.values():
        hours[row["status"]] += row.get("frames_min", 0) / 44100 / 3600
    dimensions = {
        "train_recordings": len(rows),
        "sealed_recordings": len(sealed),
        "fully_decoded_tracks": sum(len(r.get("tracks", [])) for r in rows.values()),
        "counts": dict(status_counts), "shared_span_hours": dict(hours),
        "original_role_to_status": {role: dict(Counter(r["status"] for r in rows.values()
                                                        if r["role_before"] == role))
                                    for role in {r["role_before"] for r in rows.values()}},
        "source_rail_songs": sum(any(t["rail_samples"] for t in r.get("tracks", []))
                               for r in rows.values()),
        "source_rail_tracks": sum(t["rail_samples"] > 0 for r in rows.values()
                                 for t in r.get("tracks", [])),
        "mixture_over_songs": sum(r.get("overs", {}).get("mixture", 0) > 0
                                 for r in rows.values()),
        "strict_mixture_over_songs": sum(r["overs"]["mixture"] > 0 for r in clean),
        "unequal_length_songs": sum("frames_min" in r and r["frames_min"] != r["frames_max"]
                                    for r in rows.values()),
        "equal_length_without_bwf": sum("frames_min" in r and r["frames_min"] == r["frames_max"]
                                        and r["missing_bwf"] for r in rows.values()),
        "no_reported_or_unknown_bleed": sum(all(t["bleed"] is False for t in r["tracks"])
                                          for r in rows.values() if "tracks" in r),
        "bleed_rows": dict(Counter(
            "true_and_unknown" if any(t["has_bleed"] is True for t in row["tracks"])
                                 and any(t["has_bleed"] is None for t in row["tracks"])
            else "true" if any(t["has_bleed"] is True for t in row["tracks"])
            else "unknown" if any(t["has_bleed"] is None for t in row["tracks"])
            else "all_false"
            for row in manifest.values() if row["uuid"] in rows
        )),
        "reported_bleed_by_target": dict(Counter(
            ("both" if vocal and backing else "vocals" if vocal else "instrumental")
            for row in manifest.values() if row["uuid"] in rows
            if (vocal := any(t["has_bleed"] is True and t["stem"] == "vocals" for t in row["tracks"]))
            | (backing := any(t["has_bleed"] is True and t["stem"] != "vocals" for t in row["tracks"]))
        )),
        "unequal_length_evidence": dict(Counter(
            ("missing_timestamp" if row["missing_bwf"] else "timestamp_available")
            + ("_active_tail" if any(t["tail_active_samples"] for t in row["tracks"])
               else "_below_threshold_tail")
            for row in rows.values() if "tracks" in row and row["frames_min"] != row["frames_max"]
        )),
        "repeated_pcm_stem_pairs": {
            key: [paths for h, paths in by_hash(row).items() if len(paths) > 1]
            for key, row in rows.items() if row["status"] == "exclude_duplicate_stem"
        },
        "training_artists_strict": len({assignments[r["uuid"]]["artist_key"] for r in clean}),
        "artists_strict_by_song_count": dict(Counter(assignments[r["uuid"]]["artist_key"]
                                                   for r in clean)),
        "known_silent_tail_candidates_without_other_flags": [
            row["uuid"] for row in rows.values()
            if row["status"] in {"recover_silent_tail", "review_length_without_timestamp"}
            and not any(t["tail_active_samples"] or t["rail_samples"] for t in row["tracks"])
            and all(t["bleed"] is False for t in row["tracks"])
            and not manifest[row["uuid"]]["flags"]
        ],
        "training_genres_strict": dict(Counter(manifest[r["uuid"]]["genre"] for r in clean)),
        "identical_recordings": duplicates, "acoustic_review_pairs": suspected,
        "shared_pcm_across_songs": [{"a": a, "b": b, "matches": count}
                                    for (a, b), count in shared_pairs.items()],
        "pcm16_recordings": sum(any(t["pcm_bits"] == 16 for t in row.get("tracks", []))
                                for row in rows.values()),
        "bit_depth_track_counts": dict(Counter(t["pcm_bits"] for row in rows.values()
                                              for t in row.get("tracks", []))),
        "decode_seconds": audit["seconds"],
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    write_json(ROOT / "training_eligibility.json", {
        "summary": dimensions,
        "strict_train": sorted(r["uuid"] for r in clean),
        "rows": {key: {k: v for k, v in row.items() if k != "tracks"} for key, row in rows.items()},
    })
    print(json.dumps(dimensions, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
