"""Reproduce the Exp11 training-candidate audit without opening held-out audio."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile


SEALED_ROLES = {"test", "validation", "held_out_artist_reserve"}
RATE = 44100
TAIL_LIMIT_PCM24 = 84
NEEDED_RECORDS = (
    "manifest.json",
    "split_external_final.json",
    "split_internal_20_24.json",
    "training_eligibility.json",
    "training_shards.json",
    "eligibility_complete.json",
    "tail_recovery.json",
    "dev_eval_20261114/frozen_panel.json",
    "integrity.json",
    "launch_decision.json",
    "eligibility.py",
    "audit.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows):
    result = {r["uuid"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Repeated recording UUID")
    return result


def source_indexes(ids, songs, decoded, dev):
    archive_marks, pcm_hashes = defaultdict(set), defaultdict(set)
    for key in ids:
        for track in songs[key]["tracks"]:
            archive_marks[track["bytes"], track["crc32"]].add(key)
        for track in decoded.get(key, dev.get(key, {})).get("tracks", []):
            pcm_hashes[track["pcm_sha256"]].add(key)
    return archive_marks, pcm_hashes


def dispositions(records, songs, strict, active, alternative, dev):
    decoded = records["eligibility_complete.json"]["rows"]
    tail_checks = keyed(records["tail_recovery.json"]["songs"])
    known_recovery = set(
        records["training_eligibility.json"]["summary"]["known_silent_tail_candidates_without_other_flags"]
    )
    guard = set(strict) | {
        key for key, row in active.items() if row["role"] in SEALED_ROLES
    }
    archive_marks, pcm_hashes = source_indexes(guard, songs, decoded, dev)
    musdb = records["integrity.json"]["musdb_roster"]
    from audit import artist_key, norm

    filter_artists = {songs[key]["artist_key"] for key in guard - strict}
    for credit in musdb:
        filter_artists.add(artist_key(credit["artist"]))
    remaining = []
    for key, row in sorted(decoded.items()):
        if key in strict:
            continue
        song = songs[key]
        marks = {k for t in song["tracks"] for k in archive_marks[t["bytes"], t["crc32"]]}
        hashes = {k for t in row["tracks"] for k in pcm_hashes[t["pcm_sha256"]]}
        title_matches = sorted(k for k in guard if norm(song["title"]) == norm(songs[k]["title"]))
        flags = sorted(set(song["flags"]))
        bleed = Counter(str(t["bleed"]).lower() for t in row["tracks"])
        status = row["status"]
        candidate = {
            "uuid": key, "artist": song["artist"], "artist_key": song["artist_key"],
            "title": song["title"], "genre": song["genre"],
            "active_role": active[key]["role"], "unused_internal_role": alternative[key]["role"],
            "prior_audio_status": status, "prior_full_pcm_tracks": len(row["tracks"]),
            "shared_frames": row["frames_min"], "longest_frames": row["frames_max"],
            "bleed_by_track": dict(bleed),
            "rail_sample_count": sum(t["rail_samples"] for t in row["tracks"]),
            "discarded_samples_at_or_above_84_pcm24": sum(
                t["tail_active_samples"] for t in row["tracks"]
            ),
            "bwf_starts": sorted({t["bwf_start"] for t in row["tracks"] if t["bwf_start"] is not None}),
            "missing_bwf": row["missing_bwf"],
            "manifest_flags": flags,
            "exact_pcm_matches_in_training_or_development": sorted(hashes),
            "zip_size_crc_matches_in_training_or_active_sealed_roles": sorted(marks),
            "normalized_title_matches_in_training_or_active_sealed_roles": title_matches,
            "musdb_title_matches": song["musdb_title_matches"],
            "artist_matches_in_active_sealed_roles_or_musdb": song["artist_key"] in filter_artists,
        }
        if status.startswith("exclude_"):
            candidate["disposition"] = "exclude"
        elif key in known_recovery:
            validated = tail_checks[key]
            if (status != "recover_silent_tail" or candidate["missing_bwf"]
                    or len(candidate["bwf_starts"]) != 1 or candidate["rail_sample_count"]
                    or candidate["discarded_samples_at_or_above_84_pcm24"]
                    or set(bleed) != {"false"} or flags
                    or min(row["rms"].values()) < 1e-5
                    or marks or hashes or title_matches or song["musdb_title_matches"]
                    or candidate["artist_matches_in_active_sealed_roles_or_musdb"]
                    or validated["length_policy"] != "common_span"
                    or validated["frames"] != row["frames_min"]
                    or validated["reconstruction_max_abs"] > 1e-6):
                raise ValueError(f"Prior recovery evidence disagrees for {key}")
            candidate["disposition"] = "technical_common_span_pass"
            candidate["common_span_reconstruction_max_abs"] = validated["reconstruction_max_abs"]
            candidate["common_span_trim_ms"] = 1000 * (row["frames_max"] - row["frames_min"]) / RATE
        else:
            candidate["disposition"] = "manual_review"
        remaining.append(candidate)
    if {r["uuid"] for r in remaining if r["disposition"] == "technical_common_span_pass"} != known_recovery:
        raise ValueError("Recovery cohort differs from the decoded audit")
    return remaining


def live_check(archive_path, rows, songs, decoded, archive_bytes):
    from eligibility import scan_song, classify

    if archive_path.stat().st_size != archive_bytes:
        raise ValueError("Unexpected archive size")
    live = []
    directory_tracks = 0
    with ZipFile(archive_path) as archive:
        for row in rows:
            key = row["uuid"]
            for track in songs[key]["tracks"]:
                member = archive.getinfo(track["wav"])
                if (member.file_size, f"{member.CRC:08x}") != (track["bytes"], track["crc32"]):
                    raise ValueError(f"ZIP member changed: {track['wav']}")
                directory_tracks += 1
            if (row["disposition"] != "technical_common_span_pass"
                    or row["unused_internal_role"] in SEALED_ROLES):
                continue
            before = perf_counter()
            audio, _ = scan_song(archive, songs[key])
            expected = decoded[key]
            for field in audio:
                if field == "rms":
                    if any(abs(v - expected[field][k]) > 1e-12 for k, v in audio[field].items()):
                        raise ValueError(f"RMS differs from prior evidence for {key}")
                elif audio[field] != expected[field]:
                    raise ValueError(f"{field} differs from prior evidence for {key}")
            if classify(songs[key], audio) != "recover_silent_tail":
                raise ValueError(f"Candidate fails common-span QC: {key}")
            live.append({
                "uuid": key, "tracks": len(audio["tracks"]),
                "evidence": "member CRC, PCM SHA-256, BWF start, frame counts, tail, rails and source RMS",
                "elapsed_seconds": perf_counter() - before,
                "compressed_wav_bytes": songs[key]["compressed_wav_bytes"],
            })
    return live, directory_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp09", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hashes = {name: sha256(args.exp09 / name) for name in NEEDED_RECORDS}
    records = {name: json.loads((args.exp09 / name).read_text(encoding="utf-8"))
               for name in NEEDED_RECORDS if name.endswith(".json")}
    decision = records["launch_decision.json"]
    if any(hashes[n] != digest for n, digest in decision["data_sha256"].items()):
        raise ValueError("Exp09 bound source evidence failed SHA-256")
    sys.path.insert(0, str(args.exp09.resolve()))
    sys.dont_write_bytecode = True

    songs = keyed(records["manifest.json"])
    active_split = records["split_external_final.json"]
    alternative_split = records["split_internal_20_24.json"]
    if not active_split["external_final_required"] or active_split["test_artist_groups"]:
        raise ValueError("Expected frozen external-final protocol")
    active = keyed(active_split["assignments"])
    alternative = keyed(alternative_split["assignments"])
    strict = set(records["training_eligibility.json"]["strict_train"])
    decoded = records["eligibility_complete.json"]
    dev = records["dev_eval_20261114/frozen_panel.json"]["songs"]
    reserved = {k for k, r in active.items() if r["role"] in SEALED_ROLES}
    if (set(songs) != set(active) or set(songs) != set(alternative) or len(songs) != 240
            or len(strict) != 60 or strict & reserved
            or set(dev) != {k for k, r in active.items() if r["role"] == "validation"}
            or set(decoded["rows"]) != set(songs) - reserved
            or decoded["status"] != "complete"):
        raise ValueError("Frozen rosters or decoded audit evidence disagree")
    rows = dispositions(records, songs, strict, active, alternative, dev)
    recoveries = [r for r in rows if r["disposition"] == "technical_common_span_pass"]
    rescans = []
    directory_tracks = 0
    if args.archive:
        rescans, directory_tracks = live_check(args.archive, rows, songs, decoded["rows"],
                                               records["integrity.json"]["zip_bytes"])
    if hashes != {name: sha256(args.exp09 / name) for name in NEEDED_RECORDS}:
        raise ValueError("Source evidence changed during audit")
    summary = {
        "active_split": "external-final",
        "archive_songs": len(songs), "existing_strict_songs": len(strict),
        "current_development_songs": len(dev),
        "current_development_artist_reserve_songs": len(reserved - dev.keys()),
        "remaining_training_artist_songs": len(rows),
        "additional_strict_eligible": sum(r["prior_audio_status"] == "eligible_strict" for r in rows),
        "additional_identity_cleared_for_independent_final": 0,
        "status_counts": dict(Counter(r["prior_audio_status"] for r in rows)),
        "dispositions": dict(Counter(r["disposition"] for r in rows)),
        "technical_candidate_hours": sum(r["shared_frames"] for r in recoveries) / RATE / 3600,
        "technical_candidate_artist_groups": len({r["artist_key"] for r in recoveries}),
        "new_artist_groups_if_admitted": sorted({r["artist_key"] for r in recoveries}
                                               - {songs[k]["artist_key"] for k in strict}),
        "technical_candidates_reserved_in_unused_alternative": sum(
            r["unused_internal_role"] in SEALED_ROLES for r in recoveries
        ),
        "discarded_tail_peak_bound_pcm24_exclusive": TAIL_LIMIT_PCM24,
        "wav_members_checked_in_zip_directory": directory_tracks,
        "new_pcm_checks": rescans,
    }
    payload = {"summary": summary, "source_sha256": hashes, "remaining": rows}
    if args.output.resolve().parent != Path(__file__).resolve().parent:
        raise ValueError("Write audit output inside the Exp11 audit directory")
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
