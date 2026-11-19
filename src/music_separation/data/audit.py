"""Metadata-only inventory, recording-risk audit and proposed splits."""
import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from zipfile import ZipFile

PREFIX = "moisesdb/moisesdb_v0.1/"
META = re.compile(PREFIX.replace(".", r"\.") + r"([0-9a-f-]{36})/data\.json")
PUBLISHED_SHA256 = "4cde33ce416ac7c868cffcb60eb31f5c741ab7ae5601cbb9d99ed498b72c48c1"
KNOWN_STEMS = {"bass", "guitar", "vocals", "drums", "other_keys", "other", "piano",
               "percussion", "bowed_strings", "wind", "other_plucked"}
VOCAL_TYPES = {"lead_male_singer", "lead_female_singer", "background_vocals",
               "other_(vocoder,_beatboxing_etc)"}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value.casefold()) if c.isalnum())


def artist_key(value):
    return norm(re.split(r"\s+(?:ft|feat|featuring)\.?\s+", value.casefold(), maxsplit=1)[0])


def wav_signatures(tracks):
    return tuple(sorted((t["bytes"], t["crc32"]) for t in tracks if t["available"]))


def read_musdb(root):
    roster = []
    for split, expected in (("train", 100), ("test", 50)):
        files = sorted((root / split).glob("*.stem.mp4"))
        if len(files) != expected:
            raise ValueError(f"MUSDB18 {split}: {len(files)} files, expected {expected}")
        for path in files:
            artist, title = path.name.removesuffix(".stem.mp4").split(" - ", 1)
            roster.append({"split": split, "artist": artist, "title": title})
    return roster


def inspect_archive(path, musdb):
    wav_matches, recordings, title_matches = defaultdict(list), defaultdict(list), defaultdict(list)
    songs = []
    with ZipFile(path) as archive:
        infos = archive.infolist()
        by_name = {item.filename: item for item in infos}
        if len(infos) != len(by_name):
            raise ValueError("Duplicate ZIP member paths")
        references = Counter()
        for name, info in sorted(by_name.items()):
            match = META.fullmatch(name)
            if not match:
                continue
            raw = archive.read(info)
            data = json.loads(raw)
            prefix = name.removesuffix("data.json")
            tracks = []
            for stem in data["stems"]:
                for track in stem["tracks"]:
                    category = stem["stemName"]
                    member = f'{prefix}{category}/{track["id"]}.{track["extension"]}'
                    wav = by_name.get(member)
                    references[member] += 1
                    row = {"stem": category, "stem_id": stem["id"], "track_id": track["id"],
                           "track_type": track["trackType"], "mime": track["type"],
                           "extension": track["extension"], "has_bleed": track["has_bleed"],
                           "wav": member, "available": wav is not None}
                    if wav:
                        row.update(bytes=wav.file_size, compressed_bytes=wav.compress_size,
                                   crc32=f"{wav.CRC:08x}")
                        wav_matches[(wav.file_size, wav.CRC)].append(member)
                    tracks.append(row)
            vocal = [t["wav"] for t in tracks if t["stem"] == "vocals"]
            backing = [t["wav"] for t in tracks if t["stem"] != "vocals"]
            flags = []
            for condition, flag in (
                (not vocal, "missing_vocals"),
                (not backing, "missing_instrumental"),
                (any(not t["available"] for t in tracks), "missing_wav"),
                (any(t["has_bleed"] is True for t in tracks), "reported_bleed"),
                (any(t["has_bleed"] is None for t in tracks), "unknown_bleed"),
                (any(t["stem"] not in KNOWN_STEMS or
                     (t["stem"] == "vocals") != (t["track_type"] in VOCAL_TYPES)
                     or t["extension"] != "wav" or t["mime"] != "audio/x-wav"
                     or (t["has_bleed"] is not None and type(t["has_bleed"]) is not bool)
                     for t in tracks), "unexpected_taxonomy_or_schema"),
                (any(t["track_type"] == "other_(vocoder,_beatboxing_etc)" for t in tracks),
                 "unusual_vocal_label"),
                (len(wav_signatures(tracks)) != len(set(wav_signatures(tracks))),
                 "repeated_wav_signature_within_song"),
            ):
                if condition:
                    flags.append(flag)
            notes = []
            if (any(t["track_type"] == "full_acoustic_drumkit" for t in tracks)
                    and sum(t["stem"] == "drums" for t in tracks) > 1):
                notes.append("full_drumkit_plus_other_drum_tracks")
            if len(data["stems"]) != len({s["stemName"] for s in data["stems"]}):
                notes.append("repeated_stem_category_blocks")
            uuid = match[1]
            artist = data["artist"]
            title = data["song"]
            mus_matches = [m for m in musdb if artist_key(m["artist"]) == artist_key(artist)]
            if mus_matches:
                flags.append("musdb_artist_match")
            mus_title_matches = [m for m in musdb if norm(m["title"]) == norm(title)]
            if any(m in mus_matches for m in mus_title_matches):
                flags.append("musdb_artist_title_match")
            track_sizes = {t["bytes"] for t in tracks if t["available"]}
            row = {"uuid": uuid, "artist": artist, "artist_key": artist_key(artist),
                   "title": title, "genre": data["genre"], "metadata": name,
                   "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                   "stem_categories": sorted({s["stemName"] for s in data["stems"]}),
                   "vocal_wavs": vocal, "instrumental_wavs": backing, "tracks": tracks,
                   "bleed": dict(Counter(str(t["has_bleed"]).lower() for t in tracks)),
                   "flags": flags, "notes": notes, "musdb_artist_matches": mus_matches,
                   "musdb_title_matches": mus_title_matches, "distinct_wav_sizes": len(track_sizes),
                   "archive_pcm_bytes_estimate_basis": max(track_sizes, default=0),
                   "raw_wav_bytes": sum(t.get("bytes", 0) for t in tracks),
                   "compressed_wav_bytes": sum(t.get("compressed_bytes", 0) for t in tracks)}
            songs.append(row)
            title_matches[(row["artist_key"], norm(title))].append(uuid)
            if tracks and all(t["available"] for t in tracks):
                recordings[wav_signatures(tracks)].append(uuid)

        duplicate_recordings = [v for v in recordings.values() if len(v) > 1]
        duplicate_titles = [v for v in title_matches.values() if len(v) > 1]
        duplicate_stems = [{"bytes": k[0], "crc32": f"{k[1]:08x}", "members": v}
                           for k, v in wav_matches.items() if len(v) > 1]
        cross_recording = {member.split("/")[2] for group in duplicate_stems
                           if len({m.split("/")[2] for m in group["members"]}) > 1
                           for member in group["members"]}
        for song in songs:
            uuid = song["uuid"]
            if uuid in cross_recording:
                song["flags"].append("shared_wav_across_recordings")
            if any(uuid in group for group in duplicate_recordings):
                song["flags"].append("duplicate_recording_candidate")
            if any(uuid in group for group in duplicate_titles):
                song["flags"].append("artist_title_duplicate_candidate")

        wav_names = {n for n in by_name if n.lower().endswith(".wav")}
        summary = {
            "zip": str(path.resolve()), "published_sha256_user_verified": PUBLISHED_SHA256,
            "zip_bytes": path.stat().st_size, "entries": len(infos),
            "uncompressed_bytes": sum(i.file_size for i in infos),
            "metadata_count": len(songs), "wav_entries": len(wav_names),
            "wav_references": sum(references.values()), "distinct_references": len(references),
            "missing_references": sorted(set(references) - wav_names),
            "unreferenced_wavs": sorted(wav_names - set(references)),
            "repeated_reference_paths": sorted(k for k, v in references.items() if v > 1),
            "duplicate_recording_candidates": duplicate_recordings,
            "duplicate_normalized_artist_title_groups": duplicate_titles,
            "duplicate_wav_signature_groups": duplicate_stems,
            "raw_artist_labels": len({s["artist"] for s in songs}),
            "artist_groups": len({s["artist_key"] for s in songs}),
            "genres": dict(Counter(s["genre"] for s in songs)),
            "stem_track_counts": dict(Counter(t["stem"] for s in songs for t in s["tracks"])),
            "track_bleed_counts": dict(Counter(str(t["has_bleed"]).lower()
                                               for s in songs for t in s["tracks"])),
            "song_flag_counts": dict(Counter(f for s in songs for f in s["flags"])),
            "song_note_counts": dict(Counter(f for s in songs for f in s["notes"])),
            "songs_with_varied_wav_file_sizes": sum(s["distinct_wav_sizes"] > 1 for s in songs),
            "musdb_roster": musdb,
        }
    moises_artists = {s["artist_key"]: s["artist"] for s in songs}
    mus_artists = {artist_key(m["artist"]): m["artist"] for m in musdb}
    summary["musdb_normalized_artist_count"] = len(mus_artists)
    summary["musdb_fuzzy_artist_candidates"] = [
        {"moises": moises_artists[a], "musdb": mus_artists[b], "similarity": round(ratio, 4)}
        for a in sorted(moises_artists) for b in sorted(mus_artists) if a != b
        if (ratio := SequenceMatcher(None, a, b).ratio()) >= 0.80
    ]
    return songs, summary


HARD_FLAGS = {"missing_vocals", "missing_instrumental", "missing_wav",
              "unexpected_taxonomy_or_schema", "repeated_wav_signature_within_song",
              "shared_wav_across_recordings", "artist_title_duplicate_candidate",
              "musdb_artist_match", "musdb_artist_title_match"}


def panel_rows(groups, clean, count, seed):
    """Take one recording per artist first, then use extra slots for genre breadth."""
    order = sorted(groups, key=lambda key: (len(clean[key]), key))
    panel, seen_genres, seen_artists = [], Counter(), Counter()
    chosen = set()
    rank = lambda row: hashlib.sha256(f'{seed}:{row["uuid"]}'.encode()).digest()

    def choose(rows):
        return min(rows, key=lambda row: (seen_genres[row["genre"]], seen_artists[row["artist_key"]], rank(row)))

    def add(row):
        panel.append(row)
        chosen.add(row["uuid"])
        seen_genres[row["genre"]] += 1
        seen_artists[row["artist_key"]] += 1

    for group in order:
        add(choose(clean[group]))
    while len(panel) < count:
        add(choose([row for group in groups for row in clean[group] if row["uuid"] not in chosen and seen_artists[group] < 8]))
    return panel


def proposed_split(songs, internal, seed=909):
    groups, clean = defaultdict(list), defaultdict(list)
    for song in songs:
        groups[song["artist_key"]].append(song)
        if not song["flags"]:
            clean[song["artist_key"]].append(song)
    all_eligible_genres = {s["genre"] for s in songs if not (HARD_FLAGS & set(s["flags"]))}
    active = sorted(clean)
    rng = random.Random(seed)
    group_hours_basis = {k: sum(s["archive_pcm_bytes_estimate_basis"] for s in v)
                         for k, v in groups.items()}
    group_genres = {k: {s["genre"] for s in v if not HARD_FLAGS & set(s["flags"])}
                    for k, v in groups.items()}
    clean_genres = {k: {s["genre"] for s in v} for k, v in clean.items()}
    candidates = []
    for _ in range(22000):
        test = set(rng.sample(active, 20)) if internal else set()
        if internal and sum(len(clean[k]) for k in test) < 24:
            continue
        val = set(rng.sample(sorted(set(active) - test), 8))
        if sum(min(8, len(clean[k])) for k in val) < 20:
            continue
        held = val | test
        train = set(groups) - held
        lost = all_eligible_genres - set().union(*(group_genres[k] for k in train))
        val_genres = set().union(*(clean_genres[k] for k in val))
        if len(val_genres) < 5:
            continue
        test_genres = set().union(*(clean_genres[k] for k in test))
        # Penalty units are equivalent stereo PCM24 seconds at 44.1 kHz.
        score = (sum(group_hours_basis[k] for k in held) / 264600
                 + 5000 * len(lost) - 500 * (len(val_genres) + len(test_genres)))
        candidates.append((score, tuple(sorted(val)), tuple(sorted(test))))
    if not candidates:
        raise ValueError("No feasible artist-group proposal")
    _, val, test = min(candidates)
    val_rows = panel_rows(val, clean, 20, seed)
    test_rows = panel_rows(test, clean, 24, seed + 1) if test else []
    panel = {s["uuid"]: label for label, rows in (("validation", val_rows), ("test", test_rows))
             for s in rows}
    assignments = []
    for song in songs:
        uuid, key = song["uuid"], song["artist_key"]
        if uuid in panel:
            role = panel[uuid]
        elif key in set(val) | set(test):
            role = "held_out_artist_reserve"
        elif HARD_FLAGS & set(song["flags"]):
            role = "quarantine"
        elif song["flags"]:
            role = "train_review"
        else:
            role = "train_clean"
        assignments.append({"uuid": uuid, "role": role, "artist": song["artist"],
                            "artist_key": key, "title": song["title"], "genre": song["genre"],
                            "flags": song["flags"]})
    by_id = {s["uuid"]: s for s in songs}
    metrics = {}
    for role in sorted({a["role"] for a in assignments}):
        rows = [by_id[a["uuid"]] for a in assignments if a["role"] == role]
        metrics[role] = {
            "songs": len(rows), "artists": len({s["artist_key"] for s in rows}),
            "genres": dict(Counter(s["genre"] for s in rows)),
            "hours_proxy": round(sum(s["archive_pcm_bytes_estimate_basis"] for s in rows)
                                 / (264600 * 3600), 4),
            "raw_wav_bytes": sum(s["raw_wav_bytes"] for s in rows),
        }
    return {"seed": seed, "external_final_required": not internal, "metrics": metrics,
            "validation_artist_groups": list(val), "test_artist_groups": list(test),
            "assignments": assignments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("musdb", type=Path, help="Local MUSDB18 root containing train and test")
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    songs, summary = inspect_archive(args.archive, read_musdb(args.musdb))
    if len(songs) != 240:
        raise ValueError(f"Expected 240 metadata files, got {len(songs)}")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "manifest.json", songs)
    write_json(args.out / "integrity.json", summary)
    write_json(args.out / "split_external_final.json", proposed_split(songs, internal=False))
    write_json(args.out / "split_internal_20_24.json", proposed_split(songs, internal=True))
    with (args.out / "songs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        columns = ("uuid", "artist", "artist_key", "title", "genre", "stem_categories",
                   "vocal_wavs", "instrumental_wavs", "bleed", "flags", "notes", "raw_wav_bytes",
                   "distinct_wav_sizes", "archive_pcm_bytes_estimate_basis")
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for song in songs:
            writer.writerow({k: json.dumps(song[k], ensure_ascii=False)
                             if isinstance(song[k], (dict, list)) else song[k] for k in columns})
    print(f"{len(songs)} metadata files, {summary['wav_references']} WAV references, "
          f"{len(summary['missing_references'])} missing")


if __name__ == "__main__":
    main()
