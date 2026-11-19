"""Audit the historical MUSDB18 fit roster while keeping evaluation audio sealed."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from audit import artist_key, norm, write_json
from eligibility import FP_RATE, RATE, fingerprint_candidates, pcm24, signature, wav_header, wave_match

ROOT = Path(__file__).resolve().parent
HOP = RATE * 8
VOCAL_ERRATA = {
    norm(x) for x in ("Al James - Schoolboy Facination", "Chris Durban - Celebrate",
                      "Enda Reilly - Cur An Long Ag Seol", "Hop Along - Sister Cities")
}


def moises_mix(archive, row, frames):
    proxy = np.zeros((frames + 19) // 20, dtype=np.float32)
    with ZipFile(archive) as z:
        for track in sorted(row["tracks"], key=lambda t: z.getinfo(t["wav"]).header_offset):
            with z.open(track["wav"]) as stream:
                h = wav_header(stream)
                if frames != h["frames"]:
                    raise ValueError("Strict source lengths changed")
                for start in range(0, frames, HOP):
                    n = min(HOP, frames - start)
                    raw = stream.read(n * h["stride"])
                    if len(raw) != n * h["stride"]:
                        raise ValueError("Short WAV read")
                    if h["bits"] == 24:
                        signal = pcm24(raw).mean(axis=1, dtype=np.float32) / (1 << 23)
                    else:
                        signal = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
                        signal = signal.mean(axis=1, dtype=np.float32) / (1 << 15)
                    piece = resample_poly(signal, 1, RATE // FP_RATE)
                    proxy[start // 20:start // 20 + len(piece)] += piece
    return proxy


def musdb_mix(path):
    hashes = [hashlib.sha256(), hashlib.sha256()]
    with sf.SoundFile(path / "vocals.wav") as v, sf.SoundFile(path / "instrumental.wav") as i:
        frames = len(v)
        if any((f.samplerate, f.channels, len(f)) != (RATE, 2, frames) for f in (v, i)):
            raise ValueError(f"MUSDB target format differs: {path.name}")
        peaks = np.zeros(3)
        energies = np.zeros(2)
        clips = 0
        parts = []
        for vb in v.blocks(blocksize=HOP, dtype="float32", always_2d=True):
            ib = i.read(len(vb), dtype="float32", always_2d=True)
            mix = vb + ib
            if not np.isfinite(mix).all():
                raise ValueError(f"Nonfinite MUSDB source: {path.name}")
            peaks = np.maximum(peaks, [np.abs(a).max() for a in (vb, ib, mix)])
            energies += [np.square(a, dtype=np.float64).sum() for a in (vb, ib)]
            clips += int(np.count_nonzero(np.abs(mix) > 1))
            for h, array in zip(hashes, (vb, ib)):
                h.update(array.tobytes())
            parts.append(resample_poly(mix.mean(axis=1, dtype=np.float32), 1, RATE // FP_RATE))
    return np.concatenate(parts), {"frames": frames, "peaks_v_i_mix": peaks.tolist(),
                                  "rms_v_i": np.sqrt(energies / (frames * 2)).tolist(),
                                  "mix_overs": clips, "decoded_sha256": [h.hexdigest() for h in hashes]}


def track_fields(name):
    artist, title = name.removesuffix(".stem.mp4").removesuffix(".stem").split(" - ", 1)
    return artist, title


def artist_and_title(name):
    a, t = track_fields(name)
    return artist_key(a), norm(t)


def evidence_for(pairs, audio):
    matches = []
    for a, b, reason in pairs:
        values = wave_match(audio[a], audio[b])
        high = [r for r in values if r["correlation"] >= .70]
        if len(high) >= 2 and max(r["lag_seconds"] for r in high) - min(r["lag_seconds"] for r in high) < .25:
            status = "likely_match"
        else:
            status = "no_strong_match"
        matches.append({"a": a, "b": b, "reason": reason, "status": status, "correlations": values})
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("original", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    read = lambda n: json.loads((ROOT / n).read_text(encoding="utf-8"))
    training = read("training_eligibility.json")
    moises = read("manifest.json")
    archive_rows = {r["uuid"]: r for r in moises}
    selected = {r["uuid"]: r for r in read("split_external_final.json")["assignments"]}
    sealed_artists = {r["artist_key"] for r in selected.values() if r["role"] in
                      {"validation", "test", "held_out_artist_reserve"}}
    processed = args.original / "data" / "processed" / "train"
    roster = sorted(p for p in processed.iterdir() if p.is_dir())
    if len(roster) != 100:
        raise ValueError("Expected 100 processed official training songs")
    if {p.name + ".mp4" for p in roster} != {p.name for p in
             (args.original / "data" / "raw" / "train").glob("*.stem.mp4")}:
        raise ValueError("Processed and raw MUSDB18 train rosters differ")
    random.Random(42).shuffle(roster)
    old_val = [p.name for p in roster[:15]]
    official = sorted(p.name for p in (args.original / "data" / "raw" / "test").glob("*.stem.mp4"))
    val_artists = {artist_and_title(n)[0] for n in old_val}
    final_artists = {artist_and_title(n)[0] for n in official}
    report, proxies, fingerprints = {}, {}, {}
    ids = training["strict_train"]
    comparisons = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        def audit_mus(path):
            return musdb_mix(path)

        for ix, (path, (proxy, row)) in enumerate(zip(roster[15:], pool.map(audit_mus, roster[15:])), 1):
            a, title = track_fields(path.name)
            key, title_key = artist_and_title(path.name)
            flags = []
            if key in sealed_artists:
                flags.append("moises_sealed_artist")
            if key in final_artists:
                flags.append("official_musdb_test_artist")
            if key in val_artists:
                flags.append("legacy_validation_artist")
            if norm(path.name.removesuffix(".stem")) in VOCAL_ERRATA:
                flags.append("published_target_erratum")
            if min(row["rms_v_i"]) < 1e-5:
                flags.append("missing_signal")
            for r in moises:
                if title_key == norm(r["title"]):
                    comparisons.append({"musdb": path.name, "moises": r["uuid"],
                                        "moises_artist": r["artist"], "moises_title": r["title"],
                                        "strict_train": r["uuid"] in ids})
            report[path.name] = {"artist": a, "title": title, "artist_key": key,
                                 "flags": flags, **row}
            proxies[path.name] = proxy
            fingerprints[path.name] = signature(proxy)
            if ix % 20 == 0:
                print(f"MUSDB signal audit {ix}/85", flush=True)
        for ix, (key, proxy) in enumerate(zip(ids, pool.map(
            lambda k: moises_mix(args.archive, archive_rows[k], training["rows"][k]["frames_min"]), ids)), 1):
            proxies[key] = proxy
            fingerprints[key] = signature(proxy)
            if ix % 10 == 0:
                print(f"Moises fingerprint audit {ix}/60", flush=True)
    pairs = {(a, b): f"{votes} aligned spectral hashes" for a, b, votes in
             fingerprint_candidates(fingerprints)}
    for r in comparisons:
        if r["strict_train"]:
            pairs[(r["musdb"], r["moises"])] = "matching title"
    matches = evidence_for([(a, b, reason) for (a, b), reason in pairs.items()], proxies)
    digest_checks = {}
    for mus, row in report.items():
        key = tuple(row["decoded_sha256"])
        if key in digest_checks:
            row["flags"].append("duplicate_decoded_targets")
        digest_checks[key] = mus
    for m in matches:
        if m["status"] == "likely_match":
            for k in (m["a"], m["b"]):
                if k in report:
                    report[k]["flags"].append("likely_acoustic_duplicate")
    write_json(ROOT / "musdb_eligibility.json", {
        "seconds": perf_counter() - start, "musdb_original_fit_count": len(report),
        "moises_strict_count": len(ids), "original_fit_order": [p.name for p in roster[15:]],
        "legacy_validation_metadata_only": old_val, "official_test_metadata_only": official,
        "moises_reserved_groups": sorted(sealed_artists),
        "moises_all_artist_exact_overlap": sorted({r["artist_key"] for r in moises} &
                                                  {r["artist_key"] for r in report.values()}),
        "moises_artist_similar_names": sorted((a, b) for a in {r["artist_key"] for r in report.values()}
                                              for b in {r["artist_key"] for r in moises}
                                              if a != b and SequenceMatcher(None, a, b).ratio() >= .80),
        "metadata_title_matches": comparisons, "decoded_candidate_pairs": matches,
        "flag_counts": dict(Counter(f for row in report.values() for f in row["flags"])),
        "rows": report,
    })
    print("MUSDB audit complete:", len(report), "recordings,", len(matches), "candidate pairs")


if __name__ == "__main__":
    main()
