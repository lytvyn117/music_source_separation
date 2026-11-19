"""Decode training-artist source audio in-place. Keep held-out audio sealed."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import struct
import subprocess
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.signal import correlate, resample_poly

from audit import write_json

ROOT = Path(__file__).resolve().parent
BLOCK = 262144
RATE = 44100
SCALE = 1 << 23
FP_RATE = 2205


def wav_header(stream):
    kind, _, wave = struct.unpack("<4sI4s", stream.read(12))
    if (kind, wave) != (b"RIFF", b"WAVE"):
        raise ValueError("Expected RIFF WAVE")
    fmt, start, offset = None, None, 12
    while offset < 1024**2:
        tag, size = struct.unpack("<4sI", stream.read(8))
        offset += 8
        if tag == b"data":
            if fmt is None:
                raise ValueError("Missing fmt chunk")
            codec, channels, rate, _, stride, bits = fmt
            if (codec not in (1, 65534) or channels != 2 or rate != RATE
                    or (stride, bits) not in {(4, 16), (6, 24)}):
                raise ValueError(f"Unsupported format {fmt}")
            if size == 0 or size % stride:
                raise ValueError("Invalid PCM data length")
            return {"frames": size // stride, "data_bytes": size, "start": start,
                    "offset": offset, "stride": stride, "bits": bits}
        if size > 1024**2:
            break
        padded = size + size % 2
        payload = stream.read(padded)
        if len(payload) != padded:
            raise ValueError("Truncated WAV chunk")
        if tag == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", payload)
            if fmt[0] == 65534 and (size < 40 or payload[24:40].hex() !=
                                    "0100000000001000800000aa00389b71"):
                raise ValueError("Unsupported WAVE extensible subformat")
        elif tag == b"bext" and size >= 346:
            start = struct.unpack_from("<Q", payload, 338)[0]
        offset += padded
    raise ValueError("No WAV data header within bound")


def trailing_bwf(stream):
    """Parse metadata following the audio chunk during the same CRC-checked read."""
    start = None
    while header := stream.read(8):
        if len(header) != 8:
            raise ValueError("Truncated trailing WAV header")
        tag, size = struct.unpack("<4sI", header)
        if size > 8 * 1024**2:
            raise ValueError("Trailing WAV metadata exceeds bound")
        payload = stream.read(size + size % 2)
        if len(payload) != size + size % 2:
            raise ValueError("Truncated trailing WAV metadata")
        if tag == b"bext" and size >= 346:
            start = struct.unpack_from("<Q", payload, 338)[0]
    return start


def pcm24(raw):
    bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2, 3)
    # View the top byte as signed for two's-complement sign extension.
    samples = bytes_[..., 2].view(np.int8).astype(np.int32) << 16
    samples |= bytes_[..., 1].astype(np.int32) << 8
    samples |= bytes_[..., 0]
    return samples


def scan_song(archive, song):
    tracks = sorted(song["tracks"], key=lambda t: archive.getinfo(t["wav"]).header_offset)
    if len(tracks) >= 128:
        raise ValueError("Too many sources for bounded 32-bit integer accumulation")
    headers = []
    for t in tracks:
        info = archive.getinfo(t["wav"])
        if info.file_size != t["bytes"] or f"{info.CRC:08x}" != t["crc32"]:
            raise ValueError("Archive member changed")
        with archive.open(info) as member:
            headers.append(wav_header(member))
    lengths = [h["frames"] for h in headers]
    shared = min(lengths)
    vocal = np.zeros((shared, 2), dtype=np.int32)
    backing = np.zeros_like(vocal)
    rows = []
    for t, h in zip(tracks, headers):
        sha = hashlib.sha256()
        peak = rails = tail_active = 0
        reference = vocal if t["stem"] == "vocals" else backing
        with archive.open(t["wav"]) as member:
            member.read(h["offset"])
            for begin in range(0, h["frames"], BLOCK):
                count = min(BLOCK, h["frames"] - begin)
                raw = member.read(count * h["stride"])
                if len(raw) != count * h["stride"]:
                    raise ValueError("Truncated source samples")
                rail_step = 1 << (24 - h["bits"])
                if h["bits"] == 24:
                    samples = pcm24(raw)
                    sha.update(raw)
                else:
                    encoded = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2)
                    packed = np.zeros((len(encoded), 3), dtype=np.uint8)
                    packed[:, 1:] = encoded
                    normalized = packed.tobytes()
                    samples = pcm24(normalized)
                    sha.update(normalized)
                peak = max(peak, int(samples.max()), -int(samples.min()))
                rails += int(np.count_nonzero((samples == SCALE - rail_step) | (samples == -SCALE)))
                n = max(0, min(count, shared - begin))
                if n:
                    reference[begin:begin + n] += samples[:n]
                if n < count:
                    tail = samples[n:]
                    tail_active += int(np.count_nonzero(np.abs(tail) >= 84))
            timestamp = trailing_bwf(member)
            if h["start"] is not None and timestamp is not None and h["start"] != timestamp:
                raise ValueError("Conflicting BWF time references within one track")
            h["start"] = h["start"] if h["start"] is not None else timestamp
        rows.append({"path": t["wav"], "stem": t["stem"], "type": t["track_type"],
                     "bleed": t["has_bleed"], "frames": h["frames"], "bwf_start": h["start"],
                     "pcm_bits": h["bits"],
                     "peak": peak / SCALE, "rail_samples": rails, "tail_active_samples": tail_active,
                     "pcm_sha256": sha.hexdigest()})
    peaks = [0, 0, 0]
    overs = [0, 0, 0]
    energy = [0.0, 0.0, 0.0]
    fp = []
    for begin in range(0, shared, BLOCK):
        end = min(shared, begin + BLOCK)
        v, i = vocal[begin:end], backing[begin:end]
        x = v + i
        for k, signal in enumerate((v, i, x)):
            peaks[k] = max(peaks[k], int(signal.max()), -int(signal.min()))
            overs[k] += int(np.count_nonzero((signal > SCALE) | (signal < -SCALE)))
            data = signal.astype("float64") / SCALE
            energy[k] += float(np.square(data).sum())
    # Anti-aliased mono proxy, used only for duplicate screening. Processing in
    # aligned chunks avoids constructing a full-length float64 waveform.
    step = RATE * 8
    for begin in range(0, shared, step):
        end = min(shared, begin + step)
        x = vocal[begin:end] + backing[begin:end]
        fp.append(resample_poly(x.mean(axis=1, dtype=np.float32) / SCALE, 1, RATE // FP_RATE))
    times = {r["bwf_start"] for r in rows}
    return {"tracks": rows, "frames_min": shared, "frames_max": max(lengths),
            "time_reference_count": len(times - {None}), "missing_bwf": None in times,
            "peak": dict(zip(("vocals", "instrumental", "mixture"), [p / SCALE for p in peaks])),
            "overs": dict(zip(("vocals", "instrumental", "mixture"), overs)),
            "rms": dict(zip(("vocals", "instrumental", "mixture"),
                            [float(np.sqrt(e / (shared * 2))) for e in energy]))}, np.concatenate(fp)


def signature(wave):
    """Low-rate spectral landmark hashes with position bins for broad screening."""
    stride, fft = 256, 1024
    if len(wave) < fft:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(wave, fft)[::stride]
    spectra = np.abs(np.fft.rfft(frames * np.hanning(fft), axis=1))
    bands = np.stack([spectra[:, low:high].mean(axis=1) for low, high in
                      zip((2, 4, 8, 16, 32, 64, 128, 256), (4, 8, 16, 32, 64, 128, 256, 513))], axis=1)
    loud = bands.max(axis=1)
    top = np.argsort(bands, axis=1)[:, -3:]
    code = top[:, 0] * 64 + top[:, 1] * 8 + top[:, 2]
    marks = []
    for n in range(0, len(code) - 24, 3):
        if loud[n:n+25:4].min() > loud.max() * 0.01:
            token = tuple(map(int, code[n:n+25:4]))
            if len(set(token)) >= 4:
                marks.append((token, n))
    return marks


def fingerprint_candidates(signatures):
    postings = defaultdict(list)
    for key, marks in signatures.items():
        counts = Counter(t for t, _ in marks)
        for token, frame in marks:
            if counts[token] == 1:
                postings[token].append((key, frame))
    hits = defaultdict(Counter)
    for rows in postings.values():
        if 2 <= len(rows) <= 5:
            for (a, f), (b, g) in itertools.combinations(rows, 2):
                hits[(a, b)][round((g - f) / 3)] += 1
    return [(a, b, votes) for (a, b), lag in hits.items()
            if (votes := max(lag.values())) >= 12]


def wave_match(a, b):
    """Compare three probes with offset and global-gain invariance."""
    size = FP_RATE * 12
    if min(len(a), len(b)) < size * 2:
        return []
    results = []
    sums = np.pad(np.cumsum(b.astype("float64") ** 2), (1, 0))
    norm_b = np.sqrt(np.maximum(sums[size:] - sums[:-size], 1e-12))
    for pos in (0.22, 0.50, 0.77):
        start = int((len(a) - size) * pos)
        x = a[start:start + size]
        norm = np.linalg.norm(x)
        if norm < 1e-5:
            continue
        ratios = np.abs(correlate(b, x, mode="valid", method="fft")) / (norm * norm_b)
        ix = int(ratios.argmax())
        results.append({"correlation": float(ratios[ix]), "lag_seconds": (ix - start) / FP_RATE})
    return results


def classify(song, audio):
    if not song["vocal_wavs"] or not song["instrumental_wavs"]:
        return "exclude_missing_target"
    hashes = [t["pcm_sha256"] for t in audio["tracks"]]
    if len(set(hashes)) != len(hashes):
        return "exclude_duplicate_stem"
    if audio["time_reference_count"] > 1:
        return "exclude_start_conflict"
    if audio["frames_min"] < 4 * RATE or min(audio["rms"]["vocals"], audio["rms"]["instrumental"]) < 1e-5:
        return "exclude_missing_signal"
    if audio["frames_min"] != audio["frames_max"]:
        # Canonical future padding would keep a documented, shared start.
        if audio["missing_bwf"]:
            return "review_length_without_timestamp"
        if any(t["tail_active_samples"] for t in audio["tracks"]):
            return "review_audible_tail"
        return "recover_silent_tail"
    if any(t["rail_samples"] for t in audio["tracks"]):
        return "review_source_rails"
    if any(t["bleed"] is None for t in audio["tracks"]):
        return "review_unknown_bleed"
    if any(t["bleed"] is True for t in audio["tracks"]):
        return "review_reported_bleed"
    if "unusual_vocal_label" in song["flags"]:
        return "review_vocal_taxonomy"
    return "eligible_strict"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--musdb-fire", required=True, type=Path)
    parser.add_argument("--pcm16-supplement", action="store_true")
    parser.add_argument("--output", default="eligibility.json")
    args = parser.parse_args()
    read = lambda name: json.loads((ROOT / name).read_text(encoding="utf-8"))
    split = read("split_external_final.json")
    by_id = {s["uuid"]: s for s in read("manifest.json")}
    roles = {r["uuid"]: r["role"] for r in split["assignments"]}
    reserved = set(split["validation_artist_groups"]) | set(split["test_artist_groups"])
    selected = {key: row for key, row in by_id.items() if row["artist_key"] not in reserved}
    assert set(selected) == {k for k, role in roles.items()
                             if role not in {"validation", "test", "held_out_artist_reserve"}}
    rows, features, fingerprints = {}, {}, {}
    started = perf_counter()
    report_path = ROOT / args.output
    with ZipFile(args.archive) as archive:
        keys = sorted(selected, key=lambda key: min(archive.getinfo(t["wav"]).header_offset
                                                    for t in selected[key]["tracks"]))
        if args.pcm16_supplement:
            with_pcm16 = []
            for key in keys:
                for t in selected[key]["tracks"]:
                    with archive.open(t["wav"]) as member:
                        if wav_header(member)["bits"] == 16:
                            with_pcm16.append(key)
                            break
            supplement = {}
            for key in with_pcm16:
                song = selected[key]
                audio, proxy = scan_song(archive, song)
                supplement[key] = {"uuid": key, "artist": song["artist"], "title": song["title"],
                                   "role_before": roles[key], "status": classify(song, audio),
                                   "fingerprint_marks": signature(proxy), **audio}
            write_json(ROOT / "eligibility_pcm16.json", supplement)
            print("Supplemental PCM16 songs", list(supplement), flush=True)
            return
        for number, key in enumerate(keys, 1):
            song = selected[key]
            try:
                audio, features[key] = scan_song(archive, song)
                fingerprints[key] = signature(features[key])
                status = classify(song, audio)
            except (ValueError, struct.error) as exc:
                audio, status = {"error": str(exc)}, "exclude_invalid_format"
            rows[key] = {"uuid": key, "artist": song["artist"], "title": song["title"],
                         "role_before": roles[key], "status": status, **audio}
            if number % 10 == 0:
                print(f"{number}/{len(keys)} training-artist songs checked", flush=True)
                write_json(report_path, {"status": "in_progress", "rows": rows})
    duplicates = defaultdict(list)
    for key, row in rows.items():
        if "tracks" in row:
            duplicates[tuple(sorted(t["pcm_sha256"] for t in row["tracks"]))].append(key)
    exact = [keys for keys in duplicates.values() if len(keys) > 1]
    for group in exact:
        for key in group:
            rows[key]["status"] = "exclude_duplicate_recording"
    pairs = {(a, b): f"{votes} aligned spectral hashes" for a, b, votes in
             fingerprint_candidates(fingerprints)}
    home = [key for key in features if by_id[key]["title"].strip().casefold() == "home"]
    if len(home) == 2:
        pairs[tuple(home)] = "same normalized title"
    cross = []
    for (a, b), reason in pairs.items():
        if any(a in group and b in group for group in exact):
            continue
        measurements = wave_match(features[a], features[b])
        cross.append({"a": a, "b": b, "reason": reason, "measurements": measurements})
    fire = next((key for key in features if by_id[key]["musdb_title_matches"]), None)
    fire_result = None
    if fire is not None:
        cmd = ["ffmpeg", "-v", "error", "-i", str(args.musdb_fire), "-map", "0:a:0",
               "-ac", "1", "-ar", str(FP_RATE), "-f", "f32le", "-"]
        music = np.frombuffer(subprocess.check_output(cmd), dtype="<f4")
        fire_result = {"uuid": fire, "source": args.musdb_fire.name,
                       "measurements": wave_match(features[fire], music)}
    write_json(report_path, {"status": "complete", "seconds": perf_counter() - started,
                             "sealed_moises_songs": len(by_id) - len(selected), "rows": rows,
                             "exact_duplicate_groups": exact, "acoustic_pairs": cross,
                             "timestamp_scan": "before and after PCM data",
                             "musdb_title_candidate": fire_result})
    print(Counter(row["status"] for row in rows.values()), flush=True)


if __name__ == "__main__":
    main()
