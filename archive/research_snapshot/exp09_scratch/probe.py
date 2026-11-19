"""Run bounded full-track checks and CPU loading timings on clean training audio."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import random
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from audit import write_json
from loader import RATE, Shard

GIB = 1024 ** 3


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pick_songs(manifest, splits):
    common = set.intersection(*[
        {r["uuid"] for r in s["assignments"] if r["role"] == "train_clean"} for s in splits
    ])
    eligible = [s for s in manifest if s["uuid"] in common]
    selected = []
    # Sample four distinct artists. Three controls plus one length-risk challenge.
    for count_min, count_max, varied in ((2, 6, False), (7, 11, False),
                                          (12, 100, False), (5, 100, True)):
        rows = []
        for row in eligible:
            sizes = [t["bytes"] for t in row["tracks"]]
            size_spread = max(sizes) - min(sizes)
            if (count_min <= len(sizes) <= count_max
                    and (size_spread >= 20000 if varied else size_spread < 20)
                    and row["artist_key"] not in {r["artist_key"] for r in selected}
                    and not row["musdb_title_matches"]):
                rows.append(row)
        if not rows:
            raise ValueError("No training probe candidate for requested format stratum")
        selected.append(min(rows, key=lambda s: s["raw_wav_bytes"]))
    return selected


def benchmark(shard, seed=909, windows=384):
    rng = random.Random(seed)
    frames = 4 * RATE
    available = sorted(shard.songs)
    jobs = []
    for _ in range(windows):
        key = rng.choice(available)
        jobs.append((key, rng.randrange(shard.songs[key][1] - frames + 1)))
    latencies = []
    checksum = 0.0
    with ExitStack() as stack:
        readers = {key: stack.enter_context(shard.reader(key)) for key in available}
        for key, start in jobs[:32]:
            readers[key](start, frames)
        start_total = perf_counter()
        for key, start in jobs:
            now = perf_counter()
            mix, vocal, backing = readers[key](start, frames)
            checksum += float(mix[0, 0] + vocal[0, 0] + backing[0, 0])
            latencies.append(perf_counter() - now)
        elapsed = perf_counter() - start_total
    return {"windows": windows, "warmup_windows": 32, "seconds_per_window_audio": 4,
            "elapsed_seconds": elapsed, "windows_per_second": windows / elapsed,
            "reference_MiB_per_second": windows * frames * 16 / 1024**2 / elapsed,
            "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
            "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
            "checksum": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    manifest = read_json(Path("manifest.json"))
    splits = [read_json(Path(n)) for n in ("split_external_final.json", "split_internal_20_24.json")]
    songs = pick_songs(manifest, splits)
    budget = 4 * GIB
    verify_results, stage_results, failures = [], [], []
    with ZipFile(args.archive) as archive, Shard(Path.cwd(), budget) as shard:
        for song in songs:
            label = {k: song[k] for k in ("uuid", "artist", "title", "genre")}
            try:
                row = shard.stage(archive, song, verify=True)
            except ValueError as exc:
                failures.append({**label, "strict_error": str(exc)})
                if "format or length" not in str(exc):
                    raise
                row = shard.stage(archive, song, verify=True, common_span=True)
            verify_results.append({**label, **row})
            print(label, row["length_policy"], row["frames"], row["timing"], flush=True)
        warm = [benchmark(shard, seed=909 + i) for i in range(3)]
        observed = shard.peak_reserved_bytes
    # Production path run is a repeat of the selected recordings. OS caching may
    # affect archive reads. No process or filesystem cache is flushed.
    with ZipFile(args.archive) as archive, Shard(Path.cwd(), budget) as shard:
        for song, verified in zip(songs, verify_results):
            row = shard.stage(archive, song, common_span=verified["length_policy"] == "common_span")
            stage_results.append({k: row[k] for k in
                                  ("uuid", "frames", "cache_bytes", "timing")})
        observed = max(observed, shard.peak_reserved_bytes)
    raw_bytes = sum(s["raw_wav_bytes"] for s in songs)
    compressed_bytes = sum(s["compressed_wav_bytes"] for s in songs)
    report = {"platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "numpy": np.__version__,
              "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
              "budget_bytes": budget, "peak_reserved_payload_bytes": observed,
              "raw_wav_bytes": raw_bytes, "compressed_wav_bytes": compressed_bytes,
              "strict_failures": failures, "verified": verify_results,
              "stage_without_qc_repeat": stage_results, "warm_random_reads": warm}
    write_json(Path("audio_probe.json"), report)


if __name__ == "__main__":
    main()
