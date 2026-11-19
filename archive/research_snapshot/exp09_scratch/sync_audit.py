"""Complete the decoded training audit by reading trailing Broadcast WAV timestamps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import local
from time import perf_counter
from zipfile import ZipFile

from audit import write_json
from eligibility import classify, trailing_bwf, wav_header

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    base = read("eligibility.json")
    if base["status"] != "complete":
        raise ValueError("The complete PCM evidence is required")
    manifest = {s["uuid"]: s for s in read("manifest.json")}
    rows = base["rows"]
    for key, record in read("eligibility_pcm16.json").items():
        rows[key] = {k: v for k, v in record.items() if k != "fingerprint_marks"}
    # Reuse songs already decoded by the integrated timestamp pass if present.
    integrated_path = ROOT / "eligibility_complete.json"
    if integrated_path.exists():
        integrated = read(integrated_path.name)
        completed = {key for key, row in integrated["rows"].items() if "tracks" in row}
        for key in completed:
            rows[key] = integrated["rows"][key]
    else:
        completed = set()
    pending = [key for key, row in rows.items() if key not in completed and "tracks" in row]
    archives = []
    worker = local()

    def inspect(key):
        timestamps = {}
        if not hasattr(worker, "archive"):
            worker.archive = ZipFile(args.archive)
            archives.append(worker.archive)
        archive = worker.archive
        for track in rows[key]["tracks"]:
            path = track["path"]
            info = archive.getinfo(path)
            with archive.open(info) as stream:
                header = wav_header(stream)
                if header["frames"] != track["frames"]:
                    raise ValueError(f"WAV length differs from decoded evidence for {path}")
                remainder = info.file_size - header["offset"] - header["data_bytes"]
                start = header["start"]
                if remainder:
                    stream.seek(header["data_bytes"], 1)
                    trailing = trailing_bwf(stream)
                    if start is not None and trailing is not None and start != trailing:
                        raise ValueError(f"Conflicting time references in {path}")
                    if start is None:
                        start = trailing
                timestamps[path] = start
        return key, timestamps

    started = perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for index, (key, timestamps) in enumerate(pool.map(inspect, pending), 1):
                row = rows[key]
                for track in row["tracks"]:
                    track.setdefault("pcm_bits", 24)
                    track["bwf_start"] = timestamps[track["path"]]
                times = set(timestamps.values())
                row["missing_bwf"] = None in times
                row["time_reference_count"] = len(times - {None})
                row["status"] = classify(manifest[key], row)
                if index % 20 == 0:
                    print(f"{index}/{len(pending)} remaining timestamp checks", flush=True)
    finally:
        for archive in archives:
            archive.close()
    for group in base["exact_duplicate_groups"]:
        for key in group:
            rows[key]["status"] = "exclude_duplicate_recording"
    base["timestamp_scan"] = "before and after PCM data"
    base["timestamp_seconds"] = perf_counter() - started
    base["reused_integrated_rows"] = len(completed)
    write_json(integrated_path, base)
    print("Completed timestamp evidence", len(rows), "songs", base["timestamp_seconds"], "seconds")


if __name__ == "__main__":
    main()
