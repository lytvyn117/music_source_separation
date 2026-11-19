"""Check conservative shared-span recovery without changing the strict allowlist."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from audit import write_json
from loader import Shard

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    audit = json.loads((ROOT / "training_eligibility.json").read_text(encoding="utf-8"))
    ids = set(audit["summary"]["known_silent_tail_candidates_without_other_flags"])
    rows = [r for r in json.loads((ROOT / "manifest.json").read_text(encoding="utf-8")) if r["uuid"] in ids]
    if {r["uuid"] for r in rows} != ids or ids & set(audit["strict_train"]):
        raise ValueError("Expected audited recovery candidates outside strict training")
    results = []
    with ZipFile(args.archive) as archive, Shard(ROOT, 4 * 1024**3) as shard:
        for song in rows:
            result = shard.stage(archive, song, verify=True, common_span=True)
            expected = audit["rows"][song["uuid"]]
            if (result["frames"] != expected["frames_min"]
                    or result["reconstruction_max_abs"] > 1e-6):
                raise ValueError("Common-span loader differs from decoded audit")
            results.append({**result, "title": song["title"], "artist": song["artist"]})
            print(song["title"], result["frames"], result["reconstruction_max_abs"], flush=True)
    write_json(ROOT / "tail_recovery.json", {"songs": results,
                                           "seconds": sum(r["duration_seconds"] for r in results),
                                           "policy": "explicit common span, no gain or offset changes"})


if __name__ == "__main__":
    main()
