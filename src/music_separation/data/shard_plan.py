"""Deterministic shard grouping and crop addressing by optimizer update."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from music_separation.data.audit import write_json
from music_separation.models.bs_roformer import SEGMENT, crop_for

GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parent


def partition(records, budget):
    """First-fit packing with headroom for one staged song's transient files."""
    if not records:
        raise ValueError("Empty training allowlist")
    headroom = max(32 * frames + wav + 8192 for frames, wav in records.values())
    payload_limit = budget - headroom
    groups = []
    sizes = []
    for key in sorted(records, key=lambda k: (-records[k][0], k)):
        payload = 16 * records[key][0] + 256
        if payload > payload_limit:
            raise ValueError(f"Shard budget cannot fit {key}")
        for group, size in enumerate(sizes):
            if size + payload <= payload_limit:
                groups[group].append(key)
                sizes[group] += payload
                break
        else:
            groups.append([key])
            sizes.append(payload)
    return groups


def sample_position(plan, records, segment, seed, update, effective_batch, updates_per_shard):
    if update < 0 or min(effective_batch, updates_per_shard) < 1:
        raise ValueError("Invalid update address")
    visit = update // updates_per_shard
    round_number, index = divmod(visit, len(plan))
    order = list(range(len(plan)))
    random.Random(f"{seed}:shard-round:{round_number}").shuffle(order)
    shard_index = order[index]
    lengths = {key: records[key][0] for key in plan[shard_index]}
    windows = [crop_for(lengths, segment, seed, effective_batch * update + offset)
               for offset in range(effective_batch)]
    return shard_index, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-gib", type=float, default=4.0)
    args = parser.parse_args()
    audit = ROOT / "training_eligibility.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    manifest = {s["uuid"]: s for s in json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))}
    ids = payload["strict_train"]
    records = {key: (payload["rows"][key]["frames_min"],
                     max(t["bytes"] for t in manifest[key]["tracks"])) for key in ids}
    budget = int(args.budget_gib * GIB)
    plan = partition(records, budget)
    summaries = []
    for ids in plan:
        members = [records[key] for key in ids]
        persistent = sum(16 * frames + 256 for frames, _ in members)
        transient = max(32 * frames + wav + 8192 for frames, wav in members)
        summaries.append({"songs": len(ids), "reserved_bound_bytes": persistent + transient,
                          "reference_bytes": persistent,
                          "raw_zip_wav_bytes": sum(manifest[key]["raw_wav_bytes"] for key in ids)})
    write_json(ROOT / "training_shards.json", {
        "budget_bytes": budget, "allowlist_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "segment_frames": SEGMENT, "updates_per_shard": 500, "shards": plan, "summary": summaries,
        "frames_by_uuid": {key: frames for key, (frames, _) in records.items()},
    })
    print(len(plan), "bounded shards", summaries)


if __name__ == "__main__":
    main()
