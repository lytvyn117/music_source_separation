"""Run the authorized scratch training schedule from the strict MoisesDB manifest."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from audit import write_json
from gpu_preflight import BACKENDS, device_snapshot
from loader import Shard
from loss_candidates import WEIGHTS, get_loss
from scratch_model import SEGMENT, Separator, seed_all
from scratch_train import ExactRun, model_digest
from shard_plan import sample_position

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_schedule(decision=None):
    plan = read(ROOT / "training_shards.json")
    allowlist = ROOT / "training_eligibility.json"
    if plan["allowlist_sha256"] != digest(allowlist):
        raise ValueError("Shard plan does not match the audited allowlist")
    ids = set(read(allowlist)["strict_train"])
    split = read(ROOT / "split_external_final.json")
    assigned = {row["uuid"] for row in split["assignments"] if row["role"] == "train_clean"}
    flattened = [key for shard in plan["shards"] for key in shard]
    if ids != set(flattened) or not ids <= assigned or len(flattened) != len(ids):
        raise ValueError("Shard UUIDs differ from strict training or include reserved artists")
    if plan["segment_frames"] != SEGMENT:
        raise ValueError("Unexpected segment size")
    lengths = {key: (int(n), None) for key, n in plan["frames_by_uuid"].items()}
    if set(lengths) != ids or min(n for n, _ in lengths.values()) < SEGMENT:
        raise ValueError("Invalid training lengths")
    if decision is not None and decision["loss"] != WEIGHTS[decision["objective"]]:
        raise ValueError("Selected objective and weights differ from the measured configuration")
    manifests = {r["uuid"]: r for r in read(ROOT / "manifest.json") if r["uuid"] in ids}
    if set(manifests) != ids:
        raise ValueError("Incomplete source manifest")
    for filename, expected in (decision or {}).get("data_sha256", {}).items():
        if digest(ROOT / filename) != expected:
            raise ValueError(f"Frozen decision input changed: {filename}")
    identity = hashlib.sha256()
    for name in ("scratch_model.py", "scratch_train.py", "loss_candidates.py", "shard_plan.py",
                 "loader.py", "train_exp09.py"):
        identity.update((ROOT / name).read_bytes())
    identity.update(json.dumps(decision, sort_keys=True).encode())
    return plan, lengths, manifests, identity.hexdigest()


def pack(readers, addresses):
    x, y = [], []
    for key, start in addresses:
        mix, vocal, _ = readers[key](start, SEGMENT)
        x.append(mix.T)
        y.append(vocal.T)
    return tuple(torch.from_numpy(np.stack(a)).cuda() for a in (x, y))


def reconcile_log(path, step):
    if not path.exists():
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row["step"] > step:
            break
        rows.append(line)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(line + "\n" for line in rows), encoding="utf-8")
    tmp.replace(path)


def execute(archive_path, out, decision, resume=False, staging_only=False, smoke_stop=None):
    plan, frames, manifests, identity = bind_schedule(None if staging_only else decision)
    if staging_only:
        evidence = []
        with ZipFile(archive_path) as archive:
            for group in plan["shards"]:
                t = perf_counter()
                with Shard(ROOT, plan["budget_bytes"]) as shard:
                    for key in group:
                        result = shard.stage(archive, manifests[key])
                        if result["frames"] != frames[key][0]:
                            raise ValueError("Frame count changed during staging")
                    evidence.append({"songs": len(group), "seconds": perf_counter()-t,
                                     "peak_payload_bytes": shard.peak_reserved_bytes})
                    print("Staged shard", evidence[-1], flush=True)
        write_json(out, {"shards": evidence, "total_seconds": sum(x["seconds"] for x in evidence)})
        return

    seed_all(decision["seed"], deterministic=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required")
    objective = decision["objective"]
    model = Separator(checkpoint_blocks=True).cuda()
    run = ExactRun(model, decision["loss"], decision["seed"], True, total=decision["updates"],
                   objective_fn=get_loss(objective), objective_name=objective)
    latest = out / "latest.pt"
    if resume:
        run.restore(latest, identity)
    else:
        out.mkdir(exist_ok=False, parents=True)
        write_json(out / "run.json", {"identity": identity, "decision": decision,
                                     "initial_model_sha256": model_digest(model)})
        run.save(latest, identity)
    sink = out / "steps.jsonl"
    if resume:
        reconcile_log(sink, run.index)
    # Keep one cache shard open, and rotate according to the exact update address.
    previous = None
    with ZipFile(archive_path) as archive, ExitStack() as active, sdpa_kernel(BACKENDS):
        stop = decision["updates"] if smoke_stop is None else smoke_stop
        if not run.index < stop <= decision["updates"]:
            raise ValueError("Stop must be after the saved step and within the schedule")
        while run.index < stop:
            n, addresses = sample_position(plan["shards"], frames, SEGMENT, decision["seed"],
                                            run.index, 4, plan["updates_per_shard"])
            if n != previous:
                active.close()
                shard = active.enter_context(Shard(ROOT, plan["budget_bytes"]))
                for key in plan["shards"][n]:
                    checked = shard.stage(archive, manifests[key])
                    if checked["frames"] != frames[key][0]:
                        raise ValueError("Frame count changed during staging")
                readers = {k: active.enter_context(shard.reader(k)) for k in shard.songs}
                previous = n
            t = perf_counter()
            metrics = run.step(*pack(readers, addresses))
            torch.cuda.synchronize()
            memory = device_snapshot()
            if memory["peak_reserved_gib"] > 10:
                raise RuntimeError("Peak reserved exceeded 10 GiB")
            with sink.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"step": run.index, "seconds": perf_counter()-t, "shard": n,
                                      "crops": addresses, "memory": memory, **metrics}) + "\n")
            if run.index % decision["checkpoint_every"] == 0 or run.index == stop:
                run.save(latest, identity)
            if run.index % 100 == 0:
                print(run.index, metrics["loss"], memory["peak_reserved_gib"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--decision", type=Path, default=ROOT / "launch_decision.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--staging-only", action="store_true")
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument("--authorize-full-training", action="store_true")
    launch.add_argument("--smoke-stop", type=int, choices=(1, 2), help="Bounded launcher check only")
    args = parser.parse_args()
    if not (args.staging_only or args.authorize_full_training or args.smoke_stop):
        parser.error("Choose --staging-only, --smoke-stop, or explicitly authorize the full run")
    if args.staging_only and args.resume:
        parser.error("Staging-only has no optimizer state")
    decision = None if args.staging_only else read(args.decision)
    execute(args.archive, args.out, decision, args.resume, args.staging_only, args.smoke_stop)


if __name__ == "__main__":
    main()
