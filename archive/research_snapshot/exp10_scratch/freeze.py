"""Derive the standalone runtime roster from the hash-bound Exp09 records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "preserved_exp09"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    preservation = {row["path"]: row["sha256"] for row in read(ROOT / "exp09_preservation.json")}

    def input_file(name):
        path = BASE / name
        if file_hash(path) != preservation[name]:
            raise ValueError(f"Exp09 evidence has changed: {name}")
        return read(path)

    run = input_file("runs/exp09_mse_seed909/run.json")
    settings = run["decision"]
    raw = {}
    for name, expected in settings["data_sha256"].items():
        if preservation[name] != expected:
            raise ValueError(f"Unexpected Exp09 data identity: {name}")
        raw[name] = input_file(name)
    split = raw["split_external_final.json"]
    if not split["external_final_required"]:
        raise ValueError("Expected the active external-final split")
    roles = {r["uuid"]: r["role"] for r in split["assignments"]}
    plan = raw["training_shards.json"]
    train_ids = raw["training_eligibility.json"]["strict_train"]
    panel = input_file("dev_eval_20261114/frozen_panel.json")
    scores = input_file("dev_eval_20261114/results.json")
    eval_run = input_file("dev_eval_20261114/execution_freeze.json")
    if eval_run["files_sha256"]["frozen_panel.json"] != preservation["dev_eval_20261114/frozen_panel.json"]:
        raise ValueError("Archived development execution panel differs")
    dev_ids = panel["admitted"]
    if (len(set(train_ids)) != 60 or len(set(dev_ids)) != 16
            or set(train_ids) & set(dev_ids)
            or any(roles[k] != "train_clean" for k in train_ids)
            or any(roles[k] != "validation" for k in dev_ids)
            or [row["uuid"] for row in scores["rows"]] != dev_ids):
        raise ValueError("Training and development UUIDs disagree with frozen evidence")
    manifests = {r["uuid"]: r for r in raw["manifest.json"]}
    validation_artists = set(split["validation_artist_groups"])
    if (any(manifests[k]["artist_key"] in validation_artists for k in train_ids)
            or any(manifests[k]["artist_key"] not in validation_artists for k in dev_ids)):
        raise ValueError("An artist is outside its approved role")

    def song(k):
        r = manifests[k]
        tracks = [{field: t[field] for field in ("wav", "stem", "available", "bytes", "crc32")}
                  for t in r["tracks"]]
        return {"uuid": k, "artist_key": r["artist_key"], "tracks": tracks}

    development = {}
    for k in dev_ids:
        g = panel["songs"][k]
        frames = g["frames_min"]
        if (g["reasons"] or g["frames_max"] != frames or g["missing_bwf"]
                or len({t["bwf_start"] for t in g["tracks"]}) != 1
                or {t["path"] for t in g["tracks"]} != {t["wav"] for t in manifests[k]["tracks"]}):
            raise ValueError(f"Unexpected reference or alignment evidence for {k}")
        development[k] = {
            **song(k), "frames": frames, "native_start_frame": 0,
            "bwf_start": g["tracks"][0]["bwf_start"],
            "reference_hashes": g["reference_hashes"],
        }
    roster = {
        "holdout": "external and sealed",
        "train": {k: song(k) for k in train_ids},
        "development": development,
        "shard_plan": plan,
    }
    metrics = ("instrumental_sdr_db", "v_side_sdr_db", "i_side_sdr_db", "abs_vocal_bleed_coeff")
    baselines = {name: {m: scores["aggregate"][name][m]["artist_macro"] for m in metrics}
                 for name in ("exp09", "hybrid")}
    config = {
        "exp09_settings": settings,
        "scheduler": {"name": "CosineAnnealingLR", "eta_min": .00003, "T_max": 24000},
        "exp09_identity": run["identity"],
        "initial_model_sha256": run["initial_model_sha256"],
        "frozen_inputs_sha256": {
            name: preservation[name] for name in ("dev_eval_20261114/frozen_panel.json",
                                                "dev_eval_20261114/results.json",
                                                "dev_eval_20261114/execution_freeze.json",
                                                "runs/exp09_mse_seed909/latest.pt",
                                                "split_external_final.json")
        },
        "ema_decay": .999,
        "validation": {
            "precision": "FP32", "interval": 1000, "segment_frames": 176400,
            "hop_frames": 88200, "window": "positive Hann",
            "activity_dbfs": -45, "side_activity_dbfs": -50, "sample_rate": 44100,
            "songs": 16, "artists": 7,
        },
        "selection": {
            "primary": "vocal_sdr_db",
            "guardrails": {
                **{m: {"min": baselines["hybrid"][m] - .25} for m in metrics[:3]},
                "abs_vocal_bleed_coeff": {"max": baselines["exp09"]["abs_vocal_bleed_coeff"] + .01},
            },
            "min_updates": 16000, "patience_validations": 8, "min_delta_db": .15,
        },
        "archived_development_anchors": baselines,
        "memory_guard_gib": 10,
    }
    roster_path = ROOT / "roster.json"
    with roster_path.open("x", encoding="utf-8") as handle:
        json.dump(roster, handle, indent=2)
        handle.write("\n")
    config["roster_sha256"] = file_hash(roster_path)
    with (ROOT / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
