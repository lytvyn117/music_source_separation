"""Expose only the two frozen rosters to the training and validation loaders."""
from contextlib import contextmanager
from copy import copy
import hashlib
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo

import numpy as np

from checkpoint import file_hash, read_json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "exp09_core"))

from loader import Shard, blocks, mapped

ARCHIVE = Path(r"C:\Users\YOUR_USER\Downloads\moisesdb.zip")
CACHE = ROOT.parent / "exp10_scratch" / "development_cache"


def load_roster():
    source = ROOT / "roster.json"
    if file_hash(source) != read_json(ROOT / "config.json")["roster_sha256"]:
        raise ValueError("The frozen audio allowlist changed")
    roster = read_json(source)
    train, dev, plan = (roster[k] for k in ("train", "development", "shard_plan"))
    ordered = [k for shard in plan["shards"] for k in shard]
    train_artists = {r["artist_key"] for r in train.values()}
    dev_artists = {r["artist_key"] for r in dev.values()}
    files = [{t["wav"] for r in group.values() for t in r["tracks"]} for group in (train, dev)]
    if (len(train) != 60 or len(dev) != 16 or len(dev_artists) != 7
            or train.keys() & dev.keys() or train_artists & dev_artists or files[0] & files[1]
            or len(ordered) != len(train) or set(ordered) != train.keys()
            or plan["frames_by_uuid"].keys() != train.keys()
            or roster["holdout"] != "external and sealed"):
        raise ValueError("The runtime rosters violate the frozen design")
    for group in (train, dev):
        for key, row in group.items():
            if row["uuid"] != key or not all(f"/{key}/" in t["wav"] for t in row["tracks"]):
                raise ValueError(f"Misassigned WAV path for {key}")
    return roster


class RestrictedArchive:
    """A ZIP view whose canonical member lookups are scoped to one approved role."""
    def __init__(self, path, rows):
        self.path = path
        self.members = {t["wav"] for row in rows.values() for t in row["tracks"]}

    def __enter__(self):
        self._archive = ZipFile(self.path, "r")
        return self

    def __exit__(self, *exc):
        self._archive.close()

    def _member(self, name):
        if name not in self.members:
            raise ValueError(f"ZIP member is outside the frozen role: {name}")
        return self._archive.getinfo(name)

    def getinfo(self, name):
        return copy(self._member(name))

    def open(self, name):
        if isinstance(name, ZipInfo):
            name = name.filename
        return self._archive.open(self._member(name), "r")


def reference_hashes(refs):
    names = ("mix", "vocals", "instrumental")
    hashes = {name: hashlib.sha256() for name in names}
    for begin, end in blocks(refs.shape[1]):
        v, i = refs[:, begin:end]
        for name, wave in zip(names, (v + i, v, i)):
            hashes[name].update(wave)
    return {name: digest.hexdigest() for name, digest in hashes.items()}


def verify_cache(development):
    index = read_json(CACHE / "index.json")
    if index.keys() != development.keys():
        raise ValueError("Development cache UUIDs differ from the frozen panel")
    for key, row in development.items():
        path = CACHE / f"{key}.npy"
        with mapped(path, "r") as refs:
            if (refs.shape != (2, row["frames"], 2) or refs.dtype != np.float32
                    or reference_hashes(refs) != row["reference_hashes"]):
                raise ValueError(f"Cached reference differs from the frozen samples at {key}")
        if path.stat().st_size != index[key]["bytes"] or file_hash(path) != index[key]["sha256"]:
            raise ValueError(f"Development cache provenance mismatch for {key}")
    return index


@contextmanager
def open_references(development, key):
    if key not in development:
        raise ValueError("Requested UUID is outside the frozen development panel")
    with mapped(CACHE / f"{key}.npy", "r") as refs:
        yield refs
