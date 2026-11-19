"""Immutable checkpoint payloads with one atomic latest/best commit record."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch

from .data import sha256


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def _writer_lock(path):
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield  # The OS releases the lock when the handle closes, including after a crash.


class Checkpoints:
    def __init__(self, root, fingerprint):
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.commit = self.root / "commit.json"

    def _record(self):
        record = json.loads(self.commit.read_text(encoding="utf-8"))
        if record["fingerprint"] != self.fingerprint:
            raise ValueError("Checkpoint configuration or data fingerprint changed")
        return record

    def _publish(self, record):
        temporary = self.root / f".commit-{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.commit)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, state):
        with _writer_lock(self.root / ".writer.lock"):
            epoch = state["epoch"]
            selected = state["selection"]["best_epoch"]
            previous = self._record() if self.commit.exists() else None
            if previous is None:
                if epoch != 0 or selected != 0:
                    raise ValueError("The first committed checkpoint must be epoch zero")
            elif (epoch != previous["latest"]["epoch"] + 1
                  or selected not in (epoch, previous["best"]["epoch"])):
                raise ValueError("Checkpoint publication must advance one epoch from latest")
            name = f"epoch_{epoch:03d}_{uuid4().hex}.pt"
            path = self.root / name
            with path.open("xb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            entry = {"epoch": epoch, "file": name, "sha256": sha256(path)}
            best = entry if selected == epoch else previous["best"]
            snapshots = dict(previous["snapshots"]) if previous else {}
            if epoch % 10 == 0:
                snapshots[str(epoch)] = entry
            self._publish({"fingerprint": self.fingerprint, "latest": entry,
                           "best": best, "snapshots": snapshots})
            return entry

    def load(self, which="latest"):
        if which not in ("latest", "best"):
            raise ValueError("Expected latest or best checkpoint")
        entry = self._record()[which]
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint path in commit record")
        path = self.root / name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError("Checkpoint integrity check failed")
        # Local payloads include optimizer and Python/NumPy RNG state.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["epoch"] != entry["epoch"]:
            raise ValueError("Checkpoint epoch differs from commit record")
        return state
