"""Immutable checkpoint transactions with reconstructible latest, best and history."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import stat
from tempfile import NamedTemporaryFile, mkdtemp
from time import sleep

import numpy as np
import torch

BEST_ALIASES = ("best_overall", "best_qualified")


def qualification_status(selection):
    if not selection:
        return "validation_pending"
    return "qualified" if selection["best_qualified"] is not None else "no_candidate_qualified"


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replace_file(source, destination):
    for retry in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or retry == 5:
                raise
            sleep(.1 * (retry + 1))


def atomic_file(path, write):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp = Path(handle.name)
    try:
        with temp.open("wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, data):
    payload = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    atomic_file(path, lambda handle: handle.write(payload))


def copy_atomic(source, destination):
    def copy(handle):
        with source.open("rb") as reader:
            shutil.copyfileobj(reader, handle, 2**20)
    atomic_file(destination, copy)


def capture_rng(device):
    kind, keys, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(), "numpy": (kind, keys.tolist(), position, gaussian, cached),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    kind, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((kind, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.weights[name].lerp_(param.detach(), 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "weights": self.weights}

    def load_state_dict(self, state):
        if self.decay != state["decay"] or self.weights.keys() != state["weights"].keys():
            raise ValueError("EMA state does not match the model or configured decay")
        for name, weight in self.weights.items():
            weight.copy_(state["weights"][name])
        self.updates = state["updates"]


class Store:
    """A numbered directory contains state.pt and its SHA-256 provenance."""
    def __init__(self, root, identity, interval, maximum):
        self.root, self.identity, self.interval, self.maximum = root, identity, interval, maximum
        self.numbered = root / "checkpoints"

    def __enter__(self):
        # A byte-range lock is released by the OS even if the trainer is killed.
        self.handle = (self.root / ".writer.lock").open("a+b")
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.numbered.mkdir(exist_ok=True)
        except BaseException:
            self.handle.close()
            raise
        return self

    def __exit__(self, *exc):
        self.handle.close()

    def bundle(self, step):
        return self.numbered / f"step_{step:06d}"

    def validate_header(self, state, step):
        if (state["identity"] != self.identity or state["step"] != step
                or not 0 <= step <= self.maximum):
            raise ValueError("Checkpoint provenance or update mismatch")
        boundaries = list(range(self.interval, step + 1, self.interval))
        if [row["step"] for row in state["history"]] != boundaries:
            raise ValueError("Incomplete or misordered validation history")

    def checked_path(self, bundle):
        metadata = read_json(bundle / "provenance.json")
        path = bundle / "state.pt"
        step = int(bundle.name.removeprefix("step_"))
        if (metadata["identity"] != self.identity or metadata["step"] != step
                or path.stat().st_size != metadata["bytes"]
                or file_hash(path) != metadata["sha256"]):
            raise ValueError(f"Corrupt or mismatched checkpoint at {bundle}")
        return path, metadata

    def materialize(self, state, current):
        latest, metadata = current
        copy_atomic(latest, self.root / "latest.pt")
        pointer = self.root / "checkpoint_index.json"
        selection = state["history"][-1]["selection"] if state["history"] else {}
        index = {"latest": metadata, "qualification_status": qualification_status(selection)}
        for name in BEST_ALIASES:
            winner = selection.get(name)
            alias = self.root / f"{name}.pt"
            if winner is None:
                alias.unlink(missing_ok=True)
                index[name] = None
                continue
            best_step = winner["step"]
            if best_step == state["step"]:
                best_source, best = latest, metadata
            else:
                best_source = self.bundle(best_step) / "state.pt"
                best = read_json(best_source.with_name("provenance.json"))
            if (best["step"] != best_step or best["identity"] != self.identity
                    or winner["weights"] not in ("raw", "ema")):
                raise ValueError(f"Invalid source for {name}")
            if not alias.is_file() or file_hash(alias) != best["sha256"]:
                if best_step != state["step"]:
                    self.checked_path(self.bundle(best_step))
                copy_atomic(best_source, alias)
            index[name] = {**best, **winner}
        write_json(self.root / "validation_history.json", state["history"])
        write_json(pointer, index)

    def commit(self, state):
        step = state["step"]
        self.validate_header(state, step)
        destination = self.bundle(step)
        if destination.exists():
            raise FileExistsError(destination)
        temp = Path(mkdtemp(prefix=".checkpoint-", dir=self.numbered))
        try:
            payload = temp / "state.pt"
            with payload.open("wb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            metadata = {
                "step": step, "identity": self.identity, "bytes": payload.stat().st_size,
                "sha256": file_hash(payload),
                "file": destination.relative_to(self.root).as_posix() + "/state.pt",
            }
            write_json(temp / "provenance.json", metadata)
            for file in temp.iterdir():
                file.chmod(stat.S_IREAD)
            # Rename into a new directory. The writer lock prevents a second publisher.
            temp.rename(destination)
        finally:
            if temp.exists():
                for file in temp.iterdir():
                    file.chmod(stat.S_IREAD | stat.S_IWRITE)
                shutil.rmtree(temp)
        self.materialize(state, (destination / "state.pt", metadata))

    def published(self):
        return sorted(p for p in self.numbered.iterdir()
                      if p.is_dir() and re.fullmatch(r"step_\d{6}", p.name))

    def recover(self):
        # The numbered transaction is authoritative even if publishing an alias was interrupted.
        published = self.published()
        if not published:
            raise FileNotFoundError("No committed checkpoint")
        path, metadata = self.checked_path(published[-1])
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.validate_header(state, metadata["step"])
        # Alias index is derived. A partial or manually damaged index cannot block recovery.
        (self.root / "checkpoint_index.json").unlink(missing_ok=True)
        self.materialize(state, (path, metadata))
        return state
