import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_rng_state(generator=None):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state() if generator is not None else None,
    }


def restore_rng_state(state, generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if generator is not None:
        generator.set_state(state["loader_generator"])


def _save(state, path, exclusive=False):
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(state, temporary)
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.best_value = float("inf")

    def initialize(self, state):
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Checkpoint collision in {self.directory}")
        for filename in ("initial.pt", "best.pt", "latest.pt"):
            _save(state, self.directory / filename, exclusive=True)
        self.best_value = state["validation_objective"]

    def save_latest(self, state):
        _save(state, self.directory / "latest.pt")

    def maybe_save_best(self, state, selection_value):
        if selection_value >= self.best_value:
            return False
        _save(state | {"validation_objective": selection_value}, self.directory / "best.pt")
        self.best_value = selection_value
        return True

    def save_snapshot(self, state):
        _save(state, self.directory / f"checkpoint_epoch_{state['epoch']}.pt", exclusive=True)

    def load_latest(self, expected_config):
        for name in ("initial.pt", "best.pt", "latest.pt"):
            if not (self.directory / name).is_file():
                raise FileNotFoundError(self.directory / name)
        # These are local checkpoints written by this trainer, including Python and NumPy RNG state.
        load = lambda filename: torch.load(self.directory / filename, map_location="cpu", weights_only=False)
        state = load("latest.pt")
        if state["config"] != expected_config:
            raise ValueError("Checkpoint configuration mismatch")
        self.best_value = load("best.pt")["validation_objective"]
        return state
