import random

import numpy as np
import pytest
import torch

from src.experiments.exp07_phase_aware_residual.checkpointing import (
    CheckpointManager, capture_rng_state, restore_rng_state,
)


def test_initial_is_immutable_and_best_only_replaced_on_improvement(tmp_path):
    manager = CheckpointManager(tmp_path)
    first = {"epoch": 0, "validation_objective": 1.0, "config": {"batch_size": 4}}
    manager.initialize(first)
    initial = (tmp_path / "initial.pt").read_bytes()
    with pytest.raises(FileExistsError):
        manager.initialize(first)
    manager.save_latest(first | {"epoch": 1})
    assert not manager.maybe_save_best(first | {"epoch": 1}, 1.0)
    assert manager.maybe_save_best(first | {"epoch": 2}, 0.9)
    assert torch.load(tmp_path / "best.pt")["epoch"] == 2
    assert manager.load_latest(first["config"])["epoch"] == 1
    assert (tmp_path / "initial.pt").read_bytes() == initial
    with pytest.raises(ValueError, match="configuration"):
        manager.load_latest({"batch_size": 2})
    (tmp_path / "best.pt").unlink()
    with pytest.raises(FileNotFoundError, match="best"):
        manager.load_latest(first["config"])


def test_rng_roundtrip_restores_python_numpy_torch_and_loader():
    generator = torch.Generator().manual_seed(101)
    state = capture_rng_state(generator)

    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item(), torch.rand(1, generator=generator).item()

    expected = draw()
    draw()
    restore_rng_state(state, generator)
    assert draw() == expected
