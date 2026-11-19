import json
import random

import pytest
import torch

from src.experiments.exp08_subband.checkpointing import Checkpoints, rng_state, restore_rng
from src.training.experiment_selection import SelectionState, should_stop, update_selection


def _payload(epoch, best):
    return {"epoch": epoch, "selection": {"best_epoch": best}, "rng": rng_state()}


def test_commit_preserves_selected_and_detects_corrupt_payload(tmp_path):
    store = Checkpoints(tmp_path, {"split": "a", "lr": .001})
    store.save(_payload(0, 0))
    store.save(_payload(1, 0))
    assert store.load()["epoch"] == 1
    assert store.load("best")["epoch"] == 0
    with pytest.raises(ValueError, match="fingerprint"):
        Checkpoints(tmp_path, {"split": "b", "lr": .001}).load()
    commit = json.loads((tmp_path / "commit.json").read_text())
    (tmp_path / commit["latest"]["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        store.load()


def test_interrupted_commit_can_retry_without_exposing_orphan(tmp_path, monkeypatch):
    store = Checkpoints(tmp_path, {})
    store.save(_payload(0, 0))
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError("simulated publication failure")
        patch.setattr(store, "_publish", fail)
        with pytest.raises(OSError):
            store.save(_payload(1, 1))
    assert store.load()["epoch"] == store.load("best")["epoch"] == 0
    store.save(_payload(1, 1))
    assert store.load()["epoch"] == store.load("best")["epoch"] == 1


def test_epoch_zero_ties_min_delta_and_patience():
    state = SelectionState(best_value=1.)
    for epoch in range(1, 13):
        state, improved = update_selection(state, .99995, epoch, min_delta=1e-4)
        assert not improved and state.best_epoch == 0
    assert should_stop(state, 12)
    state, improved = update_selection(state, .999, 13, min_delta=1e-4)
    assert improved and not should_stop(state, 12)


def test_cpu_rng_checkpoint_exact_continuation(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, 1, .99)

    def update():
        opt.zero_grad()
        model(torch.randn(2) * random.random()).square().backward()
        opt.step()
        sch.step()

    update()
    store = Checkpoints(tmp_path, {})
    store.save({**_payload(0, 0), "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sch.state_dict()})
    update()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    payload = store.load()
    model.load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])
    sch.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    update()
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
