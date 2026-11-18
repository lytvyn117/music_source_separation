import json

import pytest

from src.experiments.exp08_subband.data import bind_run, sha256
from src.experiments.exp08_subband.checkpointing import Checkpoints
from src.experiments.exp08_subband.train import TrainConfig, resume_epoch


def test_fit_binding_rejects_fingerprint_before_opening_data(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"train": ["a"], "validation": ["b"]}))
    with pytest.raises(ValueError, match="fingerprint"):
        bind_run(path, {"manifest_sha256": "wrong", "config": {}}, {})
    with pytest.raises(ValueError, match="fingerprint"):
        bind_run(path, {"manifest_sha256": sha256(path), "config": {"seed": 42}}, {"seed": 43})


def test_resume_rejects_changed_config_before_loading(tmp_path):
    store = Checkpoints(tmp_path, {"config": {"lr": 3e-4}})
    with pytest.raises(ValueError, match="configuration"):
        resume_epoch(store, None, None, None, TrainConfig(lr=1e-4))
