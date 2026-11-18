from pathlib import Path

import pytest
import torch

from src.training.experiment_selection import (
    SELECTION_METRIC,
    SelectionState,
    initialize_best_checkpoint,
    restore_selection_state,
    selection_value,
    should_stop,
    update_selection,
)


def test_selection_value_uses_full_objective_and_matches_exp04_legacy_alias():
    validation = {
        "val_base_composite_loss": 0.6805979421264247,
        "val_leakage_loss": 0.0046810195168578315,
        "val_total_loss": 0.685278959964451,
        "val_composite_loss": 0.685278959964451,
    }

    assert SELECTION_METRIC == "val_total_loss"
    assert selection_value(validation) == pytest.approx(0.685278959964451)
    assert selection_value(validation) == validation["val_composite_loss"]
    assert selection_value(validation) != validation["val_base_composite_loss"]


def test_best_selection_and_patience_use_the_same_min_delta_rule():
    state = SelectionState(best_value=1.0, best_epoch=0)

    state, improved = update_selection(state, current_value=0.99995, epoch=1, min_delta=1e-4)
    assert not improved
    assert state.best_value == 1.0
    assert state.epochs_without_improvement == 1
    assert not should_stop(state, patience=2)

    state, improved = update_selection(state, current_value=0.9, epoch=2, min_delta=1e-4)
    assert improved
    assert state.best_value == 0.9
    assert state.best_epoch == 2
    assert state.epochs_without_improvement == 0

    state, improved = update_selection(state, current_value=0.9001, epoch=3, min_delta=1e-4)
    assert not improved
    state, improved = update_selection(state, current_value=0.9002, epoch=4, min_delta=1e-4)
    assert not improved
    assert should_stop(state, patience=2)


def test_initial_best_checkpoint_is_preserved_when_no_epoch_improves(tmp_path: Path):
    best_path = tmp_path / "best.pt"
    initial_payload = {
        "epoch": 0,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "config": {"mask_mode": "sigmoid"},
        "selection_metric": "val_total_loss",
        "selection_value": 0.75,
    }

    initialize_best_checkpoint(best_path, initial_payload)
    before = best_path.read_bytes()
    state = SelectionState(best_value=0.75, best_epoch=0)
    state, improved = update_selection(state, current_value=0.75005, epoch=1, min_delta=1e-4)

    assert not improved
    assert best_path.read_bytes() == before
    restored = torch.load(best_path, map_location="cpu", weights_only=True)
    assert restored["epoch"] == 0
    assert restored["config"]["mask_mode"] == "sigmoid"


def test_initial_best_checkpoint_refuses_to_overwrite_existing_file(tmp_path: Path):
    best_path = tmp_path / "best.pt"
    best_path.write_bytes(b"existing checkpoint")

    with pytest.raises(FileExistsError):
        initialize_best_checkpoint(best_path, {"epoch": 0})

    assert best_path.read_bytes() == b"existing checkpoint"


def test_resume_restores_selection_counter_history_and_sigmoid_configuration():
    checkpoint = {
        "best_val_total": 0.7,
        "selection_state": {
            "best_value": 0.7,
            "best_epoch": 4,
            "epochs_without_improvement": 3,
        },
        "history": [{"epoch": 4, "val_total_loss": 0.7}],
        "config": {
            "base_channels": 16,
            "pad_mode": "exact",
            "log_input": False,
            "mask_mode": "sigmoid",
            "leakage_weight": 1.0,
            "leakage_margin": 0.05,
            "leakage_activity_threshold": 0.001,
        },
    }

    state, history = restore_selection_state(
        checkpoint,
        expected_config=checkpoint["config"],
    )

    assert state == SelectionState(
        best_value=0.7,
        best_epoch=4,
        epochs_without_improvement=3,
    )
    assert history == [{"epoch": 4, "val_total_loss": 0.7}]


def test_resume_rejects_mask_mode_mismatch():
    checkpoint = {
        "best_val_total": 0.7,
        "config": {"mask_mode": "softmax"},
    }

    with pytest.raises(ValueError, match="mask_mode"):
        restore_selection_state(
            checkpoint,
            expected_config={"mask_mode": "sigmoid"},
        )
