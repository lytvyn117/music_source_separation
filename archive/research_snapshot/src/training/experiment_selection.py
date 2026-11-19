from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch


SELECTION_METRIC = "val_total_loss"


@dataclass(frozen=True)
class SelectionState:
    best_value: float
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "best_value": float(self.best_value),
            "best_epoch": int(self.best_epoch),
            "epochs_without_improvement": int(self.epochs_without_improvement),
        }


def selection_value(validation: Mapping[str, float]) -> float:
    """Return the documented full training objective used for model selection."""
    if SELECTION_METRIC not in validation:
        raise KeyError(f"Validation results do not contain {SELECTION_METRIC!r}")
    value = float(validation[SELECTION_METRIC])
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"Selection value must be finite, got {value}")
    return value


def update_selection(
    state: SelectionState,
    current_value: float,
    epoch: int,
    min_delta: float,
) -> Tuple[SelectionState, bool]:
    """Apply the single improvement rule shared by best saving and early stopping."""
    improved = float(current_value) < (state.best_value - float(min_delta))
    if improved:
        return SelectionState(float(current_value), int(epoch), 0), True
    return SelectionState(
        state.best_value,
        state.best_epoch,
        state.epochs_without_improvement + 1,
    ), False


def should_stop(state: SelectionState, patience: int) -> bool:
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    return state.epochs_without_improvement >= patience


def initialize_best_checkpoint(path: Path, payload: Mapping) -> None:
    """Persist the initial model as a valid best checkpoint without overwriting."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def restore_selection_state(
    checkpoint: Mapping,
    expected_config: Mapping,
) -> Tuple[SelectionState, List[Dict]]:
    """Restore selection/history state and reject incompatible resume settings."""
    checkpoint_config = checkpoint.get("config", {})
    for key, expected in expected_config.items():
        if key in checkpoint_config and checkpoint_config[key] != expected:
            raise ValueError(
                f"Resume configuration mismatch for {key}: "
                f"checkpoint={checkpoint_config[key]!r}, requested={expected!r}"
            )

    raw_state = checkpoint.get("selection_state")
    if raw_state is None:
        best_value = float(
            checkpoint.get(
                "best_val_total",
                checkpoint.get("best_val_composite", float("inf")),
            )
        )
        state = SelectionState(best_value=best_value)
    else:
        state = SelectionState(
            best_value=float(raw_state["best_value"]),
            best_epoch=int(raw_state.get("best_epoch", 0)),
            epochs_without_improvement=int(
                raw_state.get("epochs_without_improvement", 0)
            ),
        )

    history = [dict(row) for row in checkpoint.get("history", [])]
    return state, history
