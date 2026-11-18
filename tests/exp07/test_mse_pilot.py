import pytest

from src.experiments.exp07_phase_aware_residual.mse_pilot import PilotSafetyStop, enforce_safety, stop_reason


def test_pilot_stops_at_approved_amplification_and_clipping_limits():
    safe = {"clipped_sample_pct": 0.1, "active_amplification_change_db": 3.,
            "high_amplification_change_db": 2., "complex_violations": 0}
    enforce_safety(safe, 10.5)
    for row in (safe | {"clipped_sample_pct": .1001}, safe | {"active_amplification_change_db": 3.01},
                safe | {"high_amplification_change_db": 3.01}, safe | {"complex_violations": 1},
                safe | {"clipped_sample_pct": float("nan")}):
        with pytest.raises(PilotSafetyStop):
            enforce_safety(row, 10.5)
    with pytest.raises(PilotSafetyStop):
        enforce_safety(safe, 10.501)


def test_epoch_budget_and_patience_are_independent_stop_conditions():
    assert stop_reason(8, 300, 151) == "early_stopping"
    assert stop_reason(0, 10_700, 151) == "runtime_budget"
    assert stop_reason(7, 100, 151) is None
