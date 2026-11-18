import pytest
import torch

from src.experiments.exp07_phase_aware_residual.complex_ops import project_equal_weight_targets
from src.experiments.exp07_phase_aware_residual.oracle_precision import ProjectionDiagnostics


def test_reports_one_ulp_fp32_projection_error_without_changing_gate():
    x = torch.tensor([[[[128.00001525878906 + 0j]]]], dtype=torch.complex64)
    v, i = torch.full_like(x, 128), torch.zeros_like(x)
    projected = project_equal_weight_targets(x, v, i)
    assert projected.vocals.dtype == torch.complex64

    diagnostic = ProjectionDiagnostics()
    diagnostic.add(x, v, i, projected)
    result = diagnostic.summary()

    assert result["fp32_max_abs"] == 2**-16
    assert result["fp32_bins_over_1e_5"] == 1
    assert result["fp32_active_ulp_p99"] == 1
    assert result["fp32_active_relative_p99"] == pytest.approx((2**-16) / x.real.item())
    assert result["fp64_projection_max_abs"] == 0
    assert result["fp64_then_cast_max_abs"] == 2**-16


def test_silent_bins_are_not_in_active_error_quantiles():
    x = torch.zeros((1, 2, 8, 9), dtype=torch.complex64)
    diagnostic = ProjectionDiagnostics()
    diagnostic.add(x, x, x, project_equal_weight_targets(x, x, x))
    result = diagnostic.summary()
    assert result["active_bins"] == 0
    assert result["fp32_active_relative_p99"] is None
    assert result["fp32_max_abs"] == 0


@pytest.mark.parametrize("x,v,i,violations", [
    (0j, 2e-5j, 0j, 1),
    (128 + 0j, 64 + 0.01j, 64 + 0j, 1),
    (128.00001525878906 + 0j, 128 + 0j, 0j, 0),
])
def test_approved_bound_rejects_real_perturbations(x, v, i, violations):
    from src.experiments.exp07_phase_aware_residual.complex_ops import complex_additivity
    x, v, i = [torch.tensor([value], dtype=torch.complex64) for value in (x, v, i)]
    error, bound = complex_additivity(x, v, i)
    expected = 1e-5 + 4 * 2**-24 * (abs(x.item()) + abs(v.item()) + abs(i.item()))
    assert bound.item() == pytest.approx(expected)
    assert (error > bound).sum().item() == violations
