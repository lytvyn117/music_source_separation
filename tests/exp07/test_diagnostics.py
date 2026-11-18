import torch

from src.experiments.exp07_phase_aware_residual.diagnostics import compute_residual_diagnostics
from src.experiments.exp07_phase_aware_residual.model import SeparationOutput


def test_reports_clipping_and_complex_error_without_rescaling():
    x = torch.tensor([[[[1 + 0j, 0j]], [[0j, 0j]]]])
    wave = torch.tensor([[[0.5, 0], [0, 0]]])
    loud = wave * 3
    out = SeparationOutput(x, None, x, x, x * 2, -x, loud, wave - loud)
    report = compute_residual_diagnostics(out, wave)
    assert report["clipped_sample_pct"] == 12.5
    assert report["spectral_amplification_active_bin_count"] == 1
    assert report["complex_violations"] == 0
    assert report["waveform_max_abs"] == 0
    assert out.vocals_waveform.max() == 1.5
