import pytest
import torch

from src.experiments.exp07_phase_aware_residual.config import Exp07Config
from src.experiments.exp07_phase_aware_residual.losses import LossWeights, PhaseAwareLoss, make_projected_targets
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator
from src.experiments.exp07_phase_aware_residual.trainer import configure_optimizer, training_step
from tests.exp07.test_model import SmallBackbone


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_head_has_finite_cuda_gradients_and_fp32_spectra():
    model = PhaseAwareResidualSeparator(SmallBackbone()).cuda()
    model.use_bf16 = torch.cuda.is_bf16_supported()
    optimizer, _ = configure_optimizer(model, Exp07Config())
    v, i = (torch.randn(4, 2, 4096, device="cuda") * gain for gain in (0.05, 0.1))
    targets = make_projected_targets(v + i, v, i)
    assert model.forward_stft(targets.mixture_stft).residual_stft.dtype == torch.complex64
    criterion = PhaseAwareLoss()
    first = training_step(model, optimizer, criterion, LossWeights(), v + i, targets)
    second = training_step(model, optimizer, criterion, LossWeights(), v + i, targets)
    assert first["final_gradient_norm"] > 0
    assert first["upstream_gradient_norm"] == 0
    assert second["upstream_gradient_norm"] > 0
    assert second["complex_violations"] == 0
