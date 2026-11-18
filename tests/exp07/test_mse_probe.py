import torch

from src.experiments.exp07_phase_aware_residual.losses import LossWeights, PhaseAwareLoss, make_projected_targets
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator
from src.experiments.exp07_phase_aware_residual.mse_probe_loss import MSEProbeLoss, normalized_complex_mse
from tests.exp07.test_model import SmallBackbone


def test_complex_mse_uses_both_components_and_per_channel_mixture_scale():
    x = torch.tensor([2 + 0j, 4 + 0j]).reshape(1, 2, 1, 1)
    d = torch.tensor([2 + 2j, 4 + 4j]).reshape_as(x).requires_grad_()
    loss = normalized_complex_mse(x, d, -d)
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(d.grad).all()
    torch.testing.assert_close(normalized_complex_mse(x * 3, d * 3, -d * 3), loss)


def test_silent_bins_have_finite_zero_loss_and_gradient():
    x = torch.zeros((1, 2, 2, 2), dtype=torch.complex64, requires_grad=True)
    loss = normalized_complex_mse(x.detach(), x, -x)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(x.grad) == 0


def test_mse_variant_leaves_the_three_auxiliaries_identical():
    gen = torch.Generator().manual_seed(42)
    v, i = [torch.randn(1, 2, 4096, generator=gen) * scale for scale in (0.1, 0.2)]
    model = PhaseAwareResidualSeparator(SmallBackbone())
    x = v + i
    targets = make_projected_targets(x, v, i)
    output = model(x)
    prior = PhaseAwareLoss()(output, targets, LossWeights())
    revised = MSEProbeLoss()(output, targets, LossWeights())
    for component in ("waveform_l1", "mr_stft", "leakage"):
        torch.testing.assert_close(getattr(prior, component), getattr(revised, component), rtol=0, atol=0)
