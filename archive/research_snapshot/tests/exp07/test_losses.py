import torch

from src.experiments.exp07_phase_aware_residual.complex_ops import complex_additivity
from src.experiments.exp07_phase_aware_residual.losses import PhaseAwareLoss, LossWeights, make_projected_targets
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator
from tests.exp07.test_model import SmallBackbone


def test_projected_targets_reconstruct_original_mixture():
    generator = torch.Generator().manual_seed(42)
    v = torch.randn(1, 2, 4096, generator=generator) * 0.05
    i = torch.randn(1, 2, 4096, generator=generator) * 0.05
    x = v + i + 0.01
    t = make_projected_targets(x, v, i)
    error, bound = complex_additivity(t.mixture_stft, t.vocals_stft, t.instrumental_stft)
    assert (error <= bound).all()
    torch.testing.assert_close(t.vocals_waveform, v + 0.005, atol=1e-6, rtol=0)
    assert PhaseAwareLoss()(t, t, LossWeights()).total.item() == 0


def test_gradient_reaches_upstream_only_after_final_projection_moves():
    model = PhaseAwareResidualSeparator(SmallBackbone())
    generator = torch.Generator().manual_seed(8)
    vocals = torch.randn(1, 2, 4096, generator=generator) * 0.03
    instrumental = torch.randn(1, 2, 4096, generator=generator) * 0.12
    target = make_projected_targets(vocals + instrumental, vocals, instrumental)
    optimizer = torch.optim.AdamW(model.refinement.parameters(), lr=1e-4)
    loss = PhaseAwareLoss()
    for step in (0, 1):
        optimizer.zero_grad()
        prediction = model.forward_stft(target.mixture_stft, 4096)
        objective = loss(prediction, target, LossWeights()).total
        objective.backward()
        assert torch.isfinite(objective)
        assert model.refinement.final.weight.grad.abs().max() > 0
        upstream = model.refinement.stem[0].weight.grad
        assert torch.isfinite(upstream).all()
        assert bool(upstream.abs().max() > 0) == bool(step)
        optimizer.step()


def test_silent_vocal_reference_gives_zero_leakage():
    noise = torch.randn(1, 2, 4096) * 0.03
    target = make_projected_targets(noise, torch.zeros_like(noise), noise)
    model = PhaseAwareResidualSeparator(SmallBackbone())
    output = model(noise)
    assert PhaseAwareLoss()(output, target, LossWeights()).leakage.item() == 0
