import pytest
import torch
from torch import nn

from src.audio.stft import compute_istft
from src.experiments.exp07_phase_aware_residual.complex_ops import complex_additivity
from src.experiments.exp07_phase_aware_residual.config import Exp07Paths
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2)

    def forward(self, magnitude):
        score = self.bn(magnitude)
        return torch.stack((score.sigmoid(), (-score).sigmoid()), dim=1)


def test_zero_projection_preserves_fp32_exp04_outputs():
    path = Exp07Paths().backbone
    if not path.is_file():
        pytest.skip("Local Exp04 checkpoint required")
    model = PhaseAwareResidualSeparator.from_exp04(path, "cpu").train()
    snapshot = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    samples = torch.randn(1, 2, 20003, generator=torch.Generator().manual_seed(42)) * 0.1
    output = model(samples)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 100000
    assert torch.count_nonzero(output.residual_stft) == 0
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
    expected = model.backbone(output.mixture_stft.abs())
    torch.testing.assert_close(output.masks, expected, rtol=0, atol=0)
    for mask, wave in zip(expected.unbind(1), (output.vocals_waveform, output.instrumental_waveform)):
        direct = compute_istft(output.mixture_stft * mask, length=samples.shape[-1])
        torch.testing.assert_close(wave, direct, rtol=0, atol=1e-6)
    error, bound = complex_additivity(output.mixture_stft, output.vocals_stft, output.instrumental_stft)
    assert (error <= bound).all()
    torch.testing.assert_close(output.vocals_waveform + output.instrumental_waveform, samples, rtol=0, atol=1e-6)
    optimizer = torch.optim.AdamW(model.refinement.parameters(), lr=1e-4)
    output.vocals_waveform.square().mean().backward()
    optimizer.step()
    assert torch.count_nonzero(model.refinement.final.weight) > 0
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)


def test_silence_preserves_stereo_shape_and_is_finite():
    model = PhaseAwareResidualSeparator(SmallBackbone())
    output = model(torch.zeros(1, 2, 321))
    assert output.vocals_waveform.shape == (1, 2, 321)
    assert output.vocals_stft.dtype == torch.complex64
    assert torch.count_nonzero(output.vocals_waveform) == 0
    assert torch.count_nonzero(output.instrumental_waveform) == 0
    with pytest.raises(ValueError, match="stereo"):
        model(torch.zeros(1, 1, 321))
