from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.audio.stft import compute_istft, compute_stft

from .complex_ops import project_equal_weight_targets, safe_complex_rms


@dataclass(frozen=True)
class ProjectedTargets:
    mixture_stft: torch.Tensor
    vocals_stft: torch.Tensor
    instrumental_stft: torch.Tensor
    vocals_waveform: torch.Tensor
    instrumental_waveform: torch.Tensor


@dataclass(frozen=True)
class LossWeights:
    complex: float = 1.0
    waveform_l1: float = 1.0
    mr_stft: float = 1.0
    leakage: float = 1.0


@dataclass(frozen=True)
class LossResult:
    complex: torch.Tensor
    waveform_l1: torch.Tensor
    mr_stft: torch.Tensor
    leakage: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class LossCalibrationReport:
    weights: LossWeights
    values: dict
    raw_gradient_norms: dict
    weighted_gradient_norms: dict
    complex_share: float


COMPONENTS = ("complex", "waveform_l1", "mr_stft", "leakage")


@torch.no_grad()
def make_projected_targets(mixture, vocals, instrumental):
    with torch.autocast(mixture.device.type, enabled=False):
        x, v, i = (compute_stft(w.float()) for w in (mixture, vocals, instrumental))
        targets = project_equal_weight_targets(x, v, i)
        waves = compute_istft(torch.cat((targets.vocals, targets.instrumental), dim=1),
                             length=mixture.shape[-1]).split(2, dim=1)
        return ProjectedTargets(x, targets.vocals, targets.instrumental, *waves)


class PhaseAwareLoss(nn.Module):
    def __init__(self, resolutions=(512, 1024, 2048)):
        super().__init__()
        self.resolutions = resolutions

    def _multires(self, predictions, targets):
        result = 0
        for n_fft in self.resolutions:
            window = torch.hann_window(n_fft, device=predictions.device, dtype=torch.float32)
            kwargs = dict(n_fft=n_fft, hop_length=n_fft // 4, window=window,
                          return_complex=True, pad_mode="constant")
            p = torch.stft(predictions.flatten(0, 1), **kwargs).abs()
            t = torch.stft(targets.flatten(0, 1), **kwargs).abs()
            result = result + F.l1_loss(p, t) + F.l1_loss(p.log1p(), t.log1p())
        return result / len(self.resolutions)

    def forward(self, output, targets, weights):
        with torch.autocast(output.mixture_stft.device.type, enabled=False):
            norm = safe_complex_rms(targets.mixture_stft, eps=1e-5)
            complex_loss = sum(
                torch.view_as_real((getattr(output, key) - getattr(targets, key)) / norm).abs().mean()
                for key in ("vocals_stft", "instrumental_stft")
            ) * 0.5
            prediction = torch.cat((output.vocals_waveform, output.instrumental_waveform), dim=1).float()
            reference = torch.cat((targets.vocals_waveform, targets.instrumental_waveform), dim=1).float()
            wave_loss = F.l1_loss(prediction, reference)
            multires = self._multires(prediction, reference)
            v = targets.vocals_waveform
            instrument_error = output.instrumental_waveform - targets.instrumental_waveform
            energy = v.square().sum(dim=(1, 2))
            projection = (instrument_error * v).sum(dim=(1, 2)) / energy.clamp_min(1e-8)
            leakage = torch.where(v.square().mean(dim=(1, 2)) > 1e-10, projection.square(), 0).mean()
            values = (complex_loss, wave_loss, multires, leakage)
            total = sum(value * getattr(weights, key) for key, value in zip(COMPONENTS, values))
        return LossResult(*values, total)


def calibrate_loss_weights(model, fixed_batches, target_ratios=None):
    ratios = target_ratios or {"waveform_l1": 0.25, "mr_stft": 0.25, "leakage": 0.10}
    values = dict.fromkeys(COMPONENTS, 0.0)
    norms = values.copy()
    criterion = PhaseAwareLoss()
    params = tuple(model.refinement.final.parameters())
    count = 0
    for mixture, vocals, instrumental in fixed_batches:
        targets = make_projected_targets(mixture, vocals, instrumental)
        output = model.forward_stft(targets.mixture_stft, mixture.shape[-1])
        result = criterion(output, targets, LossWeights())
        for index, name in enumerate(COMPONENTS):
            value = getattr(result, name)
            gradients = torch.autograd.grad(value, params, retain_graph=index < len(COMPONENTS) - 1)
            norms[name] += float(sum(g.detach().square().sum() for g in gradients).sqrt())
            values[name] += float(value.detach())
        count += 1
    if count == 0:
        raise ValueError("Loss calibration requires fixed segments")
    for name in COMPONENTS:
        norms[name] /= count
        values[name] /= count
    if not all(0 < value < float("inf") for value in norms.values()):
        raise FloatingPointError(f"Invalid calibration gradient norms: {norms}")
    weights = LossWeights(**{name: min(10.0, max(1e-4, ratio * norms["complex"] / norms[name]))
                             for name, ratio in ratios.items()})
    weighted = {name: value * getattr(weights, name) for name, value in norms.items()}
    return LossCalibrationReport(weights, values, norms, weighted, weighted["complex"] / sum(weighted.values()))
