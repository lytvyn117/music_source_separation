"""Squared complex loss for the isolated learnability probe."""

import torch
from torch.nn import functional as F

from .complex_ops import safe_complex_rms
from .losses import LossResult, PhaseAwareLoss


def normalized_complex_mse(mixture, vocal_error, instrumental_error):
    """Mean square per real component, channel, bin, and source in FP32."""
    with torch.autocast(mixture.device.type, enabled=False):
        scale = safe_complex_rms(mixture.to(torch.complex64), eps=1e-5)
        errors = (vocal_error, instrumental_error)
        return 0.5 * sum(torch.view_as_real(error.to(torch.complex64) / scale).square().mean()
                         for error in errors)


class MSEProbeLoss(PhaseAwareLoss):
    def forward(self, output, targets, weights):
        with torch.autocast(output.mixture_stft.device.type, enabled=False):
            spectral = normalized_complex_mse(targets.mixture_stft,
                output.vocals_stft - targets.vocals_stft,
                output.instrumental_stft - targets.instrumental_stft)
            prediction = torch.cat((output.vocals_waveform, output.instrumental_waveform), dim=1).float()
            reference = torch.cat((targets.vocals_waveform, targets.instrumental_waveform), dim=1).float()
            waveform = F.l1_loss(prediction, reference)
            multires = self._multires(prediction, reference)
            vocal = targets.vocals_waveform
            error = output.instrumental_waveform - targets.instrumental_waveform
            energy = vocal.square().sum(dim=(1, 2))
            projection = (error * vocal).sum(dim=(1, 2)) / energy.clamp_min(1e-8)
            leakage = torch.where(vocal.square().mean(dim=(1, 2)) > 1e-10, projection.square(), 0).mean()
            total = (weights.complex * spectral + weights.waveform_l1 * waveform
                     + weights.mr_stft * multires + weights.leakage * leakage)
            return LossResult(spectral, waveform, multires, leakage, total)
