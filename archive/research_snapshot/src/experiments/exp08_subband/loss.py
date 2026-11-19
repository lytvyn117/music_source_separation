"""Frozen Exp08 objective with common mixture scales and one shared error."""
import torch

from .model import spectral_scale

LOSS_WEIGHTS = {
    "complex_mse": 1.0, "log_magnitude": .5, "waveform": 2.0,
    "stereo_side": .25, "vocal_leakage": .25,
}


def project_vocals(mixture, vocals, instrumental):
    return vocals + .5 * (mixture - vocals - instrumental)


def normalized_mse(error, mixture):
    normalized = torch.view_as_real(error / spectral_scale(mixture))
    return normalized.square().sum(dim=-1).mean()


def _one_second_frames(wave, rate):
    length = wave.shape[-1] // rate * rate
    return wave[..., :length].reshape(wave.shape[0], wave.shape[1], length // rate, rate)


def active_vocal_frames(vocals, rate=44100):
    frames = _one_second_frames(vocals, rate)
    return frames.square().mean(dim=(1, 3)) > .005**2


def vocal_leakage(inst_error, projected_vocals, rate=44100, denominator=None):
    vocals = _one_second_frames(projected_vocals, rate)
    errors = _one_second_frames(inst_error, rate)
    energy = vocals.square().sum(dim=(1, 3))
    active = energy > vocals.shape[1] * rate * .005**2
    projection = (errors * vocals).sum(dim=(1, 3)) / energy.clamp_min(1e-8)
    total = ((projection.abs() - .05).relu().square() * active).sum()
    count = active.sum() if denominator is None else denominator
    return total / count.clamp_min(1)


def composite_loss(mixture, vocals, instrumental, projected_vocals,
                   mix_wave, vocal_wave, target_vocal_wave, rate=44100,
                   effective_active=None, accumulation=1):
    with torch.autocast(mixture.device.type, enabled=False):
        scale = spectral_scale(mixture)
        error = (vocals - projected_vocals) / scale
        spectral = torch.view_as_real(error).square().sum(dim=-1).mean()
        mag_v = (vocals.abs() / scale).log1p() - (projected_vocals.abs() / scale).log1p()
        mag_i = (instrumental.abs() / scale).log1p() - ((mixture - projected_vocals).abs() / scale).log1p()
        wave_scale = mix_wave.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).sqrt()
        wave_error = vocal_wave - target_vocal_wave
        normalized_wave = wave_error / wave_scale
        terms = {
            "complex_mse": spectral,
            "log_magnitude": .5 * (mag_v.abs().mean() + mag_i.abs().mean()),
            "waveform": normalized_wave.abs().mean(),
            "stereo_side": (.5 * (normalized_wave[:, 0] - normalized_wave[:, 1])).abs().mean(),
            "vocal_leakage": accumulation * vocal_leakage(
                -wave_error, target_vocal_wave, rate, effective_active),
        }
        return sum(LOSS_WEIGHTS[name] * value for name, value in terms.items()), terms
