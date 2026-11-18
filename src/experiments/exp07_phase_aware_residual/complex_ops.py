from dataclasses import dataclass

import numpy as np
import torch

from .config import NumericalTolerances


def complex_additivity(mixture, vocals, instrumental, tolerances=NumericalTolerances()):
    """Return the FP32 complex residual magnitude and its per-bin bound."""
    x, v, i = (value.to(torch.complex64) for value in (mixture, vocals, instrumental))
    error = (x - (v + i)).abs()
    bound = tolerances.complex_atol + tolerances.complex_rtol * (x.abs() + v.abs() + i.abs())
    return error, bound


@dataclass(frozen=True)
class ProjectedComplexTargets:
    vocals: torch.Tensor
    instrumental: torch.Tensor


def project_equal_weight_targets(mixture, vocals, instrumental) -> ProjectedComplexTargets:
    if mixture.shape != vocals.shape or mixture.shape != instrumental.shape:
        raise ValueError("Mixture and source spectra must have matching shapes")
    residual_half = (mixture - vocals - instrumental) * 0.5
    return ProjectedComplexTargets(vocals + residual_half, instrumental + residual_half)


def complex_to_channels(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value.real, value.imag), dim=1)


def channels_to_complex(value: torch.Tensor) -> torch.Tensor:
    real, imag = value.float().chunk(2, dim=1)
    return torch.complex(real, imag)


def safe_complex_rms(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return value.abs().square().mean(dim=(-2, -1), keepdim=True).clamp_min(eps**2).sqrt()


def waveform_nrmse_pct(estimate, reference, eps: float = 1e-12) -> float:
    if isinstance(reference, torch.Tensor):
        reference = reference.to(torch.float64)
        error_energy = (estimate - reference).square().sum()
        return float(100 * (error_energy / reference.square().sum().clamp_min(eps)).sqrt())
    reference = np.asarray(reference, dtype=np.float64)
    difference = estimate - reference
    return float(100 * np.sqrt(np.sum(difference**2) / max(np.sum(reference**2), eps)))


def make_vocal_ablation_stfts(initial_vocal, refined_vocal) -> dict[str, torch.Tensor]:
    return {
        "full": refined_vocal,
        "magnitude_only": torch.polar(refined_vocal.abs().float(), initial_vocal.angle().float()),
        "phase_only": torch.polar(initial_vocal.abs().float(), refined_vocal.angle().float()),
    }
