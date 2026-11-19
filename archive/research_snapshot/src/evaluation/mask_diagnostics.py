import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch


FrequencyBand = Tuple[str, float, float]

DEFAULT_FREQUENCY_BANDS: Tuple[FrequencyBand, ...] = (
    ("sub_bass_20_80_hz", 20.0, 80.0),
    ("bass_80_250_hz", 80.0, 250.0),
    ("low_mid_250_1000_hz", 250.0, 1000.0),
    ("vocal_mid_1000_4000_hz", 1000.0, 4000.0),
    ("presence_4000_8000_hz", 4000.0, 8000.0),
    ("air_8000_22050_hz", 8000.0, 22050.0001),
)


def frequency_band_indices(
    n_fft: int,
    sample_rate: int,
    bands: Sequence[FrequencyBand] = DEFAULT_FREQUENCY_BANDS,
) -> Dict[str, torch.Tensor]:
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    result: Dict[str, torch.Tensor] = {}
    for index, (name, low_hz, high_hz) in enumerate(bands):
        is_last = index == len(bands) - 1
        upper = frequencies <= high_hz if is_last else frequencies < high_hz
        result[name] = (frequencies >= low_hz) & upper
    return result


def classify_tf_regions(
    vocals_magnitude: torch.Tensor,
    instrumental_magnitude: torch.Tensor,
    dominance_threshold: float = 0.8,
    activity_threshold: float = 1e-3,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    vocals = vocals_magnitude.float()
    instrumental = instrumental_magnitude.float()
    energy = vocals + instrumental
    peak = energy.amax(dim=(-2, -1), keepdim=True)
    active = energy > (activity_threshold * peak).clamp_min(eps)
    vocal_ratio = vocals / energy.clamp_min(eps)
    vocal_dominant = active & (vocal_ratio >= dominance_threshold)
    instrumental_dominant = active & (vocal_ratio <= (1.0 - dominance_threshold))
    overlap = active & ~vocal_dominant & ~instrumental_dominant
    return {
        "active": active,
        "vocal_dominant": vocal_dominant,
        "instrumental_dominant": instrumental_dominant,
        "overlap": overlap,
    }


def _tensor_distribution(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p10": float(torch.quantile(values, 0.10).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "max": float(values.max().item()),
    }


def mask_diagnostics(
    masks: torch.Tensor,
    vocals_magnitude: torch.Tensor,
    instrumental_magnitude: torch.Tensor,
    dominance_threshold: float = 0.8,
    activity_threshold: float = 1e-3,
    eps: float = 1e-8,
) -> Dict:
    if masks.ndim != 5 or masks.shape[1] != 2:
        raise ValueError(f"Expected masks shaped (B, 2, C, F, T), got {tuple(masks.shape)}")

    tensors = (masks, vocals_magnitude, instrumental_magnitude)
    nonfinite = sum(int((~torch.isfinite(tensor)).sum().item()) for tensor in tensors)
    if nonfinite:
        return {"finite": False, "nonfinite_values": nonfinite}

    vocals = vocals_magnitude.float()
    instrumental = instrumental_magnitude.float()
    energy = vocals + instrumental
    irm_vocal = vocals / energy.clamp_min(eps)
    irm_instrumental = instrumental / energy.clamp_min(eps)
    regions = classify_tf_regions(
        vocals,
        instrumental,
        dominance_threshold=dominance_threshold,
        activity_threshold=activity_threshold,
        eps=eps,
    )
    active = regions["active"]
    if not bool(active.any()):
        return {"finite": True, "nonfinite_values": 0, "active_bins": 0, "regions": {}}

    vocal_mask = masks[:, 0].float()
    instrumental_mask = masks[:, 1].float()
    mask_sum = vocal_mask + instrumental_mask
    active_vocal_mask = vocal_mask[active]
    active_instrumental_mask = instrumental_mask[active]
    active_mask_sum = mask_sum[active]

    region_stats: Dict[str, Dict[str, float]] = {}
    for name in ("vocal_dominant", "instrumental_dominant", "overlap"):
        selected = regions[name]
        count = int(selected.sum().item())
        if count == 0:
            region_stats[name] = {"count": 0}
            continue
        region_energy = energy[selected].sum().clamp_min(eps).item()
        region_stats[name] = {
            "count": count,
            "vocal_mask_irm_mae": float(
                torch.abs(vocal_mask[selected] - irm_vocal[selected]).mean().item()
            ),
            "instrumental_mask_irm_mae": float(
                torch.abs(
                    instrumental_mask[selected] - irm_instrumental[selected]
                ).mean().item()
            ),
            "vocal_prediction_nmae": float(
                torch.abs((energy * vocal_mask)[selected] - vocals[selected]).sum().item()
                / region_energy
            ),
            "instrumental_prediction_nmae": float(
                torch.abs(
                    (energy * instrumental_mask)[selected] - instrumental[selected]
                ).sum().item()
                / region_energy
            ),
        }

    return {
        "finite": True,
        "nonfinite_values": 0,
        "active_bins": int(active.sum().item()),
        "vocal_mask_distribution": _tensor_distribution(active_vocal_mask),
        "instrumental_mask_distribution": _tensor_distribution(
            active_instrumental_mask
        ),
        "mask_sum_mean": float(active_mask_sum.mean().item()),
        "mask_sum_std": float(active_mask_sum.std(unbiased=False).item()),
        "mask_product_mean": float(
            (active_vocal_mask * active_instrumental_mask).mean().item()
        ),
        "both_masks_gt_0_5_fraction": float(
            ((active_vocal_mask > 0.5) & (active_instrumental_mask > 0.5))
            .float()
            .mean()
            .item()
        ),
        "vocal_mask_irm_mae": float(
            torch.abs(vocal_mask[active] - irm_vocal[active]).mean().item()
        ),
        "instrumental_mask_irm_mae": float(
            torch.abs(
                instrumental_mask[active] - irm_instrumental[active]
            ).mean().item()
        ),
        "regions": region_stats,
    }


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _gain_db(output_rms: float, input_rms: float, eps: float) -> float:
    return float(20.0 * math.log10((output_rms + eps) / (input_rms + eps)))


def waveform_artifact_metrics(
    mixture: np.ndarray,
    vocals_estimate: np.ndarray,
    instrumental_estimate: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    mixture = np.asarray(mixture, dtype=np.float64)
    vocals = np.asarray(vocals_estimate, dtype=np.float64)
    instrumental = np.asarray(instrumental_estimate, dtype=np.float64)
    if mixture.shape != vocals.shape or mixture.shape != instrumental.shape:
        raise ValueError(
            f"Waveform shapes must match, got {mixture.shape}, {vocals.shape}, "
            f"and {instrumental.shape}"
        )

    nonfinite = int(
        (~np.isfinite(mixture)).sum()
        + (~np.isfinite(vocals)).sum()
        + (~np.isfinite(instrumental)).sum()
    )
    if nonfinite:
        return {"finite": False, "nonfinite_values": nonfinite}

    mixture_rms = _rms(mixture)
    vocals_rms = _rms(vocals)
    instrumental_rms = _rms(instrumental)
    combined = vocals + instrumental
    residual = combined - mixture
    residual_ratio = _rms(residual) / max(mixture_rms, eps)
    denominator = float(np.linalg.norm(vocals.reshape(-1)) * np.linalg.norm(instrumental.reshape(-1)))
    waveform_cosine = (
        float(np.dot(vocals.reshape(-1), instrumental.reshape(-1)) / denominator)
        if denominator > eps
        else 0.0
    )

    return {
        "finite": True,
        "nonfinite_values": 0,
        "vocals_gain_db": _gain_db(vocals_rms, mixture_rms, eps),
        "instrumental_gain_db": _gain_db(instrumental_rms, mixture_rms, eps),
        "combined_gain_db": _gain_db(_rms(combined), mixture_rms, eps),
        "mixture_reconstruction_residual_ratio": float(residual_ratio),
        "mixture_reconstruction_residual_db": float(
            20.0 * math.log10(max(residual_ratio, eps))
        ),
        "source_waveform_cosine": waveform_cosine,
    }


def spectral_band_metrics(
    vocals_reference_stft: torch.Tensor,
    instrumental_reference_stft: torch.Tensor,
    vocals_estimate_stft: torch.Tensor,
    instrumental_estimate_stft: torch.Tensor,
    n_fft: int,
    sample_rate: int,
    bands: Sequence[FrequencyBand] = DEFAULT_FREQUENCY_BANDS,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    shapes = {
        tuple(tensor.shape)
        for tensor in (
            vocals_reference_stft,
            instrumental_reference_stft,
            vocals_estimate_stft,
            instrumental_estimate_stft,
        )
    }
    if len(shapes) != 1:
        raise ValueError(f"All STFT tensors must have the same shape, got {shapes}")

    indices = frequency_band_indices(n_fft, sample_rate, bands)
    output: Dict[str, Dict[str, float]] = {}
    for name, band_mask_cpu in indices.items():
        band_mask = band_mask_cpu.to(vocals_reference_stft.device)
        vocal_ref = vocals_reference_stft[..., band_mask, :]
        inst_ref = instrumental_reference_stft[..., band_mask, :]
        vocal_est = vocals_estimate_stft[..., band_mask, :]
        inst_est = instrumental_estimate_stft[..., band_mask, :]
        vocal_error = vocal_est - vocal_ref
        inst_error = inst_est - inst_ref
        vocal_energy = float(torch.sum(torch.abs(vocal_ref) ** 2).item())
        vocal_error_energy = float(torch.sum(torch.abs(vocal_error) ** 2).item())
        projection = float(
            torch.real(torch.sum(torch.conj(vocal_ref) * inst_error)).item()
        )
        bleed_coefficient = projection / max(vocal_energy, eps)
        output[name] = {
            "frequency_bins": int(band_mask.sum().item()),
            "vocals_reference_energy": vocal_energy,
            "vocals_error_energy": vocal_error_energy,
            "vocals_sdr_db": float(
                10.0 * math.log10((vocal_energy + eps) / (vocal_error_energy + eps))
            ),
            "vocal_to_instrumental_bleed_coefficient": bleed_coefficient,
            "vocal_to_instrumental_projection": projection,
            "vocal_to_instrumental_bleed_energy_pct": float(
                bleed_coefficient * bleed_coefficient * 100.0
            ),
        }
    return output
