"""Measure rounding on the oracle coefficients without changing the gate."""

import numpy as np
import torch

from .complex_ops import complex_additivity, project_equal_weight_targets


class ProjectionDiagnostics:
    """Collect full-bin maxima and deterministic samples of active-bin errors."""

    def __init__(self):
        self.maxima = {}
        self.active_bins = 0
        self.over_limit = 0
        self.violations = 0
        self.bins = 0
        self.high_energy_bins = 0
        self.active_samples = []

    def add(self, x, vocals, instrumental, projected):
        v, i = projected.vocals, projected.instrumental
        error, bound = complex_additivity(x, v, i)
        ratio = error / bound
        regrouped = (x - (vocals + instrumental)) * 0.5
        wide_x = x.to(torch.complex128)
        wide = project_equal_weight_targets(wide_x, vocals.to(wide_x.dtype), instrumental.to(wide_x.dtype))
        magnitude = x.abs()
        scale = torch.maximum(magnitude, v.abs() + i.abs())
        peak = magnitude.amax(dim=(-2, -1), keepdim=True)
        active = (magnitude > 0) & (magnitude >= 1e-3 * peak)
        high = active & (magnitude >= 0.1 * peak)
        metrics = {
            "fp32_max_abs": error.amax(),
            "max_error_to_bound": ratio.amax(),
            "quiet_max_error_to_bound": ratio.masked_fill(active, 0).amax(),
            "high_energy_max_error_to_bound": ratio.masked_fill(~high, 0).amax(),
            "fp32_regrouped_max_abs": ((vocals + regrouped) + (instrumental + regrouped) - x).abs().amax(),
            "fp32_complement_max_abs": (v + (x - v) - x).abs().amax(),
            "fp32_widened_sum_max_abs": (v.to(wide_x.dtype) + i.to(wide_x.dtype) - wide_x).abs().amax(),
            "fp64_projection_max_abs": (wide.vocals + wide.instrumental - wide_x).abs().amax(),
            "fp64_then_cast_max_abs": (wide.vocals.to(x.dtype) + wide.instrumental.to(x.dtype) - x).abs().amax(),
            "mixture_spectrum_max_abs": magnitude.amax(),
            "fp32_scale_normalized_max": (error / scale.clamp_min(1e-30)).amax(),
        }
        values = torch.stack([*metrics.values(), active.sum(), (error > 1e-5).sum(),
                              (~torch.isfinite(ratio) | (error > bound)).sum(), high.sum()]).tolist()
        for key, value in zip(metrics, values):
            self.maxima[key] = max(self.maxima.get(key, 0.0), value)
        self.active_bins += int(values[-4])
        self.over_limit += int(values[-3])
        self.violations += int(values[-2])
        self.high_energy_bins += int(values[-1])
        self.bins += x.numel()

        stride = 257
        sampled = active.flatten()[::stride]
        sampled_error = error.flatten()[::stride][sampled]
        sampled_mag = magnitude.flatten()[::stride][sampled]
        sampled_scale = scale.flatten()[::stride][sampled]
        ulp = torch.nextafter(sampled_scale, torch.full_like(sampled_scale, float("inf"))) - sampled_scale
        self.active_samples.append(torch.stack((sampled_error, sampled_error / sampled_mag,
                                                sampled_error / ulp)).cpu().numpy())

    def summary(self):
        result = {**self.maxima, "active_bins": self.active_bins, "fp32_bins_over_1e_5": self.over_limit,
                  "complex_violations": self.violations, "bins": self.bins,
                  "quiet_bins": self.bins - self.active_bins, "high_energy_bins": self.high_energy_bins}
        samples = np.concatenate(self.active_samples, axis=1)
        result["sampled_active_bins"] = samples.shape[1]
        for name, values in zip(("abs", "relative", "ulp"), samples):
            for quantile in (50, 95, 99, 100):
                result[f"fp32_active_{name}_p{quantile}"] = (
                    float(np.percentile(values, quantile)) if values.size else None
                )
        return result
