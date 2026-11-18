import torch

from .complex_ops import complex_additivity, waveform_nrmse_pct


def _gain(estimate, reference):
    return float(10 * torch.log10((estimate.square().mean() + 1e-12) /
                                  (reference.square().mean() + 1e-12)))


@torch.no_grad()
def compute_residual_diagnostics(output, mixture, targets=None):
    x, v0, correction = output.mixture_stft, output.initial_vocal_stft, output.residual_stft
    v, i = output.vocals_stft, output.instrumental_stft
    waves = torch.cat((output.vocals_waveform, output.instrumental_waveform), dim=1)
    error, bound = complex_additivity(x, v, i)
    summed = output.vocals_waveform + output.instrumental_waveform
    stats = {
        "correction_magnitude_ratio": float(correction.abs().mean() / v0.abs().mean().clamp_min(1e-12)),
        "correction_energy_ratio": float(correction.abs().square().mean() / v0.abs().square().mean().clamp_min(1e-12)),
        "complex_max_abs": float(error.max()),
        "complex_max_error_to_bound": float((error / bound).max()),
        "complex_violations": int((error > bound).sum()),
        "waveform_max_abs": float((summed - mixture).abs().max()),
        "waveform_nrmse_pct": waveform_nrmse_pct(summed, mixture),
        "waveform_peak": float(waves.abs().max()),
        "clipped_samples": int((waves.abs() > 1).sum()),
        "clipped_sample_pct": float((waves.abs() > 1).float().mean() * 100),
        "mixture_output_gain_db": _gain(summed, mixture),
        "nonfinite_count": sum(int((~torch.isfinite(t)).sum()) for t in (correction, v, i, waves)),
    }
    for key, baseline in (("v0", v0), ("mixture", x)):
        magnitude = baseline.abs()
        active = (magnitude > 0) & (magnitude >= magnitude.amax(dim=(-2, -1), keepdim=True) * 1e-3)
        ratios = v.abs() / magnitude.clamp_min(1e-12)
        prefix = f"spectral_amplification_vs_{key}"
        stats[prefix + "_max"] = float(ratios.masked_fill(~active, 0).max())
        sampled = ratios.flatten()[::17][active.flatten()[::17]]
        for label, q in (("p95", 0.95), ("p99", 0.99)):
            stats[prefix + "_" + label] = float(torch.quantile(sampled, q)) if sampled.numel() else 0.0
        if key == "mixture":
            stats["spectral_amplification_active_bin_count"] = int(active.sum())
    if targets is not None:
        stats["vocal_output_gain_db"] = _gain(output.vocals_waveform, targets.vocals_waveform)
        stats["instrumental_output_gain_db"] = _gain(output.instrumental_waveform, targets.instrumental_waveform)
    return stats


@torch.no_grad()
def compute_gradient_diagnostics(model, clip_threshold):
    groups = {"final": list(model.refinement.final.parameters()),
              "upstream": [p for name, p in model.refinement.named_parameters() if not name.startswith("final.")]}
    result = {}
    bad = 0
    for name, parameters in groups.items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        result[name + "_gradient_norm"] = float(sum(g.square().sum() for g in gradients).sqrt()) if gradients else 0.0
        bad += sum(int((~torch.isfinite(g)).sum()) for g in gradients)
    norm = (result["final_gradient_norm"] ** 2 + result["upstream_gradient_norm"] ** 2) ** 0.5
    return {**result, "total_gradient_norm": norm, "gradient_clip_event": int(norm > clip_threshold),
            "nonfinite_gradients": bad}
