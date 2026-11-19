"""Explicit FP32 objectives for the matched scratch loss comparison."""
import torch
from music_separation.training.engine import objective
from music_separation.models.bs_roformer import HOP, N_FFT, RATE

WEIGHTS = {
    "preflight_l1": {"wave_l1": 1., "mr_complex": 1.8375527837636136},
    "normalized_mse_bleed": {"complex_mse": 1., "logmag_l1": .5,
                             "wave_l1": 2., "side_l1": .25, "leakage": .25},
}


def projection_terms(estimate, target):
    seconds = target.size(-1) // RATE
    if not seconds:
        raise ValueError("Vocal projection requires a complete one-second frame")
    frames = seconds * RATE
    v = target[..., :frames].unflatten(-1, (seconds, RATE))
    error = (target - estimate)[..., :frames].unflatten(-1, (seconds, RATE))
    energy = v.square().sum(dim=(1, 3))
    active = energy > (0.005 ** 2 * v.size(1) * RATE)
    projection = (error * v).sum(dim=(1, 3)) / energy.clamp_min(1e-8)
    return projection, active


def normalized_mse_loss(model, mixture, estimate, target):
    """Exp08-form objective on the Exp09 STFT and additive mixture."""
    with torch.autocast(estimate.device.type, enabled=False):
        mixture, estimate, target = mixture.float(), estimate.float(), target.float()
        def spectrum(x):
            z = torch.stft(x.flatten(0, 1), N_FFT, HOP, window=model.window.float(),
                           return_complex=True, pad_mode="constant")
            return z.unflatten(0, (x.size(0), x.size(1)))

        z_x, z_v, z_est = (spectrum(x) for x in (mixture, target, estimate))
        scale = z_x.abs().square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8).sqrt()
        error = estimate - target
        wave_scale = mixture.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).sqrt()

        def logmag(z):
            return torch.log1p(z.abs() / scale)

        def log_l1(a, b):
            return (logmag(a) - logmag(b)).abs().mean()

        projection, active = projection_terms(estimate, target)
        excess = (projection.abs() - 0.05).relu().square()
        return {
            "complex_mse": ((z_est - z_v).abs() / scale).square().mean(),
            "logmag_l1": .5 * (log_l1(z_est, z_v) + log_l1(z_x - z_est, z_x - z_v)),
            "wave_l1": (error / wave_scale).abs().mean(),
            "side_l1": ((error[:, 0] - error[:, 1]) / (2 * wave_scale[:, 0])).abs().mean(),
            "leakage": (excess * active).sum() / active.sum().clamp_min(1),
        }


def get_loss(name):
    if name == "preflight_l1":
        return lambda model, mixture, estimate, target: objective(estimate, target)
    if name == "normalized_mse_bleed":
        return normalized_mse_loss
    raise ValueError(f"Unknown objective {name}")
