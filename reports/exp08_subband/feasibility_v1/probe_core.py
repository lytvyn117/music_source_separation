import torch


def spectral_scale(x):
    return x.abs().square().mean(dim=tuple(range(1, x.ndim)), keepdim=True).clamp_min(1e-8).sqrt()


def features(x):
    z = x / spectral_scale(x)
    return torch.stack((z[:, 0].real, z[:, 0].imag, z[:, 1].real, z[:, 1].imag,
                        z[:, 0].abs().log1p(), z[:, 1].abs().log1p()), dim=1)


def normalized_error(error, mixture, power):
    magnitude = (error / spectral_scale(mixture)).abs()
    return (magnitude if power == 1 else magnitude.square()).mean()


def bounded_mask(logits, edge):
    a = .5 + .65 * logits[:, 0::2].tanh()
    b = (.25 * logits[:, 1::2].tanh()).masked_fill(edge, 0)
    return torch.complex(a, b)


def edge_bins(x):
    f = torch.arange(x.shape[-2], device=x.device)
    return ((f == 0) | (f == x.shape[-2] - 1))[:, None]


def bounded_outputs(logits, x):
    v = bounded_mask(logits.to(x.real.dtype), edge_bins(x)) * x
    return v, x - v


def box_oracle(x, v):
    active = x != 0
    ratio = v / torch.where(active, x, 1)
    real = ratio.real.clamp(-.15, 1.15)
    imag = ratio.imag.clamp(-.25, .25).masked_fill(edge_bins(x), 0)
    return torch.complex(real, imag) * x
