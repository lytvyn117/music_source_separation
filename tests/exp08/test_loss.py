import pytest
import torch

from src.experiments.exp08_subband.loss import (
    composite_loss, normalized_mse, project_vocals, vocal_leakage,
)


def test_projection_complement_and_per_example_mse_scale():
    x = torch.tensor([1+0j, 4+0j]).reshape(2, 1, 1, 1)
    v = project_vocals(x, x * .1, x * .7)
    torch.testing.assert_close(v, x * .2)
    e = torch.tensor([1+1j, 2+2j]).reshape_as(x)
    assert float(normalized_mse(e, x)) == pytest.approx(1.25)


def test_source_weights_are_redundant_in_even_terms():
    torch.manual_seed(42)
    x = torch.randn(2, 2, 5, 7, dtype=torch.complex128)
    v = torch.randn_like(x, requires_grad=True)
    vp = torch.randn_like(x)
    error = v - vp
    other = (x - v) - (x - vp)
    for penalty in (lambda e: (e / x.abs()).abs().mean(), lambda e: normalized_mse(e, x)):
        lv, li = penalty(error), penalty(other)
        weighted = (1.2 * lv + li) / 2.2
        torch.testing.assert_close(weighted, lv)
        torch.testing.assert_close(torch.autograd.grad(weighted, v, retain_graph=True)[0],
                                   torch.autograd.grad(lv, v, retain_graph=True)[0])
        assert torch.autograd.gradcheck(lambda p: penalty(p - vp), v, fast_mode=True)


def test_silence_and_nonlinear_magnitude_terms():
    x = torch.zeros(1, 2, 3, 3, dtype=torch.complex64, requires_grad=True)
    w = torch.zeros(1, 2, 8)
    total, terms = composite_loss(x, x, x, x, w, w, w, rate=4)
    total.backward()
    assert total.item() == 0
    assert torch.isfinite(x.grad).all()
    x = torch.ones_like(x)
    _, terms = composite_loss(x, -.1*x, 1.1*x, .1*x, w, w, w, rate=4)
    expected = .5 * (torch.tensor(2.1).log() - torch.tensor(1.9).log())
    torch.testing.assert_close(terms["log_magnitude"], expected)


def test_vocal_leakage_sign_activity_and_gradient():
    v = torch.tensor([[[.1, .1, .1, .1, .001, .001, .001, .001],
                       [.1, .1, .1, .1, .001, .001, .001, .001]]])
    for sign in (-1, 1):
        e = (sign * .2 * v).requires_grad_()
        leak = vocal_leakage(e, v, rate=4)
        assert float(leak) == pytest.approx(.15**2)
        g = torch.autograd.grad(leak, e)[0]
        assert (g[..., 4:] == 0).all()
        assert not torch.allclose(g, torch.autograd.grad(e.square().mean(), e)[0])
    assert vocal_leakage(v, v * 0, rate=4).item() == 0
