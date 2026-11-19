import unittest

import torch

from probe_core import normalized_error, bounded_outputs, box_oracle, features


class FeasibilityTests(unittest.TestCase):
    def test_source_weight_cannot_change_normalized_reconstruction_gradient(self):
        rng = torch.Generator().manual_seed(808)
        x = torch.randn(2, 2, 5, 7, generator=rng, dtype=torch.complex128)
        target_v = torch.randn(x.shape, generator=rng, dtype=x.dtype)
        target_i = x - target_v
        logits = torch.randn(2, 4, 5, 7, generator=rng, dtype=torch.float64, requires_grad=True)
        v, i = bounded_outputs(logits, x)
        torch.testing.assert_close(i - target_i, -(v - target_v), atol=1e-14, rtol=0)
        for p in (1, 2):
            lv = normalized_error(v - target_v, x, p)
            li = normalized_error(i - target_i, x, p)
            grad = torch.autograd.grad(lv, logits, retain_graph=True)[0]
            for wv, wi in ((1.2, 1.0), (100.0, 0.25), (0.0, 1.0)):
                weighted = (wv * lv + wi * li) / (wv + wi)
                torch.testing.assert_close(weighted, lv, atol=1e-14, rtol=0)
                actual = torch.autograd.grad(weighted, logits, retain_graph=True)[0]
                torch.testing.assert_close(actual, grad, atol=1e-14, rtol=0)
            raw_sum = 1.2 * lv + li
            raw_gradient = torch.autograd.grad(raw_sum, logits, retain_graph=True)[0]
            torch.testing.assert_close(raw_gradient, 2.2 * grad, atol=1e-14, rtol=0)

    def test_nonlinear_magnitude_errors_are_not_redundant(self):
        x = torch.tensor([1 + 0j])
        target_v = torch.tensor([.1 + 0j])
        v = torch.tensor([-.1 + 0j])
        vocal_error = (v.abs().log1p() - target_v.abs().log1p()).abs()
        inst_error = ((x - v).abs().log1p() - (x - target_v).abs().log1p()).abs()
        self.assertEqual(float(vocal_error), 0)
        self.assertGreater(float(inst_error), 0)

    def test_silent_mixture_mse_has_finite_zero_gradient(self):
        x = torch.zeros(2, 2, 5, 7, dtype=torch.complex64)
        logits = torch.zeros(2, 4, 5, 7, requires_grad=True)
        v, i = bounded_outputs(logits, x)
        loss = normalized_error(v, x, 2)
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(int(logits.grad.count_nonzero()), 0)
        self.assertTrue((i == 0).all())

    def test_box_projection_and_silence(self):
        x = torch.ones(1, 2, 3, 4, dtype=torch.complex128)
        x[..., 0] = 0
        v = box_oracle(x, torch.full_like(x, 2 + 3j))
        self.assertTrue((v[..., 0] == 0).all())
        torch.testing.assert_close(v[..., 1, 1:], torch.full_like(v[..., 1, 1:], 1.15 + .25j))
        self.assertTrue((v[..., (0, 2), :].imag == 0).all())
        self.assertTrue((v.abs() <= 1.176861 * x.abs()).all())
        self.assertTrue(((x - v).abs() <= 1.176861 * x.abs()).all())

    def test_stereo_order(self):
        x = torch.tensor([[[[2+3j]], [[5+7j]]]])
        actual = features(x)[0, :4, 0, 0]
        torch.testing.assert_close(actual / actual[0], torch.tensor([1., 1.5, 2.5, 3.5]))
        logits = torch.tensor([.5, .1, -.7, -.2]).reshape(1, 4, 1, 1).expand(1, 4, 3, 2)
        v, _ = bounded_outputs(logits, torch.ones(1, 2, 3, 2, dtype=torch.complex64))
        self.assertGreater(v[0, 0, 1, 0].imag, 0)
        self.assertLess(v[0, 1, 1, 0].imag, 0)


if __name__ == "__main__":
    unittest.main()
