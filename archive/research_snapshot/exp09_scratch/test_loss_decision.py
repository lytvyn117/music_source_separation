import unittest
import torch

from loss_candidates import normalized_mse_loss
from scratch_model import Separator


class LossDecisionTests(unittest.TestCase):
    def test_projected_vocal_bleed_and_scale(self):
        torch.manual_seed(12)
        model = Separator()
        target = torch.randn(1, 2, 44100) * .03
        x = target + torch.randn_like(target) * .05
        estimate = (target * .7).requires_grad_()
        parts = normalized_mse_loss(model, x, estimate, target)
        self.assertAlmostEqual(float(parts["leakage"].detach()), .25 ** 2, places=6)
        scaled = normalized_mse_loss(model, x * 2, estimate * 2, target * 2)
        for k, v in parts.items():
            torch.testing.assert_close(v, scaled[k])
        sum(parts.values()).backward()
        self.assertTrue(estimate.grad.isfinite().all())
        perfect = normalized_mse_loss(model, x, target, target)
        self.assertTrue(all(float(v) == 0 for v in perfect.values()))

    def test_quiet_vocals_do_not_create_leakage_loss(self):
        model = Separator()
        quiet = torch.zeros(1, 2, 44100)
        x = torch.rand_like(quiet) * .05
        self.assertEqual(float(normalized_mse_loss(model, x, quiet + .01, quiet)["leakage"]), 0)
