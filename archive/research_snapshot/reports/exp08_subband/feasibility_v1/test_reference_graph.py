import unittest

import torch

from reference_graph import ReferenceGraph


class ArchitectureTests(unittest.TestCase):
    def test_exact_shapes_and_parameter_count(self):
        with torch.device("meta"):
            net = ReferenceGraph()
            self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad), 653650)
            for batch, time in ((2, 862), (1, 517)):
                logits, ledger = net(torch.empty(batch, 6, 1025, time), trace=True)
                self.assertEqual(logits.shape, (batch, 4, 1025, time))
                for key, channels, widths in (
                        ("encoder1", 32, (32, 48, 97)), ("encoder2", 128, (16, 16, 25)),
                        ("decoder2", 32, (32, 48, 97)), ("decoder1", 4, (64, 192, 769))):
                    self.assertEqual(ledger[key], [(batch, channels, w, time) for w in widths])
                self.assertEqual(ledger["bottleneck"], (batch, 128, 57, time))

    def test_exact_temporal_tap_support(self):
        net = ReferenceGraph()
        reach = {0}
        for block in net.temporal:
            self.assertEqual(block.depthwise.kernel_size, (1, 5))
            dilation = block.depthwise.dilation[1]
            self.assertEqual(block.depthwise.padding, (0, 2 * dilation))
            reach = {p + k * dilation for p in reach for k in (-2, -1, 0, 1, 2)}
        self.assertEqual(reach, set(range(-510, 511)))

    def test_right_padding_preserves_band_end_alignment(self):
        x = torch.arange(769).reshape(1, 1, 769, 1).float()
        padded = ReferenceGraph.right_pad(x, 8)
        self.assertEqual(padded.shape[-2], 776)
        torch.testing.assert_close(padded[..., :769, :], x)
        self.assertTrue((padded[..., 769:, :] == 0).all())


if __name__ == "__main__":
    unittest.main()
