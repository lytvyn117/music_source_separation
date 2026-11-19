import pytest
import torch

from src.experiments.exp08_subband.model import (
    MASK_LIMIT, SubbandSeparator, bounded_sources, stereo_features,
)


@pytest.mark.parametrize("batch,time", [(2, 862), (1, 517)])
def test_architecture_shape_ledger_and_parameter_count(batch, time):
    with torch.device("meta"):
        model = SubbandSeparator()
        assert sum(p.numel() for p in model.parameters()) == 653650
        output, ledger = model.logits(torch.empty(batch, 6, 1025, time), trace=True)
    assert output.shape == (batch, 4, 1025, time)
    for key, width, channels in [
        ("encoder1", (32, 48, 97), 32), ("encoder2", (16, 16, 25), 128),
        ("decoder2", (32, 48, 97), 32), ("decoder1", (64, 192, 769), 4),
    ]:
        assert ledger[key] == [(batch, channels, size, time) for size in width]
    assert ledger["bottleneck"] == (batch, 128, 57, time)


def test_temporal_taps_cover_contiguous_span():
    model = SubbandSeparator()
    reach = {0}
    for block in model.temporal:
        convolution = block.depthwise
        dilation = convolution.dilation[1]
        assert convolution.kernel_size == (1, 5)
        assert convolution.padding == (0, 2 * dilation)
        reach = {offset + tap * dilation for offset in reach for tap in range(-2, 3)}
    assert reach == set(range(-510, 511))


def test_band_seams_keep_full_resolution_path_aligned():
    model = SubbandSeparator()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for head in model.raw_skip:
            head.weight[0, 0] = 1
        x = torch.zeros(1, 6, 1025, 2)
        x[0, 0, [0, 63, 64, 255, 256, 1024], 0] = torch.arange(1., 7.)
        torch.testing.assert_close(model.logits(x)[:, 0], x[:, 0])


def test_stereo_complex_bound_and_edge_bins():
    x = torch.tensor([2+0j, 2+3j, 0j, 5+0j, 5+7j, 1e-9j]).reshape(1, 2, 3, 1)
    x[:, :, -1] = x[:, :, -1].real.clone()
    feat = stereo_features(x)
    assert (feat[0, :4, 1, 0] / feat[0, 0, 1, 0]).tolist() == pytest.approx([1, 1.5, 2.5, 3.5])
    z = torch.tensor([100., 100., -100., -100.]).view(1, 4, 1, 1).expand(1, 4, 3, 1)
    estimates, mask = bounded_sources(z, x)
    assert (mask[..., [0, 2], :].imag == 0).all()
    assert mask[0, 0, 1, 0].imag > 0 > mask[0, 1, 1, 0].imag
    torch.testing.assert_close(estimates.sum(1), x)
    assert (estimates.abs() <= MASK_LIMIT * x[:, None].abs() + 1e-6).all()
    assert (estimates[0, :, 0, 2] == 0).all()
