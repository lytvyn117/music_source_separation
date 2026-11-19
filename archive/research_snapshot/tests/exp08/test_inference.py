import numpy as np
import pytest
import torch

from src.experiments.exp08_subband.inference import separate


class StereoSplit(torch.nn.Module):
    def forward(self, waveform):
        vocals = waveform * waveform.new_tensor([.2, .7])[None, :, None]
        return torch.stack((vocals, waveform - vocals), 1)


@pytest.mark.parametrize("length", [7, 31, 49, 67])
def test_overlap_add_exact_length_and_stereo(length):
    mixture = np.random.default_rng(42).uniform(-1, 1, (2, length)).astype(np.float32)
    output = separate(StereoSplit(), mixture, device="cpu", chunk_samples=31, batch_size=2)
    assert output.shape == (2, 2, length)
    np.testing.assert_allclose(output.sum(0), mixture, atol=2e-7)
    np.testing.assert_allclose(output[0], mixture * np.array([[.2], [.7]]), atol=2e-7)
