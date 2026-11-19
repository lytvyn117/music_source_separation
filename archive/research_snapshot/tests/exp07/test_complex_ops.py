import math

import numpy as np
import pytest
import torch

from src.experiments.exp07_phase_aware_residual.complex_ops import (
    channels_to_complex,
    complex_to_channels,
    make_vocal_ablation_stfts,
    project_equal_weight_targets,
    safe_complex_rms,
    waveform_nrmse_pct,
)


def test_projection_splits_the_recording_residual_equally():
    x = torch.tensor([10 + 4j])
    v = torch.tensor([2 + 2j])
    i = torch.tensor([4 - 2j])
    result = project_equal_weight_targets(x, v, i)
    torch.testing.assert_close(result.vocals, torch.tensor([4 + 4j]))
    torch.testing.assert_close(result.instrumental, torch.tensor([6 + 0j]))


def test_channel_encoding_preserves_batch_and_stereo_order():
    x = torch.tensor([1 + 2j, 3 + 4j]).reshape(1, 2, 1, 1)
    encoded = complex_to_channels(x)
    assert encoded.flatten().tolist() == [1, 3, 2, 4]
    torch.testing.assert_close(channels_to_complex(encoded), x)


def test_complex_rms_is_per_channel_and_has_a_silence_floor():
    x = torch.tensor([3 + 4j, 0j]).reshape(1, 2, 1, 1)
    assert safe_complex_rms(x, 1e-8).flatten().tolist() == pytest.approx([5, 1e-8])


def test_ablation_keeps_the_intended_magnitude_and_phase():
    v0 = torch.tensor([2j, 0j])
    v1 = torch.tensor([-3 + 0j, 1j])
    variants = make_vocal_ablation_stfts(v0, v1)
    torch.testing.assert_close(variants["magnitude_only"], torch.tensor([3j, 1 + 0j]))
    torch.testing.assert_close(variants["phase_only"], torch.tensor([-2 + 0j, 0j]))
    torch.testing.assert_close(variants["full"], v1)


def test_nrmse_is_relative_to_reference_and_reported_as_percent():
    ref = np.array([3.0, 4.0])
    estimate = np.array([3.0, 3.0])
    assert math.isclose(waveform_nrmse_pct(estimate, ref), 20.0)
