import math

import numpy as np
import pytest
import torch

from src.evaluation.mask_diagnostics import (
    classify_tf_regions,
    frequency_band_indices,
    mask_diagnostics,
    spectral_band_metrics,
    waveform_artifact_metrics,
)
from src.evaluation.analyze_exp06 import load_transition_models
from src.models.unet_v2 import UNetV2


def test_frequency_band_indices_assign_each_rfft_bin_once():
    bands = (("low", 0.0, 2.0), ("high", 2.0, 4.1))

    result = frequency_band_indices(n_fft=8, sample_rate=8, bands=bands)

    assert result["low"].tolist() == [True, True, False, False, False]
    assert result["high"].tolist() == [False, False, True, True, True]
    assert torch.stack(list(result.values())).sum(dim=0).tolist() == [1, 1, 1, 1, 1]


def test_tf_regions_separate_vocal_instrumental_overlap_and_silence():
    vocals = torch.tensor([[[[9.0, 1.0, 5.0, 0.0]]]])
    instrumental = torch.tensor([[[[1.0, 9.0, 5.0, 0.0]]]])

    regions = classify_tf_regions(vocals, instrumental, dominance_threshold=0.8)

    assert regions["vocal_dominant"].flatten().tolist() == [True, False, False, False]
    assert regions["instrumental_dominant"].flatten().tolist() == [False, True, False, False]
    assert regions["overlap"].flatten().tolist() == [False, False, True, False]
    assert regions["active"].flatten().tolist() == [True, True, True, False]


def test_mask_diagnostics_report_independent_sum_overlap_and_irm_error():
    vocals = torch.tensor([[[[9.0, 1.0, 5.0]]]])
    instrumental = torch.tensor([[[[1.0, 9.0, 5.0]]]])
    masks = torch.tensor(
        [[[[[0.8, 0.3, 0.7]]], [[[0.4, 0.8, 0.7]]]]]
    )

    stats = mask_diagnostics(masks, vocals, instrumental)

    assert stats["finite"] is True
    assert stats["mask_sum_mean"] == pytest.approx((1.2 + 1.1 + 1.4) / 3)
    assert stats["both_masks_gt_0_5_fraction"] == pytest.approx(1 / 3)
    assert stats["mask_product_mean"] == pytest.approx((0.32 + 0.24 + 0.49) / 3)
    assert stats["vocal_mask_irm_mae"] == pytest.approx((0.1 + 0.2 + 0.2) / 3)
    assert stats["instrumental_mask_irm_mae"] == pytest.approx((0.3 + 0.1 + 0.2) / 3)
    assert stats["regions"]["overlap"]["count"] == 1
    assert stats["regions"]["overlap"]["vocal_mask_irm_mae"] == pytest.approx(0.2)


def test_region_prediction_error_is_normalized_by_mixture_energy():
    vocals = torch.tensor([[[[0.0]]]])
    instrumental = torch.tensor([[[[10.0]]]])
    masks = torch.tensor([[[[[0.1]]], [[[0.9]]]]])

    stats = mask_diagnostics(masks, vocals, instrumental)

    region = stats["regions"]["instrumental_dominant"]
    assert region["vocal_prediction_nmae"] == pytest.approx(0.1)
    assert region["instrumental_prediction_nmae"] == pytest.approx(0.1)


def test_waveform_artifact_metrics_use_unmodified_outputs():
    mixture = np.ones((8, 2), dtype=np.float32)
    vocals = np.full_like(mixture, 0.6)
    instrumental = np.full_like(mixture, 0.4)

    stats = waveform_artifact_metrics(mixture, vocals, instrumental)

    assert stats["finite"] is True
    assert stats["vocals_gain_db"] == pytest.approx(20.0 * math.log10(0.6))
    assert stats["instrumental_gain_db"] == pytest.approx(20.0 * math.log10(0.4))
    assert stats["combined_gain_db"] == pytest.approx(0.0, abs=1e-6)
    assert stats["mixture_reconstruction_residual_ratio"] == pytest.approx(
        0.0, abs=1e-7
    )


def test_nonfinite_outputs_are_flagged_not_summarized_as_valid():
    mixture = np.ones((4, 2), dtype=np.float32)
    vocals = mixture.copy()
    vocals[0, 0] = np.nan

    stats = waveform_artifact_metrics(mixture, vocals, np.zeros_like(mixture))

    assert stats["finite"] is False
    assert stats["nonfinite_values"] == 1


def test_spectral_band_metrics_compute_literal_energy_sdr_and_leakage():
    bands = (("all", 0.0, 4.1),)
    vocals_ref = torch.ones(1, 1, 5, 2, dtype=torch.complex64)
    inst_ref = torch.zeros_like(vocals_ref)
    vocals_est = 0.5 * vocals_ref
    inst_est = 0.25 * vocals_ref

    stats = spectral_band_metrics(
        vocals_ref,
        inst_ref,
        vocals_est,
        inst_est,
        n_fft=8,
        sample_rate=8,
        bands=bands,
    )["all"]

    assert stats["vocals_sdr_db"] == pytest.approx(20.0 * math.log10(2.0), abs=1e-5)
    assert stats["vocal_to_instrumental_bleed_coefficient"] == pytest.approx(0.25)
    assert stats["vocal_to_instrumental_bleed_energy_pct"] == pytest.approx(6.25)


def test_transition_loader_changes_only_activation_mode(tmp_path):
    checkpoint_path = tmp_path / "exp04.pt"
    original = UNetV2(base_channels=1, pad_mode="exact", mask_mode="softmax")
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "config": {
                "base_channels": 1,
                "pad_mode": "exact",
                "log_input": False,
                "mask_mode": "softmax",
                "leakage_weight": 1.0,
            },
        },
        checkpoint_path,
    )

    softmax_model, sigmoid_model, metadata = load_transition_models(
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert softmax_model.mask_mode == "softmax"
    assert sigmoid_model.mask_mode == "sigmoid"
    assert metadata["source_mask_mode"] == "softmax"
    assert metadata["probe_mask_mode"] == "sigmoid"
    for name, value in softmax_model.state_dict().items():
        assert torch.equal(value, sigmoid_model.state_dict()[name])

    sample = torch.rand(1, 2, 33, 35)
    with torch.no_grad():
        softmax_masks = softmax_model(sample)
        sigmoid_masks = sigmoid_model(sample)
    assert softmax_masks.shape == sigmoid_masks.shape == (1, 2, 2, 33, 35)
    assert torch.allclose(
        softmax_masks.sum(dim=1), torch.ones_like(softmax_masks[:, 0]), atol=1e-6
    )
    assert torch.isfinite(sigmoid_masks).all()
