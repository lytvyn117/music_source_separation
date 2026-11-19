import sys
from unittest.mock import patch

import torch

from src.training.losses_v2 import asymmetric_leakage_loss
from src.training.losses_v2 import CompositeSeparationLoss


def _capture_cli_run(extra_args):
    from src.training import train_experiment

    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return {}

    argv = ["train_experiment", "--exp-name", "cli_test", *extra_args]
    with patch.object(train_experiment, "run_experiment", fake_run_experiment):
        with patch.object(sys, "argv", argv):
            train_experiment.main()
    return captured


def test_cli_forwards_explicit_leakage_weight():
    captured = _capture_cli_run(["--leakage-weight", "3.0"])

    assert captured["leakage_weight"] == 3.0


def test_cli_preserves_default_leakage_weight():
    captured = _capture_cli_run([])


def test_vocal_dominant_instrumental_overmask_is_penalized_with_useful_gradients():
    vocals = torch.tensor([[[[9.0]]]])
    instrumental = torch.tensor([[[[1.0]]]])
    logits = torch.tensor(
        [[[[[0.0]]], [[[0.4]]]]],
        requires_grad=True,
    )
    masks = torch.softmax(logits, dim=1)

    loss = asymmetric_leakage_loss(masks, vocals, instrumental)

    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].item() > 0.0
    assert logits.grad[0, 0].item() < 0.0


def test_oracle_instrumental_mask_within_margin_has_no_penalty():
    vocals = torch.tensor([[[[9.0]]]])
    instrumental = torch.tensor([[[[1.0]]]])
    masks = torch.tensor([[[[[0.9]]], [[[0.1]]]]])

    loss = asymmetric_leakage_loss(masks, vocals, instrumental)

    assert loss.item() == 0.0


def test_silent_bins_are_ignored_and_numerically_safe():
    silence = torch.zeros(1, 2, 4, 4)
    masks = torch.full((1, 2, 2, 4, 4), 0.5)

    loss = asymmetric_leakage_loss(masks, silence, silence)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_per_example_activity_threshold_ignores_quiet_bins():
    vocals = torch.tensor([[[[1.0, 0.0005]]]])
    instrumental = torch.zeros_like(vocals)
    masks = torch.tensor([[[[[0.0, 0.0]]], [[[1.0, 1.0]]]]])

    loss = asymmetric_leakage_loss(
        masks,
        vocals,
        instrumental,
        activity_threshold=1e-3,
    )

    # Only the active bin contributes: (1.0 - 0.0 - 0.05)^2 / 2 bins.
    assert torch.isclose(loss, torch.tensor(0.45125), atol=1e-6)


def test_composite_loss_adds_exactly_the_weighted_leakage_term():
    vocals = torch.tensor([[[[9.0]]]])
    instrumental = torch.tensor([[[[1.0]]]])
    mixture = vocals + instrumental
    masks = torch.tensor([[[[[0.4]]], [[[0.6]]]]])
    dummy_stft = torch.zeros(1, 2, 1, 1, dtype=torch.complex64)
    dummy_waveform = torch.zeros(1, 2, 1)

    baseline = CompositeSeparationLoss(
        waveform_l1_weight=0.0,
        multires_stft_weight=0.0,
        leakage_weight=0.0,
    )
    leakage_aware = CompositeSeparationLoss(
        waveform_l1_weight=0.0,
        multires_stft_weight=0.0,
        leakage_weight=1.0,
    )

    baseline_loss, baseline_metrics = baseline(
        dummy_stft,
        mixture,
        masks,
        vocals,
        instrumental,
        dummy_waveform,
        dummy_waveform,
    )
    tuned_loss, tuned_metrics = leakage_aware(
        dummy_stft,
        mixture,
        masks,
        vocals,
        instrumental,
        dummy_waveform,
        dummy_waveform,
    )
    expected_leakage = asymmetric_leakage_loss(masks, vocals, instrumental)

    assert torch.allclose(tuned_loss - baseline_loss, expected_leakage)
    assert tuned_metrics["base_total"] == baseline_metrics["base_total"]
    assert tuned_metrics["leakage"] == expected_leakage.item()
    assert tuned_metrics["total"] == baseline_loss.item()
    assert tuned_metrics["objective"] == tuned_loss.item()
