import torch
import pytest

from src.experiments.exp07_phase_aware_residual.config import Exp07Config, Exp07Paths
from src.experiments.exp07_phase_aware_residual.losses import LossWeights
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator
from src.experiments.exp07_phase_aware_residual.trainer import configure_optimizer, run_pilot, train_epoch, validate_epoch
from tests.exp07.test_model import SmallBackbone


def test_training_updates_only_head_and_preserves_validation_selection_objective():
    model = PhaseAwareResidualSeparator(SmallBackbone())
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    optimizer, _ = configure_optimizer(model, Exp07Config())
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in model.refinement.parameters()
    }
    v, i = (torch.randn(1, 2, 4096) * scale for scale in (0.05, 0.1))
    batch = (v + i, v, i)
    weights = LossWeights(complex=1, waveform_l1=0.2, mr_stft=0.3, leakage=0.1)
    epoch = train_epoch(model, [batch], optimizer, weights)
    assert epoch.steps == 1 and epoch.metrics["final_gradient_norm"] > 0
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    result = validate_epoch(model, [batch], weights)
    expected = sum(result.metrics[key] * getattr(weights, key) for key in ("complex", "waveform_l1", "mr_stft", "leakage"))
    assert result.metrics["validation_objective"] == pytest.approx(expected)


def test_pilot_rejects_failed_preflight_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match="passing preflight"):
        run_pilot(Exp07Config(), Exp07Paths(tmp_path), {"passed": False})
    assert not list(tmp_path.iterdir())
