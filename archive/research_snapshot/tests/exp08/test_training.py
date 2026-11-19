import pytest
import torch

from src.experiments.exp08_subband.model import SubbandSeparator, bounded_sources
from src.experiments.exp08_subband.loss import active_vocal_frames, composite_loss
from src.experiments.exp08_subband.train import (
    TrainConfig, guard, make_scheduler, optimizer_update, validate,
)


def test_effective_batch_matches_unpartitioned_loss_and_gradients():
    torch.manual_seed(42)
    x = torch.rand(4, 2, 3, 4).to(torch.complex64)
    mix = torch.rand(4, 2, 8)
    target = mix.clone()
    target[0, :, :] = 0
    target[1, :, 4:] = 0
    parameter = torch.tensor(.1, requires_grad=True)

    def objective(start, end, denominator=None, accumulation=1):
        spec = x[start:end]
        wave = mix[start:end]
        est = parameter * spec
        return composite_loss(spec, est, spec - est, .3 * spec,
                              wave, parameter * wave, target[start:end], rate=4,
                              effective_active=denominator, accumulation=accumulation)[0]

    all_loss = objective(0, 4)
    active = active_vocal_frames(target, rate=4).sum()
    accumulated = sum(objective(i, i+2, active, 2) / 2 for i in (0, 2))
    torch.testing.assert_close(all_loss, accumulated)
    torch.testing.assert_close(torch.autograd.grad(all_loss, parameter)[0],
                               torch.autograd.grad(accumulated, parameter)[0])


@pytest.mark.parametrize("bad", ["amplification", "nonfinite", "reconstruction"])
def test_safety_stops_before_update(bad):
    x = torch.ones(1, 2, 3, 3, dtype=torch.complex64)
    estimates, _ = bounded_sources(torch.zeros(1, 4, 3, 3), x)
    wave = torch.zeros(1, 2, 8)
    prediction = torch.zeros(1, 2, 2, 8)
    if bad == "amplification":
        estimates[:, 0], estimates[:, 1] = 2 * x, -x
    elif bad == "nonfinite":
        estimates[:, 0] = float("nan")
    else:
        prediction[:, 0] = 1.
    with pytest.raises(FloatingPointError, match=f"(?i){bad}"):
        guard(x, estimates, wave, prediction)


def test_safety_detects_amplification_on_quiet_bins():
    x = torch.full((1, 2, 3, 3), 1e-9, dtype=torch.complex64)
    spectra = torch.stack((2 * x, -x), dim=1)
    wave = torch.zeros(1, 2, 8)
    with pytest.raises(FloatingPointError, match="amplification"):
        guard(x, spectra, wave, torch.stack((wave, wave), dim=1))


def test_nonfinite_gradients_raise_logged_error_type_before_parameter_update():
    torch.set_num_threads(2)
    model = SubbandSeparator()
    parameter = model.raw_skip[0].weight
    before = parameter.detach().clone()
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    audio = torch.randn(1, 3, 2, 2048) * .01
    try:
        with pytest.raises(FloatingPointError, match="gradient"):
            optimizer_update(model, optimizer, audio, TrainConfig(microbatch=1, accumulation=1))
    finally:
        hook.remove()
    assert torch.equal(parameter, before)
    assert not optimizer.state


def test_scheduler_hits_approved_peak_and_floor():
    config = TrainConfig()
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=config.lr)
    scheduler = make_scheduler(optimizer, config)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 256)
    for _ in range(255):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    for _ in range(config.steps_per_epoch * config.epochs - 255):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_optimizer_and_validation_on_short_real_graph():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    net = SubbandSeparator()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    audio = torch.randn(2, 3, 2, 2048) * .01
    before = net.raw_skip[0].weight.detach().clone()
    stats = optimizer_update(net, opt, audio, TrainConfig(microbatch=1, accumulation=2),
                             detailed=True)
    assert stats["preclip_norm"] > 0 and stats["update_norm"] > 0
    assert all(value > 0 for value in stats["branch_gradient_norms"].values())
    assert len(stats["component_gradient_norms"]) == 5
    assert not torch.equal(before, net.raw_skip[0].weight)
    val = validate(net, [{"audio": audio}], TrainConfig(microbatch=1, accumulation=2))
    assert val["val_total_loss"] > 0
