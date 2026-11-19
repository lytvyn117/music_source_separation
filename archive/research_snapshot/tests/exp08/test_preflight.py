import pytest
import torch

from src.experiments.exp08_subband.preflight import bounded_oracle, profile_summary


def test_profile_gates_use_inclusive_limits_and_real_clip_counts():
    stats = [{"seconds": 1., "preclip_norm": 2. if i < 50 else 1.,
              "clipped": i < 50, "relative_update": .01, "update_norm": .2,
              "guard": [.1, 0, .1, 2e-6], "signal_max_by_microbatch": [0.] * 10}
             for i in range(100)]
    summary = profile_summary(stats, clips_per_update=8, allocated=6., reserved=8., headroom=2.)
    assert summary["clips_per_sec"] == 8.
    assert summary["p95_step_s"] == pytest.approx(1.)
    assert summary["clipped_updates"] == 50 and all(summary["gates"].values())
    stats[99]["clipped"] = True
    assert not profile_summary(stats, 8, 6., 8., 2.)["gates"]["gradient_clipping"]


def test_bounded_oracle_projects_rectangular_mask_and_zero_bins():
    mixture = torch.tensor([1+0j, 1+0j, 0+0j, 1+0j]).reshape(1, 1, 4, 1)
    vocal = torch.full_like(mixture, 2+3j)
    estimate = bounded_oracle(mixture, vocal)
    assert estimate.flatten().tolist() == pytest.approx([1.15, 1.15+.25j, 0., 1.15])
