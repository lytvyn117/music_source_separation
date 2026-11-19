import random

import numpy as np
import torch

from src.experiments.exp07_phase_aware_residual.deterministic_mse import loader, seed_worker, save_state, restore_state


def test_worker_seed_replays_python_numpy_and_torch(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: (1 << 32) + 42)
    seed_worker(0)
    expected = (random.random(), np.random.rand())
    seed_worker(1)
    assert (random.random(), np.random.rand()) == expected


def test_resume_restores_loader_rng_and_scaler():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    rng = torch.Generator().manual_seed(42)
    scaler = torch.amp.GradScaler("cpu", init_scale=512)
    state = save_state(model, opt, scheduler, rng, scaler=scaler)
    expected = torch.rand(1, generator=rng)
    other = torch.amp.GradScaler("cpu", init_scale=1)
    restore_state(state, model, opt, scheduler, rng, scaler=other)
    assert torch.equal(torch.rand(1, generator=rng), expected)
    assert other.get_scale() == 512
    batches = list(loader(torch.arange(8), rng))
    assert [b.tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7]]
