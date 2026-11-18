"""Opt-in deterministic execution for the isolated MSE path."""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .trainer import restore_training_state, training_state


CUBLAS_WORKSPACE = ":4096:8"


def configure_determinism():
    if torch.cuda.is_initialized():
        raise RuntimeError("Deterministic Exp07 must configure cuBLAS before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"cublas_workspace_config": CUBLAS_WORKSPACE,
            "deterministic_algorithms": True, "warn_only": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True}


def seed_worker(worker_id):
    # PyTorch sets the worker's Torch seed from the supplied loader generator.
    seed = torch.initial_seed() % (1 << 32)
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, generator, *, workers=0):
    # Epoch-boundary restore regenerates worker seeds; persistent worker RNGs
    # would instead live across epochs and require their own checkpoint.
    return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=workers,
                      generator=generator, worker_init_fn=seed_worker, persistent_workers=False)


def save_state(model, optimizer, scheduler, generator, *, scaler=None, **fields):
    state = training_state(model, optimizer, scheduler, generator, **fields)
    state["scaler_state_dict"] = None if scaler is None else scaler.state_dict()
    return state


def restore_state(state, model, optimizer, scheduler, generator, *, scaler=None):
    scaler_state = state["scaler_state_dict"]
    if (scaler_state is None) != (scaler is None):
        raise ValueError("Checkpoint and runtime AMP scaler configuration differ")
    restore_training_state(state, model, optimizer, scheduler, generator)
    if scaler is not None:
        scaler.load_state_dict(scaler_state)
