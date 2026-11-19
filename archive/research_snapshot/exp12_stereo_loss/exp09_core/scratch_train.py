"""Loss calibration and replayable optimizer steps for scratch preflight."""
import hashlib
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch

RESOLUTIONS = (512, 1024, 2048)


def objective(estimate, target):
    with torch.autocast(estimate.device.type, enabled=False):
        estimate, target = estimate.float(), target.float()
        wave = (estimate - target).abs().mean()
        spec_losses = []
        for size in RESOLUTIONS:
            win = torch.hann_window(size, device=estimate.device)
            diff = torch.stft((estimate - target).flatten(0, 1), size, size // 4,
                              window=win, return_complex=True, normalized=True, pad_mode="constant")
            spec_losses.append(diff.abs().mean())
        return {"wave_l1": wave, "mr_complex": torch.stack(spec_losses).mean()}


def calibrate(model, batch_pairs):
    evidence = []
    keys = ("wave_l1", "mr_complex")
    for mixture, target in batch_pairs:
        with torch.no_grad():
            estimate, _ = model(mixture)
        estimate = estimate.detach().requires_grad_()
        parts = objective(estimate, target)
        norms = {k: float(torch.autograd.grad(parts[k], estimate, retain_graph=True)[0].norm())
                 for k in keys}
        evidence.append({"values": {k: float(parts[k].detach()) for k in keys},
                         "output_gradient_l2": norms})
    if not evidence:
        raise ValueError("Require at least one batch for calibration")
    norms = {k: float(np.median([r["output_gradient_l2"][k] for r in evidence])) for k in keys}
    if not all(np.isfinite(n) and n > 0 for n in norms.values()):
        raise ValueError("Nonfinite or zero calibration gradient")
    spectral = float(np.clip(norms["wave_l1"] / norms["mr_complex"], 0.1, 10.0))
    return {"wave_l1": 1.0, "mr_complex": spectral}, evidence


def model_digest(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class ExactRun:
    def __init__(self, model, weights, seed, amp, lr=0.0003, total=24000, objective_fn=None, objective_name="preflight_l1"):
        self.model, self.weights, self.seed, self.amp = model, dict(weights), seed, amp
        self.lr, self.total = lr, total
        self.objective_fn, self.objective_name = objective_fn, objective_name
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                           foreach=False, fused=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total, eta_min=lr / 10)
        self.index = 0
        self.device = next(model.parameters()).device

    def step(self, mixture, target, *, gradient_audit=False, accumulation=1):
        if accumulation < 1 or len(mixture) != len(target) or len(mixture) % accumulation:
            raise ValueError("Batch must divide evenly into accumulation microbatches")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        terms = {key: torch.zeros((), device=self.device) for key in self.weights}
        for x, y in zip(mixture.chunk(accumulation), target.chunk(accumulation)):
            with torch.autocast(self.device.type, enabled=self.amp, dtype=torch.bfloat16):
                vocals, _ = self.model(x)
            parts = (objective(vocals, y) if self.objective_fn is None
                     else self.objective_fn(self.model, x, vocals, y))
            loss = sum(self.weights[k] * v for k, v in parts.items()) / accumulation
            loss.backward()
            for key, value in parts.items():
                terms[key] += value.detach() / accumulation
        blocks = []
        if gradient_audit:
            for block in self.model.blocks:
                values = torch.stack([p.grad.detach().float().norm()
                                      for p in block.parameters()])
                blocks.append({"axis": block.axis, "finite": bool(values.isfinite().all()),
                               "min_tensor_gradient_l2": float(values.min())})
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0,
                                                         error_if_nonfinite=True))
        metrics = {k: float(v) for k, v in terms.items()}
        metrics["loss"] = sum(self.weights[k] * v for k, v in metrics.items())
        if not np.isfinite(metrics["loss"]):
            raise ValueError("Nonfinite objective")
        self.optimizer.step()
        self.scheduler.step()
        self.index += 1
        return {**metrics, "gradient_l2": grad_norm, **({"blocks": blocks} if gradient_audit else {})}

    def signature(self, data_identity):
        return {"version": 1, "data": data_identity, "seed": self.seed, "amp": self.amp,
                "weights": self.weights, "lr": self.lr, "total": self.total, "objective": self.objective_name,
                "checkpoint_blocks": self.model.checkpoint_blocks,
                "group_chunk": self.model.blocks[0].group_chunk,
                "deterministic": torch.are_deterministic_algorithms_enabled()}

    def save(self, path, data_identity):
        path = Path(path)
        state = {"signature": self.signature(data_identity), "index": self.index,
                 "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                 "python_rng": random.getstate(),
                 "numpy_rng": self._numpy_state()}
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            tmp = Path(handle.name)
        try:
            torch.save(state, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _numpy_state():
        kind, keys, position, gaussian, cached = np.random.get_state()
        return kind, keys.tolist(), position, gaussian, cached

    def restore(self, path, data_identity):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["signature"] != self.signature(data_identity):
            raise ValueError("Checkpoint identity or configuration differs")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.index = state["index"]
        torch.set_rng_state(state["cpu_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        random.setstate(state["python_rng"])
        kind, keys, position, gaussian, cached = state["numpy_rng"]
        np.random.set_state((kind, np.array(keys, dtype=np.uint32), position, gaussian, cached))
