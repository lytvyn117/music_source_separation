"""Full-song stereo inference with shared overlap-add weights."""
import numpy as np
import torch

from src.experiments.exp07_phase_aware_residual.chunking import OverlapAdd, chunk_batch, chunk_starts


@torch.inference_mode()
def separate(model, mixture, device, chunk_samples=441000, batch_size=2):
    mixture = np.asarray(mixture, dtype=np.float32)
    if mixture.ndim != 2 or mixture.shape[0] != 2 or not np.isfinite(mixture).all():
        raise ValueError("Expected a finite stereo mixture shaped [2, samples]")
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    starts = chunk_starts(mixture.shape[-1], chunk_samples, .5)
    accumulator = OverlapAdd(4, mixture.shape[-1], chunk_samples, len(starts) > 1)
    model.eval()
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset + batch_size]
        wave = torch.from_numpy(chunk_batch(mixture, selected, chunk_samples)).to(device)
        output = model(wave).float().cpu().numpy()
        for start, estimate in zip(selected, output, strict=True):
            accumulator.add(estimate.reshape(4, chunk_samples), start)
    result = accumulator.finish().reshape(2, 2, -1)
    tolerance = 2e-5 * max(1., np.abs(mixture).max())
    if not np.isfinite(result).all() or np.abs(result.sum(0) - mixture).max() > tolerance:
        raise FloatingPointError("Full-song mixture consistency failed")
    return result
