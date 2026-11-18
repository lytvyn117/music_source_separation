"""Synthetic CUDA smoke of the reference graph and proposed composite loss."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.audio.stft import compute_istft, compute_stft
from probe_core import bounded_outputs, features, spectral_scale
from reference_graph import ReferenceGraph


def smoke_loss(x, vocal, instrumental, target_v, target_i):
    q = spectral_scale(x)
    mse = ((vocal - target_v) / q).abs().square().mean()
    log_mag = .5 * sum(
        ((estimate.abs() / q).log1p() - (target.abs() / q).log1p()).abs().mean()
        for estimate, target in ((vocal, target_v), (instrumental, target_i)))
    est = compute_istft(vocal, length=441000)
    ref = compute_istft(target_v, length=441000)
    mix = compute_istft(x, length=441000)
    wave_scale = mix.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).sqrt()
    error = est - ref
    wave_l1 = (error / wave_scale).abs().mean()
    side_l1 = (.5 * (error[:, :1] - error[:, 1:]) / wave_scale).abs().mean()
    ref_frames = ref.reshape(-1, 2, 10, 44100)
    inst_error = -error.reshape_as(ref_frames)
    power = ref_frames.square().sum(dim=(1, 3))
    activity = power > 2 * 44100 * .005**2
    projection = (inst_error * ref_frames).sum(dim=(1, 3)) / power.clamp_min(1e-8)
    bleed = ((projection.abs() - .05).relu().square() * activity).sum() / activity.sum().clamp_min(1)
    return mse + .5 * log_mag + 2 * wave_l1 + .25 * side_l1 + .25 * bleed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(808)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_bf16_supported()
    net = ReferenceGraph().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.0003, weight_decay=.0001)
    time_axis = torch.arange(441000, device="cuda") / 44100
    voice = torch.stack((.08 * (2 * torch.pi * 220 * time_axis).sin(),
                         .05 * (2 * torch.pi * 220 * time_axis + .3).sin())).repeat(2, 1, 1)
    backing = .02 * torch.randn_like(voice)
    wave = voice + backing
    vp = compute_stft(voice)
    totals, norms = [], []
    minimum_free = torch.cuda.mem_get_info()[0]
    max_additivity = max_wave_error = max_bound_excess = 0.
    for step in range(14):
        if step == 4:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        x = compute_stft(wave)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(features(x))
        assert logits.dtype == torch.bfloat16
        v, i = bounded_outputs(logits, x)
        loss = smoke_loss(x, v, i, vp, x - vp)
        assert loss.isfinite()
        loss.backward()
        norm = clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        with torch.no_grad():
            assert all(p.grad is not None and p.grad.count_nonzero() > 0 for p in net.parameters())
            bound = 1e-5 + 4 * 2**-24 * (x.abs() + v.abs() + i.abs())
            max_additivity = max(max_additivity, float(((v + i - x).abs() / bound).max()))
            max_bound_excess = max(max_bound_excess, float(
                (torch.maximum(v.abs(), i.abs()) - 1.1768603 * x.abs()).max()))
            recon = compute_istft(v, length=441000) + compute_istft(i, length=441000)
            max_wave_error = max(max_wave_error, float((recon - wave).abs().max()))
        if step >= 4:
            totals.append(elapsed)
        norms.append(float(norm))
        minimum_free = min(minimum_free, torch.cuda.mem_get_info()[0])
    assert max_additivity <= 1 and max_wave_error <= 2e-5 and max_bound_excess <= 1e-5
    result = {"batch": 2, "samples": 441000, "parameters": sum(p.numel() for p in net.parameters()),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "warmup_steps": 4, "timed_steps": len(totals), "median_step_sec": statistics.median(totals),
              "clips_per_sec": 2 * len(totals) / sum(totals),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "minimum_sampled_free_gib": minimum_free / 2**30, "gradient_norm": norms,
              "max_complex_error_to_bound": max_additivity, "max_wave_error": max_wave_error,
              "max_mask_bound_excess": max_bound_excess, "final_loss": float(loss)}
    print(json.dumps(result, indent=2))
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
