"""Quantify the reproduced stop from saved tensors on CPU."""

import json
from pathlib import Path

import numpy as np
import torch

from config import HOP_LENGTH, N_FFT, SAMPLE_RATE
from src.audio.stft import compute_istft


ROOT = Path(__file__).resolve().parent
BANDS = (0, 60, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, SAMPLE_RATE / 2 + 1)


def db(value):
    return float(20 * np.log10(max(float(value), 1e-12)))


def percent(numerator, denominator):
    return float(100 * numerator / denominator) if denominator else None


def p99(values):
    return float(np.quantile(values, .99))


def weighted_p99(values, weights):
    order = values.argsort(kind="stable")
    cdf = np.cumsum(weights[order], dtype=np.float64)
    return float(values[order[np.searchsorted(cdf, .99 * cdf[-1])]])


def quantiles(values):
    return dict(zip(("p50", "p90", "p99", "max"), map(float, np.quantile(values, [.5, .9, .99, 1]))))


def power(value):
    return np.abs(value.astype(np.complex128)) ** 2


def main():
    torch.set_num_threads(4)
    saved = torch.load(ROOT / "capture.pt", map_location="cpu", weights_only=True)
    assert saved["match"], "Capture does not reproduce the original logged event"
    p = {name: tensor.numpy() for name, tensor in saved["prediction"].items()}
    t = {name: tensor.numpy() for name, tensor in saved["targets"].items()}
    x = p["mixture_stft"]
    magnitude = np.abs(x)
    denominator = np.maximum(magnitude, 1e-12)
    mixture_power = power(x)
    peak = magnitude.max(axis=(-2, -1), keepdims=True)
    relative = magnitude / peak
    active = (magnitude > 0) & (magnitude >= .001 * peak)
    high = (magnitude > 0) & (magnitude >= .1 * peak)
    sampled = (np.arange(x.size) % 17 == 0).reshape(x.shape)
    freq = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    sample_seconds = np.arange(x.shape[-1]) * HOP_LENGTH / SAMPLE_RATE
    residual = p["residual_stft"]
    residual_power = power(residual)
    ideal = t["vocals_stft"] - p["initial_vocal_stft"]
    ideal_power = power(ideal)
    length = saved["batch"]["mixture"].shape[-1]
    initial = np.stack((p["initial_vocal_stft"], x - p["initial_vocal_stft"]), axis=1)
    wave0 = compute_istft(torch.from_numpy(initial.reshape(-1, 2, *x.shape[-2:])), length=length).numpy()
    wave0 = wave0.reshape(x.shape[0], 2, 2, length)

    report = {"shape": list(x.shape), "segments": saved["source_reads"], "sample_rate": SAMPLE_RATE,
              "n_fft": N_FFT, "hop_length": HOP_LENGTH, "active_count": int(active.sum()),
              "high_count": int(high.sum()), "active_mixture_energy_pct": percent(mixture_power[active].sum(), mixture_power.sum()),
              "high_mixture_energy_pct": percent(mixture_power[high].sum(), mixture_power.sum()),
              "total_mix_coefficient_power": float(mixture_power.sum()), "sources": {}}
    for idx, name in enumerate(("vocals", "instrumental")):
        before, after, target = initial[:, idx], p[name + "_stft"], t[name + "_stft"]
        baseline_ratio = p["masks"][:, idx]
        output_ratio = np.abs(after) / denominator
        ideal_ratio = np.abs(target) / denominator
        err0, err1 = power(before - target), power(after - target)
        actual_ideal = target - before
        alignment = np.real((after - before).astype(np.complex128) * np.conj(actual_ideal))
        actual_ideal_power = power(actual_ideal)
        after_power, before_power = power(after), power(before)

        def measure(mask):
            mp = mixture_power[mask].sum()
            correction = residual_power[mask].sum()
            target_power = ideal_power[mask].sum()
            initial_error, final_error = err0[mask].sum(), err1[mask].sum()
            return {"count": int(mask.sum()), "mix_power": float(mp), "mix_power_pct_of_batch": percent(mp, mixture_power.sum()),
                    "correction_power_pct_of_batch_mix": percent(correction, mixture_power.sum()),
                    "correction_power_pct_of_local_mix": percent(correction, mp), "residual_coefficient_rms": float(np.sqrt(residual_power[mask].mean())) if mask.any() else None,
                    "correction_over_ideal_rms_pct": 100 * np.sqrt(correction / target_power) if target_power else None,
                    "error_before": float(initial_error), "error_after": float(final_error),
                    "error_change_pct": percent(final_error - initial_error, initial_error),
                    "error_change_pct_of_batch_initial": percent(final_error - initial_error, err0.sum()),
                    "worse_bins_pct": percent(np.count_nonzero(err1[mask] > err0[mask]), mask.sum()),
                    "net_stem_power_change_pct_of_batch_mix": percent((after_power[mask] - before_power[mask]).sum(), mixture_power.sum()),
                    "correction_ideal_cosine": float(alignment[mask].sum() / np.sqrt(correction * actual_ideal_power[mask].sum())) if correction and target_power else None}

        def gains(mask):
            a, b, w = baseline_ratio[mask], output_ratio[mask], mixture_power[mask]
            before99, after99 = p99(a), p99(b)
            bw, aw = weighted_p99(a, w), weighted_p99(b, w)
            return {"count": int(mask.sum()), "before_p99": before99, "after_p99": after99,
                    "change_db": db(after99 / before99), "weighted_before_p99": bw, "weighted_after_p99": aw,
                    "weighted_change_db": db(aw / bw), "absolute_output_p99_db_over_mix": db(after99),
                    "ideal_target_p99": p99(ideal_ratio[mask]),
                    "ideal_target_change_db": db(p99(ideal_ratio[mask]) / before99),
                    "weighted_ideal_target_change_db": db(weighted_p99(ideal_ratio[mask], w) / bw),
                    "paired_output_minus_baseline_ratio_p99": p99(b - a)}

        masks = {"sampled_active": sampled & active, "active": active, "sampled_high": sampled & high,
                 "high": high, "mid_energy": active & ~high & (relative >= .01), "lower_active": active & (relative < .01)}
        table = {label: gains(mask) | measure(mask) for label, mask in masks.items()}
        cutoff = table["sampled_active"]["after_p99"]
        sampled_tail = masks["sampled_active"] & (output_ratio >= cutoff)
        tail = active & (output_ratio >= cutoff)
        bands = [{"low_hz": low, "high_hz": min(upper, SAMPLE_RATE / 2),
                  **measure(tail & ((freq >= low) & (freq < upper))[None, None, :, None])}
                 for low, upper in zip(BANDS, BANDS[1:])]
        relative_bands = [{"low_db": lo, "high_db": hi,
                           **measure(tail & (relative >= 10 ** (lo / 20)) & (relative < 10 ** (hi / 20)))}
                          for lo, hi in ((-61, -50), (-50, -40), (-40, -30), (-30, -20), (-20, 1))]
        per_frame = np.sum(np.where(tail, mixture_power, 0), axis=(1, 2))
        frames = []
        for linear in np.argsort(per_frame.ravel())[-12:][::-1]:
            segment, frame = map(int, np.unravel_index(linear, per_frame.shape))
            frames.append({"segment": segment, "frame": frame, "seconds": float(sample_seconds[frame]),
                           "tail_mix_energy_pct": percent(per_frame[segment, frame], per_frame.sum()),
                           "tail_bins": int(tail[segment, ..., frame].sum())})
        per_segment = [{"segment": segment, **measure(tail & (np.arange(x.shape[0]) == segment)[:, None, None, None])}
                       for segment in range(x.shape[0])]
        pw, tw, bw = p[name + "_waveform"], t[name + "_waveform"], wave0[:, idx]
        wave = []
        for segment, (a, b, ref) in enumerate(zip(pw, bw, tw)):
            energy = lambda z: float(np.mean(np.square(z, dtype=np.float64)))
            before_error, after_error = energy(b - ref), energy(a - ref)
            wave.append({"segment": segment, "gain_change_db": db(np.sqrt(energy(a) / energy(b))),
                         "baseline_rms_dbfs": db(np.sqrt(energy(b))), "estimate_rms_dbfs": db(np.sqrt(energy(a))),
                         "target_rms_dbfs": db(np.sqrt(energy(ref))),
                         "mix_rms_dbfs": db(np.sqrt(energy(saved['batch']['mixture'][segment].numpy()))),
                         "peak": float(np.abs(a).max()), "clip_count": int(np.count_nonzero(np.abs(a) > 1)),
                         "nonfinite_count": int(np.count_nonzero(~np.isfinite(a))),
                         "correction_rms_dbfs": db(np.sqrt(energy(a - b))),
                         "waveform_sdr_delta_db": db(np.sqrt(before_error / after_error))})
        offsets = [(np.arange(x.size) % 17 == offset).reshape(x.shape) & active for offset in range(17)]
        report["sources"][name] = {"gains": table, "all": measure(np.ones(x.shape, bool)),
            "sampled_tail": measure(sampled_tail), "full_tail": measure(tail), "tail_threshold_output_over_mix": cutoff,
            "tail_absolute_mix_magnitude": quantiles(magnitude[tail]), "tail_residual_magnitude": quantiles(np.abs(residual[tail])),
            "tail_relative_mix_magnitude": quantiles(relative[tail]), "tail_ideal_magnitude": quantiles(np.abs(ideal[tail])),
            "tail_by_frequency": bands, "tail_by_mix_level": relative_bands, "tail_top_frames": frames,
            "tail_frames": int(np.count_nonzero(per_frame)), "total_frames": int(per_frame.size),
            "top_frame_tail_energy_pct": {count: percent(np.sort(per_frame.ravel())[-count:].sum(), per_frame.sum())
                                          for count in (1, 10, 50)},
            "by_segment": [{"segment": segment, **measure(np.broadcast_to((np.arange(x.shape[0]) == segment)
                            [:, None, None, None], x.shape))} for segment in range(x.shape[0])],
            "tail_segment_stats": per_segment, "waveform": wave,
            "offset_scores_db": [db(p99(output_ratio[m]) / p99(baseline_ratio[m])) for m in offsets]}
    (ROOT / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
