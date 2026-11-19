from music_separation.settings import MODEL
from typing import Dict, Tuple

import numpy as np


def compute_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute global Signal-to-Distortion Ratio (SDR) in dB (MDX/utterance-level definition).

    SDR = 10 * log10( ||reference||^2 / (||reference - estimate||^2 + eps) )
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)

    num = np.sum(ref ** 2)
    den = np.sum((ref - est) ** 2)

    if num < eps:
        return float("nan")

    return float(10.0 * np.log10((num + eps) / (den + eps)))


def compute_si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Removes mean and projects estimate onto reference per channel or flattened stereo.
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)

    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    ref_energy = np.sum(ref ** 2)
    if ref_energy < eps:
        return float("nan")

    alpha = np.dot(est, ref) / (ref_energy + eps)
    s_target = alpha * ref
    e_noise = est - s_target

    target_energy = np.sum(s_target ** 2)
    noise_energy = np.sum(e_noise ** 2)

    return float(10.0 * np.log10((target_energy + eps) / (noise_energy + eps)))


def compute_framewise_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = MODEL.sample_rate,
    window_seconds: float = 1.0,
    silence_threshold_db: float = -45.0,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute frame-level (1-second window) median SDR and SI-SDR following
    MUSDB18 / SiSEC evaluation conventions, ignoring silent reference frames.
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)

    if ref.ndim == 2 and ref.shape[0] == 2 and ref.shape[1] != 2:
        ref = ref.T
        est = est.T

    total_samples = ref.shape[0]
    win_samples = int(round(window_seconds * sample_rate))

    # Minimum RMS energy threshold to count a 1s frame as active
    min_frame_rms = 10.0 ** (silence_threshold_db / 20.0)

    frame_sdrs = []
    frame_si_sdrs = []

    for start in range(0, total_samples - win_samples + 1, win_samples):
        r_frame = ref[start : start + win_samples]
        e_frame = est[start : start + win_samples]

        rms = np.sqrt(np.mean(r_frame ** 2))
        if rms < min_frame_rms:
            continue

        sdr_val = compute_sdr(r_frame, e_frame, eps=eps)
        si_sdr_val = compute_si_sdr(r_frame, e_frame, eps=eps)

        if np.isfinite(sdr_val):
            frame_sdrs.append(sdr_val)
        if np.isfinite(si_sdr_val):
            frame_si_sdrs.append(si_sdr_val)

    if not frame_sdrs:
        return {
            "median_sdr_db": float("nan"),
            "mean_sdr_db": float("nan"),
            "median_si_sdr_db": float("nan"),
            "active_frames": 0,
        }

    return {
        "median_sdr_db": float(np.median(frame_sdrs)),
        "mean_sdr_db": float(np.mean(frame_sdrs)),
        "median_si_sdr_db": float(np.median(frame_si_sdrs)),
        "active_frames": int(len(frame_sdrs)),
    }


def evaluate_stem_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray,
    sample_rate: int = MODEL.sample_rate,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute comprehensive time-domain separation metrics for a single source stem
    (vocals or instrumental), including baseline mixture comparisons.
    """
    global_sdr = compute_sdr(reference, estimate, eps=eps)
    global_si_sdr = compute_si_sdr(reference, estimate, eps=eps)

    mix_sdr = compute_sdr(reference, mixture, eps=eps)
    mix_si_sdr = compute_si_sdr(reference, mixture, eps=eps)

    frame_stats = compute_framewise_metrics(
        reference, estimate, sample_rate=sample_rate, eps=eps
    )
    mix_frame_stats = compute_framewise_metrics(
        reference, mixture, sample_rate=sample_rate, eps=eps
    )

    ref_f = np.asarray(reference, dtype=np.float64)
    est_f = np.asarray(estimate, dtype=np.float64)

    waveform_mae = float(np.mean(np.abs(ref_f - est_f)))
    waveform_rmse = float(np.sqrt(np.mean((ref_f - est_f) ** 2)))

    ref_energy = float(np.sum(ref_f ** 2))
    est_energy = float(np.sum(est_f ** 2))
    energy_ratio_db = float(10.0 * np.log10((est_energy + eps) / (ref_energy + eps)))

    return {
        "global_sdr_db": global_sdr,
        "median_sdr_db": frame_stats["median_sdr_db"],
        "global_si_sdr_db": global_si_sdr,
        "median_si_sdr_db": frame_stats["median_si_sdr_db"],
        "sdr_improvement_db": float(global_sdr - mix_sdr),
        "median_sdr_improvement_db": float(
            frame_stats["median_sdr_db"] - mix_frame_stats["median_sdr_db"]
        ),
        "si_sdr_improvement_db": float(global_si_sdr - mix_si_sdr),
        "waveform_mae": waveform_mae,
        "waveform_rmse": waveform_rmse,
        "energy_ratio_db": energy_ratio_db,
        "active_frames": frame_stats["active_frames"],
    }


def compute_leakage_diagnostics(
    vocals_ref: np.ndarray,
    inst_ref: np.ndarray,
    vocals_est: np.ndarray,
    inst_est: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Quantify cross-source leakage:
    1. Fraction of ground-truth vocal energy leaking into the predicted instrumental track.
    2. Fraction of ground-truth instrumental energy leaking into the predicted vocal track.
    """
    v_ref = np.asarray(vocals_ref, dtype=np.float64).reshape(-1)
    i_ref = np.asarray(inst_ref, dtype=np.float64).reshape(-1)
    v_est = np.asarray(vocals_est, dtype=np.float64).reshape(-1)
    i_est = np.asarray(inst_est, dtype=np.float64).reshape(-1)

    # Error in instrumental prediction: i_est - i_ref
    inst_error = i_est - i_ref
    voc_error = v_est - v_ref

    v_energy = np.sum(v_ref ** 2) + eps
    i_energy = np.sum(i_ref ** 2) + eps

    # Least-squares projection of inst_error onto v_ref (how much of true vocals is present in inst_est - i_ref)
    vocal_bleed_coeff = np.dot(inst_error, v_ref) / v_energy
    vocal_bleed_energy_pct = float((vocal_bleed_coeff ** 2) * 100.0)

    inst_bleed_coeff = np.dot(voc_error, i_ref) / i_energy
    inst_bleed_energy_pct = float((inst_bleed_coeff ** 2) * 100.0)

    return {
        "vocal_to_inst_bleed_coeff": float(vocal_bleed_coeff),
        "vocal_to_inst_bleed_energy_pct": vocal_bleed_energy_pct,
        "inst_to_vocal_bleed_coeff": float(inst_bleed_coeff),
        "inst_to_vocal_bleed_energy_pct": inst_bleed_energy_pct,
    }
