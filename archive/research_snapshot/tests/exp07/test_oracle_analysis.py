import numpy as np
import pytest
import torch

from src.audio.stft import compute_istft, compute_stft
from src.experiments.exp07_phase_aware_residual.chunking import chunk_starts, overlap_add_chunks
from src.experiments.exp07_phase_aware_residual.oracle_analysis import (
    compute_oracle_chunk,
    evaluate_oracle_gate,
    score_estimate,
)


def test_oracles_use_original_stems_and_split_only_constrained_residual():
    samples = torch.arange(4096)
    v = (0.25 * torch.sin(samples * 0.1)).repeat(1, 2, 1)
    i = (0.25 * torch.cos(samples * 0.3)).repeat(1, 2, 1)
    mix = v + i + 0.02
    result = compute_oracle_chunk(mix, v, i)
    x, v_ref, i_ref = map(compute_stft, (mix, v, i))
    free, constrained = result["unconstrained_complex"], result["constrained_complex"]
    torch.testing.assert_close(free.vocals, v_ref)
    torch.testing.assert_close(free.instrumental, i_ref)
    assert (free.vocals + free.instrumental - x).abs().max() > 1
    torch.testing.assert_close(compute_istft(constrained.vocals, length=4096), v + 0.01, atol=1e-6, rtol=0)
    torch.testing.assert_close(compute_istft(constrained.instrumental, length=4096), i + 0.01, atol=1e-6, rtol=0)
    assert (constrained.vocals + constrained.instrumental - x).abs().max() <= 1e-5
    magnitude = result["mixture_phase"]
    torch.testing.assert_close(magnitude.vocals + magnitude.instrumental, x)
    active = x.abs() > 1
    effective_mask = (magnitude.vocals[active] / x[active])
    assert effective_mask.imag.abs().max() < 1e-6
    assert ((effective_mask.real >= 0) & (effective_mask.real <= 1)).all()


def test_scoring_keeps_original_reference_errors_visible():
    t = np.arange(44100, dtype=np.float32)
    vocal = np.repeat((0.25 * np.sin(0.1 * t))[:, None], 2, axis=1)
    inst = np.repeat((0.25 * np.cos(0.3 * t))[:, None], 2, axis=1)
    mixture = vocal + inst + 0.125
    score = score_estimate(mixture, vocal, inst, vocal + 0.0625, inst + 0.0625)
    assert score["vocals_waveform_rmse"] == pytest.approx(0.0625)
    assert score["instrumental_waveform_rmse"] == pytest.approx(0.0625)
    assert score["mixture_nrmse_pct"] < 0.001


def test_overlap_add_matches_non_aligned_tail_and_short_song():
    assert chunk_starts(13, 6, 0.5) == [0, 3, 6, 7]
    chunks = np.stack([np.arange(s, s + 6) for s in (0, 3, 6, 7)])[:, None, :]
    np.testing.assert_allclose(overlap_add_chunks(chunks, [0, 3, 6, 7], 13), [np.arange(13)])
    np.testing.assert_equal(overlap_add_chunks(chunks[:1], [0], 2), [[0, 1]])


def oracle_records(wins=10, vocal_gain=1.0, instrumental_gain=-0.5):
    def method(v, i):
        return {"vocals_median_sdr_db": v, "instrumental_median_sdr_db": i}

    return [{
        "track": str(idx),
        "methods": {"mixture_phase": method(8, 14),
                    "constrained_complex": method(8 + (vocal_gain if idx < wins else 0), 14 + instrumental_gain)},
        "numerical": {"unconstrained_source_nrmse_pct": 0.001, "unconstrained_source_max_abs": 1e-5,
                      "constrained_mixture_nrmse_pct": 0.001, "constrained_mixture_max_abs": 1e-5,
                      "constrained_complex_max_abs": 2e-5, "complex_max_error_to_bound": 1.0,
                      "complex_violations": 0, "nonfinite_count": 0},
    } for idx in range(15)]


@pytest.mark.parametrize("records", [
    oracle_records(wins=9), oracle_records(vocal_gain=0.99),
    oracle_records(instrumental_gain=-0.51), oracle_records()[:14],
])
def test_oracle_gate_rejects_each_invalid_headroom_condition(records):
    assert not evaluate_oracle_gate(records).passed


def test_gate_accepts_exact_bounds_but_fails_one_bad_numerical_track():
    records = oracle_records()
    assert evaluate_oracle_gate(records).passed
    records[0]["numerical"]["complex_max_error_to_bound"] = 1.01
    records[0]["numerical"]["complex_violations"] = 1
    assert not evaluate_oracle_gate(records).passed
