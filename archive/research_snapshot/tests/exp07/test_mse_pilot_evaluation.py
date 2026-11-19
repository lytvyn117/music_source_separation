import numpy as np

from src.experiments.exp07_phase_aware_residual.mse_evaluation import excerpt_start, waveform_artifacts


def test_excerpt_is_anchored_on_active_vocal_and_stays_in_track():
    v = np.zeros((50, 2), dtype=np.float32)
    v[30:40] = 1
    assert excerpt_start(v, 10, rate=1) == 30
    assert excerpt_start(v[:4], 10, rate=1) == 0


def test_waveform_diagnostics_preserve_peaks_and_find_clip_candidates():
    v = np.array([[0, 0], [1.25, -0.5], [0, 0]], dtype=np.float32)
    info = waveform_artifacts(v)
    assert info["waveform_peak"] == 1.25
    assert info["peak_sample_difference"] == 1.25
    assert info["clipped_sample_pct"] == 100 / 6
