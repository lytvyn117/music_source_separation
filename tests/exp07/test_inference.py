import numpy as np

from src.experiments.exp07_phase_aware_residual.inference import separate_waveform, measure_full_song_consistency
from src.experiments.exp07_phase_aware_residual.model import PhaseAwareResidualSeparator
from src.experiments.exp07_phase_aware_residual.oracle_analysis import CHUNK_SAMPLES
from tests.exp07.test_model import SmallBackbone


def test_stereo_full_song_and_all_ablations_sum_through_seams():
    model = PhaseAwareResidualSeparator(SmallBackbone())
    model.refinement.final.bias.data[:] = 0.02
    length = CHUNK_SAMPLES + 993
    signal = np.arange(length, dtype=np.float32)
    mixture = np.stack((0.1 * np.cos(signal * 0.12), 0.2 * np.cos(signal * 0.05)))
    mixture[:, CHUNK_SAMPLES // 2] = [0.4, -0.4]
    results = separate_waveform(model, mixture, batch_size=1, include_ablations=True)
    for name in ("full", "magnitude_only", "phase_only"):
        pair = results[name]
        assert pair.vocals.shape == mixture.shape
        score = measure_full_song_consistency(mixture, pair, results.chunk_starts)
        assert score.global_nrmse_pct <= 0.001
        assert max(score.global_max_abs, score.max_seam_window_abs) <= 1e-5
    short = separate_waveform(model, mixture[:, :1009].T)
    assert short["full"].vocals.shape == (1009, 2)
