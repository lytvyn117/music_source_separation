import numpy as np
import pytest
import torch

from src.audio.stft import compute_istft, compute_stft
from src.experiments.exp07_phase_aware_residual.chunking import (
    chunk_batch, chunk_starts, overlap_add_chunks,
)
from src.experiments.exp07_phase_aware_residual.config import Exp07Paths
from src.experiments.exp07_phase_aware_residual.oracle_analysis import CHUNK_SAMPLES, _run_oracles
from src.inference.separate_track import load_separation_model, separate_waveform


@pytest.mark.parametrize("length,signal", [
    (1537, "endpoints"), (CHUNK_SAMPLES + 513, "endpoints"),
    (CHUNK_SAMPLES + CHUNK_SAMPLES // 2 + 317, "impulses"),
    (CHUNK_SAMPLES + 19, "silence"),
])
def test_full_song_stereo_roundtrip_preserves_boundaries(length, signal):
    mix = np.zeros((2, length), dtype=np.float32)
    if signal == "endpoints":
        mix[:] = np.array([[0.25], [-0.125]], dtype=np.float32)
        mix[:, -1] *= -2
    if signal == "impulses":
        for index in (0, CHUNK_SAMPLES // 2, CHUNK_SAMPLES, length - 1):
            mix[:, index] = [0.375, -0.625]
    starts = chunk_starts(length, CHUNK_SAMPLES, 0.5)
    batches = torch.from_numpy(chunk_batch(mix, starts, CHUNK_SAMPLES))
    spectrum = compute_stft(batches)
    vocals = spectrum * 0.25
    instrumental = spectrum - vocals
    stems = [overlap_add_chunks(compute_istft(part, length=CHUNK_SAMPLES).numpy(), starts, length)
             for part in (vocals, instrumental)]
    assert all(stem.shape == mix.shape for stem in stems)
    np.testing.assert_allclose(stems[0], mix * 0.25, rtol=0, atol=2e-6)
    np.testing.assert_allclose(stems[1], mix * 0.75, rtol=0, atol=2e-6)
    np.testing.assert_allclose(stems[0] + stems[1], mix, rtol=0, atol=2e-6)


def test_exp04_matches_legacy_inference_with_only_the_endpoint_envelope_changed():
    checkpoint = Exp07Paths().backbone
    if not checkpoint.is_file():
        pytest.skip("Local Exp04 checkpoint is required")
    model, _, device = load_separation_model(checkpoint)
    length = CHUNK_SAMPLES + 101
    time = np.arange(length, dtype=np.float32) / 44100
    mix = np.stack([0.25 * np.cos(440 * np.pi * time), 0.15 * np.cos(330 * np.pi * time)], axis=1)
    legacy = separate_waveform(model, mix, device=device, overlap_ratio=0.5, batch_size=2)
    estimates, _, _ = _run_oracles((mix, mix * 0.5, mix * 0.5), 0.5, 2, device, model=model)
    corrected = estimates["exp04"]
    weight = np.hanning(CHUNK_SAMPLES + 2)[1:9].astype(np.float32)
    envelope = np.ones((length, 1), dtype=np.float32)
    envelope[:8, 0] = weight / 1e-8
    envelope[-8:, 0] = envelope[:8, 0][::-1]
    for channels, name in ((slice(0, 2), "vocals"), (slice(2, 4), "instrumental")):
        np.testing.assert_allclose(corrected[:, channels] * envelope, legacy[name], atol=1e-6, rtol=0)
    np.testing.assert_allclose(corrected[:, :2] + corrected[:, 2:], mix, atol=2e-6, rtol=0)
