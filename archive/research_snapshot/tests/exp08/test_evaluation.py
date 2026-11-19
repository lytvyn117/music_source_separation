import numpy as np
import pytest
import soundfile as sf

from src.experiments.exp08_subband.evaluate import grade, score_song, write_blind_bundle


def test_inactive_metrics_are_nan():
    wave = np.zeros((2, 44100), dtype=np.float32)
    row = score_song("quiet", np.stack([wave, wave]), np.stack([wave, wave]), wave)
    assert np.isnan(row["vocal_sdr"])
    assert row["consistency"] == 0


def test_blind_audio_uses_float_and_preserves_samples(tmp_path):
    mix = np.array([[1.3, .4, -.1], [-1.2, .2, .1]], dtype=np.float32)
    pair = np.stack((mix * .4, mix * .6))
    write_blind_bundle(tmp_path, "clip01", mix, pair, pair, pair * 2, swap=True)
    result, rate = sf.read(tmp_path / "clip01" / "A_vocals.wav", dtype="float32")
    assert rate == 44100
    np.testing.assert_array_equal(result.T, pair[0] * 2)
    assert sf.info(tmp_path / "clip01" / "mixture.wav").subtype == "FLOAT"


def test_go_no_go_joins_tracks_and_keeps_listening_pending():
    baseline = [{"track": str(i), "vocal_sdr": 6.7415, "instrumental_sdr": 12.7874,
                 "vocal_si_sdr": 5.773, "bleed": .2139, "consistency": 0.,
                 "side_error": [0., 0.], "clipped": [0., 0.]} for i in range(15)]
    candidate = [row | {"vocal_sdr": 7.2415, "instrumental_sdr": 13.0874,
                        "vocal_si_sdr": 6.173, "bleed": .1839} for row in baseline]
    assert grade(baseline, list(reversed(candidate)))["decision"] == "PENDING_LISTENING"
    listening = {"clearer": 8, "lower_bleed": 8, "instrumentals_equal_or_better": 10,
                 "material_artifact": False}
    assert grade(baseline, candidate, listening)["decision"] == "GO"
    candidate[0]["clipped"] = [.1, .1]
    assert grade(baseline, candidate, listening)["decision"] == "NO-GO"
    with pytest.raises(ValueError):
        grade(baseline, candidate[:-1])
