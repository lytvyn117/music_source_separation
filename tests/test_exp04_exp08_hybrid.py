import numpy as np
import pytest
import soundfile as sf
import sys
import torch

from src.experiments.exp04_exp08_hybrid import inference
from src.experiments.exp04_exp08_hybrid.inference import fixed_blend, read_stereo, write_pair


def test_fixed_blend_preserves_channels_indices_and_levels():
    mix = np.array([[.4, -.9, 1.2, 0.], [-.2, .7, .1, 0.]], np.float32)
    first = np.array([[.1, .2, 1.3, .25], [.4, -.3, .1, 0.]], np.float32)
    second = np.array([[.7, -.2, 1.5, 0.], [.1, .1, .5, 0.]], np.float32)
    out = fixed_blend(mix, first, second)
    assert out.shape == (2, 2, 4)
    np.testing.assert_array_equal(out[0], .5 * first + .5 * second)
    np.testing.assert_array_equal(out[1], mix - out[0])
    np.testing.assert_allclose(out.sum(0), mix, atol=1e-7)
    assert out[0, 0, 2] > 1.


def test_misaligned_and_nonfinite_inputs_fail():
    stereo = np.zeros((2, 31), np.float32)
    with pytest.raises(ValueError, match="aligned"):
        fixed_blend(stereo, stereo[:, :-1], stereo)
    with pytest.raises(ValueError, match="finite"):
        fixed_blend(stereo, stereo + np.nan, stereo)


def test_file_roundtrip_preserves_float_samples_and_refuses_overwrite(tmp_path):
    mix = np.array([[1.25, .4, -.1], [-1.2, .2, .1]], dtype=np.float32)
    path = tmp_path / "mix.wav"
    sf.write(path, mix.T, 44100, subtype="FLOAT")
    np.testing.assert_array_equal(read_stereo(path), mix)
    pair = fixed_blend(mix, mix, mix)
    out = tmp_path / "separated"
    write_pair(out, pair)
    for stem, values in zip(("vocals", "instrumental"), pair):
        wav = out / f"{stem}.wav"
        info = sf.info(wav)
        assert (info.samplerate, info.frames, info.channels, info.subtype) == (44100, 3, 2, "FLOAT")
        np.testing.assert_array_equal(sf.read(wav, dtype="float32")[0].T, values)
    with pytest.raises(FileExistsError):
        write_pair(out, pair)


@pytest.mark.parametrize("rate,channels", [(48000, 2), (44100, 1)])
def test_unsupported_input_format_is_rejected(tmp_path, rate, channels):
    path = tmp_path / "mix.wav"
    sf.write(path, np.zeros((30, channels)), rate)
    with pytest.raises(ValueError, match="44.1 kHz stereo"):
        read_stereo(path)


def test_cli_enables_determinism_before_loading_models(tmp_path, monkeypatch):
    path = tmp_path / "mix.wav"
    sf.write(path, np.zeros((20, 2)), 44100)
    out = tmp_path / "outputs"

    class CheckedModel:
        def __init__(self, device):
            assert torch.are_deterministic_algorithms_enabled()

        def separate(self, waveform):
            return np.stack((waveform, waveform))

    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    monkeypatch.setattr(inference, "HybridSeparator", CheckedModel)
    monkeypatch.setattr(sys, "argv", ["inference", str(path), str(out), "--device", "cpu"])
    torch.use_deterministic_algorithms(False)
    try:
        inference.main()
        assert (out / "vocals.wav").is_file()
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)
