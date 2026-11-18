"""Export the four replay segments at their original shared gain as float WAV."""

from pathlib import Path

import soundfile as sf
import torch

from config import SAMPLE_RATE
from src.audio.stft import compute_istft


ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    saved = torch.load(ROOT / "capture.pt", map_location="cpu", weights_only=True)
    assert saved["match"]
    p, t = saved["prediction"], saved["targets"]
    mix = saved["batch"]["mixture"]
    initial_vocal = p["initial_vocal_stft"]
    stems = {"mixture": mix}
    for name, initial in (("vocals", initial_vocal), ("instrumental", p["mixture_stft"] - initial_vocal)):
        stems["exp04_" + name] = compute_istft(initial, length=mix.shape[-1])
        stems["replay_" + name] = p[name + "_waveform"]
        stems["projected_" + name] = t[name + "_waveform"]
    for index in range(mix.shape[0]):
        directory = ROOT / "audio" / f"segment_{index + 1:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, batch in stems.items():
            wave = batch[index].T.numpy()
            assert wave.shape == (6 * SAMPLE_RATE, 2)
            output = directory / (name + ".wav")
            sf.write(output, wave, SAMPLE_RATE, subtype="FLOAT")
            decoded, rate = sf.read(output, dtype="float32")
            assert rate == SAMPLE_RATE and (decoded == wave).all()


if __name__ == "__main__":
    main()
