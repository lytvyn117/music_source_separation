from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf
import torch

from src.experiments.exp09_pretrained.inference import SonySeparator
from src.experiments.exp04_exp08_hybrid.inference import read_stereo
from reports.exp09_zero_shot_preparation.windows_probe import CHUNK, OVERLAP, load_sony
from torch.nn.attention import SDPBackend


RATE = 44100


def synthetic_stereo(length):
    t = np.arange(length, dtype=np.float64) / RATE
    return np.stack((.24 * np.sin(2 * np.pi * 211 * t),
                     .055 * np.cos(2 * np.pi * 617 * t))).astype(np.float32)


class SyntheticChunkModel:
    def inference(self, batch):
        assert batch.dtype == torch.float32
        assert batch.shape == (1, 2, CHUNK)
        vocal = batch * batch.new_tensor([.25, .625]).view(1, 2, 1)
        return torch.stack((vocal, batch - vocal), dim=1)


class AdapterTests(unittest.TestCase):
    def test_odd_tail_keeps_channel_order_and_scale(self):
        mixture = synthetic_stereo(CHUNK * 2 + 71)
        separator = SonySeparator(SyntheticChunkModel(), "cpu")
        outputs = separator.separate(mixture)
        self.assertEqual(outputs.shape, (2, 2, mixture.shape[1]))
        self.assertEqual(outputs.dtype, np.float32)
        np.testing.assert_allclose(outputs[0], mixture * [[.25], [.625]], atol=3e-8, rtol=0)
        np.testing.assert_allclose(outputs[1], mixture - outputs[0], atol=0, rtol=0)

    def test_rejects_nonstereo_empty_and_nonfinite(self):
        separator = SonySeparator(SyntheticChunkModel(), "cpu")
        for x in (np.empty((2, 0)), np.zeros((1, 20)), np.array([[np.nan], [0]])):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                separator.separate(x)

    def test_file_round_trip_and_existing_output_guard(self):
        separator = SonySeparator(SyntheticChunkModel(), "cpu")
        mixture = synthetic_stereo(139)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            output = root / "result"
            sf.write(input_path, mixture.T, RATE, subtype="FLOAT")
            separator.separate_file(input_path, output)
            for stem, expected in zip(("vocals", "instrumental"), separator.separate(mixture), strict=True):
                path = output / f"{stem}.wav"
                self.assertEqual(sf.info(path).subtype, "FLOAT")
                np.testing.assert_array_equal(read_stereo(path), expected)
            with self.assertRaises(FileExistsError):
                separator.separate_file(input_path, output)
            sf.write(input_path, mixture.T, 48000, subtype="FLOAT")
            with self.assertRaises(ValueError):
                separator.separate_file(input_path, root / "other")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for official release parity")
class UpstreamParityTests(unittest.TestCase):
    def test_real_checkpoint_matches_upstream_short_seams_and_tail(self):
        model, source, functions, *_ = load_sony()
        separator = SonySeparator.from_release()
        for module in model.modules():
            if isinstance(module, source.Attend):
                module.cuda_config = [SDPBackend.EFFICIENT_ATTENTION]

        class NoAutocast:
            autocast = staticmethod(nullcontext)

        for length in (127, CHUNK, CHUNK + OVERLAP, CHUNK * 2 + 71):
            with self.subTest(length=length):
                mixture = synthetic_stereo(length)
                with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                    expected = functions["separate"](
                        model, torch.from_numpy(mixture).cuda(), CHUNK, OVERLAP, 1, [1], NoAutocast()
                    )[0, 0].cpu().numpy()
                actual = separator.separate(mixture)
                np.testing.assert_array_equal(actual[0], expected)
                np.testing.assert_array_equal(actual[1], mixture - expected)

