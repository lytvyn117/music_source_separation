import unittest
from unittest.mock import patch
import numpy as np
import torch
from music_separation.inference.separate import inference_device, separate, verify_audio

class HalfModel(torch.nn.Module):
    def forward(self, wave):
        return wave * .5, wave * .5

class CpuInferenceTests(unittest.TestCase):
    def test_device_resolution(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.set_device") as set_device:
            self.assertEqual(inference_device(requested="auto").type, "cpu")
            self.assertEqual(inference_device(requested="cpu").type, "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                inference_device(requested="cuda")
            set_device.assert_not_called()
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_device"):
            self.assertEqual(inference_device(requested="auto").type, "cuda")
            self.assertEqual(inference_device(requested="cpu").type, "cpu")

    def test_cpu_overlap_and_short_inputs_without_cuda_calls(self):
        for length in (3, 220501):
            mix = np.random.default_rng(42).normal(0, .03, (2, length)).astype("float32")
            with patch("torch.cuda.synchronize", side_effect=AssertionError("CUDA called")), patch("torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("CUDA called")), patch("torch.cuda.max_memory_allocated", side_effect=AssertionError("CUDA called")):
                vocals, backing, stats = separate(HalfModel(), mix, torch.device("cpu"))
            np.testing.assert_allclose(vocals, mix * .5, atol=1e-8)
            verify_audio(mix, vocals, backing)
            self.assertIsNone(stats["peak_gpu_allocated_mib"])
