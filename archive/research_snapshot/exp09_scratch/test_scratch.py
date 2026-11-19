import unittest
import io
import struct
import tempfile
from pathlib import Path

import torch
from scratch_model import Separator, BANDS, seed_all, crop_for
from scratch_train import ExactRun, calibrate
from eligibility import pcm24, wave_match, trailing_bwf, FP_RATE
from shard_plan import partition, sample_position
import numpy as np


class ScratchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_trailing_bwf_timestamp_and_padding(self):
        timestamp = 158794358
        bext = bytearray(603)
        struct.pack_into("<Q", bext, 338, timestamp)
        chunks = b"LIST" + struct.pack("<I", 3) + b"abc\x00"
        chunks += b"bext" + struct.pack("<I", len(bext)) + bytes(bext) + b"\x00"
        self.assertEqual(trailing_bwf(io.BytesIO(chunks)), timestamp)
        with self.assertRaises(ValueError):
            trailing_bwf(io.BytesIO(chunks[:-2]))

    def test_pcm_sign_and_offset_gain_invariant_duplicate_probe(self):
        codes = np.array([[-8388608, -1], [0, 8388607], [83, -84]], dtype="<i4")
        encoded = codes.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        np.testing.assert_array_equal(pcm24(encoded), codes)
        rng = np.random.default_rng(9)
        original = rng.normal(size=FP_RATE * 30).astype("float32")
        padded = np.pad(original * .4, (900, 300))
        match = wave_match(original, padded)
        self.assertEqual(len(match), 3)
        self.assertTrue(all(r["correlation"] > .9999 for r in match))
        self.assertTrue(all(r["lag_seconds"] == 900 / FP_RATE for r in match))

    def test_dimensions_and_round_trip(self):
        self.assertEqual((len(BANDS), sum(BANDS)), (62, 1025))
        model = Separator()
        self.assertTrue(11_500_000 <= sum(p.numel() for p in model.parameters()) <= 12_500_000)
        x = torch.randn(1, 2, 5131)
        z = model.spectrum(x)
        self.assertEqual(z.shape, (1, 2, 1025, 12))
        torch.testing.assert_close(model.waveform(z, x.shape[-1]), x, atol=1.5e-6, rtol=1e-5)

    def test_stereo_impulse_timing_and_silent_channel(self):
        x = torch.zeros(1, 2, 5307)
        x[0, 0, 99] = .5
        x[0, 1, 3007] = -.25
        model = Separator()
        z = model.spectrum(x)
        wave = model.waveform(z, x.shape[-1])
        self.assertEqual(wave.abs().argmax(dim=-1).tolist(), [[99, 3007]])
        torch.testing.assert_close(x, wave, atol=1e-7, rtol=1e-6)
        x[:, 1].zero_()
        with torch.no_grad():
            v, i = model(x)
        self.assertEqual((v[:, 1].count_nonzero() + i[:, 1].count_nonzero()).item(), 0)

    def test_shard_budget_and_resumed_sampling(self):
        records = {key: (frames, 100) for key, frames in (("a", 4000), ("b", 5000), ("c", 6000))}
        plan = partition(records, 380000)
        self.assertEqual(sorted(k for shard in plan for k in shard), ["a", "b", "c"])
        self.assertEqual(plan, partition(dict(reversed(list(records.items()))), 380000))
        stream = [sample_position(plan, records, 1000, 9, step, 4, 3) for step in range(10)]
        self.assertEqual(stream[7:], [sample_position(plan, records, 1000, 9, step, 4, 3)
                                      for step in range(7, 10)])
        self.assertTrue(all(all(key in plan[shard] for key, _ in samples) for shard, samples in stream))
        with self.assertRaises(ValueError):
            partition(records, 1000)

    def test_seed_and_sampling(self):
        seed_all(76)
        a = Separator().state_dict()
        seed_all(76)
        b = Separator().state_dict()
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        seed_all(77)
        self.assertFalse(torch.equal(a["project.0.weight"], Separator().state_dict()["project.0.weight"]))
        rows = {"b": 3000, "a": 2000}
        self.assertEqual(crop_for(rows, 1000, 9, 2), crop_for(dict(reversed(list(rows.items()))), 1000, 9, 2))
        with self.assertRaises(ValueError):
            crop_for(rows, 4000, 9, 2)

    def test_stability_and_every_block_gradient(self):
        seed_all(17)
        model = Separator(checkpoint_blocks=True)
        x = torch.randn(1, 2, 4410) * .1
        v, i, mask = model(x, return_mask=True)
        self.assertTrue(mask.abs().max() <= 2.000001)
        self.assertEqual(mask.dtype, torch.complex64)
        torch.testing.assert_close(v + i, x, rtol=0, atol=1e-7)
        (v.square().mean() + .5 * i.abs().mean()).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(p.grad.isfinite().all(), name)
            self.assertGreater(p.grad.abs().max().item(), 0, name)
        with torch.no_grad():
            v, i = model(torch.zeros_like(x))
            self.assertEqual(v.count_nonzero().item() + i.count_nonzero().item(), 0)

    def test_calibration_and_exact_cpu_resume(self):
        seed_all(81, deterministic=True)
        try:
            model = Separator()
            x = torch.randn(1, 2, 4410) * .05
            target = x.roll(3, dims=-1) * .3
            weights, evidence = calibrate(model, [(x, target)])
            self.assertTrue(all(value > 0 for value in weights.values()))
            run = ExactRun(model, weights, seed=81, amp=False)
            run.step(x, target)
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
                file = Path(temp) / "test.pt"
                run.save(file, "synthetic-v1")
                rng = torch.rand(6)
                a = run.step(x, target)
                expected = {k: v.clone() for k, v in model.state_dict().items()}
                replay = ExactRun(Separator(), weights, seed=81, amp=False)
                with self.assertRaisesRegex(ValueError, "identity"):
                    replay.restore(file, "wrong")
                replay.restore(file, "synthetic-v1")
                self.assertTrue(torch.equal(rng, torch.rand(6)))
                self.assertEqual(replay.index, 1)
                self.assertEqual(a, replay.step(x, target))
                self.assertTrue(all(torch.equal(v, replay.model.state_dict()[k])
                                    for k, v in expected.items()))
        finally:
            torch.use_deterministic_algorithms(False)
