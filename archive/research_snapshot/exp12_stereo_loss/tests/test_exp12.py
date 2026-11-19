"""Synthetic Exp12 loss and frozen protocol checks."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import random
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "exp10_scratch"
sys.path.insert(0, str(ROOT / "exp09_core"))


def setUpModule():
    torch.set_num_threads(2)


def fixture(samples=44100):
    t = torch.arange(samples) / 44100
    mid = 0.08 * torch.sin(2 * torch.pi * 394 * t)
    side = 0.02 * torch.cos(2 * torch.pi * 677 * t)
    return torch.stack((mid + side, mid - side))[None]


class LossTests(unittest.TestCase):
    def setUp(self):
        from loss_candidates import get_loss
        from scratch_model import N_FFT

        self.loss = get_loss("normalized_mse_side_aware")
        self.window = type("Window", (), {"window": torch.hann_window(N_FFT)})()
        self.wave = fixture()
        self.zero = torch.zeros_like(self.wave)

    def check(self, mix, vocal, estimate, expected=None):
        from loss_candidates import WEIGHTS

        estimate = estimate.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            parts = self.loss(self.window, mix, estimate, vocal)
        self.assertTrue(
            all(x.ndim == 0 and x.dtype == torch.float32 and x.isfinite() for x in parts.values())
        )
        if expected is not None:
            torch.testing.assert_close(
                parts["side_l1"], torch.tensor(expected), atol=2e-7, rtol=1e-6
            )
        g = torch.autograd.grad(
            sum(parts[k] * w for k, w in WEIGHTS["normalized_mse_bleed"].items()), estimate
        )[0]
        self.assertTrue(g.isfinite().all())
        return parts

    def test_identical_stereo_channels(self):
        x = torch.full_like(self.wave, 0.125)
        self.check(x, x, x / 2, 0.0)

    def test_opposite_phase_channels(self):
        x = torch.tensor([0.125, -0.125])[None, :, None].expand_as(self.wave).contiguous()
        self.check(x, x, x / 2, 0.5)

    def test_silent_side_target_and_loud_mid(self):
        mix = torch.full_like(self.wave, 0.125)
        e = torch.tensor([0.03125, -0.03125])[None, :, None]
        self.check(mix, mix, mix + e, 0.5)

    def test_silent_vocal_target(self):
        self.assertEqual(float(self.check(self.wave, self.zero, self.wave / 5)["leakage"]), 0.0)

    def test_silent_instrumental_target(self):
        self.check(self.wave, self.wave, self.wave * 0.5)

    def test_low_amplitude_side_and_common_gain(self):
        m, s = self.wave.mean(1), (self.wave[:, 0] - self.wave[:, 1]) * 1e-5
        x = torch.stack((m + s, m - s), 1)
        a = self.check(x, x, self.wave)
        b = self.check(x * 0.1, x * 0.1, self.wave * 0.1)
        torch.testing.assert_close(a["side_l1"], b["side_l1"], atol=1e-6, rtol=1e-6)

    def test_unilateral_left_right(self):
        for levels in ([0.125, 0.0], [0.0, 0.125]):
            x = torch.tensor(levels)[None, :, None].expand_as(self.wave).contiguous()
            self.check(x, x, x * 0.5, 0.5 / math.sqrt(1.25))

    def test_exact_reconstruction_and_gradient(self):
        y = self.wave.clone().requires_grad_()
        parts = self.loss(self.window, self.wave * 2, y, self.wave)
        self.assertTrue(all(float(v.detach()) == 0.0 for v in parts.values()))
        self.assertEqual(
            int(torch.count_nonzero(torch.autograd.grad(sum(parts.values()), y)[0])), 0
        )

    def test_known_error_under_absolute_floor(self):
        eps = torch.tensor([1e-5, -1e-5])[None, :, None]
        self.check(self.zero, self.zero, self.zero + eps, 0.2)
        self.check(self.zero, self.zero, self.zero, 0.0)

    def test_gradient_direction_and_batch_time_reduction(self):
        x = torch.stack((torch.full_like(self.wave[0], 0.25), torch.full_like(self.wave[0], 0.5)))
        y = x.clone()
        y[0, 0, :22050] += 0.125
        y[0, 1, :22050] -= 0.125
        y[1, 0] -= 0.125
        y[1, 1] += 0.125
        y.requires_grad_()
        side = self.loss(self.window, x, y, x)["side_l1"]
        torch.testing.assert_close(side, torch.tensor(0.5))
        (g,) = torch.autograd.grad(side, y)
        torch.testing.assert_close(g[0, 0, 0], torch.tensor(1 / (44100 * 0.5)))
        torch.testing.assert_close(g[1, 0, 0], torch.tensor(-1 / 44100))
        self.assertTrue(torch.equal(g[:, 0], -g[:, 1]))
        self.assertEqual(int(torch.count_nonzero(g[0, :, 22050:])), 0)

    def test_disabled_path_exactly_reproduces_exp10_and_only_side_changes(self):
        from loss_candidates import normalized_mse_loss

        spec = spec_from_file_location("original_loss", BASE / "exp09_core" / "loss_candidates.py")
        original = module_from_spec(spec)
        spec.loader.exec_module(original)
        y = (self.wave * 0.62).requires_grad_()
        args = (self.window, self.wave, y, self.wave * 0.55)
        baseline = original.normalized_mse_loss(*args)
        disabled = normalized_mse_loss(*args, side_aware=False)
        selected = self.loss(*args)
        for name in baseline:
            (g1,) = torch.autograd.grad(baseline[name], y, retain_graph=True)
            (g2,) = torch.autograd.grad(disabled[name], y, retain_graph=True)
            self.assertTrue(torch.equal(baseline[name], disabled[name]))
            self.assertTrue(torch.equal(g1, g2))
            if name != "side_l1":
                (g3,) = torch.autograd.grad(selected[name], y, retain_graph=True)
                self.assertTrue(torch.equal(baseline[name], selected[name]))
                self.assertTrue(torch.equal(g1, g3))
        self.assertGreater(float(selected["side_l1"].detach()), float(baseline["side_l1"].detach()))


class ProtocolTests(unittest.TestCase):
    def test_model_parameter_count_seed_and_mixture_consistency(self):
        from scratch_model import Separator, seed_all
        from scratch_train import model_digest
        from train_exp12 import bind

        cfg, _ = bind()
        seed_all(909, deterministic=True)
        model = Separator(checkpoint_blocks=True).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11662596)
        self.assertEqual(model_digest(model), cfg["initial_model_sha256"])
        x = fixture(4096)
        with torch.inference_mode():
            v, i = model(x)
        torch.testing.assert_close(v + i, x, rtol=0.0, atol=1e-7)

    def test_training_addresses_match_all_24000_logged_exp10_batches(self):
        from data import ARCHIVE, load_roster
        from train_exp12 import BatchSource

        s = BatchSource(ARCHIVE, None, load_roster(), 909)
        with (BASE / "runs" / "exp10_seed909" / "steps.jsonl").open(encoding="utf-8") as handle:
            count = 0
            for index, line in enumerate(handle):
                row = json.loads(line)
                group, crops = s.address(index)
                self.assertEqual(
                    (row["step"], row["shard"], row["crops"]),
                    (index + 1, group, [list(x) for x in crops]),
                )
                count += 1
        self.assertEqual(count, 24000)

    def test_train_and_dev_archive_reject_unapproved_roles(self):
        from data import RestrictedArchive, load_roster, open_references

        roster = load_roster()
        train, dev = roster["train"], roster["development"]
        self.assertFalse(train.keys() & dev.keys())
        self.assertFalse(
            {r["artist_key"] for r in train.values()} & {r["artist_key"] for r in dev.values()}
        )
        self.assertEqual(roster["holdout"], "external and sealed")
        dev_name = next(iter(dev.values()))["tracks"][0]["wav"]
        train_name = next(iter(train.values()))["tracks"][0]["wav"]
        with TemporaryDirectory() as temp:
            archive = Path(temp) / "synthetic.zip"
            names = (train_name, dev_name, "synthetic_holdout/test.wav")
            with ZipFile(archive, "w") as z:
                for name in names:
                    z.writestr(name, b"fixture")
            for allowed, own, other in ((train, train_name, dev_name), (dev, dev_name, train_name)):
                with RestrictedArchive(archive, allowed) as restricted:
                    with restricted.open(own) as stream:
                        self.assertEqual(stream.read(), b"fixture")
                    for blocked in (other, names[-1]):
                        with self.assertRaises(ValueError):
                            restricted.open(blocked)
        with self.assertRaises(ValueError), open_references(dev, next(iter(train))):
            pass

    def test_resume_and_deterministic_replay_with_synthetic_module(self):
        from checkpoint import Store
        from scratch_model import seed_all
        from scratch_train import model_digest
        from train_exp12 import Experiment, bind, open_state

        class SmallSeparator(torch.nn.Module):
            def __init__(self, checkpoint_blocks):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(0.5))
                self.register_buffer("window", torch.hann_window(2048), persistent=False)

            def forward(self, x):
                vocal = x * self.gain
                return vocal, x - vocal

        def draws():
            return [random.random(), float(np.random.rand()), float(torch.rand(1))]

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for k in a:
                    equal(a[k], b[k])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                self.assertEqual(a, b)

        cfg, _ = bind()
        cfg["initial_model_sha256"] = model_digest(SmallSeparator(True))
        device = torch.device("cpu")
        record = {"identity": "synthetic-resume"}
        x = fixture()
        seed_all(909, deterministic=True)
        with TemporaryDirectory() as temp, patch("train_exp12.Separator", SmallSeparator):
            path = Path(temp)
            a = Experiment(cfg, record["identity"], device)
            with Store(path, record["identity"], 1, 6000) as store:
                self.assertIsNone(open_state(store, record, a, resume=False))
                a.update((x, x * 0.8))
                a.history.append({"step": 1, "selection": {}})
                store.commit(a.state())
            expected_draws = draws()
            expected = a.update((x, x * 0.6))
            a.history.append({"step": 2, "selection": {}})
            expected_state = deepcopy(a.state())
            b = Experiment(cfg, record["identity"], device)
            with Store(path, record["identity"], 1, 6000) as store:
                b.restore(open_state(store, record, b, resume=True))
                self.assertEqual(b.raw.scheduler.T_max, 24000)
                self.assertEqual(b.step, 1)
                self.assertEqual(draws(), expected_draws)
                self.assertEqual(b.update((x, x * 0.6)), expected)
                b.history.append({"step": 2, "selection": {}})
                equal(expected_state, b.state())
                with self.assertRaises(ValueError):
                    open_state(store, {"identity": "different"}, b, resume=True)


class EvaluationTests(unittest.TestCase):
    def test_matched_report_reads_committed_step6000_and_checks_all_crops(self):
        from checkpoint import file_hash, read_json, write_json
        from evaluate_pilot import BASELINE, SIDE, report
        from train_exp12 import bind, provenance, replay_log, track_compute_failure
        from validation import artist_macro

        config, _ = bind()
        record = provenance(config, {}, {}, "pilot")
        history = read_json(BASELINE / "validation_history.json")[:6]
        for arm in history[-1]["validation"].values():
            for row in arm["songs"]:
                for metric in SIDE:
                    row[metric] += 0.3
            arm["artist_macro"] = artist_macro(arm["songs"])
        done = {"step": 6000, "reason": "pilot_boundary", "selection": history[-1]["selection"]}
        with TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "run.json", record)
            write_json(run / "completion.json", done)
            write_json(run / "validation_history.json", history)
            path = run / "latest.pt"
            torch.save(
                {
                    "step": 6000,
                    "identity": record["identity"],
                    "history": history,
                    "ema": {"updates": 6000},
                    "scheduler": {"last_epoch": 6000, "T_max": 24000},
                },
                path,
            )
            write_json(
                run / "checkpoint_index.json",
                {
                    "latest": {
                        "step": 6000,
                        "identity": record["identity"],
                        "bytes": path.stat().st_size,
                        "sha256": file_hash(path),
                    }
                },
            )
            with (
                (BASELINE / "steps.jsonl").open(encoding="utf-8") as src,
                (run / "steps.jsonl").open("w", encoding="utf-8") as dst,
            ):
                for _ in range(6000):
                    dst.write(next(src))
            result = report(run)
            self.assertEqual(result["stability_and_sampler"]["finite_matched_updates"], 6000)
            self.assertTrue(result["both_arms_pass_screening"])
            self.assertTrue(result["stability_and_sampler"]["clean_compute_history"])
            for arm in result["arms"].values():
                self.assertIsNone(arm["vocal_si_sdr_paired_delta"])
                self.assertEqual((len(arm["songs"]), len(arm["artists"])), (16, 7))
                self.assertTrue(arm["screening_gates"]["vocal_side_sdr"])
                self.assertTrue(arm["screening_gates"]["mixture_consistency"])
            with (run / "steps.jsonl").open("a", encoding="utf-8") as dst:
                dst.write('{"step":6001}\n')
            with self.assertRaisesRegex(ValueError, "exactly 6000"):
                report(run)
            with self.assertRaises(KeyboardInterrupt), track_compute_failure(run, 1000, "update"):
                raise KeyboardInterrupt
            replay_log(run / "steps.jsonl", 6000)
            self.assertTrue(report(run)["both_arms_pass_screening"])
            with self.assertRaises(RuntimeError), track_compute_failure(run, 1000, "validation"):
                raise RuntimeError("synthetic compute failure")
            replay_log(run / "steps.jsonl", 6000)
            result = report(run)
            self.assertFalse(result["both_arms_pass_screening"])
            checks = result["stability_and_sampler"]
            self.assertFalse(checks["clean_compute_history"])
            self.assertEqual(checks["first_compute_failure"]["phase"], "validation")

    def test_side_gain_requires_artist_spread_and_bleed_nonregression(self):
        from data import load_roster
        from checkpoint import read_json
        from evaluate_pilot import BASELINE, SIDE, BLEED, compare
        from validation import artist_macro

        old = read_json(BASELINE / "validation_history.json")[5]["validation"]["ema"]
        dev = load_roster()["development"]
        candidate = deepcopy(old)

        def judge():
            candidate["artist_macro"] = artist_macro(candidate["songs"])
            return compare(candidate, old, dev)

        for row in candidate["songs"]:
            for key in SIDE:
                row[key] += 0.3
        self.assertTrue(judge()["passes_screening"])
        for row in candidate["songs"]:
            row[BLEED] += 0.011
        self.assertFalse(judge()["screening_gates"]["absolute_vocal_bleed"])
        for row in candidate["songs"]:
            row["mixture_max_abs"] += 1e-5
        self.assertFalse(judge()["screening_gates"]["mixture_consistency"])
        candidate = deepcopy(old)
        lucky = candidate["songs"][0]["artist_key"]
        for row in candidate["songs"]:
            if row["artist_key"] == lucky:
                for key in SIDE:
                    row[key] += 2.0
        report = judge()
        self.assertTrue(report["screening_gates"]["vocal_side_sdr"])
        self.assertFalse(report["screening_gates"]["artist_spread"])
        self.assertFalse(report["passes_screening"])


class CudaTests(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
        "requires CUDA BF16",
    )
    def test_full_size_cuda_bf16_backward_without_updates(self):
        from time import perf_counter
        from torch.nn.attention import sdpa_kernel
        from loss_candidates import WEIGHTS, get_loss
        from scratch_model import SEGMENT, Separator
        from scratch_train import model_digest
        from train_exp12 import BACKENDS, setup

        setup(909)
        model = Separator(checkpoint_blocks=True).cuda().train()
        before = model_digest(model)
        source = fixture(SEGMENT)
        mono = source.mean(1, keepdim=True).expand_as(source)
        left = source.clone()
        left[:, 1] = 0.0
        x = torch.cat((source, mono, left, torch.zeros_like(source))).cuda()
        target = x * 0.77
        params = tuple(model.parameters())
        for objective in ("normalized_mse_bleed", "normalized_mse_side_aware"):
            fn = get_loss(objective)
            times = []
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for index in range(4):
                torch.cuda.synchronize()
                start = perf_counter()
                with sdpa_kernel(BACKENDS):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        v, i = model(x)
                    parts = fn(model, x, v, target)
                    weights = WEIGHTS["normalized_mse_bleed"]
                    gradients = torch.autograd.grad(
                        sum(weights[key] * val for key, val in parts.items()), params
                    )
                torch.cuda.synchronize()
                if index:
                    times.append(perf_counter() - start)
                self.assertTrue(
                    all(t.dtype == torch.float32 and bool(t.isfinite()) for t in parts.values())
                )
                self.assertTrue(all(bool(t.isfinite().all()) for t in gradients))
                torch.testing.assert_close(v + i, x, rtol=0.0, atol=1e-7)
                del v, i, parts, gradients
            print(
                f"{objective}: median forward/backward {np.median(times):.4f}s, "
                f"peak allocated {torch.cuda.max_memory_allocated()/1024**3:.3f} GiB, "
                f"peak reserved {torch.cuda.max_memory_reserved()/1024**3:.3f} GiB",
                flush=True,
            )
        self.assertEqual(before, model_digest(model))
        self.assertTrue(all(p.grad is None for p in params))


if __name__ == "__main__":
    unittest.main()
