from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import torch

from remix import RemixSource, distribution
from train_exp11 import Experiment, PILOT_UPDATES, bind
from checkpoint import Store, file_hash, read_json
from evaluate_pilot import EXP10_RUN, additional_metrics, archived_baseline, compare, pilot_diagnostics
from scratch_model import N_FFT, RATE, SEGMENT, seed_all


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "exp10_scratch"


def address_in_worker(payload):
    roster, update = payload
    return RemixSource(None, None, roster, 909).address(update)


def fixture(folder, nonfinite=False):
    archive_path = folder / "fixture.zip"
    rows, frames = {}, {}
    with ZipFile(archive_path, "w") as archive:
        for number, key in enumerate(("a", "b", "c", "development", "holdout"), 1):
            n = SEGMENT + 10 * number
            timeline = np.arange(n, dtype=np.float32) / n
            vocal = np.stack((number * .12 + timeline * .01, -.05 - timeline * .03), axis=1)
            inst = np.stack((.35 + number * .04 + timeline * .01, number * .11 - timeline * .02), axis=1)
            if nonfinite and key == "b":
                inst[1, 0] = np.nan
            rows[key] = {"uuid": key, "tracks": []}
            frames[key] = n
            for name, audio in (("vocals", vocal), ("drums", inst)):
                member = f"data/{key}/{name}.wav"
                output = io.BytesIO()
                sf.write(output, audio, RATE, format="WAV", subtype="FLOAT")
                archive.writestr(member, output.getvalue())
                info = archive.getinfo(member)
                rows[key]["tracks"].append({
                    "wav": member, "stem": name, "available": True,
                    "bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                })
    train = {k: rows[k] for k in ("a", "b", "c")}
    roster = {"train": train, "development": {"development": rows["development"]},
              "holdout": "external and sealed",
              "shard_plan": {"shards": [["a", "b"], ["c"]],
                             "frames_by_uuid": {k: frames[k] for k in train},
                             "updates_per_shard": 2, "budget_bytes": 40 * 1024**2}}
    return archive_path, roster


class DataTests(unittest.TestCase):
    def test_frozen_addresses_isolate_all_approved_training_rows(self):
        _, roster = bind()
        approved = set(roster["train"])
        seen_v, seen_i = set(), set()
        remixed = 0
        with (BASELINE / "runs/exp10_seed909/steps.jsonl").open(encoding="utf-8") as log:
            source = RemixSource(None, None, roster, 909)
            for update, line in enumerate(log):
                baseline = json.loads(line)
                group, crops = source.address(update)
                self.assertEqual(group, baseline["shard"])
                self.assertEqual([[r.vocal_uuid, r.vocal_start] for r in crops], baseline["crops"])
                for position, crop in enumerate(crops):
                    self.assertEqual((crop.update, crop.position), (update, position))
                    self.assertLessEqual(crop.instrumental_start + SEGMENT,
                                         source.frames[crop.instrumental_uuid][0])
                    if crop.remixed:
                        self.assertNotEqual(crop.vocal_uuid, crop.instrumental_uuid)
                    else:
                        self.assertEqual((crop.vocal_uuid, crop.vocal_start),
                                         (crop.instrumental_uuid, crop.instrumental_start))
                    remixed += crop.remixed
                    seen_v.add(crop.vocal_uuid)
                    seen_i.add(crop.instrumental_uuid)
            self.assertEqual(update + 1, 24000)
        self.assertEqual(seen_v, approved)
        self.assertEqual(seen_i, approved)
        self.assertLess(abs(remixed / 96000 - .5), .01)

    def test_process_workers_and_resume_addresses(self):
        _, roster = bind()
        updates = [5999, 0, 6000, 500, 999, 4, 5999, 23999]
        expected = [address_in_worker((roster, u)) for u in updates]
        with ProcessPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(address_in_worker, [(roster, u) for u in updates]))
        self.assertEqual(actual, expected)

    def test_real_reader_remix_targets_mixture_and_stereo(self):
        with TemporaryDirectory(dir=ROOT) as name:
            folder = Path(name)
            archive_path, roster = fixture(folder)
            with ZipFile(archive_path) as raw:
                direct = {}
                for key in roster["train"]:
                    for stem in ("vocals", "drums"):
                        wave, rate = sf.read(io.BytesIO(raw.read(f"data/{key}/{stem}.wav")),
                                             dtype="float32", always_2d=True)
                        self.assertEqual(rate, RATE)
                        direct[key, stem] = wave
            with RemixSource(archive_path, folder, roster, 909) as source:
                self.assertNotIn("holdout", source.readers)
                self.assertNotIn("development", source.readers)
                for key in ("holdout", "development"):
                    with self.assertRaises(ValueError):
                        source.archive.getinfo(f"data/{key}/vocals.wav")
                both = set()
                for update in range(5):
                    _, crops, mixtures, vocals, instruments = source.arrays(update)
                    for ix, crop in enumerate(crops):
                        v = direct[crop.vocal_uuid, "vocals"][crop.vocal_start:crop.vocal_start + SEGMENT].T
                        i = direct[crop.instrumental_uuid, "drums"][crop.instrumental_start:crop.instrumental_start + SEGMENT].T
                        np.testing.assert_array_equal(vocals[ix], v)
                        np.testing.assert_array_equal(instruments[ix], i)
                        np.testing.assert_array_equal(mixtures[ix], v + i)
                        self.assertEqual(vocals.dtype, np.float32)
                        self.assertFalse(np.array_equal(v[0], v[1]))
                        both.add(crop.remixed)
                self.assertEqual(both, {False, True})
                before = source.arrays(7)
            with RemixSource(archive_path, folder, roster, 909) as reopened:
                after = reopened.arrays(7)
                self.assertEqual(before[:2], after[:2])
                for left, right in zip(before[2:], after[2:]):
                    np.testing.assert_array_equal(left, right)

    def test_remix_disabled_matches_actual_exp10_batch(self):
        spec = spec_from_file_location("baseline_launcher", BASELINE / "train_exp10.py")
        baseline = module_from_spec(spec)
        spec.loader.exec_module(baseline)
        with TemporaryDirectory(dir=ROOT) as name:
            folder = Path(name)
            archive_path, roster = fixture(folder)
            with RemixSource(archive_path, folder, roster, 909, enabled=False) as source:
                with baseline.BatchSource(archive_path, folder, roster, 909) as original:
                    with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
                        for update in (0, 1, 2, 17):
                            group, crops, mix, vocal, _ = source.arrays(update)
                            old_group, old_crops, (expected_mix, expected_vocal) = original.batch(update)
                            self.assertEqual(group, old_group)
                            self.assertEqual([(r.vocal_uuid, r.vocal_start) for r in crops], old_crops)
                            for actual, expected in ((mix, expected_mix), (vocal, expected_vocal)):
                                np.testing.assert_array_equal(actual.view(np.uint32),
                                                              expected.numpy().view(np.uint32))

    def test_nonfinite_audio_fails_during_staging(self):
        with TemporaryDirectory(dir=ROOT) as name:
            folder = Path(name)
            archive_path, roster = fixture(folder, nonfinite=True)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                with RemixSource(archive_path, folder, roster, 909):
                    self.fail("Corrupt audio entered the training path")

    def test_peak_rms_and_no_peak_clipping(self):
        wave = np.array([[2., 0.], [-2., 0.]], dtype=np.float32)
        summary = distribution(wave)
        self.assertEqual(summary["peak"], 2.0)
        self.assertAlmostEqual(summary["rms"], np.sqrt(2))
        self.assertEqual(summary["fraction_over_full_scale"], .5)


class TrainingContractTests(unittest.TestCase):
    def test_seed_parameter_count_unchanged_objective_and_cosine_horizon(self):
        config, _ = bind()
        self.assertEqual(config, read_json(BASELINE / "config.json"))
        seed_all(909, deterministic=True)
        run = Experiment(config, "exp11-test", torch.device("cpu"))
        self.assertEqual(sum(p.numel() for p in run.raw.model.parameters()), 11662596)
        self.assertEqual(PILOT_UPDATES, 6000)
        self.assertEqual(run.raw.weights, config["exp09_settings"]["loss"])
        self.assertEqual(run.raw.optimizer.defaults["weight_decay"], .01)
        self.assertEqual(run.raw.optimizer.defaults["lr"], .0003)
        self.assertEqual(run.raw.scheduler.T_max, 24000)
        self.assertAlmostEqual(run.raw.scheduler.eta_min, .00003)
        for filename in ("config.json", "roster.json", "checkpoint.py", "validation.py",
                         "selection.py", *[p.relative_to(ROOT) for p in (ROOT / "exp09_core").glob("*.py")]):
            self.assertEqual(file_hash(ROOT / filename), file_hash(BASELINE / filename), filename)

    def test_checkpoint_schema_inference_and_resume_identity(self):
        config, _ = bind()
        seed_all(909, deterministic=True)
        cpu = torch.device("cpu")
        run = Experiment(config, "exp11-test", cpu)
        with TemporaryDirectory(dir=ROOT) as name:
            folder = Path(name)
            with Store(folder, "exp11-test", 1000, 6000) as store:
                store.commit(run.state())
                state = store.recover()
            expected = torch.rand(4)
            run.restore(state)
            torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)
            self.assertEqual(run.step, 0)
            reference = torch.load(BASELINE / "runs/exp10_seed909/latest.pt",
                                   map_location="cpu", weights_only=True)
            self.assertEqual(state.keys(), reference.keys())
            self.assertEqual(state["raw_model"].keys(), reference["raw_model"].keys())
            self.assertEqual(state["ema"]["weights"].keys(), reference["ema"]["weights"].keys())
            with self.assertRaisesRegex(ValueError, "does not match"):
                run.restore(reference)

    def test_nonzero_checkpoint_replays_toy_update_and_sampler_bitwise(self):
        class TinySeparator(torch.nn.Module):
            def __init__(self, checkpoint_blocks):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor([.2, .3]).view(1, 2, 1))
                self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

            def forward(self, mix):
                estimate = self.gain * mix * torch.rand(())
                return estimate, mix - estimate

        torch.set_num_threads(2)
        config, _ = bind()
        seed_all(909, deterministic=True)
        cpu = torch.device("cpu")
        with (
            TemporaryDirectory(dir=ROOT) as name,
            patch("train_exp11.Separator", TinySeparator),
            patch("train_exp11.model_digest", return_value=config["initial_model_sha256"]),
        ):
            folder = Path(name)
            archive_path, roster = fixture(folder)
            run = Experiment(config, "toy-run", cpu)
            with RemixSource(archive_path, folder, roster, 909) as source:
                def batch(update):
                    _, _, x, y, _ = source.arrays(update)
                    return tuple(torch.from_numpy(w[:1, :, :RATE].copy()) for w in (x, y))

                run.update(batch(0))
                policy = config["selection"]
                scores = {name: 0. for name in policy["guardrails"]}
                scores[policy["primary"]] = 0.
                run.record({arm: {"artist_macro": scores} for arm in ("raw", "ema")})
                with Store(folder, "toy-run", 1, 6000) as store:
                    store.commit(run.state())
                    expected_log = run.update(batch(run.step))
                    expected = run.state()
                    resumed = Experiment(config, "toy-run", cpu)
                    resumed.restore(store.recover())
                    self.assertEqual(resumed.step, 1)
                    self.assertEqual(resumed.update(batch(resumed.step)), expected_log)
                    actual = resumed.state()
                    self.assertEqual(actual["scheduler"], expected["scheduler"])
                    self.assertEqual(actual["history"], expected["history"])
                    for a, b in ((actual["raw_model"], expected["raw_model"]),
                                 (actual["ema"]["weights"], expected["ema"]["weights"]),
                                 (actual["optimizer"]["state"][0], expected["optimizer"]["state"][0])):
                        for key in a:
                            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


class EvaluationReportingTests(unittest.TestCase):
    def test_full_metrics_keep_the_original_song_scores_and_projection(self):
        r = np.random.default_rng(2)
        vocals = r.normal(0, .1, (2, RATE * 2)).astype(np.float32)
        inst = r.normal(0, .1, vocals.shape).astype(np.float32)
        mix, pred_v = vocals + inst, .75 * vocals
        result = additional_metrics(mix, vocals, inst, pred_v, mix - pred_v)
        self.assertAlmostEqual(result["vocal_retention"], .75, places=6)
        self.assertAlmostEqual(result["vocal_bleed_energy_pct"], 6.25, places=5)
        self.assertTrue(np.isfinite(result["vocal_si_sdr_db"]))

    def test_paired_fallback_reconstructs_archived_macro_and_all_participants(self):
        fallback = archived_baseline(EXP10_RUN)
        history = read_json(EXP10_RUN / "validation_history.json")
        for w, arm in fallback.items():
            original = next(row for row in history if row["step"] == 6000)["validation"][w]
            self.assertEqual(len(arm["songs"]), 16)
            self.assertEqual(len(arm["artists"]), 7)
            self.assertEqual({k: arm["artist_macro"][k] for k in original["artist_macro"]},
                             original["artist_macro"])
            comparison = compare(arm, arm)
            self.assertTrue(all(comparison["preliminary_nonregression"].values()))
            self.assertIsNone(comparison["artist_macro_delta"]["vocal_si_sdr_db"])

    def test_logged_amplitudes_are_grouped_and_nonfinite_updates_fail(self):
        config, _ = bind()
        settings = config["exp09_settings"]
        crops = [{**distribution(np.full((2, 10), peak)), "remixed": bool(i % 2)}
                 for i, peak in enumerate((.5, .6, .7, 1.2))]
        row = {**dict.fromkeys(settings["loss"], .2), "step": 1, "crops": crops,
               "loss": .8, "gradient_l2": 1.4, "peak_reserved_gib": 3.}
        with TemporaryDirectory(dir=ROOT) as name, patch("evaluate_pilot.PILOT_UPDATES", 1):
            run = Path(name)
            log = run / "steps.jsonl"
            log.write_text(json.dumps(row), encoding="utf-8")
            result = pilot_diagnostics(run, settings)
            self.assertEqual(result["finite_logged_updates"], 1)
            self.assertEqual(result["amplitudes"]["original"]["crops_over_full_scale"], 0)
            self.assertEqual(result["amplitudes"]["remixed"]["crops_over_full_scale"], 1)
            log.write_text(json.dumps({**row, "gradient_l2": float("inf")}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                pilot_diagnostics(run, settings)
