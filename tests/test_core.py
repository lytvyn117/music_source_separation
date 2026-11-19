import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch


class ValidationTests(unittest.TestCase):
    def test_fixed_frames_bleed_and_artist_macro(self):
        from music_separation.evaluation.evaluator import song_metrics, artist_macro
        rng = np.random.default_rng(3)
        v = rng.normal(0, .04, (2, 88200)).astype("float32")
        i = rng.normal(0, .07, v.shape).astype("float32")
        row = song_metrics(v + i, v, i, .75 * v, i + .25 * v)
        self.assertAlmostEqual(row["vocal_sdr_db"], 20 * np.log10(4), places=5)
        self.assertAlmostEqual(row["v_side_sdr_db"], 20 * np.log10(4), places=5)
        self.assertAlmostEqual(row["abs_vocal_bleed_coeff"], .25, places=6)
        summary = artist_macro([{"artist_key": a, **dict(row, vocal_sdr_db=s)}
                               for a, s in [("a", 1), ("a", 3), ("b", 8)]])
        self.assertEqual(summary["vocal_sdr_db"], 5)

    def test_missing_artist_metric_fails(self):
        from music_separation.evaluation.evaluator import artist_macro, METRICS
        with self.assertRaises(ValueError):
            artist_macro([{"artist_key": "a", **dict.fromkeys(METRICS, None)}])

    def test_guardrails_best_ties_and_patience(self):
        from music_separation.training.selection import decision
        policy = {"guardrails": {"i": {"min": 2}, "bleed": {"max": .2}},
                  "primary": "v", "min_delta_db": .15, "min_updates": 16000,
                  "patience_validations": 8}
        history = []
        for n in range(1, 18):
            m = {"v": 99 if n == 1 else (6.1 if n >= 9 else 6),
                 "i": 1 if n == 1 else 2, "bleed": .2}
            raw = {**m, "v": 100, "i": 0}
            state = decision(history, n * 1000, {"raw": raw, "ema": m}, policy)
            history.append({"step": n * 1000, "selection": state})
        self.assertFalse(history[0]["selection"]["candidates"]["ema"]["eligible"])
        self.assertFalse(history[8]["selection"]["stop"])
        self.assertEqual(history[13]["selection"]["best_qualified"]["step"], 9000)
        self.assertFalse(history[14]["selection"]["stop"])
        self.assertTrue(history[15]["selection"]["stop"])
        self.assertEqual(history[-1]["selection"]["anchor_step"], 2000)
        self.assertEqual(history[-1]["selection"]["best_qualified"],
                         {"step": 9000, "weights": "ema", "vocal_sdr_db": 6.1})
        self.assertEqual(history[-1]["selection"]["best_overall"],
                         {"step": 1000, "weights": "raw", "vocal_sdr_db": 100})

    def test_raw_can_win_qualified_and_same_update_tie_prefers_ema(self):
        from music_separation.training.selection import decision
        policy = {"guardrails": {"i": {"min": 2}}, "primary": "v", "min_delta_db": .15,
                  "min_updates": 16000, "patience_validations": 8}
        m = {"v": 6, "i": 2}
        start = decision([], 1000, {"raw": m, "ema": m}, policy)
        self.assertEqual(start["best_qualified"]["weights"], "ema")
        new = decision([{"step": 1000, "selection": start}], 2000,
                       {"raw": {"v": 8, "i": 2}, "ema": {"v": 9, "i": 1}}, policy)
        self.assertEqual(new["best_overall"], {"step": 2000, "weights": "ema", "vocal_sdr_db": 9})
        self.assertEqual(new["best_qualified"], {"step": 2000, "weights": "raw", "vocal_sdr_db": 8})
        self.assertEqual(new["anchor_step"], 1000)

    def test_raw_only_qualification_does_not_start_ema_patience(self):
        from music_separation.training.selection import decision
        policy = {"guardrails": {"i": {"min": 2}}, "primary": "v", "min_delta_db": .15,
                  "min_updates": 16000, "patience_validations": 8}
        metrics = {"raw": {"v": 6, "i": 2}, "ema": {"v": 9, "i": 1}}
        history = []
        for step in range(1000, 25000, 1000):
            state = decision(history, step, metrics, policy)
            history.append({"step": step, "selection": state})
        self.assertEqual(state["best_qualified"]["weights"], "raw")
        self.assertIsNone(state["anchor_step"])
        self.assertFalse(state["stop"])


class RunTests(unittest.TestCase):
    def test_ema_and_rng_restore(self):
        from music_separation.training.checkpoints import EMA, capture_rng, restore_rng
        model = torch.nn.Linear(2, 1, bias=False)
        ema = EMA(model, .999)
        initial = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(2)
        weights = model.weight.detach().clone()
        ema.update(model)
        torch.testing.assert_close(ema.weights["weight"], initial + .002)
        self.assertTrue(torch.equal(model.weight, weights))
        self.assertFalse(ema.weights["weight"].requires_grad)
        state = capture_rng(torch.device("cpu"))
        first = [random.random(), float(np.random.rand()), torch.rand(1).item()]
        restore_rng(state, torch.device("cpu"))
        self.assertEqual(first, [random.random(), float(np.random.rand()), torch.rand(1).item()])

    def test_interrupted_initial_publish_recovers_and_ignores_partial_number(self):
        from music_separation.training.checkpoints import Store, write_json
        from music_separation.training.trainer import open_state

        class Initial:
            identity = "test"

            def state(self):
                return {"identity": self.identity, "step": 0, "history": []}

        with TemporaryDirectory() as folder:
            path = Path(folder)
            record = {"identity": "test", "mode": "full"}
            with Store(path, "test", 1000, 24000) as store:
                (path / "run.json").write_text('{"identity":"test","mode":"full"}')
                partial = store.numbered / ".checkpoint-interrupted"
                partial.mkdir()
                (partial / "state.pt").write_bytes(b"partial")
            with Store(path, "test", 1000, 24000) as store:
                resumed = open_state(store, record, Initial(), resume=True)
                self.assertIsNone(resumed)
                self.assertEqual(store.recover()["step"], 0)
                import stat
                self.assertFalse(store.bundle(0).joinpath("state.pt").stat().st_mode & stat.S_IWRITE)

    def test_immutable_commit_recovery_and_writer_lock(self):
        from music_separation.training.checkpoints import Store, read_json
        with TemporaryDirectory() as folder:
            path = Path(folder)
            with Store(path, "test", 1000, 24000) as store:
                with self.assertRaises(OSError), Store(path, "test", 1000, 24000):
                    pass
                store.commit({"identity": "test", "step": 0, "history": []})
                qualified = {"step": 1000, "weights": "ema", "vocal_sdr_db": 6.}
                row = {"step": 1000, "selection": {"best_overall": qualified,
                                                    "best_qualified": qualified}}
                store.commit({"identity": "test", "step": 1000, "history": [row]})
                with self.assertRaises(FileExistsError):
                    store.commit({"identity": "test", "step": 1000, "history": [row]})
                (path / "latest.pt").unlink()
                (path / "best_overall.pt").unlink()
                (path / "best_qualified.pt").write_bytes(b"partial alias")
                (path / "validation_history.json").write_text("partial")
            with Store(path, "test", 1000, 24000) as store:
                state = store.recover()
                self.assertEqual(state["step"], 1000)
                self.assertEqual(read_json(path / "validation_history.json"), [row])
                for alias in ("best_overall", "best_qualified"):
                    self.assertEqual((path / f"{alias}.pt").read_bytes(), (path / "latest.pt").read_bytes())

                overall = {"step": 2000, "weights": "raw", "vocal_sdr_db": 9.}
                new = {"step": 2000, "selection": {"best_overall": overall,
                                                    "best_qualified": qualified}}
                store.commit({"identity": "test", "step": 2000, "history": [row, new]})
                store.recover()
                index = read_json(path / "checkpoint_index.json")
                for alias, winner in [("best_overall", overall), ("best_qualified", qualified)]:
                    self.assertTrue(all(index[alias][key] == value for key, value in winner.items()))
                self.assertEqual((path / "best_overall.pt").read_bytes(), (path / "latest.pt").read_bytes())
                self.assertEqual((path / "best_qualified.pt").read_bytes(),
                                 (store.bundle(1000) / "state.pt").read_bytes())

    def test_no_qualified_checkpoint_means_no_best_alias(self):
        from music_separation.training.checkpoints import Store, read_json
        with TemporaryDirectory() as folder:
            path = Path(folder)
            with Store(path, "test", 1000, 24000) as store:
                store.commit({"identity": "test", "step": 0, "history": []})
                overall = {"step": 1000, "weights": "raw", "vocal_sdr_db": 9.}
                row = {"step": 1000, "selection": {"best_overall": overall,
                                                    "best_qualified": None}}
                store.commit({"identity": "test", "step": 1000, "history": [row]})
                index = read_json(path / "checkpoint_index.json")
                self.assertIsNone(index["best_qualified"])
                self.assertEqual(index["qualification_status"], "no_candidate_qualified")
                self.assertTrue((path / "best_overall.pt").is_file())
                self.assertFalse((path / "best_qualified.pt").exists())


if __name__ == "__main__":
    unittest.main()
