"""Regression checks for the package migration."""
import importlib.util
from pathlib import Path
import unittest
import torch
from music_separation.models.bs_roformer import Separator
from music_separation.paths import PROJECT_ROOT
from music_separation.training.trainer import bind, provenance

class LayoutTests(unittest.TestCase):
    def test_model_matches_frozen_source(self):
        source = PROJECT_ROOT / "archive/research_snapshot/exp10_scratch/exp09_core/scratch_model.py"
        spec = importlib.util.spec_from_file_location("frozen_model", source)
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            torch.manual_seed(909)
            original = frozen.Separator(group_chunk=8).eval()
            current = Separator(group_chunk=8).eval()
            current.load_state_dict(original.state_dict(), strict=True)
            x = torch.randn(1, 2, 2048) * .03
            with torch.inference_mode():
                expected = original(x)
                actual = current(x)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            torch.set_num_threads(threads)

    def test_provenance_binds_shared_sources(self):
        config, roster = bind()
        evidence = provenance(config, {}, {}, "layout-test")
        self.assertEqual(evidence["schema"], 3)
        self.assertIn("src/music_separation/models/bs_roformer.py", evidence["files_sha256"])
        self.assertIn("configs/training/exp10/roster.json", evidence["files_sha256"])
        self.assertEqual(len(roster["train"]), 60)
