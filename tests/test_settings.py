import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from music_separation.settings import MODEL, load_inference_settings
from music_separation.paths import INFERENCE_CONFIG

class SettingsTests(unittest.TestCase):
    def test_invalid_or_architecture_overrides_are_rejected(self):
        baseline = json.loads(INFERENCE_CONFIG.read_text(encoding="utf-8"))
        for change in ({"hop_samples": 0}, {"hop_samples": MODEL.segment_samples + 1},
                       {"threads": True}, {"seed": -1}, {"width": 128},
                       {"attention_backends": []}, {"attention_backends": "MATH"}):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**baseline, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_inference_settings(path)

    def test_custom_runtime_settings_do_not_change_model_profile(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"hop_samples": 44100, "threads": 2,
                "seed": 909, "attention_backends": ["MATH"]}), encoding="utf-8")
            settings = load_inference_settings(path)
            self.assertEqual(settings.hop_samples, 44100)
            self.assertEqual(MODEL.segment_samples, 176400)
