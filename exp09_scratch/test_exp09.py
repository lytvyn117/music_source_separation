import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from audit import artist_key, wav_signatures
from loader import Shard


class PreparationTests(unittest.TestCase):
    def test_artist_alias_and_audio_signature(self):
        self.assertEqual(artist_key(" Sunny SIde Up "), artist_key("Sunny Side Up"))
        self.assertEqual(artist_key("Strugzy ft. Mr Peabody"), artist_key("Strugzy"))
        a = {"bytes": 40, "crc32": "08", "available": True}
        b = {"bytes": 40, "crc32": "09", "available": True}
        self.assertEqual(wav_signatures([a, b, a]), wav_signatures([b, a, a]))
        self.assertNotEqual(wav_signatures([a, b, a]), wav_signatures([a, b]))

    def test_staging_preserves_gain_and_alignment_and_enforces_budget(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            vocal = np.array([[.125, -.25], [.25, -.125], [0, .125]], dtype="float32")
            backing = vocal * -2
            for name, signal in (("v", vocal), ("i", backing), ("short", backing[:2])):
                sf.write(root / (name + ".wav"), signal, 44100, subtype="PCM_16")
            with ZipFile(root / "source.zip", "w") as archive:
                for name in ("v", "i", "short"):
                    archive.write(root / (name + ".wav"), name + ".wav")
            with ZipFile(root / "source.zip") as archive:
                def track(name, stem):
                    member = name + ".wav"
                    info = archive.getinfo(member)
                    return {"wav": member, "stem": stem, "bytes": info.file_size,
                            "crc32": f"{info.CRC:08x}", "available": True}
                song = {"uuid": "song", "tracks": [track("v", "vocals"), track("i", "bass")]}
                with Shard(root, 10000) as shard:
                    result = shard.stage(archive, song, verify=True)
                    self.assertEqual(result["reconstruction_max_abs"], 0)
                    x, v, i = shard.window("song", 1, 2)
                    np.testing.assert_array_equal(v, vocal[1:])
                    np.testing.assert_array_equal(i, backing[1:])
                    np.testing.assert_array_equal(x, vocal[1:] + backing[1:])
                    with self.assertRaises(ValueError):
                        shard.window("song", -1, 1)
                    short = {"uuid": "bad", "tracks": [track("v", "vocals"), track("short", "bass")]}
                    with self.assertRaisesRegex(ValueError, "format or length"):
                        shard.stage(archive, short)
                    self.assertNotIn("bad", shard.songs)
                with Shard(root, 40) as shard:
                    with self.assertRaisesRegex(ValueError, "budget"):
                        shard.stage(archive, song)
                    self.assertEqual(shard.songs, {})
