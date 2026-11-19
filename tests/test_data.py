from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile


class RuntimeRosterTests(unittest.TestCase):
    def test_frozen_training_and_development_members_are_disjoint(self):
        from music_separation.data.moisesdb import load_roster

        roster = load_roster()
        training = roster["train"]
        development = roster["development"]
        self.assertEqual(len(training), 60)
        self.assertEqual(len(development), 16)
        self.assertEqual(roster["holdout"], "external and sealed")
        self.assertEqual(len({song["artist_key"] for song in training.values()}), 26)
        self.assertEqual(len({song["artist_key"] for song in development.values()}), 7)
        self.assertFalse(training.keys() & development.keys())
        self.assertFalse({song["artist_key"] for song in training.values()}
                         & {song["artist_key"] for song in development.values()})
        training_wavs = {t["wav"] for song in training.values() for t in song["tracks"]}
        development_wavs = {t["wav"] for song in development.values() for t in song["tracks"]}
        self.assertTrue(training_wavs)
        self.assertTrue(development_wavs)
        self.assertFalse(training_wavs & development_wavs)

    def test_archive_rejects_unapproved_names_and_zipinfo(self):
        from music_separation.data.moisesdb import RestrictedArchive

        with TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.zip"
            approved_name = "training/vocals.wav"
            sealed_name = "sealed/vocals.wav"
            with ZipFile(path, "w") as archive:
                archive.writestr(approved_name, b"approved payload")
                archive.writestr(sealed_name, b"sealed payload")
                approved = archive.getinfo(approved_name)
                sealed = archive.getinfo(sealed_name)
            rows = {"allowed": {"tracks": [{"wav": approved_name, "available": True,
                                            "bytes": approved.file_size,
                                            "crc32": f"{approved.CRC:08x}"}]}}
            with RestrictedArchive(path, rows) as checked:
                with checked.open(checked.getinfo(approved_name)) as member:
                    self.assertEqual(member.read(), b"approved payload")
                with checked.open(approved_name) as member:
                    self.assertEqual(member.read(), b"approved payload")
                with self.assertRaises(ValueError):
                    checked.getinfo(sealed_name)
                with self.assertRaises(ValueError):
                    checked.open(sealed_name)
                with self.assertRaises(ValueError):
                    checked.open(sealed)
                # A caller-provided ZipInfo can retain the sealed local-header offset
                # and original name even if its public filename is made to look safe.
                sealed.filename = approved_name
                try:
                    with checked.open(sealed) as member:
                        self.assertEqual(member.read(), b"approved payload")
                except ValueError:
                    pass

    def test_single_role_cannot_read_other_role_wavs(self):
        from music_separation.data.moisesdb import RestrictedArchive, load_roster

        roster = load_roster()
        training_name = next(iter(roster["train"].values()))["tracks"][0]["wav"]
        validation_name = next(iter(roster["development"].values()))["tracks"][0]["wav"]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr(training_name, b"training")
                archive.writestr(validation_name, b"validation")
                tiny_roster = {
                    role: {role: {"tracks": [{"wav": name, "available": True,
                                              "bytes": archive.getinfo(name).file_size,
                                              "crc32": f"{archive.getinfo(name).CRC:08x}"}]}}
                    for role, name in (("train", training_name), ("development", validation_name))
                }
            with RestrictedArchive(path, tiny_roster["train"]) as training:
                with self.assertRaises(ValueError):
                    training.open(validation_name)
            with RestrictedArchive(path, tiny_roster["development"]) as validation:
                with self.assertRaises(ValueError):
                    validation.open(training_name)


if __name__ == "__main__":
    unittest.main()
