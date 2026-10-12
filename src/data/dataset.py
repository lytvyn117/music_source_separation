import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import (
    PROCESSED_DIR,
    SAMPLE_RATE,
    SEGMENT_LENGTH_SECONDS,
    SAMPLES_PER_EPOCH,
)


class MUSDBDataset(Dataset):

    def __init__(
        self,
        data_dir=PROCESSED_DIR / "train",
        track_files=None,
        segment_length=SEGMENT_LENGTH_SECONDS,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        random_segments=True,
        seed=42,
    ):
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.samples_per_epoch = samples_per_epoch
        self.random_segments = random_segments
        self.seed = seed

        if track_files is not None:
            self.track_files = list(track_files)
        else:
            self.track_files = sorted(
                path
                for path in self.data_dir.iterdir()
                if path.is_dir()
            )

        if not self.track_files:
            raise FileNotFoundError(
                f"No processed MUSDB tracks found in {self.data_dir}"
            )

        self.segment_samples = int(
            self.segment_length * SAMPLE_RATE
        )

        self.track_info = []

        for track_path in self.track_files:
            mixture_path = track_path / "mixture.wav"

            info = sf.info(mixture_path)

            self.track_info.append({
                "path": track_path,
                "frames": info.frames,
            })

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):

        if self.random_segments:
            track = random.choice(self.track_info)
            rng = random
        else:
            track_index = index % len(self.track_info)
            track = self.track_info[track_index]
            rng = random.Random(self.seed + index)

        track_path = track["path"]
        total_frames = track["frames"]

        max_start = max(
            0,
            total_frames - self.segment_samples
        )

        if max_start > 0:
            start_frame = rng.randint(0, max_start)
        else:
            start_frame = 0

        mixture = self._read_segment(
            track_path / "mixture.wav",
            start_frame,
        )

        vocals = self._read_segment(
            track_path / "vocals.wav",
            start_frame,
        )

        instrumental = self._read_segment(
            track_path / "instrumental.wav",
            start_frame,
        )

        return {
            "mixture": mixture,
            "vocals": vocals,
            "instrumental": instrumental,
            "track_name": track_path.name,
            "start": start_frame / SAMPLE_RATE,
        }

    def _read_segment(self, path, start_frame):

        with sf.SoundFile(path) as audio_file:
            audio_file.seek(start_frame)

            audio = audio_file.read(
                frames=self.segment_samples,
                dtype="float32",
                always_2d=True,
            )

        return torch.from_numpy(
            audio.T.copy()
        )