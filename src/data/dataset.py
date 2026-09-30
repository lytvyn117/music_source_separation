import random

import stempeg
import torch
from torch.utils.data import Dataset

from config import (
    TRAIN_DIR,
    SAMPLE_RATE,
    SEGMENT_LENGTH_SECONDS,
    SAMPLES_PER_EPOCH,
)


class MUSDBDataset(Dataset):
    """
    PyTorch dataset for random MUSDB18 training segments.

    Each sample contains:
    - mixture
    - vocals
    - instrumental

    Instrumental is constructed as:
    drums + bass + other
    """

    def __init__(
        self,
        data_dir=TRAIN_DIR,
        segment_length=SEGMENT_LENGTH_SECONDS,
        samples_per_epoch=SAMPLES_PER_EPOCH,
    ):
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.samples_per_epoch = samples_per_epoch

        self.track_files = sorted(
            self.data_dir.glob("*.mp4")
        )

        if not self.track_files:
            raise FileNotFoundError(
                f"No MUSDB18 files found in {self.data_dir}"
            )

        # Read track metadata once.
        # This avoids loading full songs just to determine their duration.
        self.track_info = []

        for track_path in self.track_files:
            info = stempeg.Info(str(track_path))

            duration = float(
                info.info["format"]["duration"]
            )

            self.track_info.append(
                {
                    "path": track_path,
                    "duration": duration,
                    "info": info,
                }
            )

    def __len__(self):
        """
        Number of random segments generated per epoch.
        """
        return self.samples_per_epoch

    def __getitem__(self, index):
        """
        Generate one random 6-second training segment.
        """

        # Random song
        track = random.choice(self.track_info)

        track_path = track["path"]
        track_duration = track["duration"]
        info = track["info"]

        # Random start position
        max_start = track_duration - self.segment_length

        if max_start > 0:
            start = random.uniform(0, max_start)
        else:
            start = 0.0

        # Read only the required segment
        stems, sample_rate = stempeg.read_stems(
            str(track_path),
            start=start,
            duration=self.segment_length,
            info=info,
            dtype="float32",
        )

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected sample rate {SAMPLE_RATE}, "
                f"but got {sample_rate}"
            )

        # MUSDB18:
        # 0 = mixture
        # 1 = drums
        # 2 = bass
        # 3 = other
        # 4 = vocals

        mixture = stems[0]
        drums = stems[1]
        bass = stems[2]
        other = stems[3]
        vocals = stems[4]

        instrumental = drums + bass + other

        # NumPy format:
        # (samples, channels)

        # PyTorch format:
        # (channels, samples)

        mixture = torch.from_numpy(
            mixture.T.copy()
        )

        vocals = torch.from_numpy(
            vocals.T.copy()
        )

        instrumental = torch.from_numpy(
            instrumental.T.copy()
        )

        return {
            "mixture": mixture,
            "vocals": vocals,
            "instrumental": instrumental,
            "track_name": track_path.stem,
            "start": start,
        }