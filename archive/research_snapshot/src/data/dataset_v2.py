import random
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(r"C:\Users\YOUR_USER\music_source_separation")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import (
    PROCESSED_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_EPOCH,
    SEGMENT_LENGTH_SECONDS,
)


class AugmentedMUSDBDataset(Dataset):
    """
    Enhanced MUSDB18 dataset supporting:
    1. Active-vocal segment rejection sampling (prevents wasting batches on silent vocal intros/outros).
    2. Cross-track stem remixing (mixes vocals from song A with instrumental from song B).
    3. Independent random stem gain scaling ([0.75, 1.25]) and peak clipping protection.
    4. Random stereo channel swap (L <-> R) and polarity inversion.
    """

    def __init__(
        self,
        data_dir=PROCESSED_DIR / "train",
        track_files=None,
        segment_length: float = SEGMENT_LENGTH_SECONDS,
        samples_per_epoch: int = SAMPLES_PER_EPOCH,
        random_segments: bool = True,
        augment: bool = False,
        remix_prob: float = 0.5,
        active_vocal_prob: float = 0.8,
        min_vocal_rms: float = 0.005,
        gain_range: Tuple[float, float] = (0.75, 1.25),
        channel_swap_prob: float = 0.5,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.samples_per_epoch = samples_per_epoch
        self.random_segments = random_segments
        self.augment = augment
        self.remix_prob = remix_prob
        self.active_vocal_prob = active_vocal_prob
        self.min_vocal_rms = min_vocal_rms
        self.gain_range = gain_range
        self.channel_swap_prob = channel_swap_prob
        self.seed = seed

        if track_files is not None:
            self.track_files = list(track_files)
        else:
            self.track_files = sorted(
                path for path in self.data_dir.iterdir() if path.is_dir()
            )

        if not self.track_files:
            raise FileNotFoundError(
                f"No processed MUSDB tracks found in {self.data_dir}"
            )

        self.segment_samples = int(round(self.segment_length * SAMPLE_RATE))

        self.track_info = []
        for track_path in self.track_files:
            mixture_path = track_path / "mixture.wav"
            info = sf.info(mixture_path)
            self.track_info.append(
                {
                    "path": track_path,
                    "frames": info.frames,
                }
            )

    def __len__(self):
        return self.samples_per_epoch

    def _read_numpy_segment(self, path: Path, start_frame: int) -> np.ndarray:
        with sf.SoundFile(path) as audio_file:
            audio_file.seek(start_frame)
            audio = audio_file.read(
                frames=self.segment_samples,
                dtype="float32",
                always_2d=True,
            )
        # Return (2, segment_samples)
        return audio.T.copy()

    def _sample_track_and_start(self, rng: random.Random):
        track = rng.choice(self.track_info)
        max_start = max(0, track["frames"] - self.segment_samples)
        start_frame = rng.randint(0, max_start) if max_start > 0 else 0
        return track, start_frame

    def __getitem__(self, index: int):
        if self.random_segments:
            rng = random
        else:
            rng = random.Random(self.seed + index)

        # Deterministic validation mode (matches original MUSDBDataset exactly)
        if not self.random_segments or not self.augment:
            if self.random_segments:
                track, start_frame = self._sample_track_and_start(rng)
            else:
                track = self.track_info[index % len(self.track_info)]
                max_start = max(0, track["frames"] - self.segment_samples)
                start_frame = rng.randint(0, max_start) if max_start > 0 else 0

            track_path = track["path"]
            mixture = self._read_numpy_segment(track_path / "mixture.wav", start_frame)
            vocals = self._read_numpy_segment(track_path / "vocals.wav", start_frame)
            instrumental = self._read_numpy_segment(
                track_path / "instrumental.wav", start_frame
            )
            return {
                "mixture": torch.from_numpy(mixture),
                "vocals": torch.from_numpy(vocals),
                "instrumental": torch.from_numpy(instrumental),
                "track_name": track_path.name,
                "start": start_frame / SAMPLE_RATE,
            }

        # Augmented training mode:
        # 1. Sample vocals segment (with rejection sampling against silent vocal regions)
        want_active_vocals = rng.random() < self.active_vocal_prob
        voc_track, voc_start = self._sample_track_and_start(rng)
        vocals = self._read_numpy_segment(
            voc_track["path"] / "vocals.wav", voc_start
        )

        if want_active_vocals:
            for _ in range(5):
                rms = float(np.sqrt(np.mean(vocals ** 2)))
                if rms >= self.min_vocal_rms:
                    break
                voc_track, voc_start = self._sample_track_and_start(rng)
                vocals = self._read_numpy_segment(
                    voc_track["path"] / "vocals.wav", voc_start
                )

        # 2. Sample instrumental segment (either from same track/position or remixed from another track)
        if rng.random() < self.remix_prob:
            inst_track, inst_start = self._sample_track_and_start(rng)
        else:
            inst_track, inst_start = voc_track, voc_start

        instrumental = self._read_numpy_segment(
            inst_track["path"] / "instrumental.wav", inst_start
        )

        # 3. Random gain scaling per stem
        g_voc = rng.uniform(self.gain_range[0], self.gain_range[1])
        g_inst = rng.uniform(self.gain_range[0], self.gain_range[1])
        vocals = vocals * g_voc
        instrumental = instrumental * g_inst

        # 4. Random stereo channel swap
        if rng.random() < self.channel_swap_prob:
            vocals = vocals[::-1, :].copy()
        if rng.random() < self.channel_swap_prob:
            instrumental = instrumental[::-1, :].copy()

        # 5. Construct consistent linear mixture
        mixture = vocals + instrumental

        # Prevent clipping while preserving exact linear additivity (mixture == vocals + instrumental)
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.99:
            scale = 0.99 / peak
            mixture *= scale
            vocals *= scale
            instrumental *= scale

        return {
            "mixture": torch.from_numpy(mixture),
            "vocals": torch.from_numpy(vocals),
            "instrumental": torch.from_numpy(instrumental),
            "track_name": voc_track["path"].name,
            "start": voc_start / SAMPLE_RATE,
        }
