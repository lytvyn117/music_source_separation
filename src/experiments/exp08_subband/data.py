"""Frozen MUSDB18 partition and deterministic stereo segment sampling."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import PROCESSED_DIR, PROJECT_ROOT
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split

SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 441000
STEMS = ("mixture", "vocals", "instrumental")
BASELINE = PROJECT_ROOT / "models/experiments/exp04_asymmetric_leakage/best.pt"
BASELINE_SHA256 = "c2bddec2b084a4c022b269aa19c8460c6273dda91bf1cfe1256065edf3a8427f"
FEASIBILITY = PROJECT_ROOT / "reports/exp08_subband/feasibility_v1/probe_results.json"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_partition(root, tracks, expected):
    names = [[path.name for path in side] for side in tracks]
    all_paths = [path.resolve() for side in tracks for path in side]
    if (names != [list(side) for side in expected]
            or len(set(all_paths)) != len(all_paths)
            or any(path.parent != root.resolve() for path in all_paths)):
        raise ValueError("Track order, uniqueness, or training partition changed")


def frozen_partition():
    evidence = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
    tracks = create_train_validation_split(seed=42)
    if tuple(map(len, tracks)) != (85, 15):
        raise ValueError("Expected the original 85/15 split")
    verify_partition(PROCESSED_DIR / "train", tracks, (evidence["train"], evidence["validation"]))
    return tracks


def reserve_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_segment(path, start, frames):
    audio, rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 2:
        raise ValueError(f"Expected 44.1 kHz stereo at {path}")
    if frames >= 0 and len(audio) < frames:
        audio = np.pad(audio, ((0, frames - len(audio)), (0, 0)))
    return audio.T.copy()


def _rms(wave):
    return np.sqrt(np.mean(wave**2))


def validation_manifest(tracks):
    rng = random.Random(42)
    records, excerpts = [], []
    for track in tracks:
        vocal_file = track / "vocals.wav"
        length = sf.info(vocal_file).frames
        limit = max(0, length - SEGMENT_SAMPLES)
        for index in range(10):
            low, high = index * limit // 10, (index + 1) * limit // 10
            candidates = [rng.randint(low, high) for _ in range(3)]
            select = min if index % 2 == 0 else max
            start = select(candidates, key=lambda s: _rms(read_segment(vocal_file, s, SEGMENT_SAMPLES)))
            records.append({"track": track.name, "start_sample": start})
        max_start = max(0, length - 20 * SAMPLE_RATE)
        candidates = [j * max_start // 4 for j in range(5)]
        start = max(candidates, key=lambda s: _rms(read_segment(vocal_file, s, 20 * SAMPLE_RATE)))
        excerpts.append({"track": track.name, "start_sample": start, "samples": min(length, 20 * SAMPLE_RATE)})
    return records, excerpts[:12]


def fingerprint_manifest(train, validation):
    wavs = {}
    for track in (*train, *validation):
        for stem in STEMS:
            path = track / f"{stem}.wav"
            if path.resolve().parent != track.resolve():
                raise ValueError(f"Audio resolves outside its training track: {path}")
            wavs[f"{track.name}/{stem}.wav"] = sha256(path)
    baseline_sha = sha256(BASELINE)
    if baseline_sha != BASELINE_SHA256:
        raise ValueError("Exp04 checkpoint fingerprint changed")
    val_clips, listening = validation_manifest(validation)
    return {"train": [p.name for p in train], "validation": [p.name for p in validation],
            "wavs": wavs, "baseline_sha256": baseline_sha,
            "validation_clips": val_clips, "listening": listening, "source": source_fingerprints()}


def source_fingerprints():
    files = [p.relative_to(PROJECT_ROOT).as_posix() for p in sorted(Path(__file__).parent.glob("*.py"))]
    files.extend(("config.py", "src/audio/stft.py", "src/audio/separation.py", "src/data/splits.py",
                  "src/data/dataset_v2.py", "src/training/experiment_selection.py",
                  "src/evaluation/metrics.py", "src/evaluation/compare_experiments.py",
                  "src/inference/separate_track.py", "src/models/unet.py", "src/models/unet_v2.py",
                  "src/experiments/exp07_phase_aware_residual/chunking.py"))
    return {path: sha256(PROJECT_ROOT / path) for path in files}


def bind_run(manifest_path, checkpoint_fingerprint, config):
    """Verify current inputs against the manifest committed with the checkpoints."""
    manifest_path = Path(manifest_path)
    if checkpoint_fingerprint != {"config": config, "manifest_sha256": sha256(manifest_path)}:
        raise ValueError("Run configuration or manifest fingerprint does not match the checkpoint store")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, validation = frozen_partition()
    if fingerprint_manifest(train, validation) != manifest:
        raise ValueError("Current source, WAV, split, or baseline fingerprints changed")
    return train, validation, manifest


class TrainClips(AugmentedMUSDBDataset):
    """Reuse the base sampling settings with per-item RNG and joint stereo augmentation."""

    def __init__(self, tracks, epoch, samples=2048, seed=42):
        super().__init__(track_files=tracks, segment_length=10., samples_per_epoch=samples,
                         augment=True, seed=seed)
        self.epoch = epoch

    def __getitem__(self, index):
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        active = rng.random() < self.active_vocal_prob
        for _ in range(6 if active else 1):
            vocal_track, vocal_start = self._sample_track_and_start(rng)
            vocal = read_segment(vocal_track["path"] / "vocals.wav", vocal_start, SEGMENT_SAMPLES)
            if _rms(vocal) >= self.min_vocal_rms:
                break
        if rng.random() < self.remix_prob:
            inst_track, inst_start = self._sample_track_and_start(rng)
        else:
            inst_track, inst_start = vocal_track, vocal_start
        instrumental = read_segment(inst_track["path"] / "instrumental.wav", inst_start, SEGMENT_SAMPLES)
        vocal_gain, inst_gain = [rng.uniform(*self.gain_range) for _ in range(2)]
        vocal *= vocal_gain
        instrumental *= inst_gain
        swapped = rng.random() < self.channel_swap_prob
        if swapped:
            vocal, instrumental = vocal[::-1], instrumental[::-1]
        mixture = vocal + instrumental
        scale = min(1., .99 / max(float(np.abs(mixture).max()), 1e-12))
        audio = torch.from_numpy(np.stack((mixture, vocal, instrumental)))
        audio *= scale
        return {"audio": audio, "index": index, "vocals_track": vocal_track["path"].name,
                "instrumental_track": inst_track["path"].name, "vocal_start": vocal_start,
                "instrumental_start": inst_start, "vocal_gain": vocal_gain,
                "instrumental_gain": inst_gain, "swapped": swapped, "peak_scale": scale}


class RecordedClips(Dataset):
    def __init__(self, tracks, records):
        self.tracks = {path.name: path for path in tracks}
        self.records = records
        if any(row["track"] not in self.tracks for row in records):
            raise ValueError("Clip manifest uses a track outside its partition")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        track = self.tracks[row["track"]]
        audio = np.stack([read_segment(track / f"{stem}.wav", row["start_sample"], SEGMENT_SAMPLES)
                          for stem in STEMS])
        return {"audio": torch.from_numpy(audio), **row}
