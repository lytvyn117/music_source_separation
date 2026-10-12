import random

from config import PROCESSED_DIR


def create_train_validation_split(
    validation_ratio=0.15,
    seed=42
):
    train_dir = PROCESSED_DIR / "train"

    track_files = sorted(
        path
        for path in train_dir.iterdir()
        if path.is_dir()
    )

    if not track_files:
        raise FileNotFoundError(
            f"No processed MUSDB18 tracks found in {train_dir}"
        )

    rng = random.Random(seed)
    rng.shuffle(track_files)

    validation_size = int(
        len(track_files) * validation_ratio
    )

    validation_files = track_files[:validation_size]
    train_files = track_files[validation_size:]

    return train_files, validation_files