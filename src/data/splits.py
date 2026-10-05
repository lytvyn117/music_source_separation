import random

from config import TRAIN_DIR


def create_train_validation_split(
    validation_ratio=0.15,
    seed=42
):
    """
    Split MUSDB18 training tracks into
    training and validation sets.

    The split happens on song level.
    """

    track_files = sorted(
        TRAIN_DIR.glob("*.mp4")
    )

    if not track_files:
        raise FileNotFoundError(
            f"No MUSDB18 files found in {TRAIN_DIR}"
        )

    # Reproducible shuffle
    rng = random.Random(seed)
    rng.shuffle(track_files)

    validation_size = int(
        len(track_files) * validation_ratio
    )

    validation_files = track_files[:validation_size]
    train_files = track_files[validation_size:]

    return train_files, validation_files