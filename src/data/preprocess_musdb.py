from pathlib import Path

import soundfile as sf
import stempeg

from config import TRAIN_DIR, TEST_DIR, PROCESSED_DIR


def process_split(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    track_files = sorted(input_dir.glob("*.mp4"))

    print(f"Found {len(track_files)} tracks in {input_dir}")

    for i, track_path in enumerate(track_files, start=1):
        print(f"[{i}/{len(track_files)}] Processing {track_path.name}")

        stems, sample_rate = stempeg.read_stems(
            str(track_path),
            dtype="float32"
        )

        mixture = stems[0]
        drums = stems[1]
        bass = stems[2]
        other = stems[3]
        vocals = stems[4]

        instrumental = drums + bass + other

        track_output_dir = output_dir / track_path.stem
        track_output_dir.mkdir(parents=True, exist_ok=True)

        mixture_path = track_output_dir / "mixture.wav"
        vocals_path = track_output_dir / "vocals.wav"
        instrumental_path = track_output_dir / "instrumental.wav"

        if (
            mixture_path.exists()
            and vocals_path.exists()
            and instrumental_path.exists()
        ):
            print(f"Skipping already processed: {track_path.name}")
            continue

        sf.write(
            mixture_path,
            mixture,
            sample_rate,
            subtype="FLOAT"
        )

        sf.write(
            vocals_path,
            vocals,
            sample_rate,
            subtype="FLOAT"
        )

        sf.write(
            instrumental_path,
            instrumental,
            sample_rate,
            subtype="FLOAT"
        )


def main():
    process_split(
        TRAIN_DIR,
        PROCESSED_DIR / "train"
    )

    process_split(
        TEST_DIR,
        PROCESSED_DIR / "test"
    )


if __name__ == "__main__":
    main()