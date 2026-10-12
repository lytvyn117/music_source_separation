from pathlib import Path

import soundfile as sf
import torch

from config import (
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
    SAMPLE_RATE,
    OUTPUTS_DIR,
)

from src.data.dataset import MUSDBDataset
from src.models.unet import UNet
from src.audio.stft import compute_stft, compute_magnitude
from src.audio.separation import reconstruct_source


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    dataset = MUSDBDataset(
        data_dir=PROCESSED_DIR / "test",
        samples_per_epoch=1,
        random_segments=False,
        seed=42,
    )

    sample = dataset[0]

    mixture = sample["mixture"].unsqueeze(0).to(device)

    model = UNet().to(device)

    checkpoint = torch.load(
        CHECKPOINTS_DIR / "best_model.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        mixture_stft = compute_stft(mixture)
        mixture_mag = compute_magnitude(mixture_stft)

        masks = model(mixture_mag)

        vocal_mask = masks[:, 0]
        instrumental_mask = masks[:, 1]

        predicted_vocals = reconstruct_source(
            mixture_stft,
            vocal_mask,
            length=mixture.shape[-1]
        )

        predicted_instrumental = reconstruct_source(
            mixture_stft,
            instrumental_mask,
            length=mixture.shape[-1]
        )

    output_dir = OUTPUTS_DIR / "one_track_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    mixture_cpu = mixture[0].cpu().numpy().T
    vocals_cpu = predicted_vocals[0].cpu().numpy().T
    instrumental_cpu = predicted_instrumental[0].cpu().numpy().T

    sf.write(
        output_dir / "mixture.wav",
        mixture_cpu,
        SAMPLE_RATE
    )

    sf.write(
        output_dir / "predicted_vocals.wav",
        vocals_cpu,
        SAMPLE_RATE
    )

    sf.write(
        output_dir / "predicted_instrumental.wav",
        instrumental_cpu,
        SAMPLE_RATE
    )

    print("Track:", sample["track_name"])
    print("Start:", sample["start"])
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()