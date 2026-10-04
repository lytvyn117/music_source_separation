# src/training/train.py

import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SAMPLES_PER_EPOCH,
)
from src.audio.stft import compute_stft, compute_magnitude
from src.data.dataset import MUSDBDataset
from src.models.unet import UNet
from src.training.losses import SeparationLoss


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = MUSDBDataset(
        samples_per_epoch=SAMPLES_PER_EPOCH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNet().to(device)

    loss_function = SeparationLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_index, batch in enumerate(dataloader):

            mixture = batch["mixture"].to(device)
            vocals = batch["vocals"].to(device)
            instrumental = batch["instrumental"].to(device)

            # -----------------------------
            # STFT
            # -----------------------------

            mixture_stft = compute_stft(mixture)
            vocals_stft = compute_stft(vocals)
            instrumental_stft = compute_stft(instrumental)

            # -----------------------------
            # Magnitude
            # -----------------------------

            mixture_mag = compute_magnitude(mixture_stft)
            vocals_mag = compute_magnitude(vocals_stft)
            instrumental_mag = compute_magnitude(instrumental_stft)

            # -----------------------------
            # Forward pass
            # -----------------------------

            masks = model(mixture_mag)

            # -----------------------------
            # Loss
            # -----------------------------

            loss = loss_function(
                mixture_mag,
                masks,
                vocals_mag,
                instrumental_mag
            )

            # -----------------------------
            # Backpropagation
            # -----------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_index % 1 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Batch {batch_index}/{len(dataloader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        epoch_loss = (
            running_loss / len(dataloader)
        )

        print(
            f"\nEpoch {epoch + 1} finished "
            f"| Average Loss: {epoch_loss:.4f}\n"
        )


if __name__ == "__main__":
    train()