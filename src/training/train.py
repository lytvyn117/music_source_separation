# src/training/train.py

import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SAMPLES_PER_EPOCH,
    CHECKPOINTS_DIR,
    VALIDATION_SAMPLES,
    RESUME_TRAINING,
    RESUME_CHECKPOINT,
)

from src.audio.stft import compute_stft, compute_magnitude
from src.data.dataset import MUSDBDataset
from src.models.unet import UNet
from src.training.losses import SeparationLoss
from src.training.validate import validate
from src.data.splits import create_train_validation_split


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    train_loss = checkpoint["train_loss"]
    validation_loss = checkpoint["validation_loss"]

    best_validation_loss = checkpoint.get(
        "best_validation_loss",
        validation_loss
    )

    print(
        f"Checkpoint loaded: {checkpoint_path}"
    )

    print(
        f"Resume from epoch {start_epoch + 1}"
    )

    return (
        start_epoch,
        train_loss,
        validation_loss,
        best_validation_loss
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    best_validation_loss
):
    CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
    }

    path = (
        CHECKPOINTS_DIR
        / f"checkpoint_epoch_{epoch}.pt"
    )

    torch.save(
        checkpoint,
        path
    )

    print(f"Checkpoint saved: {path}")


def save_best_model(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss
):
    CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
    }

    path = CHECKPOINTS_DIR / "best_model.pt"

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Best model saved: {path}"
    )


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_files, validation_files = (
        create_train_validation_split()
    )

    print(f"Training tracks: {len(train_files)}")
    print(f"Validation tracks: {len(validation_files)}")


    train_dataset = MUSDBDataset(
        track_files=train_files,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        random_segments=True
    )

    validation_dataset = MUSDBDataset(
        track_files=validation_files,
        samples_per_epoch=VALIDATION_SAMPLES,
        random_segments=False,
        seed=42
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
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

    start_epoch = 0
    best_validation_loss = float("inf")

    if RESUME_TRAINING and RESUME_CHECKPOINT is not None:

        (
            start_epoch,
            _,
            _,
            best_validation_loss
        ) = load_checkpoint(
            RESUME_CHECKPOINT,
            model,
            optimizer,
            device
        )


    for epoch in range(start_epoch, EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):

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
                    f"| Batch {batch_index}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        epoch_loss = (
            running_loss / len(train_loader)
        )

        print(
            f"\nEpoch {epoch + 1} finished "
            f"| Average Loss: {epoch_loss:.4f}\n"
        )

        validation_loss = validate(
            model,
            validation_loader,
            device
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            save_best_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=epoch_loss,
                validation_loss=validation_loss
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            train_loss=epoch_loss,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss
        )


if __name__ == "__main__":
    train()