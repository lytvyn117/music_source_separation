import torch

from src.audio.stft import compute_stft, compute_magnitude
from src.training.losses import SeparationLoss


def validate(
    model,
    dataloader,
    device
):
    model.eval()

    loss_function = SeparationLoss()

    running_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            mixture = batch["mixture"].to(device)
            vocals = batch["vocals"].to(device)
            instrumental = batch["instrumental"].to(device)

            mixture_stft = compute_stft(mixture)
            vocals_stft = compute_stft(vocals)
            instrumental_stft = compute_stft(instrumental)

            mixture_mag = compute_magnitude(
                mixture_stft
            )

            vocals_mag = compute_magnitude(
                vocals_stft
            )

            instrumental_mag = compute_magnitude(
                instrumental_stft
            )

            masks = model(
                mixture_mag
            )

            loss = loss_function(
                mixture_mag,
                masks,
                vocals_mag,
                instrumental_mag
            )

            running_loss += loss.item()

    average_loss = (
        running_loss / len(dataloader)
    )

    return average_loss