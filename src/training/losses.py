import torch
import torch.nn as nn


class SeparationLoss(nn.Module):
    """
    L1 loss for vocals/instrumental source separation.

    The model predicts source masks.
    These masks are applied to the mixture magnitude
    and compared with the true source magnitudes.
    """

    def __init__(self):
        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(
        self,
        mixture_magnitude,
        masks,
        vocals_target,
        instrumental_target
    ):
        """
        Parameters
        ----------
        mixture_magnitude:
            Shape:
            (batch, channels, freq, time)

        masks:
            Shape:
            (batch, sources, channels, freq, time)

        vocals_target:
            True vocal magnitude spectrogram.

        instrumental_target:
            True instrumental magnitude spectrogram.
        """

        vocal_mask = masks[:, 0]
        instrumental_mask = masks[:, 1]

        predicted_vocals = (
            mixture_magnitude * vocal_mask
        )

        predicted_instrumental = (
            mixture_magnitude * instrumental_mask
        )

        vocal_loss = self.l1(
            predicted_vocals,
            vocals_target
        )

        instrumental_loss = self.l1(
            predicted_instrumental,
            instrumental_target
        )

        total_loss = (
            vocal_loss
            + instrumental_loss
        )

        return total_loss