import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks:
    Conv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block used in the decoder.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Spectrogram dimensions such as 1025 x 517 are not
        # perfectly divisible by powers of 2.
        # Resize to exactly match the corresponding skip connection.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for vocals / instrumental mask prediction.

    Input:
        (batch, 2, frequency, time)

    Output:
        (batch, 2 sources, 2 channels, frequency, time)

    Source 0 = vocals
    Source 1 = instrumental
    """

    def __init__(self, input_channels=2, base_channels=16):
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(
            input_channels,
            base_channels
        )

        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(
            base_channels,
            base_channels * 2
        )

        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(
            base_channels * 2,
            base_channels * 4
        )

        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(
            base_channels * 4,
            base_channels * 8
        )

        # Decoder
        self.decoder3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4
        )

        self.decoder2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2
        )

        self.decoder1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels
        )

        # 2 sources × 2 stereo channels = 4 outputs
        self.output_layer = nn.Conv2d(
            base_channels,
            4,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool1(e1)
        )

        e3 = self.encoder3(
            self.pool2(e2)
        )

        # Bottleneck
        bottleneck = self.bottleneck(
            self.pool3(e3)
        )

        # Decoder + skip connections
        d3 = self.decoder3(
            bottleneck,
            e3
        )

        d2 = self.decoder2(
            d3,
            e2
        )

        d1 = self.decoder1(
            d2,
            e1
        )

        masks = self.output_layer(d1)

        batch, _, freq, time = masks.shape

        # 4 channels ->
        # 2 sources × 2 stereo channels
        masks = masks.reshape(
            batch,
            2,
            2,
            freq,
            time
        )

        # Normalize masks across the two sources.
        masks = torch.softmax(
            masks,
            dim=1
        )

        return masks