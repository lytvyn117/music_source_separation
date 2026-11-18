import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import DoubleConv


class ExactUpBlock(nn.Module):
    """
    Decoder upsampling block that supports either:
    - pad_mode="exact": zero/replicate-pads the 1 dropped high-frequency bin / final time frame
      so lower frequency bins and earlier time frames maintain 1-to-1 alignment with skip connections.
    - pad_mode="interpolate": original bilinear interpolation behavior.
    Weight shapes are 100% identical to UpBlock in src/models/unet.py.
    """

    def __init__(self, in_channels, skip_channels, out_channels, pad_mode="exact"):
        super().__init__()
        self.pad_mode = pad_mode

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            if self.pad_mode == "exact":
                diff_f = skip.shape[-2] - x.shape[-2]
                diff_t = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [0, diff_t, 0, diff_f], mode="replicate")
            else:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetV2(nn.Module):
    """
    Improved U-Net for stereo vocals / instrumental separation.

    Supports:
    - State-dict compatibility with original UNet (when base_channels=16)
    - pad_mode: 'exact' (fixes UpBlock spatial misalignment) or 'interpolate'
    - log_input: if True, applies log1p compression before the encoder so low and high
      frequencies have balanced dynamic range
    - mask_mode: 'softmax' (original zero-sum mask) or 'sigmoid' (decoupled [0, 1] masks)
    """

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 16,
        pad_mode: str = "exact",
        log_input: bool = False,
        mask_mode: str = "softmax",
    ):
        super().__init__()
        self.base_channels = base_channels
        self.pad_mode = pad_mode
        self.log_input = log_input
        self.mask_mode = mask_mode

        # Encoder
        self.encoder1 = DoubleConv(input_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.encoder5 = DoubleConv(base_channels * 8, base_channels * 16)
        self.pool5 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 16, base_channels * 32)

        # Decoder
        self.decoder5 = ExactUpBlock(
            base_channels * 32, base_channels * 16, base_channels * 16, pad_mode=pad_mode
        )
        self.decoder4 = ExactUpBlock(
            base_channels * 16, base_channels * 8, base_channels * 8, pad_mode=pad_mode
        )
        self.decoder3 = ExactUpBlock(
            base_channels * 8, base_channels * 4, base_channels * 4, pad_mode=pad_mode
        )
        self.decoder2 = ExactUpBlock(
            base_channels * 4, base_channels * 2, base_channels * 2, pad_mode=pad_mode
        )
        self.decoder1 = ExactUpBlock(
            base_channels * 2, base_channels, base_channels, pad_mode=pad_mode
        )

        self.output_layer = nn.Conv2d(base_channels, 4, kernel_size=1)

    def forward(self, x):
        if self.log_input:
            x = torch.log1p(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        e5 = self.encoder5(self.pool4(e4))

        bottleneck = self.bottleneck(self.pool5(e5))

        d5 = self.decoder5(bottleneck, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        masks = self.output_layer(d1)
        batch, _, freq, time = masks.shape
        masks = masks.reshape(batch, 2, 2, freq, time)

        if self.mask_mode == "sigmoid":
            masks = torch.sigmoid(masks)
        else:
            masks = torch.softmax(masks, dim=1)

        return masks
