"""Reference shape graph for feasibility only."""
import torch
from torch import nn
from torch.nn import functional as F

SIZES = (64, 192, 769)
COMPRESSED = (16, 16, 25)
STRIDES_1 = (2, 4, 8)
STRIDES_2 = (2, 3, 4)


def channel_norm(norm, x):
    return norm(x.movedim(1, -1)).movedim(-1, 1)


class Residual(nn.Module):
    def __init__(self, channels, hidden, kernel, dilation=1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel, groups=hidden,
                                  padding=(kernel[0] // 2, kernel[1] // 2 * dilation),
                                  dilation=(1, dilation))
        self.reduce = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        z = F.silu(self.expand(channel_norm(self.norm, x)))
        return x + self.reduce(F.silu(self.depthwise(z)))


class FrequencyMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.expand = nn.Linear(57, 96)
        self.reduce = nn.Linear(96, 57)

    def forward(self, x):
        z = channel_norm(self.norm, x).transpose(2, 3)
        return x + self.reduce(F.silu(self.expand(z))).transpose(2, 3)


def frequency_convs(in_channels, out_channels, strides, cls):
    return nn.ModuleList(cls(in_channels, out_channels, (s, 1), stride=(s, 1))
                         for s in strides)


class ReferenceGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = frequency_convs(6, 32, STRIDES_1, nn.Conv2d)
        self.down2 = frequency_convs(32, 128, STRIDES_2, nn.Conv2d)
        self.up2 = frequency_convs(128, 32, STRIDES_2, nn.ConvTranspose2d)
        self.up1 = frequency_convs(32, 4, STRIDES_1, nn.ConvTranspose2d)
        self.locals1 = nn.ModuleList(nn.Sequential(*(Residual(32, 16, (3, 3)) for _ in range(d)))
                                     for d in (2, 2, 1))
        self.locals2 = nn.ModuleList(Residual(32, 16, (3, 3)) for _ in SIZES)
        self.temporal = nn.ModuleList(Residual(128, 256, (1, 5), 2**j) for j in range(8))
        self.mixers = nn.ModuleList(FrequencyMixer() for _ in range(2))
        self.raw_skip = nn.ModuleList(nn.Conv2d(6, 4, 1) for _ in SIZES)
        for head in (*self.up1, *self.raw_skip):
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)

    @staticmethod
    def right_pad(x, stride):
        return F.pad(x, (0, 0, 0, -x.shape[-2] % stride))

    def forward(self, x, trace=False):
        split = x.split(SIZES, dim=2)
        skips = [local(down(self.right_pad(band, stride)))
                 for local, down, band, stride in zip(self.locals1, self.down1, split, STRIDES_1)]
        down2 = [down(self.right_pad(band, stride))
                 for down, band, stride in zip(self.down2, skips, STRIDES_2)]
        z = torch.cat(down2, dim=2)
        z = self.mixers[0](z)
        for layer in self.temporal[:4]:
            z = layer(z)
        z = self.mixers[1](z)
        for layer in self.temporal[4:]:
            z = layer(z)
        up2 = [local(up(band)[:, :, :skip.shape[-2]] + skip)
               for local, up, band, skip in zip(self.locals2, self.up2, z.split(COMPRESSED, dim=2), skips)]
        up1 = [up(band)[:, :, :raw.shape[-2]] + linear(raw)
               for up, band, linear, raw in zip(self.up1, up2, self.raw_skip, split)]
        logits = torch.cat(up1, dim=2)
        if not trace:
            return logits
        return logits, {"encoder1": [tuple(a.shape) for a in skips],
                        "encoder2": [tuple(a.shape) for a in down2],
                        "bottleneck": tuple(z.shape),
                        "decoder2": [tuple(a.shape) for a in up2],
                        "decoder1": [tuple(a.shape) for a in up1]}
