from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from config import N_FFT
from src.audio.stft import compute_istft, compute_stft
from src.inference.separate_track import load_separation_model

from .complex_ops import channels_to_complex, complex_to_channels, safe_complex_rms


@dataclass(frozen=True)
class SeparationOutput:
    mixture_stft: torch.Tensor
    masks: torch.Tensor
    initial_vocal_stft: torch.Tensor
    residual_stft: torch.Tensor
    vocals_stft: torch.Tensor
    instrumental_stft: torch.Tensor
    vocals_waveform: torch.Tensor | None = None
    instrumental_waveform: torch.Tensor | None = None


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        return x + F.silu(self.norm(self.pointwise(self.depthwise(x))))


class RefinementHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(12, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(32, dilation) for dilation in (1, 2, 4, 1)))
        self.final = nn.Conv2d(32, 4, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, features):
        return self.final(self.blocks(self.stem(features)))


class PhaseAwareResidualSeparator(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.float().requires_grad_(False).eval()
        self.refinement = RefinementHead()
        self.use_bf16 = False

    @classmethod
    def from_exp04(cls, checkpoint_path, device=None):
        backbone, metadata, device = load_separation_model(checkpoint_path, device=device)
        if metadata.get("exp_name") != "exp04_asymmetric_leakage" or backbone.mask_mode != "softmax":
            raise ValueError("A softmax Exp04 checkpoint is required")
        return cls(backbone).to(device)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def backbone_masks(self, x):
        with torch.autocast(x.device.type, enabled=False):
            mag = x.abs()
            freq, frames = mag.shape[-2:]
            # Exp04 has five pools; tiny test/audio segments still need a valid input.
            padding = (0, max(0, 32 - frames), 0, max(0, 32 - freq))
            if any(padding):
                mag = F.pad(mag, padding, mode="replicate")
            return self.backbone(mag)[..., :freq, :frames]

    def forward_stft(self, mixture_stft, length=None, *, masks=None):
        if mixture_stft.ndim != 4 or mixture_stft.shape[1] != 2 or not mixture_stft.is_complex():
            raise ValueError("Expected a batched stereo complex spectrum")
        x = mixture_stft.to(torch.complex64)
        with torch.autocast(x.device.type, enabled=False):
            masks = self.backbone_masks(x) if masks is None else masks
            v0 = x * masks[:, 0]
            scale = safe_complex_rms(x)
            features = torch.cat((complex_to_channels(x / scale), complex_to_channels(v0 / scale),
                                  masks.flatten(1, 2)), dim=1)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda and self.use_bf16):
            correction = self.refinement(features)
        with torch.autocast(x.device.type, enabled=False):
            residual = channels_to_complex(correction) * scale
            vocal = v0 + residual
            instrumental = x - vocal
            waves = (None, None)
            if length is not None:
                waves = compute_istft(torch.cat((vocal, instrumental), dim=1), length=length).split(2, dim=1)
        return SeparationOutput(x, masks, v0, residual, vocal, instrumental, *waves)

    def forward(self, mixture_waveform):
        if mixture_waveform.ndim != 3 or mixture_waveform.shape[1] != 2 or mixture_waveform.shape[-1] == 0:
            raise ValueError("Expected nonempty batched stereo audio")
        length = mixture_waveform.shape[-1]
        wave = mixture_waveform.float()
        if length <= N_FFT // 2:
            wave = F.pad(wave, (0, N_FFT // 2 + 1 - length))
        with torch.autocast(wave.device.type, enabled=False):
            x = compute_stft(wave)
        return self.forward_stft(x, length)
