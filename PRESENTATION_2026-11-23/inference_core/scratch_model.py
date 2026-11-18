"""Independent stereo BS-RoFormer-inspired separator."""
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

RATE = 44100
N_FFT = 2048
HOP = 441
SEGMENT = 4 * RATE
BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
WIDTH = 256
HEADS = 8
PAIRS = 6


def seed_all(seed, deterministic=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def crop_for(lengths, segment, seed, index):
    keys = sorted(key for key, frames in lengths.items() if frames >= segment)
    if not keys or index < 0:
        raise ValueError("No eligible segment or invalid sample index")
    rng = random.Random(f"{seed}:{index}")
    key = rng.choice(keys)
    return key, rng.randrange(lengths[key] - segment + 1)


def rope_table(length, channels, device):
    inverse = 10000.0 ** (-torch.arange(0, channels, 2, device=device, dtype=torch.float32) / channels)
    phase = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inverse
    return phase.cos(), phase.sin()


def rotate(x, table):
    cosine, sine = (part.to(x.dtype) for part in table)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine),
                       dim=-1).flatten(-2)


class AxisBlock(nn.Module):
    def __init__(self, axis, group_chunk):
        super().__init__()
        self.axis = axis
        self.group_chunk = group_chunk
        self.pre_attn = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.attn_out = nn.Linear(WIDTH, WIDTH)
        self.pre_ff = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(),
                                nn.Linear(4 * WIDTH, WIDTH))

    def attend(self, seq, table):
        groups, length, _ = seq.shape
        q, k, v = self.qkv(seq).reshape(groups, length, 3, HEADS, WIDTH // HEADS).unbind(2)
        q = rotate(q.transpose(1, 2), table)
        k = rotate(k.transpose(1, 2), table)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(groups, length, WIDTH)

    def forward(self, tokens):
        layout = tokens.transpose(1, 2) if self.axis == "time" else tokens
        shape = layout.shape
        seq = layout.reshape(-1, shape[-2], WIDTH)
        table = rope_table(seq.shape[1], WIDTH // HEADS, seq.device)
        norm = self.pre_attn(seq)
        parts = norm.split(self.group_chunk) if self.group_chunk else (norm,)
        attn = torch.cat([self.attend(part, table) for part in parts], dim=0)
        seq = seq + self.attn_out(attn)
        seq = (seq + self.ff(self.pre_ff(seq))).reshape(shape)
        return seq.transpose(1, 2) if self.axis == "time" else seq


class Separator(nn.Module):
    def __init__(self, checkpoint_blocks=False, group_chunk=0):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.project = nn.ModuleList(nn.Linear(4 * band, WIDTH) for band in BANDS)
        self.blocks = nn.ModuleList(AxisBlock(axis, group_chunk)
                                    for _ in range(PAIRS) for axis in ("time", "band"))
        self.mask_features = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU())
        self.readout = nn.ModuleList(nn.Linear(WIDTH, 4 * band) for band in BANDS)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        # All trainable tensors, including affine offsets, start from seeded draws.
        # Norm scales draw around one. Readout weights start small.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.normal_(module.bias, std=0.002)
            if isinstance(module, nn.LayerNorm):
                nn.init.normal_(module.weight, mean=1.0, std=0.005)
                nn.init.normal_(module.bias, std=0.002)
        for head in self.readout:
            nn.init.normal_(head.weight, std=0.006)

    def spectrum(self, x):
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[-1] <= N_FFT // 2:
            raise ValueError("Expected [batch, 2, samples] with >1024 samples")
        with torch.autocast(x.device.type, enabled=False):
            spec = torch.stft(x.float().flatten(0, 1), N_FFT, HOP, window=self.window.float(),
                              center=True, return_complex=True)
        return spec.unflatten(0, (x.shape[0], 2))

    def waveform(self, z, length):
        with torch.autocast(z.device.type, enabled=False):
            wave = torch.istft(z.to(torch.complex64).flatten(0, 1), N_FFT, HOP,
                               window=self.window.float(), center=True, length=length)
        return wave.unflatten(0, (z.shape[0], 2))

    def forward(self, x, return_mask=False):
        z = self.spectrum(x)
        values = torch.view_as_real(z.permute(0, 3, 2, 1)).flatten(-2)
        # Preserve useful amplitude cues with fixed scaling and soft compression.
        values = torch.asinh(values)
        tokens = torch.stack([projection(band.flatten(-2)) for projection, band in
                              zip(self.project, values.split(BANDS, dim=2))], dim=2)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False, preserve_rng_state=False)
            else:
                tokens = block(tokens)
        features = self.mask_features(tokens)
        raw = torch.cat([head(features[:, :, k]).unflatten(-1, (band, 2, 2))
                         for k, (head, band) in enumerate(zip(self.readout, BANDS))], dim=2)
        with torch.autocast(x.device.type, enabled=False):
            parts = raw.float()
            parts = parts * (2.0 * torch.rsqrt(16.0 + parts.square().sum(dim=-1, keepdim=True)))
            mask = torch.view_as_complex(parts.contiguous()).permute(0, 3, 2, 1)
            vocals = self.waveform(mask * z, x.shape[-1])
            instrumental = x.float() - vocals
        return (vocals, instrumental, mask) if return_mask else (vocals, instrumental)
