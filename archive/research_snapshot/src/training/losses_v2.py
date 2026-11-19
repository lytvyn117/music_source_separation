import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(r"C:\Users\YOUR_USER\music_source_separation")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.audio.separation import reconstruct_source


def asymmetric_leakage_loss(
    masks: torch.Tensor,
    vocals_mag: torch.Tensor,
    instrumental_mag: torch.Tensor,
    margin: float = 0.05,
    activity_threshold: float = 1e-3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize excess instrumental mask energy in vocal-dominant TF bins."""
    with torch.no_grad():
        vocals = vocals_mag.detach().float()
        instrumental = instrumental_mag.detach().float()
        energy = vocals + instrumental
        denominator = energy.clamp_min(eps)
        vocal_ratio = vocals / denominator
        instrumental_ratio = instrumental / denominator
        peak = energy.amax(dim=(-2, -1), keepdim=True)
        active = (energy > activity_threshold * peak).float()
        weight = active * vocal_ratio.square()

    predicted_instrumental = masks[:, 1].float()
    excess = F.relu(predicted_instrumental - instrumental_ratio - margin)
    return (weight * excess.square()).mean()


class CompositeSeparationLoss(nn.Module):
    """
class CompositeSeparationLoss(nn.Module):
class CompositeSeparationLoss(nn.Module):
    Composite loss function for stereo music source separation addressing:
    1. Linear STFT Magnitude L1 Loss (captures overall energy balance)
    2. Log1p-Compressed STFT Magnitude L1 Loss (upweights mid/high-frequency vocal
       formants, harmonics, sibilance, and quiet passages relative to loud sub-bass)
    3. Multi-Resolution STFT + Time-Domain Waveform L1 Loss after iSTFT reconstruction
       (enforces phase/STFT consistency and penalizes musical noise / vocal bleed in waveform)
    """

    def __init__(
        self,
        linear_mag_weight: float = 1.0,
        log_mag_weight: float = 0.5,
        waveform_l1_weight: float = 5.0,
        multires_stft_weight: float = 0.5,
        multires_ffts: Tuple[int, ...] = (512, 1024, 2048),
        vocal_weight: float = 1.2,
        instrumental_weight: float = 1.0,
        leakage_weight: float = 0.0,
        leakage_margin: float = 0.05,
        leakage_activity_threshold: float = 1e-3,
    ):
        super().__init__()
        self.linear_mag_weight = linear_mag_weight
        self.log_mag_weight = log_mag_weight
        self.waveform_l1_weight = waveform_l1_weight
        self.multires_stft_weight = multires_stft_weight
        self.multires_ffts = multires_ffts
        self.vocal_weight = vocal_weight
        self.instrumental_weight = instrumental_weight
        self.leakage_weight = leakage_weight
        self.leakage_margin = leakage_margin
        self.leakage_activity_threshold = leakage_activity_threshold
        self.l1 = nn.L1Loss()

    def _multires_stft_loss(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-resolution STFT linear + log1p magnitude loss on stereo waveforms
        in float32 for numerical stability.
        """
        pred_f32 = pred_wav.float().reshape(-1, pred_wav.shape[-1])
        targ_f32 = target_wav.float().reshape(-1, target_wav.shape[-1])

        total = 0.0
        for n_fft in self.multires_ffts:
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=pred_f32.device, dtype=torch.float32)
            p_stft = torch.stft(
                pred_f32,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=win,
                return_complex=True,
            )
            t_stft = torch.stft(
                targ_f32,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=win,
                return_complex=True,
            )
            p_mag = torch.abs(p_stft)
            t_mag = torch.abs(t_stft)

            lin_l1 = F.l1_loss(p_mag, t_mag)
            log_l1 = F.l1_loss(torch.log1p(p_mag), torch.log1p(t_mag))
            total = total + (lin_l1 + log_l1)

        return total / len(self.multires_ffts)

    def forward(
        self,
        mixture_stft: torch.Tensor,
        mixture_magnitude: torch.Tensor,
        masks: torch.Tensor,
        vocals_magnitude: torch.Tensor,
        instrumental_magnitude: torch.Tensor,
        vocals_waveform: torch.Tensor,
        instrumental_waveform: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        vocal_mask = masks[:, 0]
        instrumental_mask = masks[:, 1]

        pred_vocals_mag = mixture_magnitude * vocal_mask
        pred_inst_mag = mixture_magnitude * instrumental_mask

        # 1. Linear magnitude L1
        lin_voc = self.l1(pred_vocals_mag, vocals_magnitude)
        lin_inst = self.l1(pred_inst_mag, instrumental_magnitude)
        linear_loss = (
            self.vocal_weight * lin_voc + self.instrumental_weight * lin_inst
        )

        # 2. Log1p magnitude L1 (equalizes frequency band importance across 20Hz-22kHz)
        log_voc = self.l1(
            torch.log1p(pred_vocals_mag.float()),
            torch.log1p(vocals_magnitude.float()),
        )
        log_inst = self.l1(
            torch.log1p(pred_inst_mag.float()),
            torch.log1p(instrumental_magnitude.float()),
        )
        log_loss = (
            self.vocal_weight * log_voc + self.instrumental_weight * log_inst
        )

        total_loss = (
            self.linear_mag_weight * linear_loss
            + self.log_mag_weight * log_loss
        )

        metrics = {
            "linear_l1": float((lin_voc + lin_inst).detach().item()),
            "log_l1": float((log_voc + log_inst).detach().item()),
        }

        # 3. Time-domain waveform & Multi-Resolution STFT consistency loss (in fp32)
        if self.waveform_l1_weight > 0.0 or self.multires_stft_weight > 0.0:
            with torch.amp.autocast("cuda", enabled=False):
                target_len = vocals_waveform.shape[-1]
                mix_stft_f32 = mixture_stft
                pred_voc_wav = reconstruct_source(
                    mix_stft_f32, vocal_mask.float(), length=target_len
                )
                pred_inst_wav = reconstruct_source(
                    mix_stft_f32, instrumental_mask.float(), length=target_len
                )

                if self.waveform_l1_weight > 0.0:
                    wav_voc = self.l1(pred_voc_wav, vocals_waveform.float())
                    wav_inst = self.l1(pred_inst_wav, instrumental_waveform.float())
                    wav_loss = (
                        self.vocal_weight * wav_voc
                        + self.instrumental_weight * wav_inst
                    )
                    total_loss = total_loss + self.waveform_l1_weight * wav_loss
                    metrics["wav_l1"] = float((wav_voc + wav_inst).detach().item())

                if self.multires_stft_weight > 0.0:
                    mr_voc = self._multires_stft_loss(
                        pred_voc_wav, vocals_waveform
                    )
                    mr_inst = self._multires_stft_loss(
                        pred_inst_wav, instrumental_waveform
                    )
                    mr_loss = (
                        self.vocal_weight * mr_voc
                        + self.instrumental_weight * mr_inst
                    )
                    total_loss = total_loss + self.multires_stft_weight * mr_loss
                    metrics["mr_stft"] = float((mr_voc + mr_inst).detach().item())

        metrics["base_total"] = float(total_loss.detach().item())
        leak_loss = asymmetric_leakage_loss(
            masks=masks,
            vocals_mag=vocals_magnitude,
            instrumental_mag=instrumental_magnitude,
            margin=self.leakage_margin,
            activity_threshold=self.leakage_activity_threshold,
        )
        total_loss = total_loss + self.leakage_weight * leak_loss
        metrics["leakage"] = float(leak_loss.detach().item())

        metrics["total"] = float(total_loss.detach().item())
        metrics["objective"] = metrics["total"]
        metrics["total"] = metrics["base_total"]
        return total_loss, metrics
