import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

PROJECT_ROOT = Path(r"C:\Users\YOUR_USER\music_source_separation")
STAGING_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(STAGING_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
import src
if str(STAGING_ROOT / "src") not in src.__path__:
    src.__path__.append(str(STAGING_ROOT / "src"))

import numpy as np
import soundfile as sf
import torch

from config import (
    CHECKPOINTS_DIR,
    SAMPLE_RATE,
    SEGMENT_LENGTH_SECONDS,
    SEPARATED_TRACKS_DIR,
)
from src.audio.separation import reconstruct_source
from src.audio.stft import compute_magnitude, compute_stft
from src.models.unet import UNet


def load_separation_model(
    checkpoint_path: Union[str, Path] = CHECKPOINTS_DIR / "best_model.pt",
    device: Optional[Union[str, torch.device]] = None,
    base_channels: int = 16,
) -> Tuple[UNet, Dict, torch.device]:
    """
    Safely load a trained U-Net checkpoint in evaluation mode (read-only).

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the saved model checkpoint (.pt).
    device : str or torch.device, optional
        Target device ('cuda' or 'cpu'). Auto-detected if None.
    base_channels : int
        Base channel width of the U-Net architecture.

    Returns
    -------
    model : UNet
        Loaded model in eval() mode.
    metadata : dict
        Checkpoint metadata (epoch, train_loss, validation_loss, etc.).
    device : torch.device
        Device the model resides on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        from src.models.unet_v2 import UNetV2

        cfg = checkpoint["config"]
        model = UNetV2(
            input_channels=2,
            base_channels=cfg.get("base_channels", base_channels),
            pad_mode=cfg.get("pad_mode", "exact"),
            log_input=cfg.get("log_input", False),
            mask_mode=cfg.get("mask_mode", "softmax"),
        ).to(device)
    else:
        model = UNet(input_channels=2, base_channels=base_channels).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        k: v
        for k, v in checkpoint.items()
        if k not in ("model_state_dict", "optimizer_state_dict")
    }

    return model, metadata, device


@torch.no_grad()
def separate_waveform(
    model: torch.nn.Module,
    mixture: Union[np.ndarray, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    sample_rate: int = SAMPLE_RATE,
    segment_length_seconds: float = SEGMENT_LENGTH_SECONDS,
    overlap_ratio: float = 0.5,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Perform full-song stereo source separation using overlapping chunks
    and smooth Hann window overlap-add reconstruction.

    Parameters
    ----------
    model : torch.nn.Module
        Trained separation model taking magnitude spectrograms of shape
        (batch, 2, freq, time) and returning masks of shape
        (batch, 2, 2, freq, time).
    mixture : np.ndarray or torch.Tensor
        Input stereo mixture waveform. Accepted shapes:
        - (samples, 2) [soundfile convention]
        - (2, samples) [PyTorch convention]
        - (samples,)   [mono; duplicated to stereo and returned as stereo]
    device : str or torch.device, optional
        Compute device. Inferred from model parameters if None.
    sample_rate : int
        Audio sample rate in Hz (default: 44100).
    segment_length_seconds : float
        Chunk duration in seconds (default: 6.0, matching training).
    overlap_ratio : float
        Overlap fraction between consecutive chunks in [0.0, 0.95) (default: 0.5).
    batch_size : int
        Number of chunks processed simultaneously on GPU.

    Returns
    -------
    dict
        Dictionary with keys ``"vocals"`` and ``"instrumental"``, each a
        float32 numpy array matching the input orientation.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    if isinstance(mixture, torch.Tensor):
        mix_np = mixture.detach().cpu().numpy().astype(np.float32)
    else:
        mix_np = np.asarray(mixture, dtype=np.float32)

    # Normalize shape to (channels=2, samples) internally
    transposed_input = False
    if mix_np.ndim == 1:
        mix_ch_first = np.stack([mix_np, mix_np], axis=0)
        transposed_input = True
    elif mix_np.ndim == 2:
        if mix_np.shape[0] == 2 and mix_np.shape[1] != 2:
            mix_ch_first = mix_np
            transposed_input = False
        elif mix_np.shape[1] == 2:
            mix_ch_first = mix_np.T.copy()
            transposed_input = True
        elif mix_np.shape[0] == 1:
            mix_ch_first = np.repeat(mix_np, 2, axis=0)
            transposed_input = False
        elif mix_np.shape[1] == 1:
            mix_ch_first = np.repeat(mix_np.T, 2, axis=0)
            transposed_input = True
        else:
            raise ValueError(f"Unsupported stereo mixture shape: {mix_np.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {mix_np.shape}")

    total_samples = mix_ch_first.shape[1]
    chunk_samples = int(round(segment_length_seconds * sample_rate))

    if not (0.0 <= overlap_ratio < 0.95):
        raise ValueError(f"overlap_ratio must be in [0.0, 0.95), got {overlap_ratio}")

    hop_samples = max(1, int(round(chunk_samples * (1.0 - overlap_ratio))))

    # Short audio fallback: single padded chunk
    if total_samples <= chunk_samples:
        pad_amount = chunk_samples - total_samples
        padded = np.pad(mix_ch_first, ((0, 0), (0, pad_amount)), mode="constant")
        x = torch.from_numpy(padded).unsqueeze(0).to(device)
        mix_stft = compute_stft(x)
        mag = compute_magnitude(mix_stft)
        masks = model(mag)
        voc = (
            reconstruct_source(mix_stft, masks[:, 0], length=chunk_samples)[
                0, :, :total_samples
            ]
            .cpu()
            .numpy()
        )
        inst = (
            reconstruct_source(mix_stft, masks[:, 1], length=chunk_samples)[
                0, :, :total_samples
            ]
            .cpu()
            .numpy()
        )
        if transposed_input:
            return {"vocals": voc.T.copy(), "instrumental": inst.T.copy()}
        return {"vocals": voc, "instrumental": inst}

    # Generate chunk start positions covering the full track
    starts = list(range(0, total_samples - chunk_samples + 1, hop_samples))
    if starts[-1] + chunk_samples < total_samples:
        starts.append(total_samples - chunk_samples)

    # Smooth window for overlap-add (Hann window with non-zero endpoints)
    if overlap_ratio > 0.0 and len(starts) > 1:
        window = np.hanning(chunk_samples + 2)[1:-1].astype(np.float32)
    else:
        window = np.ones(chunk_samples, dtype=np.float32)

    window_2d = window[np.newaxis, :]  # (1, chunk_samples)

    vocals_acc = np.zeros((2, total_samples), dtype=np.float32)
    inst_acc = np.zeros((2, total_samples), dtype=np.float32)
    weight_acc = np.zeros((1, total_samples), dtype=np.float32)

    # Process chunks in batches on GPU
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        batch_chunks = np.stack(
            [mix_ch_first[:, s : s + chunk_samples] for s in batch_starts],
            axis=0,
        )  # (B, 2, chunk_samples)

        x_batch = torch.from_numpy(batch_chunks).to(device)
        mix_stft = compute_stft(x_batch)
        mag = compute_magnitude(mix_stft)
        masks = model(mag)

        voc_batch = (
            reconstruct_source(mix_stft, masks[:, 0], length=chunk_samples)
            .cpu()
            .numpy()
        )
        inst_batch = (
            reconstruct_source(mix_stft, masks[:, 1], length=chunk_samples)
            .cpu()
            .numpy()
        )

        for idx, s in enumerate(batch_starts):
            end = s + chunk_samples
            vocals_acc[:, s:end] += voc_batch[idx] * window_2d
            inst_acc[:, s:end] += inst_batch[idx] * window_2d
            weight_acc[:, s:end] += window_2d

    # Normalize by accumulated window weights (partition of unity)
    weight_acc = np.maximum(weight_acc, 1e-8)
    vocals_out = vocals_acc / weight_acc
    inst_out = inst_acc / weight_acc

    if transposed_input:
        return {
            "vocals": vocals_out.T.copy(),
            "instrumental": inst_out.T.copy(),
        }
    return {
        "vocals": vocals_out,
        "instrumental": inst_out,
    }


def separate_track_file(
    input_path: Union[str, Path],
    output_dir: Union[str, Path] = SEPARATED_TRACKS_DIR,
    checkpoint_path: Union[str, Path] = CHECKPOINTS_DIR / "best_model.pt",
    segment_length_seconds: float = SEGMENT_LENGTH_SECONDS,
    overlap_ratio: float = 0.5,
    batch_size: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    save_mixture_copy: bool = False,
) -> Dict[str, Path]:
    """
    Separate an audio file into vocals.wav and instrumental.wav.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {SAMPLE_RATE} Hz, but {input_path.name} has {sr} Hz."
        )

    model, _, dev = load_separation_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    preds = separate_waveform(
        model=model,
        mixture=audio,
        device=dev,
        sample_rate=sr,
        segment_length_seconds=segment_length_seconds,
        overlap_ratio=overlap_ratio,
        batch_size=batch_size,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "instrumental.wav"

    sf.write(vocals_path, preds["vocals"], sr, subtype="FLOAT")
    sf.write(instrumental_path, preds["instrumental"], sr, subtype="FLOAT")

    result_paths = {
        "vocals": vocals_path,
        "instrumental": instrumental_path,
    }

    if save_mixture_copy:
        mix_path = output_dir / "mixture.wav"
        sf.write(mix_path, audio, sr, subtype="FLOAT")
        result_paths["mixture"] = mix_path

    return result_paths


def main():
    parser = argparse.ArgumentParser(
        description="Full-track stereo music source separation (vocals & instrumental)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input mixture WAV file (or MUSDB processed track directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save separated stems (default: outputs/separated_tracks/<track_name>).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "best_model.pt",
        help="Path to model checkpoint (default: models/checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=SEGMENT_LENGTH_SECONDS,
        help="Chunk length in seconds (default: 6.0).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio between chunks in [0.0, 0.9] (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks processed per GPU batch (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda' or 'cpu').",
    )

    args = parser.parse_args()

    input_path = args.input_path
    if input_path.is_dir():
        track_name = input_path.name
        input_wav = input_path / "mixture.wav"
    else:
        track_name = input_path.stem
        input_wav = input_path

    out_dir = args.output_dir or (SEPARATED_TRACKS_DIR / track_name)

    print(f"Separating: {input_wav}")
    print(f"Checkpoint: {args.checkpoint}")
    paths = separate_track_file(
        input_path=input_wav,
        output_dir=out_dir,
        checkpoint_path=args.checkpoint,
        segment_length_seconds=args.segment_length,
        overlap_ratio=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Saved vocals to:       {paths['vocals']}")
    print(f"Saved instrumental to: {paths['instrumental']}")


if __name__ == "__main__":
    main()
