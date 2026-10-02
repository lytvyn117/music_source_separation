from torch.utils.data import DataLoader

from config import BATCH_SIZE
from src.data.dataset import MUSDBDataset
from src.audio.stft import compute_stft, compute_magnitude


# Small dataset for testing
dataset = MUSDBDataset(
    samples_per_epoch=32
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Get first batch
batch = next(iter(dataloader))

mixture = batch["mixture"]
vocals = batch["vocals"]
instrumental = batch["instrumental"]


print("Waveform shapes:")
print(f"Mixture:      {mixture.shape}")
print(f"Vocals:       {vocals.shape}")
print(f"Instrumental: {instrumental.shape}")


# Compute STFT
mixture_stft = compute_stft(mixture)
vocals_stft = compute_stft(vocals)
instrumental_stft = compute_stft(instrumental)


print("\nSTFT shapes:")
print(f"Mixture:      {mixture_stft.shape}")
print(f"Vocals:       {vocals_stft.shape}")
print(f"Instrumental: {instrumental_stft.shape}")


# Compute magnitude spectrograms
mixture_magnitude = compute_magnitude(mixture_stft)
vocals_magnitude = compute_magnitude(vocals_stft)
instrumental_magnitude = compute_magnitude(instrumental_stft)


print("\nMagnitude shapes:")
print(f"Mixture:      {mixture_magnitude.shape}")
print(f"Vocals:       {vocals_magnitude.shape}")
print(f"Instrumental: {instrumental_magnitude.shape}")


print("\nData types:")
print(f"STFT:      {mixture_stft.dtype}")
print(f"Magnitude: {mixture_magnitude.dtype}")