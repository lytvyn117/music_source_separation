import torch

from src.audio.stft import compute_stft
from src.audio.separation import reconstruct_source


sample_rate = 44100

audio = torch.randn(sample_rate)

mixture_stft = compute_stft(audio)


# Maske mit nur Einsen:
# sollte das Originalsignal praktisch unverändert rekonstruieren
mask = torch.ones_like(
    mixture_stft.abs()
)


reconstructed = reconstruct_source(
    mixture_stft,
    mask,
    length=len(audio)
)


error = torch.mean(
    torch.abs(audio - reconstructed)
)


print("Original shape:")
print(audio.shape)

print("\nReconstructed shape:")
print(reconstructed.shape)

print("\nReconstruction error:")
print(error.item())