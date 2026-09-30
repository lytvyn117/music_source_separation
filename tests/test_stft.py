import torch

from src.audio.stft import compute_stft, compute_istft


# 1 second test signal
sample_rate = 44100

audio = torch.randn(sample_rate)

stft = compute_stft(audio)

reconstructed = compute_istft(
    stft,
    length=len(audio)
)


print("Original shape:")
print(audio.shape)

print("\nSTFT shape:")
print(stft.shape)

print("\nReconstructed shape:")
print(reconstructed.shape)


error = torch.mean(
    torch.abs(audio - reconstructed)
)

print("\nReconstruction error:")
print(error.item())