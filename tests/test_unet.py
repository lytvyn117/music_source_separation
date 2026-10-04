import torch

from src.models.unet import UNet


model = UNet()

# Simulates one magnitude spectrogram
x = torch.randn(
    1,
    2,
    1025,
    517
)


with torch.no_grad():
    masks = model(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(masks.shape)


print("\nMask minimum:")
print(masks.min().item())

print("\nMask maximum:")
print(masks.max().item())


# Sum across source dimension
mask_sum = masks.sum(dim=1)

print("\nMask sum shape:")
print(mask_sum.shape)

print("\nMean mask sum:")
print(mask_sum.mean().item())