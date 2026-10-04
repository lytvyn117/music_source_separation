import torch

from src.training.losses import SeparationLoss


loss_function = SeparationLoss()


batch = 2
channels = 2
freq = 1025
time = 517


mixture = torch.rand(
    batch,
    channels,
    freq,
    time
)


# Fake masks for two sources
raw_masks = torch.rand(
    batch,
    2,
    channels,
    freq,
    time
)

masks = torch.softmax(
    raw_masks,
    dim=1
)


vocals_target = torch.rand(
    batch,
    channels,
    freq,
    time
)

instrumental_target = torch.rand(
    batch,
    channels,
    freq,
    time
)


loss = loss_function(
    mixture,
    masks,
    vocals_target,
    instrumental_target
)


print("Loss:")
print(loss.item())