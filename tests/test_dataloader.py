from torch.utils.data import DataLoader

from config import BATCH_SIZE
from src.data.dataset import MUSDBDataset


dataset = MUSDBDataset(
    samples_per_epoch=32
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


batch = next(iter(dataloader))


print("Mixture:")
print(batch["mixture"].shape)

print("\nVocals:")
print(batch["vocals"].shape)

print("\nInstrumental:")
print(batch["instrumental"].shape)

print("\nTrack names:")
print(batch["track_name"])

print("\nStart positions:")
print(batch["start"])