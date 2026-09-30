from src.data.dataset import MUSDBDataset


dataset = MUSDBDataset(
    samples_per_epoch=10
)

print(f"Dataset length: {len(dataset)}")


sample = dataset[0]


print("\nTrack:")
print(sample["track_name"])

print("\nStart:")
print(sample["start"])

print("\nMixture:")
print(sample["mixture"].shape)

print("\nVocals:")
print(sample["vocals"].shape)

print("\nInstrumental:")
print(sample["instrumental"].shape)