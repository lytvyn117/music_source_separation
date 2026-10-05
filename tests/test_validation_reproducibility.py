from src.data.dataset import MUSDBDataset
from src.data.splits import create_train_validation_split


_, validation_files = create_train_validation_split()

dataset = MUSDBDataset(
    track_files=validation_files,
    samples_per_epoch=10,
    random_segments=False,
    seed=42
)


sample_1 = dataset[0]
sample_2 = dataset[0]

print("First call:")
print(sample_1["track_name"])
print(sample_1["start"])

print("\nSecond call:")
print(sample_2["track_name"])
print(sample_2["start"])

sample_3 = dataset[1]

print("\nDifferent index:")
print(sample_3["track_name"])
print(sample_3["start"])