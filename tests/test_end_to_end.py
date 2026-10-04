import soundfile as sf
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, SAMPLE_RATE, SAMPLES_DIR
from src.audio.separation import reconstruct_source
from src.audio.stft import compute_magnitude, compute_stft
from src.data.dataset import MUSDBDataset
from src.models.unet import UNet


# ---------------------------------------------------------
# 1. Dataset + DataLoader
# ---------------------------------------------------------

dataset = MUSDBDataset(
    samples_per_epoch=8
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

batch = next(iter(dataloader))

mixture = batch["mixture"]

print("Mixture waveform:")
print(mixture.shape)


# ---------------------------------------------------------
# 2. STFT
# ---------------------------------------------------------

mixture_stft = compute_stft(mixture)

print("\nMixture STFT:")
print(mixture_stft.shape)


# ---------------------------------------------------------
# 3. Magnitude
# ---------------------------------------------------------

mixture_magnitude = compute_magnitude(
    mixture_stft
)

print("\nMixture magnitude:")
print(mixture_magnitude.shape)


# ---------------------------------------------------------
# 4. U-Net
# ---------------------------------------------------------

model = UNet()

model.eval()

with torch.no_grad():
    masks = model(
        mixture_magnitude
    )

print("\nMasks:")
print(masks.shape)


# ---------------------------------------------------------
# 5. Separate source masks
# ---------------------------------------------------------

vocal_mask = masks[:, 0]
instrumental_mask = masks[:, 1]

print("\nVocal mask:")
print(vocal_mask.shape)

print("\nInstrumental mask:")
print(instrumental_mask.shape)


# ---------------------------------------------------------
# 6. Reconstruct audio
# ---------------------------------------------------------

audio_length = mixture.shape[-1]

predicted_vocals = reconstruct_source(
    mixture_stft,
    vocal_mask,
    length=audio_length
)

predicted_instrumental = reconstruct_source(
    mixture_stft,
    instrumental_mask,
    length=audio_length
)

print("\nPredicted vocals:")
print(predicted_vocals.shape)

print("\nPredicted instrumental:")
print(predicted_instrumental.shape)


# ---------------------------------------------------------
# 7. Export first sample from batch
# ---------------------------------------------------------

SAMPLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

mixture_sample = (
    mixture[0]
    .T
    .cpu()
    .numpy()
)

vocals_sample = (
    predicted_vocals[0]
    .T
    .cpu()
    .numpy()
)

instrumental_sample = (
    predicted_instrumental[0]
    .T
    .cpu()
    .numpy()
)


sf.write(
    SAMPLES_DIR / "untrained_mixture.wav",
    mixture_sample,
    SAMPLE_RATE
)

sf.write(
    SAMPLES_DIR / "untrained_vocals.wav",
    vocals_sample,
    SAMPLE_RATE
)

sf.write(
    SAMPLES_DIR / "untrained_instrumental.wav",
    instrumental_sample,
    SAMPLE_RATE
)


print("\nFiles exported:")
print("data/samples/untrained_mixture.wav")
print("data/samples/untrained_vocals.wav")
print("data/samples/untrained_instrumental.wav")