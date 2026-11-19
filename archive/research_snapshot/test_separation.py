
from pathlib import Path
import torch
import soundfile as sf

from src.models.unet import UNet
from src.audio.stft import compute_stft, compute_magnitude
from src.audio.separation import reconstruct_source

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load best trained model
model = UNet().to(device)
checkpoint = torch.load(
    "models/checkpoints/best_model.pt",
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Pick an unseen test track
test_dir = Path("data/processed/test")
track = sorted(p for p in test_dir.iterdir() if p.is_dir())[0]

print("Testing:", track.name)

# Read 6 seconds
audio, sr = sf.read(
    track / "mixture.wav",
    frames=6 * 44100,
    dtype="float32",
    always_2d=True
)

x = torch.from_numpy(audio.T.copy()).unsqueeze(0).to(device)

with torch.no_grad():
    mixture_stft = compute_stft(x)
    magnitude = compute_magnitude(mixture_stft)
    masks = model(magnitude)

    vocals = reconstruct_source(
        mixture_stft, masks[:, 0], length=x.shape[-1]
    )
    instrumental = reconstruct_source(
        mixture_stft, masks[:, 1], length=x.shape[-1]
    )

out = Path("outputs/test_results")
out.mkdir(parents=True, exist_ok=True)

sf.write(out / "predicted_vocals.wav",
         vocals[0].T.cpu().numpy(), sr)

sf.write(out / "predicted_instrumental.wav",
         instrumental[0].T.cpu().numpy(), sr)

sf.write(out / "original_mix.wav", audio, sr)

for source in ["vocals", "instrumental"]:
    reference, _ = sf.read(
        track / f"{source}.wav",
        frames=len(audio),
        dtype="float32",
        always_2d=True
    )
    sf.write(out / f"real_{source}.wav", reference, sr)

print("Done! Results saved to:", out.resolve())
