import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\YOUR_USER\music_source_separation")
STAGING_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(STAGING_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
import src
if str(STAGING_ROOT / "src") not in src.__path__:
    src.__path__.append(str(STAGING_ROOT / "src"))

import torch
from src.audio.stft import compute_magnitude, compute_stft
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split
from src.inference.separate_track import separate_waveform
from src.models.unet_v2 import UNetV2
from src.training.losses_v2 import CompositeSeparationLoss


def test_unet_v2():
    print("Testing UNetV2 forward passes...", end="", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 2, 1025, 517, device=device)

    # 1. Softmax with exact pad
    m1 = UNetV2(base_channels=16, pad_mode="exact", mask_mode="softmax").to(device)
    out1 = m1(x)
    assert out1.shape == (2, 2, 2, 1025, 517), f"Expected shape (2, 2, 2, 1025, 517), got {out1.shape}"
    assert torch.allclose(out1.sum(dim=1), torch.ones_like(out1[:, 0]), atol=1e-5), "Softmax mask sum != 1"

    # 2. Sigmoid with exact pad
    m2 = UNetV2(base_channels=16, pad_mode="exact", mask_mode="sigmoid").to(device)
    out2 = m2(x)
    assert out2.shape == (2, 2, 2, 1025, 517)
    assert out2.min() >= 0.0 and out2.max() <= 1.0, "Sigmoid mask out of range [0, 1]"

    # 3. Log input
    m3 = UNetV2(base_channels=16, log_input=True).to(device)
    out3 = m3(torch.abs(x))
    assert out3.shape == (2, 2, 2, 1025, 517)
    print(" OK!")


def test_composite_loss_gradients():
    print("Testing CompositeSeparationLoss and backward pass...", end="", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetV2(base_channels=16, pad_mode="exact").to(device)
    loss_fn = CompositeSeparationLoss().to(device)

    B = 2
    T = 44100 * 2  # 2 seconds
    mix = torch.randn(B, 2, T, device=device)
    voc = torch.randn(B, 2, T, device=device)
    inst = torch.randn(B, 2, T, device=device)

    mix_stft = compute_stft(mix)
    voc_stft = compute_stft(voc)
    inst_stft = compute_stft(inst)

    mix_mag = compute_magnitude(mix_stft)
    voc_mag = compute_magnitude(voc_stft)
    inst_mag = compute_magnitude(inst_stft)

    masks = model(mix_mag)
    loss, metrics = loss_fn(
        mixture_stft=mix_stft,
        mixture_magnitude=mix_mag,
        masks=masks,
        vocals_magnitude=voc_mag,
        instrumental_magnitude=inst_mag,
        vocals_waveform=voc,
        instrumental_waveform=inst,
    )

    assert torch.isfinite(loss), f"Loss is non-finite: {loss}"
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"
    print(" OK! (All gradients finite)")


def test_augmented_dataset():
    print("Testing AugmentedMUSDBDataset...", end="", flush=True)
    train_files, _ = create_train_validation_split()
    ds = AugmentedMUSDBDataset(
        track_files=train_files,
        samples_per_epoch=4,
        random_segments=True,
        augment=True,
    )
    s = ds[0]
    assert "mixture" in s and "vocals" in s and "instrumental" in s
    assert s["mixture"].shape == (2, 44100 * 6), f"Expected (2, 264600), got {s['mixture'].shape}"
    # Verify linear additivity: mixture == vocals + instrumental
    diff = torch.max(torch.abs(s["mixture"] - (s["vocals"] + s["instrumental"])))
    assert diff < 1e-4, f"Mixture is not additive: max diff = {diff}"
    print(" OK!")


def test_separate_waveform_overlap_add():
    print("Testing separate_waveform overlap-add reconstruction...", end="", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetV2(base_channels=16).to(device).eval()

    # Test with 15 seconds stereo audio
    sr = 44100
    mix = torch.randn(sr * 15, 2).numpy()
    preds = separate_waveform(
        model=model,
        mixture=mix,
        device=device,
        sample_rate=sr,
        overlap_ratio=0.5,
        batch_size=4,
    )
    assert "vocals" in preds and "instrumental" in preds
    assert preds["vocals"].shape == mix.shape, f"Expected {mix.shape}, got {preds['vocals'].shape}"
    assert preds["instrumental"].shape == mix.shape
    assert not torch.isnan(torch.from_numpy(preds["vocals"])).any(), "NaN in separated vocals"
    assert not torch.isnan(torch.from_numpy(preds["instrumental"])).any(), "NaN in separated instrumental"
    print(" OK!")


if __name__ == "__main__":
    test_unet_v2()
    test_composite_loss_gradients()
    test_augmented_dataset()
    test_separate_waveform_overlap_add()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
