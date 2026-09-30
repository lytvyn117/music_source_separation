from pathlib import Path

import soundfile as sf
import stempeg


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# Erste MUSDB18-Datei finden
files = list(TRAIN_DIR.glob("*.mp4"))

if not files:
    raise FileNotFoundError(
        f"Keine .mp4-Dateien gefunden in: {TRAIN_DIR}"
    )

audio_file = files[0]

print(f"Lade Datei: {audio_file.name}")


# Alle 5 Stems laden
stems, sample_rate = stempeg.read_stems(str(audio_file))


print("\n--- Dataset Info ---")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Shape: {stems.shape}")
print(f"Datentyp: {stems.dtype}")


# MUSDB18-Reihenfolge:
# 0 = mixture
# 1 = drums
# 2 = bass
# 3 = other
# 4 = vocals

mixture = stems[0]
drums = stems[1]
bass = stems[2]
other = stems[3]
vocals = stems[4]

instrumental = drums + bass + other


print("\n--- Stem Shapes ---")
print(f"Mixture:      {mixture.shape}")
print(f"Drums:        {drums.shape}")
print(f"Bass:         {bass.shape}")
print(f"Other:        {other.shape}")
print(f"Vocals:       {vocals.shape}")
print(f"Instrumental: {instrumental.shape}")


duration_seconds = mixture.shape[0] / sample_rate

print(f"\nTrack length: {duration_seconds:.2f} seconds")


# Erste 10 Sekunden als WAV exportieren
preview_length = int(sample_rate * 10)

sf.write(
    SAMPLE_DIR / "mixture_preview.wav",
    mixture[:preview_length],
    sample_rate
)

sf.write(
    SAMPLE_DIR / "vocals_preview.wav",
    vocals[:preview_length],
    sample_rate
)

sf.write(
    SAMPLE_DIR / "instrumental_preview.wav",
    instrumental[:preview_length],
    sample_rate
)

print("\nPreview-Dateien erstellt:")
print("data/samples/mixture_preview.wav")
print("data/samples/vocals_preview.wav")
print("data/samples/instrumental_preview.wav")