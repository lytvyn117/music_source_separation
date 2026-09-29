# Music Source Separation – Vocals vs. Instrumental

## Project Overview

This project focuses on **music source separation** using machine learning and deep learning.

The goal is to take a complete music track as input and separate it into two output stems:

- **Vocals**
- **Instrumental accompaniment**

The model is trained on the **MUSDB18** dataset and uses a neural network based on an **encoder-decoder / U-Net architecture**.

The project is developed as part of university coursework in the fields of:

- Artificial Intelligence
- Machine Learning
- Deep Learning

---

## Project Goal

Given a mixed music track:

```text
song.wav
```

the trained model should generate:

```text
vocals.wav
instrumental.wav
```

The project therefore solves an **audio-to-audio separation problem** rather than a classification problem.

---

## Dataset

The project uses the **MUSDB18** dataset.

MUSDB18 contains **150 full-length songs**:

- **100 training songs**
- **50 test songs**

Each song is stored in the Native Instruments **STEMS** format and contains five stereo audio streams:

| Stream | Content |
|---|---|
| 0 | Mixture |
| 1 | Drums |
| 2 | Bass |
| 3 | Other accompaniment |
| 4 | Vocals |

For this project, the target instrumental signal is created as:

```text
instrumental = drums + bass + other
```

The two target sources are therefore:

```text
vocals
instrumental
```

All MUSDB18 signals are stereo and sampled at **44.1 kHz**.

The raw dataset is stored under:

```text
data/raw/
├── train/
├── test/
└── README.md
```

The original dataset files in `data/raw/` should remain unchanged.

---

## Project Structure

```text
music_source_separation/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── test/
│   │   └── README.md
│   │
│   ├── processed/
│   └── samples/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_audio_analysis.ipynb
│   ├── 03_spectrogram_visualization.ipynb
│   └── 04_results_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── splits.py
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── stft.py
│   │   └── separation.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── losses.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── separate_track.py
│
├── models/
│   ├── checkpoints/
│   └── final/
│
├── outputs/
│   ├── separated_tracks/
│   ├── spectrograms/
│   └── logs/
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── results.md
│
└── tests/
    ├── test_dataset.py
    ├── test_stft.py
    └── test_model.py
```

---

## Processing Pipeline

The basic processing pipeline is:

```text
Music Track
    ↓
Audio Loading
    ↓
Short-Time Fourier Transform (STFT)
    ↓
Magnitude Spectrogram
    ↓
U-Net
    ↓
Vocal Mask + Instrumental Mask
    ↓
Separated Spectrograms
    ↓
Inverse STFT
    ↓
vocals.wav + instrumental.wav
```

---

## STFT and Spectrograms

A music waveform contains information about signal amplitude over time.

The **Short-Time Fourier Transform (STFT)** divides the audio into short time windows and determines which frequencies are present in each window.

This produces a time-frequency representation.

The magnitude of this representation can be visualized as a **spectrogram**.

The neural network uses this representation to learn which regions belong primarily to vocals and which belong to the accompaniment.

---

## Separation Model

The planned model is a **U-Net-style encoder-decoder neural network**.

The encoder extracts increasingly abstract features from the input spectrogram.

The decoder reconstructs a separation mask.

Skip connections preserve fine-grained time-frequency information between corresponding encoder and decoder layers.

The network predicts masks such as:

```text
Vocal Mask
Instrumental Mask
```

These masks are applied to the mixture spectrogram:

```text
mixture × vocal_mask = vocal_spectrogram

mixture × instrumental_mask = instrumental_spectrogram
```

The separated spectrograms are converted back into audio using the **inverse STFT**.

---

## Training Strategy

The model should not necessarily use full songs as individual training samples.

Instead, random short segments can be loaded dynamically during training.

Example:

```text
segment length: 6–10 seconds
```

This allows the same song to provide many different training examples.

The training pipeline should approximately follow:

```text
MUSDB18 song
    ↓
Random audio segment
    ↓
Mixture + target sources
    ↓
STFT
    ↓
U-Net
    ↓
Predicted masks
    ↓
Loss calculation
    ↓
Backpropagation
```

The validation split should be created **on song level**, not on segment level, to prevent data leakage.

---

## Input and Targets

For each training example:

### Input

```text
mixture
```

### Targets

```text
vocals
instrumental
```

with:

```text
instrumental = drums + bass + other
```

---

## Evaluation

The separation quality should be evaluated using suitable source-separation metrics.

Possible metrics include:

- SDR – Signal-to-Distortion Ratio
- SI-SDR – Scale-Invariant Signal-to-Distortion Ratio
- Validation loss

Qualitative evaluation is also useful by listening to separated tracks and comparing predicted stems with the original MUSDB18 targets.

---

## Inference

After training, the model should be able to process music tracks that were **not part of the training dataset**.

Example:

```text
python -m src.inference.separate_track path/to/song.wav
```

Expected output:

```text
outputs/separated_tracks/
├── vocals.wav
└── instrumental.wav
```

The goal is to test whether the model generalizes to completely unseen music.

---

## Technology Stack

Planned technologies include:

- Python
- PyTorch
- NumPy
- librosa
- stempeg
- FFmpeg
- soundfile
- matplotlib
- Jupyter Notebook

Additional libraries may be added during development.

---

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Because MUSDB18 uses the STEMS format, **FFmpeg** must also be installed and available in the system path.

---

## Configuration

Important project parameters should be stored in:

```text
config.yaml
```

Possible configuration values include:

```yaml
data:
  raw_dir: data/raw

audio:
  sample_rate: 44100
  segment_length: 6.0

stft:
  n_fft: 2048
  hop_length: 512

training:
  batch_size: 8
  learning_rate: 0.001
  epochs: 50
```

These values are initial examples and may change during experimentation.

---

## Planned Development Steps

1. Verify that MUSDB18 files can be decoded with `stempeg` and FFmpeg.
2. Load mixture, vocals, drums, bass and other stems.
3. Construct the instrumental target.
4. Visualize waveforms and spectrograms.
5. Implement STFT and inverse STFT.
6. Build the PyTorch dataset pipeline.
7. Implement the U-Net.
8. Train the first separation model.
9. Evaluate separation quality.
10. Test the model on unseen music tracks.
11. Compare results and document limitations.

---

## Expected Result

The final system should perform:

```text
unseen music track
        ↓
trained model
        ↓
vocals.wav
instrumental.wav
```

The project aims to demonstrate the complete machine-learning workflow from raw audio data to a trained deep-learning model and practical inference on new audio files.

---

## Current Status

- [x] Project structure created
- [x] MUSDB18 downloaded
- [x] MUSDB18 extracted into `data/raw/`
- [ ] Python environment finalized
- [ ] MUSDB18 loading tested with `stempeg`
- [ ] STFT pipeline implemented
- [ ] Dataset class implemented
- [ ] U-Net implemented
- [ ] Model training completed
- [ ] Evaluation completed
- [ ] Inference on unseen tracks completed
