# Music Source Separation

The current reference is **Exp10**, an independently implemented BS-RoFormer-inspired stereo separator trained from scratch on MoisesDB. Historical U-Net implementations are archived.

## Start here

- [Two-computer workflow and execution matrix](docs/WORKFLOW.md)
- [Setup, training, evaluation and inference](EXP10_REPRODUCTION.md)
- [Architecture](docs/architecture.md)
- [Settings and configuration](docs/SETTINGS.md)
- [Laptop demo on CPU](docs/CPU_DEMO.md)
- [Experiment index](experiments/README.md)
- [Migration and reproducibility](docs/MIGRATION.md)

## Project layout

| Directory | Purpose |
| --- | --- |
| `src/music_separation/models/` | Shared band-split Transformer |
| `src/music_separation/audio/` | Overlap-add reconstruction; STFT/iSTFT remain model methods |
| `src/music_separation/data/` | MoisesDB loading, frozen rosters and sampling |
| `src/music_separation/training/` | Trainer, losses, EMA and checkpoint selection |
| `src/music_separation/evaluation/` | Metrics and full-song evaluation |
| `src/music_separation/inference/` | Audio separation CLI |
| `configs/training/exp10/` | Recorded training configuration and roster |
| `tests/` | Active CPU unit and migration tests |
| `notebooks/` | Original first steps: dataset exploration, audio analysis and spectrogram visualization |
| `demos/` | Presentation launcher using the shared package |
| `experiments/` | Experiment index and links to historical evidence |
| `archive/research_snapshot/` | Original research layout and frozen implementations |
| `provenance/` | Original run and checkpoint records |
| `models/`, `data/`, `outputs/`, `runs/` | Local weights, data and generated artifacts; excluded from ordinary Git tracking |

## Selected reference

The selected checkpoint is **step 24,000 Raw**, with **7.0884 dB artist-macro vocal SDR** on 16 development recordings from seven artists. This is `best_overall`; no candidate passed all `best_qualified` guardrails. The external final holdout remains separate. Earlier MUSDB18 scores are not directly comparable. See the [recorded endpoint](provenance/development_endpoint.json).

MoisesDB can remain on the training computer. Inference on your own audio only requires a checkpoint and 44.1 kHz stereo WAV/FLAC. Use `--device auto` (default), `--device cpu` or `--device cuda`. Auto selects CUDA when available and otherwise uses CPU.

Install the shared package in an existing environment with `python -m pip install -e . --no-deps`. Run active unit tests from the repository root with `python -B -m unittest discover -s tests -v`. See the reproduction guide for environment setup.

Original [source origins](SOURCE_ORIGINS.json) and [handoff checksums](SOURCE_SHA256SUMS.txt) refer to the historical layout, not the refactored tree. The pre-refactor Git tag is `pre-layout-refactor-2026-11-19`.

## Introductory notebooks

The notebooks in [notebooks/](notebooks/) document the first steps of the project, from MUSDB18 exploration to audio and spectrogram analysis. They remain part of the main project as educational material. They describe the original data workflow, not the current Exp10 training pipeline; running them requires their original data and dependencies.
