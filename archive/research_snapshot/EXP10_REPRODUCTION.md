# Music Source Separation

This source package follows the final Exp10 stereo separator from data loading through training, checkpoint selection, inference, and scoring. The selected checkpoint is step **24,000 Raw**. It scored **7.0884 dB** artist-macro vocal SDR on the frozen development panel, and none of the candidates met all qualification thresholds.

## Start here

For development without the dataset or a CUDA GPU, see the [two-computer workflow and execution matrix](docs/WORKFLOW.md). MoisesDB can remain on the training computer. The inference commands below require CUDA even when processing your own audio; they do not require MoisesDB. The isolated unit suite uses synthetic data and can run on CPU with compatible dependencies.

| Task | Required external inputs |
| --- | --- |
| Separate your audio | Exp10 `best_overall.pt`, 44.1 kHz stereo WAV or FLAC, and the CUDA environment below |
| Repeat frozen-panel evaluation | The checkpoint and the approved MoisesDB archive to build the 16 development references, or an existing cache that passes all reference checks |
| Reproduce training from scratch | The original approved MoisesDB archive, sufficient storage, and a CUDA GPU with native BF16 support |

The handoff contains source, frozen train/development configuration and allowlist metadata, and small provenance records. The checkpoint and all audio are supplied separately. The frozen training workflow uses 60 MoisesDB training recordings and 16 artist-disjoint development recordings. The external final holdout stays outside this workflow.

## Environment

Use 64-bit Python 3.13 on Windows and an NVIDIA driver compatible with the PyTorch CUDA 12.8 wheel. The recorded Windows 11 run used Python 3.13.5, PyTorch 2.7.1+cu128, NumPy 2.5.4, SoundFile 0.14.0 with libsndfile 1.2.2, and an RTX 3080 Ti with driver 591.86. The scripts use the first visible CUDA device. `nvidia-smi` must be on PATH for training provenance. Other hardware and software stacks can change numerical results.

Run in PowerShell from the extracted package root with Python 3.13 on PATH.

```powershell
python -m venv .venv
$Python = (Resolve-Path .venv/Scripts/python.exe).Path
& $Python -m pip install -r requirements-exp10.txt
& $Python -c "import torch, numpy, soundfile; print(torch.__version__, numpy.__version__, soundfile.__version__); print(torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

The package uses the official CUDA wheel. [PyTorch 2.7.1 installation](https://docs.pytorch.org/get-started/previous-versions/#v271)

## Run inference

Keep the separately supplied checkpoint outside the source tree. It contains both the selected Raw weights and the same-step EMA weights. Its expected SHA-256 is `bcad32866b37f98a19e1add45ae28c59f2740f88870fb0e6cad9a5fa021f1bef`. Use an existing 44.1 kHz stereo WAV or FLAC, and choose a new output directory.

```powershell
$Checkpoint = 'D:/Models/best_overall.pt'
Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint
& $Python -B exp10_scratch/infer.py --checkpoint $Checkpoint --input-audio 'D:/Audio/mix.wav' --output-dir 'D:/Audio/exp10-stems'
```

The default selects Raw. Use `--weights ema` for the paired EMA arm. The runner verifies the file digest, loads on the CPU with `weights_only=True`, checks the frozen run identity, step, and tensor digest, and strictly loads the selected state dict before CUDA inference. To load a checkpoint from your own Exp10 run, pass its trusted index digest through `--checkpoint-sha256` and select the corresponding weight arm from its index.

The command writes `vocals.wav`, `instrumental.wav`, and `inference_manifest.json`. It preserves length, stereo channel order, and native gain in IEEE float WAVs. Convert audio to 44.1 kHz stereo before running if needed. Playback software or integer PCM conversion may clip float samples above full scale.

The model takes four-second windows, applies FP32 complex-mask inference, and reconstructs the vocal waveform with inverse STFT. Short inputs are right-padded and trimmed to native length. Long inputs use a two-second inference hop with an extra end-aligned window when needed. Positive Hann overlap-add divides by the actual accumulated weights. Instrumental is reconstructed as mixture minus vocals.

## Prepare data and evaluate

Supply the approved MoisesDB v0.1 ZIP without repacking it. Header order and aligned native samples are part of the executed recipe. The recorded archive size is 88,822,120,667 bytes and its SHA-256 is `4cde33ce416ac7c868cffcb60eb31f5c741ab7ae5601cbb9d99ed498b72c48c1`. Confirm the archive against the recorded digest before using it. [MoisesDB checksum and terms](https://github.com/moises-ai/moises-db#dataset-integrity)

```powershell
$Archive = 'D:/Datasets/moisesdb.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath $Archive
& $Python -B exp10_scratch/reproduce.py prepare --archive $Archive
& $Python -B exp10_scratch/reproduce.py evaluate --checkpoint $Checkpoint --weights both --output exp10-development.json
```

The loader checks role membership, ZIP member sizes and CRCs, native stereo format, and sample alignment. It sums vocal-tagged sources into the vocal reference and the remaining sources into the instrumental reference at native gain. Preparation builds `exp10_scratch/development_cache/` and verifies decoded-reference digests. If an existing cache fails verification, move that cache aside before rebuilding.

The evaluation command writes per-song results and artist macros for Raw and EMA. Compare macros with [the archived endpoint](provenance/development_endpoint.json). Stem SDR uses the median across complete one-second windows with active references at −45 dBFS or above. Stereo Side SDR uses the full-track half-difference of left and right with a −50 dBFS reference threshold. Vocal bleed is the projection of instrumental error onto reference vocals, reported as a coefficient. The macro averages songs within artist, then averages artists equally. The metrics module also implements SI-SDR. Exp10's checkpoint selection used vocal SDR and the guardrails specified in `config.json`.

## Train or resume

Prepare the development references first. Allow roughly 12 GiB of working space in addition to the source archive for staged training shards, development references, and run checkpoints. The original run took about six hours on the recorded GPU.

```powershell
$Run = 'exp10_scratch/runs/exp10_seed909'
& $Python -B -u exp10_scratch/reproduce.py train --archive $Archive --out $Run
```

The fresh run uses a new output directory directly under `exp10_scratch/runs/`. Append `--resume` to the same command to recover that run from its last intact numbered checkpoint. Resume checks source, references, configuration, and runtime provenance. The frozen checkpoint separately supplied for inference is a complete selected-state alias. Training recovery uses the numbered bundles and the run's provenance.

The configuration defines seed 909, four four-second crops per update, deterministic CUDA BF16 with FP32 spectral operations, and 24,000 updates from random initialization. AdamW uses 3e−4 initial learning rate, 0.01 weight decay, L2 gradient clipping at 1.0, and cosine annealing to 3e−5. The five objective weights are 1.0 complex MSE, 0.5 log-magnitude L1, 2.0 waveform L1, 0.25 Side L1, and 0.25 vocal leakage. EMA uses decay 0.999. Each 1,000-update checkpoint is validated in Raw and EMA. Use `checkpoint_index.json` to resolve each winning step and weight arm. Reproducing the original outcome requires the frozen data and the recorded environment. Exact agreement across different builds or GPUs is unverified.

## Tests and source map

```powershell
Push-Location exp10_scratch
& $Python -B -m unittest discover -s tests -v
Pop-Location
& $Python -B exp10_scratch/smoke_inference.py --checkpoint $Checkpoint
```

The unit suite uses synthetic inputs and frozen roster metadata. The CUDA smoke test generates its own audio, invokes the inference command, and verifies the output stems and mixture reconstruction.

| Path | Contents |
| --- | --- |
| `exp10_scratch/exp09_core/scratch_model.py` | 11,662,596-parameter band-split Transformer with 62 frequency bands, six time/band attention pairs, RoPE, and stereo complex-mask reconstruction |
| `exp10_scratch/data.py`, `roster.json`, `exp09_core/loader.py`, `shard_plan.py` | Role-specific archive access, WAV and BWF checks, aligned reference construction, bounded temporary shards, and deterministic crop addressing |
| `exp10_scratch/train_exp10.py`, `config.json`, `exp09_core/scratch_train.py`, `loss_candidates.py` | Original training entry point, exact configuration, optimizer step, and objectives |
| `exp10_scratch/checkpoint.py`, `selection.py` | Atomic checkpoints, EMA, RNG restoration, selection, and guardrails |
| `exp10_scratch/infer.py`, `exp09_core/chunking.py` | File inference, strict checkpoint loader, overlap-add, and WAV export |
| `exp10_scratch/validation.py`, `exp09_core/metrics.py` | Full-song development evaluation and metric calculations |
| `exp10_scratch/reproduce.py` | Portable dataset path and standalone development evaluation commands |
| `history/`, `provenance/` | Experiment differences, selected checkpoint index, archived runtime and source hashes, and endpoint metrics |

Fourteen of the fifteen production files match the recorded Exp10 byte hashes. This publication resolves the default dataset path in `data.py` from the current user's home directory. The original hashes remain in `provenance/frozen_run_excerpt.json`, and a new training run records its own identity for this source revision. Resume each run with the source revision that created it. The added `reproduce.py` command passes the supplied archive path to the original trainer at runtime. The inference CLI is adapted from the frozen demo and imports the production model and overlap-add source directly. The roster test uses the digest-bound runtime roster and synthetic rejected-member cases so it can run independently of the original split records. `SOURCE_SHA256SUMS.txt` covers the files in this publication.

[Experiment history](history/EXPERIMENTS.md)
