# Music Source Separation

This repository contains the baseline and experimental implementations through Exp14. The active reference is **Exp10**, an independently implemented BS-RoFormer-inspired stereo separator trained from random initialization on MoisesDB. The U-Net code is retained for historical experiments and comparisons; it is not the final Exp10 model.

Start with the [two-computer workflow and execution matrix](docs/WORKFLOW.md), the [Exp10 reproduction guide](EXP10_REPRODUCTION.md), or the [experiment history](history/EXPERIMENTS.md). MoisesDB is stored on the training computer. Its absence from a development checkout is expected. Audio, model weights, environments, caches and the external final holdout are managed separately from source code.

## Current reference and interpretation

Exp10 uses 62 frequency bands, alternating time/band attention, RoPE and complex vocal masks. Instrumental is reconstructed as mixture minus predicted vocals. It has 11,662,596 parameters and uses four-second windows with a two-second inference hop.

The selected checkpoint is **step 24,000 Raw**, with **7.0884 dB artist-macro vocal SDR** on 16 development recordings from seven artists. This is `best_overall`; no candidate passed all `best_qualified` guardrails. The external final holdout is separate. Do not compare this number directly with earlier MUSDB18 scores: the dataset and scoring protocol changed. See the [recorded endpoint](provenance/development_endpoint.json).

Exp11 and Exp12 pilots were not adopted, Exp13 did not proceed to training, and Exp14 retained the two-second hop. A larger experiment number therefore does not indicate the selected model.

## Source map

| Project stage | Source and configuration |
| --- | --- |
| Baseline and Exp01–Exp06 | [src/](src/) contains the U-Net variants, preprocessing and data loaders, experiment trainer, objectives, separation and evaluation. [experiment_configs/](experiment_configs/) holds the recorded run configurations. [notebooks/](notebooks/) contains the original analysis cells with outputs cleared. |
| Exp07 | [Phase-aware residual implementation](src/experiments/exp07_phase_aware_residual/), [diagnostic scripts](reports/exp07/), and [pilot settings](experiment_configs/exp07_mse_pilot.json) |
| Exp08 and fixed Exp04/Exp08 blend | [Exp08 model and trainer](src/experiments/exp08_subband/), [launch and feasibility code](reports/exp08_subband/), [run settings](experiment_configs/exp08_subband_seed42.json), and [blend inference](src/experiments/exp04_exp08_hybrid/) |
| Exp09 | [Scratch model, preprocessing, training, and development evaluator](exp09_scratch/). The [separate pretrained adapter](src/experiments/exp09_pretrained/) and its [Windows probe](reports/exp09_zero_shot_preparation/windows_probe.py) document the alternate branch. Its upstream implementation and weights come from the separately obtained release. |
| Exp10 | [Complete final implementation](exp10_scratch/) and [verified reproduction guide](EXP10_REPRODUCTION.md) |
| Exp11–Exp12 | [Cross-song remix pilot](exp11_scratch/) and [stereo loss pilot](exp12_stereo_loss/), each with its own scripts, tests, settings, and frozen train/development roster |
| Exp13 | [Conditional design and decision](exp13_data_expansion/). The audit ended before an implementation or launch configuration was produced. |
| Exp14 | [Inference hop ablation code and protocol](exp14_inference_hop/). The final recommendation retains the Exp10 two-second hop. |
| Demonstration | [Presentation runner source](PRESENTATION_2026-11-23/) and [standalone Exp10 command](exp10_scratch/infer.py) |

## Presenting or running the final model

Follow [EXP10_REPRODUCTION.md](EXP10_REPRODUCTION.md) to set up Python and CUDA and run tests or inference. The package root has separate dependencies for the legacy MUSDB code in `requirements.txt` and final Exp10 in `requirements-exp10.txt`. For an inference demo, supply `best_overall.pt` separately and use a 44.1 kHz stereo WAV or FLAC. Original training and development scoring require authorized access to the matching dataset.

Historical source is available for inspection. Some historical entry points, particularly Exp09/Exp10 roster freezing and the full Exp14 replay, expect the original protected manifests, dataset caches, specific checkpoints, run records, or third-party releases. Where the corresponding files are omitted, the original scripts retain these checks. The portable Exp10 CLI is the checked path for live inference. Host-specific paths in historical source or docs can be updated for the destination machine. The provenance index lists the files whose local profile path was replaced.

[File origins and packaging changes](SOURCE_ORIGINS.json) · [Original handoff checksums](SOURCE_SHA256SUMS.txt)

The provenance and checksum files describe the recorded handoff or frozen run. Subsequent documentation edits are tracked by Git; these original records are not a continuously regenerated checksum of the working tree.
