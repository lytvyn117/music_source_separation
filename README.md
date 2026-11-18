# Music Source Separation — whole-project source handoff

This source archive covers the baseline and all available experimental implementations through Exp14, including the source that lived outside the original Git worktree. Start with [the experiment history](history/EXPERIMENTS.md), then follow the source map below. Configuration files record the settings used in each experiment. Actual audio, datasets, model weights, virtual environments, results caches, and sealed evaluation records are external to this handoff.

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
| Demonstration | [Presentation runner source](FINAL_PRESENTATION_2026-11-23/) and [standalone Exp10 command](exp10_scratch/infer.py) |

## Presenting or running the final model

Follow [EXP10_REPRODUCTION.md](EXP10_REPRODUCTION.md) to set up Python and CUDA and run tests or inference. The package root has separate dependencies for the legacy MUSDB code in `requirements.txt` and final Exp10 in `requirements-exp10.txt`. For an inference demo, supply `best_overall.pt` separately and use a 44.1 kHz stereo WAV or FLAC. Original training and development scoring require authorized access to the matching dataset.

Historical source is available for inspection. Some historical entry points, particularly Exp09/Exp10 roster freezing and the full Exp14 replay, expect the original protected manifests, dataset caches, specific checkpoints, run records, or third-party releases. Where the corresponding files are omitted, the original scripts retain these checks. The portable Exp10 CLI is the checked path for live inference. Host-specific paths in historical source or docs can be updated for the destination machine. The provenance index lists the files whose local profile path was replaced.

[File origins and packaging changes](SOURCE_ORIGINS.json) · [Source checksums](SOURCE_SHA256SUMS.txt)
