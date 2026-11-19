# Exp10 scratch separation

This experiment prepares a fresh run of the Exp09 architecture and optimization schedule with EMA, deterministic development selection, and resumable training. The project, cache, and new run outputs live under this directory. `preserved_exp09` contains the 156-file byte-for-byte backup, including both copies of the final checkpoint, the original run log, and all evaluation results and listening examples. `exp09_preservation.json` lists the SHA-256 of every copied file.

## Training configuration

- Seed 909. The 11,662,596-parameter separator starts from random initialization, with the same initial tensor digest as Exp09. There is no checkpoint load on a new run. Resume accepts only a numbered checkpoint from the same Exp10 run.
- The same 60 MoisesDB training recordings, shard order and window selection, native 44.1 kHz stereo, four-second crops and batch size four. Activation checkpointing, deterministic BF16 autocast and FP32 spectral operations use the Exp09 code.
- Normalized complex MSE with the same five objective terms and weights. AdamW uses learning rate 0.0003, weight decay 0.01 and gradient L2 clipping at 1. Cosine annealing reaches 0.00003 at 24,000 updates.
- After each raw optimizer step, EMA moves the previous shadow parameters 0.001 of the way to the updated raw weights. EMA starts at the seed-909 initialization and accumulates no gradients. Validation uses a separate model and restores the training RNG state around both evaluations.

## Frozen development protocol

Every 1,000 updates, score raw and EMA weights on the same 16 admitted Exp09 songs across seven artists, in the same song order. Each song uses the archived BWF alignment, native sample zero, and matching SHA-256 hashes for the reconstructed mixture and both references. Both references keep their native gains throughout. FP32 inference covers the full song with fixed four-second windows starting every two seconds and one end-aligned window when needed. Combine the windows with normalized positive Hann weights. The instrumental estimate is mixture minus estimated vocals.

For vocal and instrumental SDR, take the median across complete one-second frames with reference RMS at least −45 dBFS. Side SDR uses the full-song (L−R)/2 signal with a −50 dBFS reference threshold. Bleed is the absolute projection of instrumental error onto reference vocals, divided by vocal energy. Each metric is averaged over songs within an artist and then equally across artists. The recorded validation row includes song-level scores and the recipe hashes travel with the run. A missing metric or a failed mixture consistency check stops the run at the last committed checkpoint.

`best_overall.pt` retains the highest artist-macro vocal SDR across all raw and EMA candidates. `best_qualified.pt` retains the highest score among candidates that meet every guardrail. Exact ties favor the earlier update, then EMA within the same update. The following guardrails were fixed before preflight.

| Artist-macro metric | Required |
| --- | ---: |
| Instrumental SDR | ≥ 13.7312 dB |
| Vocal side SDR | ≥ 2.3929 dB |
| Instrumental side SDR | ≥ 10.1437 dB |
| Absolute vocal bleed coefficient | ≤ 18.8318% |

The SDR floors allow a 0.25 dB margin below archived Hybrid performance. The bleed ceiling permits one percentage point above archived Exp09. The configuration holds the unrounded thresholds. Early stopping still monitors qualified EMA scores and requires at least 16,000 updates and eight consecutive validations without an eligible gain of at least 0.15 dB over the last such improvement. Any eligible gain can update `best_qualified.pt`, including those smaller than 0.15 dB. If no raw or EMA candidate meets the guardrails, the launcher reports `no_candidate_qualified` in `checkpoint_index.json` and `completion.json`, prints that outcome at completion, and retains `best_overall.pt`.

The production launcher only sees the 60 training and 16 development recordings from the active external-final design. It has no external holdout path or loader, and archive access rejects all WAVs outside the role-specific allowlist. The 16 MoisesDB artist reserves and four development exclusions are also outside the runtime roster. Exp09 already exposed recordings from the retired internal split proposal. This access seal protects the training workflow. Recording-level independence of the external final panel still requires an authorized identity audit.

## Checkpoints and recovery

Each numbered checkpoint directory contains the state and its SHA-256 provenance. Step 0 and every 1,000 updates are published via a same-volume directory rename after both files have been written and flushed. Files are read-only and the launcher never overwrites a numbered checkpoint. One writer holds an OS file lock. The `latest.pt`, `best_overall.pt` and `best_qualified.pt` aliases are published by atomic file replacement. Each alias is a copy of a numbered checkpoint and contains both weight sets. For either best alias, read its `weights` field in `checkpoint_index.json` to choose `raw_model` or `ema.weights` from the checkpoint. After an interrupted publication, resume rebuilds the aliases, index and `validation_history.json` from the last numbered transaction. A checkpoint also contains AdamW buffers, scheduler state, update count and Python/NumPy/PyTorch CPU and CUDA RNG. Use the same code, references, Torch/CUDA build, GPU and driver for exact resume. Interrupted work since the last numbered save is replayed. A hard process exit may leave temporary shard or checkpoint staging files in that run directory. Recovery after storage corruption or power loss was outside preflight.

## Launch after explicit authorization

The command uses the existing local PyTorch 2.7.1+cu128 environment and the verified MoisesDB ZIP at `C:\Users\YOUR_USER\Downloads\moisesdb.zip`.

```powershell
Set-Location 'C:\Users\YOUR_USER\music_source_separation\exp10_scratch'
& '..\.venv\Scripts\python.exe' -B -u .\train_exp10.py --authorize-full-training --out .\runs\exp10_seed909
```

To resume, append `--resume` and use the same output directory. If the cache needs to be rebuilt from the audited authorized dataset, run `& '..\.venv\Scripts\python.exe' -B .\train_exp10.py --prepare-development-cache`. A synthetic seal/recovery suite is available with `& '..\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v`.

[Exp11 candidate audit](exp11_audit/REPORT.md) · [MoisesDB license](https://github.com/moises-ai/moises-db#licensing)

## Verification and resource estimate

The paired RTX 3080 Ti preflight ran 12 optimizer updates and replayed six after removing `latest.pt` and `best_overall.pt` and damaging the derived validation history. It restored the training state exactly, recreated the aliases, and reproduced both weight sets' scores on the fixed repeat song. It matched Exp09's first 12 loss terms and gradient norms, and the prior EMA-only preflight's training state and EMA scores bit-for-bit. CPU checks regenerated all 24,000 recorded step addresses and 96,000 crops. Twelve Exp10 tests and 15 archived Exp09 tests passed, including a qualifying winner, recovery of both best aliases and the no-qualifier case. No full training ran. [Paired preflight](preflight/c/result.json) · [Run provenance](preflight/c/run.json)

- Median training update **0.537 s**. Each validation scores 16 songs and 1,459 windows per weight set. Raw took **119.1 s** and EMA took **118.3 s**.
- GPU peak **2.22 GiB allocated and 3.06 GiB reserved** in PyTorch across the paired preflight.
- At 24,000 updates, budget **6–7 hours elapsed**, including shard rotation and checkpoints. Paired validation adds about **95 minutes** over a run with no validation, or **48 minutes** over EMA-only scoring. The measured cost fits full-panel scoring at every checkpoint.
- Each mature state takes about **179 MiB**. The 25 numbered states, three aliases and a temporary save/copy can use **5.1 GiB**. Allow another **4 GiB** for the active training shard and **1.93 GiB** for the development cache. Plan for **12 GiB** of workspace space for a maximum-length run, including the cache. The existing ZIP is **82.72 GiB** and is read in place. Backup and preflight outputs are already stored and are outside the new-run estimate.

On the same stack and inputs, Exp10's raw weights are expected to follow Exp09's trajectory up to the stopping update. The short replay cannot establish equality at update 24,000, which Exp10 might not reach. The sources of a potential development improvement are EMA smoothing and selection of an earlier checkpoint; the paired scores show how much each contributes.
