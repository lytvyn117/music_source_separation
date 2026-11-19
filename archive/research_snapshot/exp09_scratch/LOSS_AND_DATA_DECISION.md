# Exp09 loss decision and training eligibility

14 November 2026

## Decision

Recommend the normalized Complex MSE objective with explicit vocal leakage, using the 60 strict MoisesDB recordings. Keep the 85 historical MUSDB18 recordings outside this launch. The proposed 24,000-update run awaits authorization. The model architecture was unchanged.

The available evidence supports a MoisesDB-only research training run. Independent benchmark clearance still requires an authorized recording-level comparison with the sealed development panel and the eventual final roster. No assumptions about an external panel were used to admit songs.

## Loss reconciliation

The previous Exp09 scratch preflight implemented vocal waveform L1 plus multi-resolution mean complex residual magnitude at weights 1.0 and 1.8375527837636136. The unitary Fourier normalization in that term controls FFT scale. It uses neither per-mixture RMS normalization nor squared complex error, and it has no leakage term.

The earlier Exp08 source-separation spec froze a normalized Complex MSE-centered objective with a vocal-specific projection penalty. The Exp07 variant also used normalized Complex MSE with a different leakage definition and component calibration. The archived pretrained Exp09 adaptation plan specified complex L1. The scratch preflight followed an L1 reconstruction design and did not carry forward the earlier MSE/leakage objective.

The selected arm uses the Exp08 weights and loss definitions on the approved Exp09 FFT 2,048 / hop 441. The MoisesDB mixture is already the sum of its targets, so the equal-residual target projection has zero correction apart from float arithmetic. This implementation computes loss on the synthesized waveform and its STFT. Spectral loss uses constant boundary padding to retain strict deterministic CUDA training. The separator uses its original analysis and synthesis.

| Component | Weight | Definition |
|---|---:|---|
| Normalized Complex MSE | 1.0 | Squared complex magnitude of vocal residual divided by per-example mixture spectral RMS squared, with RMS floor 1e−4 |
| Log-magnitude L1 | 0.5 | Average of both targets after applying log1p to the mixture-RMS-scaled spectral magnitudes |
| Waveform L1 | 2.0 | Vocal waveform residual over mixture waveform RMS, floor 1e−4 |
| Stereo side L1 | 0.25 | L1 on (left residual − right residual)/2 with the same waveform RMS scale |
| Vocal-specific leakage | 0.25 | On one-second frames with vocal RMS above 0.005, square the positive excess of the absolute instrumental-error projection coefficient onto the vocal reference above 0.05 |

The objective is explicitly identified and weight-bound in the new launch config and checkpoint. The old preflight configuration remains historical. [Loss code](loss_candidates.py) · [Launch decision](launch_decision.json)

## Matched train-only comparison

Both arms used the same random initial tensor digest, 12 optimizer-update songs from 12 artists, the same indexed four-second crop schedule, batch four, deterministic BF16, activation checkpointing, and the 24,000-step LR schedule truncated at 160 updates. The six probe-only songs came from six other artists within the strict MoisesDB training pool, with four fixed active vocal/backing crops per song. Their UUIDs never entered either optimizer schedule. Means are macro averages across the six songs. SDR here is signal-to-residual energy on the fixed crops, with no alignment or gain fitted.

| Probe-only outcome | Initialization | L1 at 160 | MSE-centered at 160 |
|---|---:|---:|---:|
| Vocal SDR | 0.006 dB | 4.646 dB | **5.079 dB** |
| Instrumental SDR | 1.496 dB | 6.137 dB | **6.570 dB** |
| Common normalized waveform MSE | 0.41785 | 0.14851 | **0.13301** |
| Absolute vocal projection in instrumental error | 0.99938 | 0.47336 | **0.41341** |
| L1 objective | 0.03778 | 0.02495 | **0.02436** |
| MSE-centered objective, selected arm | 1.56942 | — | **0.73943** |

The selected arm decreased its own proxy validation objective **52.9%**, improved proxy vocal SDR **5.07 dB** from initialization, and beat L1 by **0.43 dB mean** across all six songs. The projection metric dropped by **6.0 percentage points** versus L1. The first MSE attempt encountered nondeterministic reflection-padding backward. After the constant-padding correction it completed 160 updates and exact replay. The L1 arm's training objective and common scoring metrics were already valid. Its auxiliary MSE-objective trace was generated before the boundary change, so the table omits cross-arm comparison of that auxiliary.

At the measured checkpoint boundary, both arms replayed the next crop schedule, CUDA RNG, scalar metrics, model tensors, AdamW moments and scheduler exactly. The selected arm had eight gradient clips at the configured norm of one over 160 updates. Probe results are short-run evidence from one seed and six artists. The six songs may rejoin optimizer use in full training and then cease to serve as independent validation. [Per-song results and timing trace](multisong_probe.json)

## Training roster and MUSDB18

The launch allowlist is **60 MoisesDB songs**, **3.62 h**, **26 artist groups**, and **10 genres**. It retains native source amplitudes with additive stereo mixture construction and three hash-bound shards, one staged at a time with a 4 GiB payload budget. All 20 MoisesDB development songs and the 16 other songs from their artists were excluded from the audit's decode, optimizer, and scoring paths. The external final set was outside this work.

The seed-42 historical MUSDB18 split reproduced 85 fit songs from the official 100-song training roster. Read-only decoding and acoustic candidate screening compared the 85 candidates with each other and the 60 strict MoisesDB songs. No strong acoustic or duplicate-target hash match and no normalized artist match with MoisesDB were found. The only cross-dataset title collision involved an already-excluded MoisesDB track. This screening can miss relabeled or heavily modified copies, and sealed audio was not decoded for cross-split proof.

Of the 85 MUSDB18 songs, 17 share a normalized artist with the legacy 15-song validation list, six share a normalized artist with the official 50-song test list, and two have documented vocal target errors. The six official-test collisions include two Triviul tracks after punctuation-aware normalization. These 25 flags apply to distinct songs. The remaining 60 provisionally unflagged recordings would add about **4.19 h** and 49 artist groups. They are outside the selected run pending source terms and recording clearance. The MUSDB official publication describes academic or educational use and per-track source terms. [MUSDB audit](musdb_eligibility.json) · [SigSep dataset and errata](https://sigsep.github.io/datasets/musdb.html) · [Official source roster](https://github.com/sigsep/website/blob/master/content/datasets/assets/tracklist.csv)

## Runtime and launch

The selected arm used **2.09 GiB allocated / 2.96 GiB reserved** peak on the RTX 3080 Ti. Median end-to-end timed update was **0.637 s**, mean **0.639 s**, p90 **0.676 s**, including cached crops, host transfer, forward/backward, loss and optimizer. At the measured mean, 24,000 updates project to **4.26 h of optimizer work**. The three production shards staged in 201.7 seconds total and peaked at 3.83 GiB of reserved cache payload. With 500 updates per visit, 24,000 updates make 16 visits to each shard, adding about 0.90 h at the measured staging rate. Nominal combined time is **5.16 h before checkpoint overhead**. Budget about **5.5–7 h** for ordinary machine-load variation. The sealed-panel scoring time is outside this fixed-step launch. [Shard staging timings](shard_staging.json)

The launcher uses deterministic batch four, activation checkpointing, AdamW 3e−4 with weight decay 0.01, norm clip 1.0, cosine decay to 3e−5, seed 909, and atomic saves every 1,000 updates. It reads only strict training data. The existing objective selection experiment used the proxy-only recordings and this launch can run for the fixed update budget without opening the development panel. [Launcher](train_exp09.py) · [Two-update launcher and resume smoke](launcher_smoke.json)

After authorization, use PowerShell with a fresh output directory name.

```powershell
Set-Location -LiteralPath 'C:\Users\YOUR_USER\music_source_separation_exp09_scratch'
& 'C:\Users\YOUR_USER\music_source_separation\.venv\Scripts\python.exe' -B -u .\train_exp09.py --archive 'C:\Users\YOUR_USER\Downloads\moisesdb.zip' --decision .\launch_decision.json --out .\runs\exp09_mse_seed909 --authorize-full-training
```

To resume, add `--resume` to the same command and preserve the output folder and bound inputs.
