# Exp13 conditional pilot design

**NOT_READY**. This is a pre-authorization design contingent on an eligible expansion. No Exp13 implementation, launch configuration, checkpoint, training or new test suite was produced.

## Baseline and sole intervention

Freeze a roster of the existing 60 songs plus only cleared additions. For approved unequal-length recordings, use their verified native common span at original gain and sample zero. Existing songs continue through strict equal-length loading. The common-span exception is limited to specifically approved IDs and is required to supply those recordings as training data. All WAV member reads stay inside an exact training allowlist. Reference cache files and the frozen 16-song, seven-artist development panel remain byte-identical to Exp10.

| Component | Locked Exp10 setting |
| --- | --- |
| Model | Scratch separator from seed 909, 11,662,596 parameters, 62 bands, width 256, eight heads, six time/band pairs, 2,048 FFT and hop 441. Use the bound initial model digest. |
| Objective | `normalized_mse_bleed`. Complex MSE 1.0, log-magnitude L1 0.5, waveform L1 2.0, baseline Side L1 0.25 and leakage 0.25. Use Exp10 mixture-RMS normalization in the Side term. |
| Optimizer | AdamW, LR 0.0003, weight decay 0.01, clip gradient L2 norm at 1.0. Deterministic CUDA BF16, FP32 spectral operations, activation checkpointing. |
| Training | Four same-song reference crops of four seconds at 44.1 kHz stereo per optimizer step, gradient accumulation one. Exp11 remixing and Exp12 Side normalization disabled. |
| Schedule and smoothing | Cosine `T_max=24000`, `eta_min=0.00003`, EMA 0.999 following each optimizer step. Stop the pilot only after committing update 6,000 and its validation on the 24,000-step schedule. LR there is about 0.000260459. |
| Validation and inference | Raw and EMA at every 1,000 updates. Same FP32 four-second windows and two-second hop, end-aligned final window and normalized positive Hann, instrumental estimate as mixture minus vocals. |
| Scoring and selection | Artist macro of per-song metrics. Median complete-second SDR at −45 dBFS target RMS, full-song Side SDR at −50 dBFS reference threshold, absolute vocal bleed coefficient and mixture residual. Keep the Exp10 checkpoint qualification floors and selection logic. |

[Exp10 run configuration](../exp10_scratch/config.json) · [Architecture](../exp10_scratch/preserved_exp09/ARCHITECTURE.md) · [Validation implementation](../exp10_scratch/validation.py)

## Exposure at fixed compute

Exp10 partitions by duration descending with UUID tie-break and first-fit under the 4 GiB staging budget. Every 500 updates it rotates shards. A seed-keyed permutation visits each shard once per round. A sample-index seeded RNG chooses uniformly among viable songs within the current shard and then uniformly among valid start frames for that song. A shard with fewer songs gives each song a larger expected fraction of crops. Repacking changes probabilities for incumbent songs and sample addresses even when the algorithm and seed are identical. At fixed steps the treatment trades repeated exposure to the existing roster for exposure to added songs.

Replay of the frozen baseline reproduced every recorded shard and all 96,000 addresses. An all-seven counterfactual used the baseline partitioner and exact sampler without model execution. The table shows the exposure shift from naive packing. It is an audit calculation, with the hypothetical grouping excluded from launch use.

| Sampler quantity | Exp10, 6k | All seven, 6k | Exp10, 24k | All seven, 24k |
| --- | ---: | ---: | ---: | ---: |
| Shard sizes | 19 / 25 / 16 | 18 / 23 / 26 | 19 / 25 / 16 | 18 / 23 / 26 |
| Visits per shard | 4 | 4 | 16 | 16 |
| Total crops | 24,000 | 24,000 | 96,000 | 96,000 |
| Distinct addresses | 23,999 | 23,999 | 95,987 | 95,989 |
| Candidate crops | 0 | 2,632 (10.97%) | 0 | 10,639 (11.08%) |
| Mean crops per original song | 400.0 | 356.1 | 1,600.0 | 1,422.7 |
| Per-song crop count min / median / max | 299 / 408 / 540 | 276 / 342 / 492 | 1,220 / 1,649.5 / 2,103 | 1,166 / 1,364 / 1,871 |
| Rock share of crops | 48.23% | 51.13% | 48.05% | 51.58% |
| Firefly share of crops | 15.40% | 13.65% | 15.37% | 13.45% |

The naive all-seven grouping's largest conservative staging bound is 4,296,992,680 bytes against 4,294,967,296. The baseline packer budgets using common reference frames, while the loader can allocate using the longer first WAV. For the cleared roster, preserve first-fit order and the 4 GiB cap with a bound based on native staging lengths, reserved bytes for the largest raw WAV, and all stored targets. Keep shard rotation and RNG laws fixed and rerun full address/exposure checks on the actual packing before authorizing. Record each UUID and artist's shares, genre mix, crop-level source and joint activity, Side-to-Mid energy, overlap in crop addresses and exposure of original songs at both endpoints. A change to shard visits, duration weighting or activity weighting would confound this roster comparison. [Sampler](../exp10_scratch/exp09_core/shard_plan.py) · [Loader](../exp10_scratch/exp09_core/loader.py) · [Archived baseline addresses](../exp10_scratch/runs/exp10_seed909/steps.jsonl)

## Matched pilot endpoint and screening

Use update 6,000 for both Raw and EMA, arm for arm. The archived baseline validation history SHA-256 is `cc997976ed2bf37a847656843d8c504292b4f3a6486067062c19add4c0ffc86d`. The recorded panel maximum mixture residual is 2.98e−8 in both arms. Use the artist and song rows to compute paired values. [Archived Exp10 development history](../exp10_scratch/runs/exp10_seed909/validation_history.json)

| Artist macro at 6,000 | Exp10 Raw | Exp10 EMA |
| --- | ---: | ---: |
| Vocal SDR, dB | 6.208139 | 6.237916 |
| Instrumental SDR, dB | 12.144889 | 12.687914 |
| Vocal Side SDR, dB | 1.016294 | 1.247420 |
| Instrumental Side SDR, dB | 8.767087 | 8.998213 |
| Absolute vocal bleed | 19.30596% | 20.79343% |

Proposed predeclared pilot thresholds use the prior matched-pilot non-regression margins. Require each arm to have vocal and instrumental SDR deltas at least −0.10 dB, both Side SDR deltas at least −0.15 dB, and absolute bleed worsening at most 1.00 percentage point. Require EMA artist-macro vocal SDR gain of at least +0.15 dB and positive EMA vocal SDR deltas in at least four of seven groups. The positive-signal rule is a design choice for data expansion, using the Exp10 0.15 dB meaningful-gain convention. Require both arms to have finite histories and valid, complete validation, satisfy the original inference consistency limit, and have each song's and the panel's peak mixture residual within one FP32 epsilon (1.1921e−7) above the matching archived value. Publish per-song and per-artist deltas and an artist-block bootstrap interval for context. Lock the thresholds and reference hashes before any run and apply them unchanged. A pass supports review for a longer run. The data size, one seed and seven development groups limit conclusions.

The existing Exp10 checkpoint qualification rules remain the unrounded values in its configuration, approximately 13.7312 dB instrumental SDR, 2.3929 dB vocal Side SDR, 10.1437 dB instrumental Side SDR, and at most 18.8318% bleed. The pilot's relative screening does not replace those rules. [Exp10 guardrails](../exp10_scratch/config.json) · [Prior matched-pilot margins](../exp11_scratch/REPORT.md)

## Verification

| Check | Current evidence | Needed on an admitted roster |
| --- | --- | --- |
| Artist split and duplicates | UUID, normalized artist, available exact PCM, size/CRC and title screening complete on the documented scopes. External independence and some transformed-duplicate checks are open. | Freeze identities and enforce all train/development, reserve and external clearance before launch. |
| Alignment, reconstruction, complete stems and loading | Reconciled archived PCM and reconstruction for seven, live checks on Firefly with repeated in-bounds crops, exact additivity and rejected invalid crops. All 2,585 declared ZIP entries present. | Check all cleared sources and receipts, use strict fallback for existing recordings, probe clamp boundaries and native stage bounds for each addition. |
| Determinism | Replayed the 24,000-step baseline crop log and evaluated counterfactual addresses. | Assert stable final packing and addresses across restart and shard boundaries, and document measured crop exposure. |
| Checkpoint and resume | Existing Exp10 CPU suite passed all 12 tests including atomic publish, interrupted-transaction recovery, RNG restore, aliases and the archive role seal. Existing archived GPU preflight records exact Exp10 resume. | Check the new arm's identity-bound optimizer, scheduler, EMA, RNG, addresses and histories through interruption and recovery before any approved launch. |
| Model and loss parity | All 15 archived Exp10 run-file hashes and the frozen baseline data bindings matched. | Confirm Exp10 model/loss hashes, initial weights and forward/loss on identical deterministic reference batches before enabling the changed roster. |
| No final-holdout access | ZIP payloads restricted to 12 Firefly WAVs. The 16 development cache hashes were verified. Retired protected WAVs and all active final audio stayed closed. | Use exact role allowlists, test denial of non-roster members, and preserve the seal in launch and resume. |

[Existing CPU tests](../exp10_scratch/tests) · [Archived Exp10 GPU replay](../exp10_scratch/preflight/c/result.json)

## Cost

Exp10 recorded 56.85 minutes of update compute and 24.17 minutes of paired validation to step 6,000. Budget 1.4–1.7 hours on the RTX 3080 Ti after clearance and bounded staging, including checkpoint publication and ZIP I/O. Budget 3–3.5 GiB PyTorch GPU reserved based on Exp10. A 24,000-step run would take roughly 6–7 hours if later approved.

Allow 7 GiB free for a pilot run with the existing development cache available read-only, or 9 GiB if staging its own verified cache. This includes up to 4 GiB for one training shard, seven numbered checkpoints including step zero, up to three aliases and temporary publication files. Exp10 states are approximately 179 MiB each and the dev cache is 1.93 GiB. The seven common-span stereo target pairs alone would represent 1.17 GiB if all staged simultaneously. Budget 12 GiB for a longer run including cache per Exp10. The existing 82.72 GiB archive is used in place. The C drive had 131.15 GiB available at audit. Time and peak resources are baseline-derived estimates, as no Exp13 pilot ran. [Exp10 resource record](../exp10_scratch/README.md) · [Archived preflight](../exp10_scratch/preflight/c/result.json)

