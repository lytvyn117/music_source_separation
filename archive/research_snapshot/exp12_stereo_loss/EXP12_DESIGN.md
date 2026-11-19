# Exp12 design

## Hypothesis

Weighting the existing Side reconstruction error by a bounded mixture-Side scale improves matched Side SDR without materially worsening vocal SDR, instrumental SDR, or absolute vocal bleed. Exp12 tests one run with seed 909 and one fixed denominator. The run is a development pilot.

## Controlled setup

Use the byte-identical Exp10 roster, model, seed and starting-weight digest, same-song sampler, strict WAV loading and original gain, four-second stereo crops at 44.1 kHz, batch four and accumulation one. Keep 11,662,596 parameters, deterministic CUDA BF16 with FP32 spectral/loss operations, AdamW at 0.0003 and weight decay 0.01, gradient clipping at 1.0, cosine `T_max=24000` and `eta_min=0.00003`, and EMA decay 0.999. Keep the four other loss terms, all five coefficients, provenance-bound checkpoints, and the Exp10 checkpoint-selection policy. No Exp11 remix, other augmentation, warm start, new audio, or final-holdout access.

The trainer stops at update 6,000 after publishing the step-6,000 checkpoint, with the full 24,000-update cosine horizon (LR about 0.00026046 at update 6,000). All six 1,000-update boundaries retain committed Raw and EMA metrics on the 16-song, seven-artist panel with FP32 inference, Exp10 windowing and overlap-add, and identical activity thresholds. Resume uses the last numbered checkpoint and restores model, optimizer, scheduler, EMA, RNG, and history before replay. Compare step 6,000 in both arms regardless of which checkpoint ranks best.

## Primary decision

| Exp10 artist macro at update 6,000 | Raw | EMA |
| --- | ---: | ---: |
| Vocal SDR (dB) | 6.2081 | 6.2379 |
| Instrumental SDR (dB) | 12.1449 | 12.6879 |
| Vocal Side SDR (dB) | 1.0163 | 1.2474 |
| Instrumental Side SDR (dB) | 8.7671 | 8.9982 |
| Absolute vocal bleed (%) | 19.3060 | 20.7934 |
| Vocal SI-SDR | Unavailable | Unavailable |

For **each** arm, require vocal and instrumental SDR deltas ≥ −0.10 dB and absolute vocal bleed delta ≤ +1.00 percentage point. Predeclare meaningful Side improvement as **≥ +0.20 dB on each Side SDR**, with both Side deltas strictly positive in at least **four of seven** artist groups. The +0.20 dB threshold corresponds to about 4.5% less Side error power for a fixed reference. With complementary outputs, both Side SDR deltas measure nearly the same error change. Artist spread checks whether that gain extends across the panel. The ± tolerances apply to artist macros. Report all paired song and artist values/deltas and review the distribution and trade-off before declaring GO.

Verify all 6,000 steps are present, finite and have the exact Exp10 crop addresses and shard order. Any unstable step, failed validation, or mixture-consistency regression blocks GO. The first training-computation or validation failure is recorded outside the replayed update log so a later resume cannot hide it. Apply the existing inference residual check per song. Compare every song and the panel maximum against archived residuals with one FP32 machine epsilon at full scale (1.1921e−7) tolerance for addition/subtraction rounding. Baseline metrics and residuals come from the hash-bound archived step-6,000 history. A paired SI-SDR delta remains unavailable. Report Raw and EMA separately. Require both arms to clear the screening thresholds, then review the overall trade-off. Avoid retuning against the same development panel.

## Limits and expected cost

Final-weight local gradients do not represent the early trajectory. A single seed and seven groups provide limited evidence. Capped rescaling cannot resolve bounded-mask or phase-cancellation limits. Quiet Side targets may still create unwanted pressure, and the observed parameter-gradient conflict makes bleed regression plausible. Pilot success would justify review of this formulation. It would leave final generalization and model promotion open.

Exp10 logged 56.85 minutes in update computations and 24.17 minutes in six paired validations through step 6,000. Allow about 1.4–1.7 hours on the RTX 3080 Ti including ZIP staging and checkpoint I/O. This is a baseline-derived estimate. Exp10 measured 2.13 GiB allocated and 3.00 GiB reserved in its update logs. Expect roughly 3–3.5 GiB PyTorch reserved for Exp12. Additional device and display usage is separate. A short, full-size synthetic CUDA BF16 forward/backward check measured about 0.46 seconds and 2.82 GiB reserved for Exp12, with no optimizer step or validation. Timing at this sample size cannot establish a runtime difference.

## Preparation checks

1. Capture the diagnostics and compare all three formulations in the loss decision.
2. Copy the frozen model, sampler, optimizer, evaluator and checkpoint machinery. Apply the Side denominator to the local loss implementation and cap only the pilot loop at 6,000.
3. Run synthetic loss, plumbing, and resume tests under the project Python. Use CUDA BF16 for a forward/backward comparison without updates. Verify source hashes and the evaluation path with the archived records.
4. Deliver the future pilot commands for user authorization.

