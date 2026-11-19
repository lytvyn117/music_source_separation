# Experiment 07 Phase-Aware Complex Residual Design

## Objective

Experiment 07 tests whether a small complex-spectrogram residual module can improve the existing Exp04 separator by correcting magnitude and phase errors without changing the trained Exp04 backbone. The experiment must preserve stereo audio at 44.1 kHz, maintain source additivity, remain practical on an RTX 3080 Ti with 12 GB VRAM, and improve measured and audible separation quality rather than validation loss alone.

The Exp04 source checkpoint is `models/experiments/exp04_asymmetric_leakage/best.pt`. Exp01 through Exp06 checkpoints, logs, reports, and audio exports are read-only inputs to this experiment.

## Isolation and Outputs

All new implementation code lives below `src/experiments/exp07_phase_aware_residual/`, with tests in a dedicated Exp07 test module or directory. Existing general training and inference scripts must not be modified unless an isolated adapter cannot provide a required capability.

The controlled experiment may write only to Exp07-specific locations:

- `models/experiments/exp07_phase_aware_residual/`
- `outputs/logs/exp07_phase_aware_residual/`
- `reports/exp07/`
- Exp07-named files below `reports/tables/`
- `reports/audio/exp07_vs_exp04_20s/`

Non-resume launches must refuse non-empty output directories. The implementation records SHA-256 hashes of the Exp01 through Exp06 checkpoints before and after the pilot and fails if any protected checkpoint changes.

## Phase 1: Oracle Headroom Analysis

The oracle analysis uses exactly the existing 15-track validation split and the same 6-second chunk length, 50% overlap, overlap-add reconstruction, sample rate, channel layout, and metric definitions used for Exp04 evaluation.

Four estimates are compared:

1. **Exp04 baseline.** Load the original Exp04 checkpoint strictly and run its current softmax-mask inference path.
2. **Mixture-phase magnitude oracle.** For each source, form a nonnegative ideal ratio mask from the original reference magnitudes and apply it to the recorded mixture STFT. This oracle retains mixture phase and constrains the source magnitudes to nonnegative masks.
3. **Unconstrained ground-truth complex oracle.** Independently reconstruct each original reference stem from its own complex STFT. This measures STFT/iSTFT and overlap-add fidelity. It is not mixture-consistent because the recorded mixture is not exactly the sum of the delivered stems.
4. **Constrained mixture-consistent complex oracle.** Let `X`, `V`, and `I` be the mixture, vocal, and instrumental complex STFTs and let `R = X - V - I`. Use `V_mc = V + R / 2` and `I_mc = I + R / 2`. This is the equal-weight least-squares projection that minimizes the summed squared complex reconstruction error subject to `V_mc + I_mc = X`.

The unconstrained and constrained complex oracles must be labeled separately in every table and report. All oracle audio is evaluated against the original recorded reference stems, never against the projected targets.

The existing validation data has a median mixture-versus-stem-sum NRMSE of 2.64% and a median stem-sum reconstruction SDR of 31.57 dB. The analysis must report this mismatch per track and in aggregate because it limits the score of any exactly additive separator and explains the difference between the two complex oracles.

For every method and track, report vocal and instrumental SDR and SI-SDR, waveform MAE/RMSE, vocal leakage coefficient and energy, source reconstruction error, output gain, and mixture-reconstruction SDR/NRMSE. Report both aggregate metrics and paired per-track differences.

### Oracle Go/No-Go Rule

Training is allowed only if all of the following hold:

- The constrained complex oracle improves vocal median SDR over the mixture-phase magnitude oracle by at least 1.0 dB in the median of paired per-track differences.
- At least 10 of the 15 tracks have a positive paired vocal SDR difference for the constrained complex oracle versus the mixture-phase magnitude oracle.
- The constrained complex oracle's aggregate instrumental median SDR is no more than 0.5 dB below the mixture-phase magnitude oracle.
- Unconstrained complex-oracle reconstruction and the constrained oracle's waveform mixture consistency must have NRMSE no greater than 0.001% and peak absolute error no greater than `1e-5`. The complex mixture check must pass in every bin using FP32 arithmetic and `abs(X - (V + I)) <= 1e-5 + 4 * (2**-24) * (abs(X) + abs(V) + abs(I))`. Here `V` and `I` are the actual estimated sources being checked. No original reference coefficients enter the scale term. Non-finite values fail the gate. Report absolute error, error-to-bound ratio, active-bin p99 relative error, and violations. The bound applies equally to quiet and high-energy bins.

If any condition fails, Exp07 stops after producing the oracle report. No training is launched.

The scale-aware CUDA rerun checked 1,253,802,550 bins with zero violations. Maximum absolute error was `2.158e-5` and the maximum error-to-bound ratio was `0.2231`. Quiet bins had a maximum ratio of `0.1076` and high-energy bins reached `0.2231`. The highest per-track sampled active-bin p99 relative error was `1.1774e-7`. After positive-weight normalization, the worst waveform peak and seam error was `5.364e-7` and the worst constrained waveform NRMSE was `2.466e-5%`. Full per-track measurements are in `reports/exp07/oracle_scale_aware/`. Earlier oracle runs remain available in their original directories.

## PhaseAwareResidualSeparator

`PhaseAwareResidualSeparator` contains the strictly loaded Exp04 `UNetV2` and a small residual CNN. The wrapper accepts stereo mixture waveforms or complex mixture STFTs and exposes structured outputs needed by training, inference, diagnostics, and ablation.

The backbone configuration is read from the Exp04 checkpoint, its weights are loaded strictly, every backbone parameter has `requires_grad=False`, and the backbone remains in evaluation mode even when the wrapper is placed in training mode. This is required because Exp04 contains BatchNorm layers whose running state must not change. Backbone inference is performed without gradient recording.

For a complex mixture STFT `X`, the backbone receives `abs(X)` and produces the original Exp04 masks. The initial vocal estimate is `V0 = M_v * X`. The residual head receives normalized real and imaginary components of `X` and `V0` plus both stereo source masks. Complex features are normalized by a per-example, per-channel RMS scale clamped to a fixed epsilon. No feature divides by an individual time-frequency bin, so silent bins remain finite.

The refinement head uses a 32-channel convolutional stem followed by a small sequence of depthwise-separable residual blocks with dilated 3x3 time-frequency convolutions, GroupNorm, and SiLU activations. A final 1x1 projection emits four channels representing real and imaginary corrections for left and right vocals. The final projection's weights and bias are initialized to exactly zero. The normalized correction is rescaled to the mixture's complex RMS scale and added to `V0`:

`V_refined = V0 + delta_V`

`I_refined = X - V_refined`

The instrumental subtraction is the only post-refinement source construction, so the complex outputs are additive by construction. Complex STFT operations, subtraction, target projection, and iSTFT are performed in FP32. Mixed precision may be used only around numerically safe real-valued CNN operations after the CUDA benchmark confirms finite behavior.

## Initialization and Compatibility Contract

At zero residual initialization, the Exp04 masks and initial vocal complex estimate must match direct Exp04 inference within numerical tolerance. The reconstructed vocal waveform must match the Exp04 vocal waveform with NRMSE no greater than 0.001% and maximum absolute error no greater than `1e-5`; the report records the measured values rather than only a pass/fail result.

The instrumental estimate is verified separately. `X - V0` is not required to be bitwise identical to `M_i * X`, because subtraction and multiplication use different floating-point operation order. It must match the Exp04 instrumental estimate with waveform NRMSE no greater than 0.001% and maximum absolute error no greater than `1e-5`, reconstruct stably through iSTFT, and sum with the vocal estimate to the reconstructed mixture within the same waveform tolerances and the per-bin FP32 mixture bound above.

Tests also verify that wrapper training mode cannot update backbone parameters or BatchNorm running statistics.

## Mixture-Consistent Training Targets

For every training and validation segment, compute the original reference STFTs and project them into the model's additive output space:

`R = X - V - I`

`V_target = V + R / 2`

`I_target = I + R / 2`

The complex reconstruction, waveform L1, and multi-resolution STFT objectives use these projected targets. Projected target waveforms are produced by FP32 iSTFT with the exact source length. This avoids asking an exactly additive model to optimize mutually inconsistent source targets.

Final validation metrics and audio judgments always use the original recorded reference stems. Reports quantify both model error and the irreducible effect of the 2.64% median mixture/stem inconsistency. Projected targets are never substituted for the original references in the final Exp07-versus-Exp04 comparison.

## Loss and Weight Calibration

The training objective contains:

- A normalized complex real/imaginary reconstruction loss for both projected sources.
- Waveform L1 loss for both projected source waveforms.
- Multi-resolution STFT loss for both projected source waveforms.
- A differentiable vocal-leakage term based on the projection of predicted instrumental error onto the projected vocal target. Its denominator is clamped, and silent-vocal examples contribute zero rather than an unstable ratio.

Loss weights are not copied from Exp04. Before training, run a deterministic calibration probe over fixed training segments. Record each unweighted component's value and its gradient norm with respect to the residual head's final projection. Select fixed weights that keep complex reconstruction dominant, give waveform and multi-resolution losses material but smaller gradient contributions, and prevent leakage from dominating the update. Save raw measurements, selected weights, and the resulting combined gradient contribution in the preflight report and experiment configuration.

Zero-initializing the final projection intentionally blocks gradients to earlier residual blocks on the first backward pass. The gradient test therefore requires finite, nonzero gradients on the final projection at initialization, performs one optimizer step, and then verifies finite, nonzero gradients in upstream residual blocks. This behavior must not be misreported as a gradient failure.

### Fixed-Segment Learnability Gate

Oracle headroom alone does not prove that the residual architecture can learn useful corrections. Before the pilot, run a disposable overfitting probe on four fixed, active-vocal six-second segments drawn only from the training split, with augmentation disabled. Use the calibrated loss weights, batch size 4, learning rate `0.0001`, and at most 300 optimizer steps. The probe writes only beneath `outputs/logs/exp07_phase_aware_residual_preflight/`, outside the pilot log directory, and must not replace `initial.pt`, `best.pt`, or any pilot checkpoint.

The learnability gate passes only if all of the following hold relative to step zero:

- The fixed-batch total projected-target objective falls by at least 20%.
- The normalized complex reconstruction component falls by at least 10%.
- Vocal SDR against the projected vocal target improves by at least 1.0 dB on at least three of the four segments.
- The learned correction is nonzero and all correction, gain, amplification, clipping, waveform-consistency, loss, and gradient diagnostics remain finite.

Failure stops the experiment before the full pilot and is reported as evidence that the proposed head or optimization setup is not sufficiently learnable under the approved configuration.

## Residual Stability Diagnostics

Training and validation logs include:

- Complex correction magnitude and energy relative to `V0`.
- Per-track and aggregate output gain relative to the original references and mixture.
- Maximum and high-percentile spectral amplification relative to both `V0` and `X`, using clamped denominators and active-bin reporting.
- Waveform peak, samples exceeding `[-1, 1]`, and clipped-sample percentage without silently clipping the model output.
- Total and per-component gradient norms, gradient clipping events, and non-finite counts.
- Complex- and waveform-domain mixture reconstruction error.
- Loss-component values and the fixed calibrated weights.

Any non-finite loss, correction, waveform, or gradient stops training and preserves the latest diagnosable checkpoint. Excessive amplification or clipping is reported and treated as a quality regression even if SDR improves.

## Preflight Requirements

The pilot remains blocked until all checks pass:

1. Oracle analysis passes the fixed go/no-go rule.
2. Zero-residual compatibility tests pass for vocal and instrumental complex estimates and waveforms.
3. Backbone weights and BatchNorm state remain frozen.
4. Stereo, silence, shape, variable-length, STFT/iSTFT, and mixture-additivity unit tests pass.
5. Gradient tests pass using the two-step zero-projection procedure.
6. CUDA smoke training has finite losses and gradients.
7. The fixed-segment learnability gate passes.
8. Batch size 4 is benchmarked on the RTX 3080 Ti, including peak allocated and reserved VRAM, step time, and projected epoch/runtime. If it does not fit safely, training stops for review rather than silently changing the approved effective batch.
9. Initial, best, latest, and snapshot checkpoint saving and resume behavior pass.
10. Full-song chunked overlap-add inference handles the Exp07 model and preserves output length, stereo layout, and waveform-domain mixture consistency globally and at chunk boundaries.
11. Output collision checks, free-disk checks, conflicting-process checks, and Exp01 through Exp06 checkpoint hashes pass.

## Checkpoint and Resume Contract

Before the first optimizer step, save `initial.pt` and never overwrite it. Initialize `best.pt` from the same epoch-zero state so a run with no improvement still has a valid best checkpoint. Save `latest.pt` at every completed epoch and permanent snapshots every 10 epochs.

Best-checkpoint selection and early stopping use the same explicitly named validation objective: the fixed weighted sum of the calibrated projected-target loss components. Validation loss selects checkpoints but cannot establish final experiment success.

Checkpoints include the complete model state, optimizer, scheduler, mixed-precision scaler when enabled, epoch, global step, selection value and best epoch, early-stopping counter, calibrated loss configuration, training history, Exp04 checkpoint path and SHA-256, and all available Python, NumPy, Torch CPU, Torch CUDA, DataLoader-generator, and sampling state. Resume is allowed only from this experiment's `latest.pt` and must preserve the existing `initial.pt` and `best.pt`.

Resume tests compare uninterrupted and resumed execution at an epoch boundary. Bitwise reproducibility is claimed only if the full data sampling path and CUDA kernels are demonstrably deterministic. Otherwise the report states the verified level of reproducibility and reports numerical agreement without claiming bitwise identity.

## Controlled Pilot

Only one pilot named `exp07_phase_aware_residual` is authorized. The Exp04 backbone remains frozen for the entire run. There is no automatic unfreezing, second stage, or follow-up experiment.

Initial approved parameters are:

- Maximum epochs: 30
- Samples per epoch: 2,000
- Batch size: 4
- Learning rate: 0.0001
- Early-stopping patience: 8
- Maximum runtime: 3 hours
- Snapshot interval: 10 epochs
- Automatic learning-rate scheduling
- Mixed precision only for safe real-valued CNN operations
- FP32 complex STFT, target projection, subtraction, loss-critical complex math, and iSTFT

The CUDA benchmark verifies these settings before launch. The runtime cap triggers a graceful epoch-boundary stop with a resumable latest checkpoint.

## Full-Song Inference and Phase Ablation

Exp07 full-song inference uses the existing 6-second, 50%-overlap chunk grid and divides by the actual positive accumulated Hann weights, including at the file endpoints. It reconstructs the refined vocal and subtractive instrumental for every chunk and returns stereo waveforms with the exact input length.

Waveform-domain mixture consistency is verified after complete overlap-add, not inferred only from complex STFT additivity. Compare `vocals_out + instrumental_out` directly with the original input mixture over the whole song and within a 4,096-sample window centered on every chunk seam. The global waveform NRMSE must be no greater than 0.001%, and the global and seam-window maximum absolute residual must be no greater than `1e-5`. Tests include impulses and tonal energy placed on chunk boundaries so overlap-add errors cannot hide in low-energy seams.

After training, evaluate four outputs from the same best Exp07 checkpoint:

1. Exp04 baseline `V0`.
2. Full complex refinement `V_refined`.
3. Magnitude-only ablation: use `abs(V_refined)` with the phase of `V0`.
4. Phase-only ablation: use `abs(V0)` with the phase of `V_refined`.

For each ablation, derive the instrumental as `X - V_variant` before iSTFT so additivity is preserved. Handle near-zero magnitudes without division by constructing the variants with FP32 polar operations. Score all variants against the original recorded stems. Report paired per-track changes relative to Exp04, the number of improved and regressed tracks, and the non-additive interaction between magnitude and phase changes. This distinguishes gains attributable primarily to phase correction, magnitude refinement, or their interaction.

## Final Evaluation and Success Criteria

Evaluate Exp07 and Exp04 on exactly the same 15 validation tracks with identical chunking, overlap, batch policy, reference stems, and metric implementations. Report:

- Vocal SDR and SI-SDR, globally and as active-frame medians.
- Instrumental SDR and SI-SDR.
- Vocal leakage coefficient and energy.
- Complex- and waveform-domain source reconstruction consistency.
- Output gain, spectral amplification, waveform peak, and clipping rate.
- Per-track improvements and regressions.
- Full, magnitude-only, and phase-only ablation results.
- Runtime and peak VRAM.
- The impact of mixture/stem inconsistency on attainable scores.

Export matched 20-second Exp04, Exp07, mixture, and original-reference audio excerpts for representative improvements, regressions, and difficult overlap cases. Excerpt boundaries must match across methods.

Exp07 is successful only if the best checkpoint satisfies every pre-registered quantitative threshold:

- Vocal median SDR is at least 7.00 dB.
- Instrumental median SDR is at least 12.65 dB.
- Vocal bleed coefficient is no greater than 21.39%.
- Vocal median SDR improves over Exp04 on at least 10 of the 15 validation tracks.
- There is no meaningful waveform mixture-consistency, output-gain, spectral-amplification, clipping, or perceptual regression.

The first four conditions are hard numeric gates. For the final condition, full-song mixture consistency must meet the existing 0.001% NRMSE and `1e-5` maximum-error limits; the median absolute source output-gain error may worsen by no more than 0.25 dB versus Exp04; clipped-sample percentage may increase by no more than 0.01 percentage points and no track may newly exceed 0.1% clipped samples; and the 99th-percentile active-bin spectral amplification may rise by no more than 3 dB versus Exp04. Matched listening excerpts must reveal no clearly audible new artifact. Any failed stability threshold or perceptual regression fails the experiment even if SDR passes. Validation loss alone is insufficient. The final report must state whether the module should be kept, rejected, or investigated further and must report a negative result without launching another experiment.

## Safety and Reproducibility

- Do not install dependencies without approval.
- Do not push to GitHub.
- Do not overwrite or mutate Exp01 through Exp06 artifacts.
- Do not start a second experiment or unfreeze Exp04 automatically.
- Preserve all configuration, hashes, metrics, logs, checkpoints, and commands needed to reproduce the oracle analysis, preflight, pilot, ablation, and final evaluation.
