# Experiment 06 Independent Sigmoid Masks Design

## Objective

Determine whether replacing the two-source softmax mask activation with independent sigmoid masks improves MUSDB18 vocal separation without materially degrading instrumental quality or creating source-overlap and mixture-reconstruction artifacts.

## Controlled Variable

Experiment 06 changes only `UNetV2.mask_mode` from `softmax` to `sigmoid`. It warm-starts from `models/experiments/exp04_asymmetric_leakage/best.pt`. Architecture dimensions, padding, input scaling, composite-loss weights, asymmetric leakage settings, augmentation, data split, sample rate, inference overlap, and evaluation definitions remain unchanged.

No mixture-consistency projection, mask renormalization, log-input compression, architectural change, or loss modification is permitted.

## Checkpoint Selection

The implementation must first characterize the existing validation metrics. In the current trainer, `CompositeSeparationLoss.metrics["base_total"]` is the composite loss before leakage, `metrics["leakage"]` is the unweighted leakage term, and `metrics["objective"]` is `base_total + leakage_weight * leakage`. Historical `metrics["total"]` is retained as the base composite value.

Exp04 selected checkpoints through the field named `val_composite_loss`. The investigation must prove whether that field actually contained the full objective. Exp06 must use a clearly named and documented selection metric that is mathematically equivalent to Exp04's effective selection objective; it must not silently change objectives.

The selection implementation must preserve a serialized initial sigmoid checkpoint when no epoch improves, use the same metric for best-checkpoint selection and early stopping, and restore the best value, no-improvement counter, optimizer, scheduler, RNG state, history, and sigmoid configuration when resuming.

## Diagnostics

Run Exp04 over the unchanged 15-track validation split and report:

- Vocal leakage and vocal spectral SDR in fixed frequency bands.
- Errors in active vocal-dominant, instrumental-dominant, and overlapping time-frequency bins.
- Predicted-mask error relative to ideal ratio masks.
- Whether the largest errors concentrate in overlapping regions.

Before training, load Exp04 weights into an otherwise identical sigmoid-mode model and compare it with the original softmax model on the same inputs. Measure mask distributions, mask sums, output gain, mixture reconstruction residual, source overlap, non-finite values, output dimensions, and weight compatibility. This probe deliberately overrides only the activation mode in memory; Exp04 on disk remains unchanged.

## Training

The full run is named `exp06_independent_sigmoid` and writes only beneath:

- `models/experiments/exp06_independent_sigmoid/`
- `outputs/logs/exp06_independent_sigmoid/`

Configuration: base channels 16, exact padding, sigmoid masks, log input disabled, batch size 8, 4000 samples per epoch, 40 maximum epochs, learning rate `0.00003`, minimum learning rate `0.000003`, existing cosine scheduler, patience 10, minimum improvement `0.0001`, BF16 AMP enabled, snapshots every 10 epochs, five-hour runtime cap, and 15 GB minimum free disk. Existing loss and augmentation settings remain unchanged, including asymmetric leakage weight 1.0.

The full run starts only after unit tests, finite-gradient checks, CUDA/GPU/process checks, disk checks, an isolated smoke run, save/resume verification, output-directory collision checks, and hashes of Exp01-Exp05 checkpoints have passed.

## Evaluation and Listening Set

Evaluate Exp06 `best.pt` and Exp04 `best.pt` with the same 15-track split, sample rate, chunk size, 0.5 overlap, alignment, and metric definitions. Compare vocal and instrumental SDR, vocal SI-SDR, bleed coefficient and energy, output gain, mixture reconstruction residual, source overlap, distortion, and per-track wins/regressions.

Export one matched 20-second excerpt from each of six challenging tracks. Every excerpt includes mixture, ground-truth stems, Exp04 estimates, and Exp06 estimates. Models are not independently normalized.

## Decision Rule

Exp06 replaces Exp04 only if it improves vocal separation while avoiding a substantial instrumental regression or increased reconstruction/overlap artifacts. Training loss alone cannot establish success.

## Safety

Preserve all prior checkpoints and the dataset, never overwrite Exp04, never run concurrent GPU training, do not push, and do not start Exp07. Stop before full training if compatibility, evaluation correctness, checkpoint selection, checkpoint integrity, CUDA availability, GPU availability, disk capacity, smoke training, gradients, saving, or resume behavior remains unresolved.
