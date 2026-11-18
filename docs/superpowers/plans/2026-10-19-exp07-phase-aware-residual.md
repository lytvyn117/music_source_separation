# EXP07 Phase-Aware Complex Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, gate, train, and evaluate an isolated phase-aware complex residual separator that improves Exp04 without modifying Exp01-Exp06 artifacts.

**Architecture:** A frozen, evaluation-mode Exp04 `UNetV2` produces the initial softmax-mask estimate. A zero-initialized lightweight CNN predicts a complex vocal correction from normalized mixture, initial-vocal, and mask features; the instrumental is always the complex mixture minus the refined vocal. Oracle analysis, projected-target losses, preflight gates, training, overlap-add inference, phase/magnitude ablations, and evaluation live in one isolated experiment package.

**Tech Stack:** Python 3, PyTorch complex64/FP32 STFT and iSTFT, NumPy, SoundFile, pytest, CUDA on NVIDIA RTX 3080 Ti.

**Spec:** `docs/superpowers/specs/2026-10-19-exp07-phase-aware-residual-design.md`

## Global Constraints

- Preserve stereo audio at 44.1 kHz with `n_fft=2048`, `hop_length=512`, and `win_length=2048`.
- Load `models/experiments/exp04_asymmetric_leakage/best.pt` strictly; keep every backbone parameter frozen and every BatchNorm layer in evaluation mode.
- Perform complex STFT, target projection, source subtraction, loss-critical complex math, and iSTFT in FP32.
- Check every complex mixture bin with `abs(X - (V + I)) <= 1e-5 + 4 * (2**-24) * (abs(X) + abs(V) + abs(I))` using the estimated sources only. Report maximum absolute error, maximum error-to-bound ratio, active-bin p99 relative error, and violation count. Preserve the original waveform tolerances. Use the actual positive accumulated Hann denominator for all four matched estimates.
- Write experiment artifacts only to Exp07-specific model, log, report, table, and audio paths named in the specification.
- Refuse output collisions unless `--resume` is explicit; resume only from Exp07 `latest.pt`.
- Never modify Exp01-Exp06 checkpoints, logs, reports, or audio; verify protected checkpoint hashes before and after the pilot.
- Do not install dependencies and do not push to GitHub.
- Do not unfreeze Exp04, start a second stage, or launch another experiment automatically.
- Do not start the full pilot while executing this planning request; the pilot command remains behind the oracle, preflight, and human review gates.

## File Structure

- `src/experiments/__init__.py` - experiment package marker.
- `src/experiments/exp07_phase_aware_residual/__init__.py` - public Exp07 interfaces.
- `src/experiments/exp07_phase_aware_residual/config.py` - immutable experiment, path, loss-weight, and threshold dataclasses.
- `src/experiments/exp07_phase_aware_residual/complex_ops.py` - complex-channel conversion, stable normalization, target projection, and ablation math.
- `src/experiments/exp07_phase_aware_residual/chunking.py` - shared waveform chunking and overlap-add primitives.
- `src/experiments/exp07_phase_aware_residual/oracle_analysis.py` - four-way oracle evaluation and fixed go/no-go decision.
- `src/experiments/exp07_phase_aware_residual/model.py` - `PhaseAwareResidualSeparator`, residual CNN, and structured outputs.
- `src/experiments/exp07_phase_aware_residual/losses.py` - projected targets, loss components, and deterministic gradient-scale calibration.
- `src/experiments/exp07_phase_aware_residual/diagnostics.py` - correction, gain, amplification, clipping, gradient, and consistency metrics.
- `src/experiments/exp07_phase_aware_residual/inference.py` - full-song inference and full/magnitude-only/phase-only variants.
- `src/experiments/exp07_phase_aware_residual/checkpointing.py` - initial/best/latest/snapshot persistence and resume restoration.
- `src/experiments/exp07_phase_aware_residual/trainer.py` - frozen-head training and validation loops.
- `src/experiments/exp07_phase_aware_residual/preflight.py` - integrity checks, gradient tests, CUDA benchmark, and fixed-segment overfit gate.
- `src/experiments/exp07_phase_aware_residual/evaluate.py` - paired Exp04/Exp07/ablation evaluation, success decision, audio export, and report generation.
- `tests/exp07/` - isolated unit, integration, checkpoint, inference, and CUDA tests.

## Review Focus

- Silent or near-silent time-frequency bins must remain finite and must not create inflated ratio diagnostics; pinned in Tasks 1, 4, and 5.
- Recorded mixtures do not exactly equal the delivered stems; equal `R/2` projection is used for training while all final scores use original stems; pinned in Tasks 1, 2, and 4.
- A zero final projection blocks upstream gradients on the first step by design; the final layer must learn first and upstream layers must receive gradients after one update; pinned in Tasks 3 and 4.
- Complex additivity can still be spoiled by chunked waveform reconstruction; whole-song and seam-window tests use impulses and tones at boundaries; pinned in Task 6.
- Resume can restore all serialized state without proving full-pipeline bitwise determinism; uninterrupted-versus-resumed tests and explicit reproducibility labeling are pinned in Task 7.

## Expected Resource Envelope

- Frozen Exp04 backbone: 7,787,562 parameters and approximately 31 MB of FP32 weights.
- Residual head: enforce fewer than 100,000 trainable parameters; expected to be roughly 10,000-30,000 with width 32.
- Estimated batch-size-4 peak CUDA reservation: 7-10 GB. The blocking limit is 10.5 GiB reserved, leaving approximately 1.5 GiB on the 12 GB GPU.
- Existing Exp04/Exp06 runs used about 5.46 GB at batch size 8 and roughly 98-115 seconds for 500 training steps. Exp07 adds full-resolution residual activations but removes backbone backward storage, so the provisional estimate is 2-5 minutes per 500-step epoch and 1-2.5 hours for 30 epochs plus validation. Task 8 replaces this estimate with measured step time and projected runtime.
- Reserve at least 15 GB of free disk for self-contained checkpoints, reports, and matched audio.

---

### Task 1: Complex-domain foundations and immutable configuration

**Files:**
- Create: `src/experiments/__init__.py`
- Create: `src/experiments/exp07_phase_aware_residual/__init__.py`
- Create: `src/experiments/exp07_phase_aware_residual/config.py`
- Create: `src/experiments/exp07_phase_aware_residual/complex_ops.py`
- Test: `tests/exp07/test_complex_ops.py`

**Interfaces:**
- Produces: `Exp07Config`, `Exp07Paths`, `OracleThresholds`, `SuccessThresholds`, and `NumericalTolerances` frozen dataclasses.
- Produces: `project_equal_weight_targets(mixture, vocals, instrumental) -> ProjectedComplexTargets`.
- Produces: `complex_to_channels(value) -> torch.Tensor`, `channels_to_complex(value) -> torch.Tensor`, and `safe_complex_rms(value, eps) -> torch.Tensor`.
- Produces: `waveform_nrmse_pct(estimate, reference, eps=1e-12) -> float` with the reference energy in the denominator.
- Produces: `make_vocal_ablation_stfts(initial_vocal, refined_vocal) -> dict[str, torch.Tensor]` with `full`, `magnitude_only`, and `phase_only` keys.

- [ ] **Step 1: Write projection, channel-roundtrip, silence, and ablation tests**

```python
def test_equal_weight_projection_is_additive_and_splits_residual():
    targets = project_equal_weight_targets(X, V, I)
    residual = X - V - I
    assert torch.allclose(targets.vocals - V, residual / 2)
    assert torch.allclose(targets.instrumental - I, residual / 2)
    assert torch.allclose(targets.vocals + targets.instrumental, X)

def test_silent_complex_normalization_is_finite():
    scale = safe_complex_rms(torch.zeros(2, 2, 17, 11, dtype=torch.complex64), 1e-8)
    assert torch.isfinite(scale).all()
    assert (scale >= 1e-8).all()

def test_ablation_preserves_requested_magnitude_and_phase():
    variants = make_vocal_ablation_stfts(V0, V1)
    assert torch.allclose(variants["magnitude_only"].abs(), V1.abs(), atol=1e-6)
    assert torch.allclose(variants["phase_only"].abs(), V0.abs(), atol=1e-6)
```

- [ ] **Step 2: Run the new tests and verify they fail because the package does not exist**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_complex_ops.py -q`

Expected: FAIL during import of `src.experiments.exp07_phase_aware_residual`.

- [ ] **Step 3: Implement the dataclasses and minimal complex operations**

Use complex64 tensors, preserve batch/stereo/frequency/time axes, clamp only normalization denominators, and construct ablations with FP32 `torch.polar` rather than complex division.

- [ ] **Step 4: Run the focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_complex_ops.py -q`

Expected: PASS.

- [ ] **Step 5: Commit only Task 1 files**

```powershell
git add src/experiments/__init__.py src/experiments/exp07_phase_aware_residual/__init__.py src/experiments/exp07_phase_aware_residual/config.py src/experiments/exp07_phase_aware_residual/complex_ops.py tests/exp07/test_complex_ops.py
git commit -m "feat: add exp07 complex-domain foundations"
```

### Task 2: Oracle analysis and immutable go/no-go report

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/chunking.py`
- Create: `src/experiments/exp07_phase_aware_residual/oracle_analysis.py`
- Test: `tests/exp07/test_oracle_analysis.py`

**Interfaces:**
- Consumes: Task 1 target projection and thresholds.
- Produces: `chunk_starts(total_samples, chunk_samples, overlap_ratio) -> list[int]` and `overlap_add_chunks(chunks, starts, total_samples) -> np.ndarray`.
- Produces: `compute_oracle_chunk(mixture, vocals, instrumental) -> dict[str, SourcePair]` for `mixture_phase`, `unconstrained_complex`, and `constrained_complex`.
- Produces: `evaluate_oracle_gate(records, thresholds) -> GateDecision`.
- Produces: CLI `python -m src.experiments.exp07_phase_aware_residual.oracle_analysis` writing JSON, CSV, Markdown, and optional audio only under `reports/exp07/oracle/` and `reports/tables/exp07_oracle_*`.

- [ ] **Step 1: Write failing tests for oracle identity, constrained additivity, original-reference scoring, and gate boundaries**

```python
def test_constrained_oracle_is_r_over_two_projection():
    result = compute_oracle_chunk(mix, vocals, instrumental)
    constrained = result["constrained_complex"]
    error, bound = complex_additivity(compute_stft(mix), constrained.vocals, constrained.instrumental)
    assert (error <= bound).all()

def test_oracle_gate_requires_all_three_headroom_conditions():
    decision = evaluate_oracle_gate(records_with_10_wins_and_1db_median, OracleThresholds())
    assert decision.passed
    assert not evaluate_oracle_gate(records_with_9_wins, OracleThresholds()).passed
    assert not evaluate_oracle_gate(records_with_0_99db_median, OracleThresholds()).passed
```

Also assert that the unconstrained oracle is labeled non-additive when `mixture != vocals + instrumental` and that metric functions receive original, not projected, references.

- [ ] **Step 2: Run the oracle tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_oracle_analysis.py -q`

Expected: FAIL with missing chunking/oracle interfaces.

- [ ] **Step 3: Implement shared overlap-add and the four-way oracle analyzer**

Reuse `src.evaluation.metrics` for SDR, SI-SDR, and leakage. Run Exp04 through its existing loader, but implement oracle chunking through the shared 6-second/50%-overlap path so all four methods have matched boundary treatment. Include per-track mixture/stem mismatch and numerical reconstruction diagnostics.

- [ ] **Step 4: Run focused and existing evaluation tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_oracle_analysis.py tests\test_compare_experiments.py tests\test_validation_reproducibility.py -q`

Expected: PASS.

- [ ] **Step 5: Run the full 15-track oracle analysis and enforce the stop rule**

Run:

```powershell
.\.venv\Scripts\python.exe -u -m src.experiments.exp07_phase_aware_residual.oracle_analysis --checkpoint models\experiments\exp04_asymmetric_leakage\best.pt --val-split --overlap 0.5 --output-dir reports\exp07\oracle --tag exp07_oracle_15tracks
```

Expected: a report containing all four methods, paired per-track deltas, the measured mixture/stem inconsistency, numerical-tolerance checks, and an explicit `GO` or `STOP`. If the result is `STOP`, halt the plan before Task 3 and report it.

- [ ] **Step 6: Commit Task 2 code and tests; do not commit generated large audio**

```powershell
git add src/experiments/exp07_phase_aware_residual/chunking.py src/experiments/exp07_phase_aware_residual/oracle_analysis.py tests/exp07/test_oracle_analysis.py
git commit -m "feat: add exp07 oracle headroom gate"
```

### Task 3: Zero-initialized PhaseAwareResidualSeparator

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/model.py`
- Test: `tests/exp07/test_model.py`

**Interfaces:**
- Consumes: Task 1 normalization utilities and Exp04 `UNetV2`.
- Produces: `SeparationOutput` containing mixture STFT, masks, initial vocal STFT, residual STFT, refined vocal STFT, subtractive instrumental STFT, and optional waveforms.
- Produces: `PhaseAwareResidualSeparator.from_exp04(checkpoint_path, device) -> PhaseAwareResidualSeparator`.
- Produces: `forward_stft(mixture_stft, length=None) -> SeparationOutput` and `forward(mixture_waveform) -> SeparationOutput`.

- [ ] **Step 1: Write failing architecture and compatibility tests**

```python
def test_final_projection_is_exactly_zero_and_head_is_small(model):
    assert torch.count_nonzero(model.refinement.final.weight) == 0
    assert torch.count_nonzero(model.refinement.final.bias) == 0
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 100_000

def test_train_keeps_backbone_frozen_and_in_eval_mode(model):
    before = capture_batchnorm_state(model.backbone)
    model.train()
    model(torch.randn(1, 2, 44100))
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert capture_batchnorm_state(model.backbone) == before

def test_zero_residual_matches_exp04_and_is_additive(model, segment):
    output = model(segment)
    assert waveform_nrmse_pct(output.vocals_waveform, direct_exp04_vocals) <= 0.001
    assert waveform_nrmse_pct(output.instrumental_waveform, direct_exp04_instrumental) <= 0.001
    error, bound = complex_additivity(output.mixture_stft, output.vocals_stft, output.instrumental_stft)
    assert (error <= bound).all()
```

Add separate all-silence, stereo-shape, non-finite, actual-checkpoint, and maximum-absolute-error assertions.

- [ ] **Step 2: Run model tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_model.py -q`

Expected: FAIL with missing model interfaces.

- [ ] **Step 3: Implement the residual head and wrapper**

Use 12 input feature channels, width 32, four depthwise-separable residual blocks with dilations `(1, 1)`, `(2, 2)`, `(4, 4)`, and `(1, 1)`, GroupNorm, SiLU, and a zeroed four-channel 1x1 output projection. Override `train()` so the backbone always returns to evaluation mode. Keep backbone inference inside `torch.no_grad()` and complex operations outside autocast.

- [ ] **Step 4: Run model and existing STFT/U-Net tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_model.py tests\test_stft.py tests\test_unet.py tests\test_v2_modules.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3 files**

```powershell
git add src/experiments/exp07_phase_aware_residual/model.py tests/exp07/test_model.py
git commit -m "feat: add exp07 phase-aware residual model"
```

### Task 4: Projected-target loss and reproducible gradient calibration

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/losses.py`
- Test: `tests/exp07/test_losses.py`

**Interfaces:**
- Consumes: `SeparationOutput` and Task 1 projection.
- Produces: `ProjectedTargets` containing projected complex sources and exact-length projected waveforms.
- Produces: `PhaseAwareLoss.forward(output, targets, weights) -> LossResult` with tensor components `complex`, `waveform_l1`, `mr_stft`, `leakage`, and `total`.
- Produces: `calibrate_loss_weights(model, fixed_batches, target_ratios) -> LossCalibrationReport`.

- [ ] **Step 1: Write failing tests for projected objectives, silence, and the two-step gradient rule**

```python
def test_projected_targets_sum_to_mixture_in_complex_and_waveform_domains():
    targets = make_projected_targets(mixture, vocals, instrumental)
    error, bound = complex_additivity(compute_stft(mixture), targets.vocals_stft, targets.instrumental_stft)
    assert (error <= bound).all()
    assert waveform_nrmse_pct(targets.vocals_waveform + targets.instrumental_waveform, mixture) <= 0.001

def test_zero_projection_gradient_reaches_upstream_after_one_step(model, batch):
    first = backward_once(model, batch)
    assert first.final_projection_norm > 0
    assert first.upstream_norm == 0
    optimizer.step()
    second = backward_once(model, batch)
    assert second.upstream_norm > 0
    assert second.all_finite
```

Also assert that loss is zero when predictions equal projected targets, leakage returns zero for silent vocals, and every loss-critical tensor is float32/complex64.

- [ ] **Step 2: Run loss tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_losses.py -q`

Expected: FAIL with missing loss interfaces.

- [ ] **Step 3: Implement projected targets and loss components**

Normalize complex real/imaginary L1 by clamped per-example complex RMS. Compute waveform and multi-resolution losses from projected waveforms. Implement leakage as the squared, clamped least-squares projection of predicted instrumental error onto the projected vocal target, returning zero for silent-vocal examples.

- [ ] **Step 4: Implement deterministic weight calibration**

Use eight fixed, unaugmented training segments. Set complex weight to `1.0`; target auxiliary gradient norms relative to complex are `0.25` for waveform L1, `0.25` for multi-resolution STFT, and `0.10` for leakage. Compute `weight = target_ratio * complex_grad_norm / max(component_grad_norm, eps)`, clamp each auxiliary weight to `[1e-4, 10.0]`, then remeasure. Fail calibration if any norm is zero/non-finite or the complex contribution is outside 50-80% of the sum of component gradient norms. Serialize raw values, raw norms, selected weights, and remeasured contributions.

- [ ] **Step 5: Run the focused loss and gradient suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_losses.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 4 files**

```powershell
git add src/experiments/exp07_phase_aware_residual/losses.py tests/exp07/test_losses.py
git commit -m "feat: add exp07 projected-target losses"
```

### Task 5: Residual stability diagnostics

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/diagnostics.py`
- Test: `tests/exp07/test_diagnostics.py`

**Interfaces:**
- Consumes: model output, mixture waveform, projected targets, and optional gradients.
- Produces: `compute_residual_diagnostics(...) -> dict[str, float]`.
- Produces: `compute_gradient_diagnostics(model, clip_threshold) -> dict[str, float]`.

- [ ] **Step 1: Write failing exact-value tests for correction ratios, gains, active-bin amplification, clipping, and consistency**

```python
def test_diagnostics_do_not_inflate_silent_bin_amplification():
    stats = compute_residual_diagnostics(output_with_silent_bins, mixture, targets)
    assert stats["nonfinite_count"] == 0
    assert stats["spectral_amplification_active_bin_count"] == expected_active

def test_clipping_and_output_gain_are_reported_without_clamping():
    stats = compute_residual_diagnostics(output_over_one, mixture, targets)
    assert stats["clipped_sample_pct"] == pytest.approx(expected_pct)
    assert output_over_one.vocals_waveform.max() > 1.0
```

- [ ] **Step 2: Run diagnostics tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_diagnostics.py -q`

Expected: FAIL with missing diagnostics module.

- [ ] **Step 3: Implement diagnostics with clamped denominators and explicit active-bin masks**

Log correction magnitude/energy versus `V0`, output gain, 95th/99th/max spectral amplification versus `V0` and `X`, peaks, clipped-sample percentage, complex and waveform reconstruction residuals, gradient norms, clip events, and non-finite counts.

- [ ] **Step 4: Run diagnostics and prior tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_diagnostics.py tests\exp07\test_complex_ops.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 5 files**

```powershell
git add src/experiments/exp07_phase_aware_residual/diagnostics.py tests/exp07/test_diagnostics.py
git commit -m "feat: add exp07 residual stability diagnostics"
```

### Task 6: Full-song inference, overlap-add consistency, and ablations

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/inference.py`
- Test: `tests/exp07/test_inference.py`

**Interfaces:**
- Consumes: Task 2 chunking, Task 3 model, and Task 1 ablation math.
- Produces: `separate_waveform(model, mixture, overlap_ratio=0.5, batch_size=4, include_ablations=False) -> InferenceResult`.
- Produces: `measure_full_song_consistency(mixture, result, starts, seam_window=4096) -> ConsistencyReport`.
- Produces: checkpoint-aware CLI for one input file without changing `src/inference/separate_track.py`.

- [ ] **Step 1: Write failing short-song, multi-chunk, orientation, and seam tests**

```python
def test_full_song_outputs_sum_to_input_across_chunk_seams(identity_model):
    mixture = tone_with_impulses_at_chunk_starts()
    result = separate_waveform(identity_model, mixture, overlap_ratio=0.5, batch_size=4)
    report = measure_full_song_consistency(mixture, result, result.chunk_starts)
    assert report.global_nrmse_pct <= 0.001
    assert report.global_max_abs <= 1e-5
    assert report.max_seam_window_abs <= 1e-5

def test_ablation_instrumentals_are_subtractive(model, segment):
    result = separate_waveform(model, segment, include_ablations=True)
    for variant in ("full", "magnitude_only", "phase_only"):
        assert waveform_nrmse_pct(result[variant].vocals + result[variant].instrumental, segment) <= 0.001
```

- [ ] **Step 2: Run inference tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_inference.py -q`

Expected: FAIL with missing inference interfaces.

- [ ] **Step 3: Implement inference using the shared chunk starts and Hann overlap-add weights**

Perform model/STFT work on device, return NumPy float32 in the input orientation, preserve exact sample length, and construct each ablation in complex STFT space before deriving its instrumental by subtraction.

- [ ] **Step 4: Run inference tests plus existing overlap-add tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_inference.py tests\test_v2_modules.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 6 files**

```powershell
git add src/experiments/exp07_phase_aware_residual/inference.py tests/exp07/test_inference.py
git commit -m "feat: add exp07 full-song inference and ablations"
```

### Task 7: Checkpoint-safe trainer and resume restoration

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/checkpointing.py`
- Create: `src/experiments/exp07_phase_aware_residual/trainer.py`
- Test: `tests/exp07/test_checkpointing.py`
- Test: `tests/exp07/test_trainer.py`

**Interfaces:**
- Consumes: model, projected loss, diagnostics, existing `AugmentedMUSDBDataset`, and approved hyperparameters.
- Produces: `CheckpointManager.initialize(state)`, `save_latest(state)`, `maybe_save_best(state, selection_value)`, `save_snapshot(state)`, and `load_latest(expected_config)`.
- Produces: `capture_rng_state() -> dict`, `restore_rng_state(state) -> None`, `train_epoch(...) -> EpochMetrics`, `validate_epoch(...) -> EpochMetrics`, and CLI `python -m src.experiments.exp07_phase_aware_residual.trainer`.

- [ ] **Step 1: Write failing checkpoint tests**

```python
def test_initial_is_immutable_and_best_exists_at_epoch_zero(tmp_path):
    manager.initialize(state_epoch_zero)
    initial_hash = sha256(tmp_path / "initial.pt")
    manager.save_latest(state_epoch_one)
    manager.maybe_save_best(worse_state, worse_value)
    assert sha256(tmp_path / "initial.pt") == initial_hash
    assert torch.load(tmp_path / "best.pt", weights_only=False)["epoch"] == 0

def test_resume_restores_all_available_state(tmp_path):
    restored = manager.load_latest(expected_config)
    assert restored.optimizer_state == saved.optimizer_state
    assert restored.scheduler_state == saved.scheduler_state
    assert restored.selection_state == saved.selection_state
    assert restored.history == saved.history
    assert restored.rng_state.keys() >= {"python", "numpy", "torch_cpu", "torch_cuda"}
```

Also test configuration mismatch refusal, missing best-checkpoint refusal, snapshot interval, and output collision behavior.

- [ ] **Step 2: Write failing trainer tests**

Assert that the optimizer contains only residual-head parameters, the backbone state is unchanged after an epoch, best selection and early stopping use the same named weighted validation objective, runtime cap exits at an epoch boundary, and non-finite diagnostics stop with `latest.pt` preserved.

- [ ] **Step 3: Run checkpoint/trainer tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_checkpointing.py tests\exp07\test_trainer.py -q`

Expected: FAIL with missing checkpoint/trainer interfaces.

- [ ] **Step 4: Implement complete checkpoint state and atomic writes**

Write to a sibling temporary file and replace the target only after serialization succeeds. Store complete model, optimizer, scheduler, scaler if enabled, epoch/global step, selection metric/value/best epoch, patience counter, calibrated weights, history, Exp04 path/hash, experiment config, and available RNG/DataLoader generator state. Never overwrite `initial.pt`; initialize `best.pt` independently from epoch zero.

- [ ] **Step 5: Implement the trainer with frozen-backbone enforcement**

Use AdamW on `model.refinement.parameters()` only, cosine scheduling, gradient clipping, 30-epoch/2,000-sample/batch-4 defaults, learning rate `1e-4`, patience 8, three-hour cap, and snapshots every 10 epochs. Default to `num_workers=0` for restorable Python sampling state. Record whether deterministic CUDA algorithms are enabled and never claim bitwise resume unless the uninterrupted-versus-resumed test is exactly equal.

- [ ] **Step 6: Run checkpoint, trainer, and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_checkpointing.py tests\exp07\test_trainer.py tests\test_experiment_selection.py tests\test_asymmetric_leakage.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 7 files**

```powershell
git add src/experiments/exp07_phase_aware_residual/checkpointing.py src/experiments/exp07_phase_aware_residual/trainer.py tests/exp07/test_checkpointing.py tests/exp07/test_trainer.py
git commit -m "feat: add resumable exp07 trainer"
```

### Task 8: Blocking preflight, fixed-segment learnability, and CUDA benchmark

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/preflight.py`
- Test: `tests/exp07/test_preflight.py`
- Test: `tests/exp07/test_cuda_smoke.py`

**Interfaces:**
- Consumes: Tasks 3-7 and the Task 2 oracle report.
- Produces: `select_fixed_active_segments(train_files, count=4) -> FixedSegmentManifest` using deterministic highest-vocal-RMS selection.
- Produces: `run_overfit_gate(...) -> LearnabilityDecision`, `benchmark_cuda(...) -> CudaBenchmark`, `hash_protected_checkpoints(...) -> dict`, and `run_preflight(...) -> PreflightDecision`.
- Produces: `reports/exp07/preflight.json`, `reports/exp07/preflight.md`, and disposable learnability logs below `outputs/logs/exp07_phase_aware_residual_preflight/`, with a single blocking `passed` field and per-gate evidence.

- [ ] **Step 1: Write failing preflight policy tests**

```python
def test_overfit_gate_requires_loss_complex_and_three_sdr_wins():
    assert evaluate_learnability(valid_probe).passed
    assert not evaluate_learnability(replace(valid_probe, total_reduction_pct=19.99)).passed
    assert not evaluate_learnability(replace(valid_probe, projected_sdr_wins=2)).passed

def test_cuda_oom_does_not_silently_change_batch_size():
    with pytest.raises(PreflightFailure, match="batch size 4"):
        benchmark_cuda(ooming_step, batch_size=4)
```

Also test missing/stale oracle report, checkpoint hash mismatch, less than 15 GB disk, non-empty outputs, conflicting process detection, failed overlap-add seam report, and reserved VRAM above 10.5 GiB. Implement Windows process inspection with the standard library invoking `Get-CimInstance`; do not add `psutil` or another dependency.

- [ ] **Step 2: Run preflight tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_preflight.py tests\exp07\test_cuda_smoke.py -q`

Expected: FAIL with missing preflight interfaces or CUDA tests skipped on non-CUDA hosts.

- [ ] **Step 3: Implement deterministic fixed-segment selection and the 300-step overfit probe**

Choose four training-only tracks deterministically, scan six-second windows at fixed strides, select the highest vocal-RMS window per chosen track, disable augmentation, and save track/start metadata. Use the calibrated weights, batch size 4, learning rate `1e-4`, and a disposable fresh residual head. Enforce 20% total-loss reduction, 10% complex-loss reduction, projected-target vocal SDR gain of at least 1 dB on at least three segments, nonzero corrections, and finite stability metrics.

- [ ] **Step 4: Implement CUDA benchmark and complete preflight aggregation**

Run 10 warm-up and 30 measured training steps at batch size 4, reset CUDA peak statistics before measurement, and report median/p95 step time, peak allocated/reserved VRAM, and 30-epoch projected runtime. Fail if reserved VRAM exceeds 10.5 GiB, any value is non-finite, or projected runtime exceeds three hours. Include actual-checkpoint initialization, two-step gradients, checkpoint/resume, full-song seam inference, disk/process/collision checks, and before hashes for Exp01-Exp06.

- [ ] **Step 5: Run the full CPU/unit suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07 -m "not cuda" -q`

Expected: PASS.

- [ ] **Step 6: Run CUDA smoke tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_cuda_smoke.py -m cuda -q`

Expected: PASS on the RTX 3080 Ti with finite gradients and peak reservation below 10.5 GiB.

- [ ] **Step 7: Generate the complete preflight report without starting the pilot**

Run:

```powershell
.\.venv\Scripts\python.exe -u -m src.experiments.exp07_phase_aware_residual.preflight --checkpoint models\experiments\exp04_asymmetric_leakage\best.pt --oracle-report reports\exp07\oracle_scale_aware\exp07_oracle_scale_aware_summary.json --batch-size 4 --overfit-steps 300 --output reports\exp07\preflight.json
```

Expected: `passed: true`, the calibrated fixed loss weights, learnability evidence, checkpoint/resume evidence, seam consistency, protected hashes, and measured GPU/runtime data. If false, stop and report; do not launch training.

- [ ] **Step 8: Commit Task 8 code and tests**

```powershell
git add src/experiments/exp07_phase_aware_residual/preflight.py tests/exp07/test_preflight.py tests/exp07/test_cuda_smoke.py
git commit -m "feat: add exp07 blocking preflight"
```

### Task 9: Controlled pilot launch - deferred until post-preflight review

**Files:**
- Runtime outputs only: `models/experiments/exp07_phase_aware_residual/`
- Runtime outputs only: `outputs/logs/exp07_phase_aware_residual/`

**Interfaces:**
- Consumes: `reports/exp07/preflight.json` with `passed: true` and recorded protected hashes.
- Produces: immutable `initial.pt`, independently initialized `best.pt`, per-epoch `latest.pt`, epoch-10/20/30 snapshots when reached, training CSV/JSON logs, and post-run protected hashes.

- [ ] **Step 1: Present the oracle and preflight measurements for human review**

Include the oracle go/no-go evidence, overfit curves, selected loss weights, initialization errors, gradient checks, resume result, overlap-add seam result, peak VRAM, step time, and projected runtime. Do not continue unless the full-pilot gate is acknowledged after implementation.

- [ ] **Step 2: Recheck exact output targets and conflicting processes immediately before launch**

Expected: no existing non-empty Exp07 pilot outputs, no conflicting training/evaluation process, at least 15 GB free disk, and unchanged Exp01-Exp06 hashes.

- [ ] **Step 3: Launch exactly one approved pilot**

Run only after the review gate:

```powershell
.\.venv\Scripts\python.exe -u -m src.experiments.exp07_phase_aware_residual.trainer --launch-approved-pilot --require-preflight reports\exp07\preflight.json
```

Expected: only the residual head changes; runtime stops on early stopping, epoch 30, or the three-hour cap. No second run or backbone unfreezing occurs automatically.

- [ ] **Step 4: Verify post-pilot checkpoint integrity**

Expected: `initial.pt` hash unchanged since launch, `best.pt` selection metadata matches the minimum named validation objective, `latest.pt` is resumable, snapshot cadence is correct, and every Exp01-Exp06 checkpoint hash matches the preflight manifest.

### Task 10: Paired evaluation, phase contribution, matched audio, and conclusion

**Files:**
- Create: `src/experiments/exp07_phase_aware_residual/evaluate.py`
- Test: `tests/exp07/test_evaluate.py`
- Runtime outputs: `reports/tables/exp07_*`
- Runtime outputs: `reports/audio/exp07_vs_exp04_20s/`
- Runtime output: `reports/exp07/final_report.md`

**Interfaces:**
- Consumes: Exp04 checkpoint, best Exp07 checkpoint, original 15-track references, Task 6 inference variants, and Task 5 diagnostics.
- Produces: `evaluate_checkpoint_pair(...) -> EvaluationBundle`, `evaluate_success(bundle, thresholds) -> SuccessDecision`, matched audio manifest, JSON/CSV tables, and technical Markdown conclusion.

- [ ] **Step 1: Write failing tests for paired metrics, hard thresholds, ablation attribution, and excerpt alignment**

```python
def test_success_requires_every_preregistered_numeric_gate():
    assert evaluate_success(passing_bundle, SuccessThresholds()).passed
    assert not evaluate_success(replace(passing_bundle, vocals_median_sdr_db=6.999), SuccessThresholds()).passed
    assert not evaluate_success(replace(passing_bundle, improved_track_count=9), SuccessThresholds()).passed

def test_matched_audio_uses_identical_boundaries_for_all_methods():
    manifest = build_audio_manifest(records)
    for item in manifest:
        assert len({clip.start_sample for clip in item.clips}) == 1
        assert len({clip.num_samples for clip in item.clips}) == 1
```

Also test that final metrics use original stems, not projected targets; every ablation derives its instrumental subtractively; and perceptual/stability regressions can fail an otherwise numerically passing run.

- [ ] **Step 2: Run evaluation tests and verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\exp07\test_evaluate.py -q`

Expected: FAIL with missing evaluation interfaces.

- [ ] **Step 3: Implement paired Exp04/Exp07/ablation evaluation**

Evaluate all 15 tracks with identical chunking and original reference stems. Report global and active-frame median SDR/SI-SDR, instrumental metrics, leakage coefficient and energy, output gain, spectral amplification, clipping, global and seam mixture consistency, runtime, and per-track deltas. Compare Exp04, full Exp07, magnitude-only, and phase-only estimates and report interaction rather than treating contributions as additive.

- [ ] **Step 4: Implement fixed quantitative success decision and matched export selection**

Require vocal median SDR `>= 7.00`, instrumental median SDR `>= 12.65`, vocal bleed coefficient `<= 0.2139`, and at least 10/15 vocal-SDR track wins. Also require full-song mixture NRMSE `<= 0.001%` and maximum/seam residual `<= 1e-5`; median absolute source output-gain error no more than 0.25 dB worse than Exp04; clipped-sample percentage no more than 0.01 percentage points higher with no newly affected track above 0.1%; active-bin 99th-percentile spectral amplification no more than 3 dB above Exp04; and no audible new artifact in matched excerpts. Select representative improvement, regression, and difficult-overlap excerpts; export identical 20-second boundaries for mixture, references, Exp04, full Exp07, and both ablations.

- [ ] **Step 5: Run the complete test suite before evaluation**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: PASS with CUDA-specific tests either passed on GPU or explicitly selected separately.

- [ ] **Step 6: Run final 15-track evaluation after Task 9 finishes**

Run:

```powershell
.\.venv\Scripts\python.exe -u -m src.experiments.exp07_phase_aware_residual.evaluate --exp04-checkpoint models\experiments\exp04_asymmetric_leakage\best.pt --exp07-checkpoint models\experiments\exp07_phase_aware_residual\best.pt --val-split --overlap 0.5 --batch-size 4 --export-audio reports\audio\exp07_vs_exp04_20s --output-dir reports\exp07
```

Expected: paired CSV/JSON, matched audio manifest, ablation tables, protected-hash verification, and `reports/exp07/final_report.md` stating keep/reject/investigate without launching further work.

- [ ] **Step 7: Commit Task 10 source and tests; keep large generated audio out of Git unless explicitly requested**

```powershell
git add src/experiments/exp07_phase_aware_residual/evaluate.py tests/exp07/test_evaluate.py
git commit -m "feat: add exp07 paired evaluation and ablations"
```

## Final Execution Order and Stop Points

1. Implement Task 1 foundations.
2. Implement and run Task 2 oracle diagnostics. **STOP permanently if the oracle gate fails.**
3. Implement Tasks 3-7 with red/green tests and focused commits.
4. Implement and run Task 8. **STOP if any compatibility, learnability, gradient, resume, seam-consistency, integrity, VRAM, or runtime gate fails.**
5. Present the measured oracle and preflight report. **Do not start Task 9 until reviewed.**
6. Run the single Task 9 pilot only after that review.
7. Implement/run Task 10 evaluation and report the result. **Do not start a second experiment regardless of outcome.**
