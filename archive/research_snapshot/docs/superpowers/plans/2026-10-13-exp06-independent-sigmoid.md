# Experiment 06 Independent Sigmoid Masks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a controlled, reproducible comparison of independent sigmoid masks against Exp04 softmax masks and decide whether Exp06 should replace Exp04.

**Architecture:** Extract the trainer's checkpoint-selection lifecycle into a small testable state component while retaining Exp04's effective objective. Add reusable tensor/audio diagnostic functions and thin CLIs for the Exp04 analysis, sigmoid transition probe, paired evaluation, and listening exports. Keep all training and report outputs in Exp06-specific directories.

**Tech Stack:** Python 3, PyTorch, NumPy, SoundFile, pytest, existing MUSDB18 preprocessing and UNetV2 pipeline.

**Spec:** `docs/superpowers/specs/2026-10-13-exp06-independent-sigmoid-design.md`

## Global Constraints

- Change only mask activation from softmax to sigmoid for the full experiment.
- Warm-start from `models/experiments/exp04_asymmetric_leakage/best.pt`.
- Preserve every prior experiment, checkpoint, dataset file, and report.
- Do not add mask normalization, mixture-consistency projection, log-input compression, architecture changes, or loss changes.
- Use the exact requested training configuration and the unchanged 15-track validation split.
- Stop before full training on any unresolved compatibility, correctness, integrity, resource, or smoke-test failure.
- Do not push and do not start Exp07.

## Review Focus

- A run with no improving epoch still has a valid sigmoid-configured `best.pt` representing initial weights.
- Best-checkpoint selection and early stopping consume the identical full objective with `min_delta=0.0001`.
- Resume restores selection state and history without changing sigmoid mode or scheduler progress.
- Sigmoid checkpoint inference restores independent masks rather than silently falling back to softmax.
- Diagnostic aggregation excludes inactive bins from ratio-based conclusions and never reports non-finite values as valid statistics.

---

### Task 1: Checkpoint-selection lifecycle

**Files:**
- Create: `src/training/experiment_selection.py`
- Modify: `src/training/train_experiment.py`
- Create: `tests/test_experiment_selection.py`

**Interfaces:**
- Produces: `SelectionState`, `selection_value(validation: dict) -> float`, `update_selection(state, current, min_delta) -> tuple[SelectionState, bool]`, and checkpoint payload/restore behavior used by the trainer.
- Consumes: existing `val_base_composite_loss`, `val_leakage_loss`, and `val_total_loss` outputs.

- [ ] Write regression tests that prove Exp04's historical `val_composite_loss` equaled the full objective and that the explicit Exp06 selection value is `val_total_loss`.
- [ ] Run the focused tests and verify they fail because the selection component does not exist.
- [ ] Implement the minimal pure selection component and wire best selection plus early stopping to it.
- [ ] Add failing tests for initial-best preservation, no-improvement patience, improvement reset, and resume state/history/config restoration.
- [ ] Implement initial `best.pt` serialization and complete resumable state without modifying the loss formula.
- [ ] Run focused and full tests.

### Task 2: Mask and frequency-band diagnostics

**Files:**
- Create: `src/evaluation/mask_diagnostics.py`
- Create: `src/evaluation/analyze_exp06.py`
- Create: `tests/test_mask_diagnostics.py`

**Interfaces:**
- Produces: band definitions, TF-region aggregation, mask/IRM statistics, output-gain/reconstruction/overlap metrics, and JSON/CSV/Markdown report generation.
- Consumes: existing STFT helpers, `UNetV2`, checkpoint loader, validation split, and Task 1 checkpoint semantics.

- [ ] Write failing literal-fixture tests for band assignment, dominance/overlap regions, IRM mask error, mask sums, source overlap, gain, reconstruction residual, and non-finite detection.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement the diagnostic functions without altering inference outputs.
- [ ] Write and test a CLI that compares Exp04 softmax predictions with the same weights under sigmoid and analyzes all 15 validation tracks.
- [ ] Run focused and full tests.

### Task 3: Paired evaluation and listening exports

**Files:**
- Create: `src/evaluation/compare_experiments.py`
- Create: `tests/test_compare_experiments.py`

**Interfaces:**
- Produces: per-track delta CSV/JSON, aggregate comparison, six-track selection, and matched 20-second export manifest.
- Consumes: existing evaluation metrics, Task 2 artifact metrics, Exp04/Exp06 checkpoints, and validation split.

- [ ] Write failing tests for per-track joins, win/regression counts, deterministic six-track selection, identical excerpt windows, and absence of model-specific normalization.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement paired comparison and export utilities.
- [ ] Run focused and full tests.

### Task 4: Safety preflight and smoke/resume run

**Files:**
- Create: `reports/exp06/preflight.json`
- Create: `reports/exp06/checkpoint_hashes_before.json`
- Create: Exp06 smoke artifacts only under `models/experiments/exp06_smoke_sigmoid/` and `outputs/logs/exp06_smoke_sigmoid/`.

**Interfaces:**
- Produces: machine-readable go/no-go evidence for Task 5.
- Consumes: Tasks 1-3, CUDA status, process state, free disk, prior checkpoint hashes.

- [ ] Verify the full test suite, CUDA, BF16, finite gradients, idle compute GPU, at least 15 GB free, and absence of full Exp06 output paths.
- [ ] Hash every Exp01-Exp05 `.pt` checkpoint.
- [ ] Run an interrupted one-epoch sigmoid smoke phase followed by an actual resume to the next epoch with the same scheduler horizon.
- [ ] Verify `best.pt`, `latest.pt`, history preservation, sigmoid metadata, inference loading, output dimensions, and finite outputs.
- [ ] Record a go/no-go result; stop on any failure.

### Task 5: Full Exp06, evaluation, audio, and report

**Files:**
- Create: `models/experiments/exp06_independent_sigmoid/*`
- Create: `outputs/logs/exp06_independent_sigmoid/*`
- Create: `reports/tables/exp06_val_15tracks_*`
- Create: `reports/audio/exp06_vs_exp04_20s/*`
- Create: `reports/exp06/final_report.md`
- Create: `reports/exp06/checkpoint_hashes_after.json`

**Interfaces:**
- Consumes: Task 4 go decision and all earlier diagnostic/evaluation interfaces.
- Produces: final scientific result and Exp04 replacement recommendation.

- [ ] Launch exactly one full training process with the specified CLI values and monitor runtime, disk, numerical stability, and checkpoints.
- [ ] Evaluate the selected Exp06 checkpoint on the same 15 validation tracks as Exp04.
- [ ] Generate paired metrics, wins/regressions, artifact analysis, and six matched audio excerpts.
- [ ] Re-hash Exp01-Exp05 and prove they are unchanged.
- [ ] Write the final report with configuration, selection semantics, best epoch/runtime, comparisons, artifact findings, audio paths, recommendation, and next measured improvement.
- [ ] Run final tests and artifact-integrity checks.
