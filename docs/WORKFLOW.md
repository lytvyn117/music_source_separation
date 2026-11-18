# Two-computer workflow

Exp10 is the project's current reference implementation. Source code is shared through Git; MoisesDB and compute-intensive runs remain on the training computer. The full dataset does not need to be copied to every development machine.

## Execution matrix

“Local” means a development computer that does not necessarily have CUDA. This table describes the requirements of the current programs, not the results of rerunning them.

| Task | Local development computer | Training computer | Required files / conditions | Entry point |
| --- | --- | --- | --- | --- |
| Inspect code, configurations and recorded results | Yes | Yes | Git checkout; no dataset or weights needed | `README.md`, `history/EXPERIMENTS.md`, `provenance/` |
| Exp10 unit tests | Yes, CPU | Yes | Python, PyTorch, NumPy, SoundFile; included `config.json` and `roster.json`; synthetic inputs | From `exp10_scratch`: `python -B -m unittest discover -s tests -v` |
| Prepare development references | Only with the archive; preparation itself does not require CUDA | Yes, preferred location | Original approved MoisesDB ZIP, disk space and Exp10 dependencies | `exp10_scratch/reproduce.py prepare --archive ...` |
| Train Exp10 from scratch | Not supported by the current CPU workflow | Yes | MoisesDB, verified development cache, CUDA with native BF16, configuration and roster | `exp10_scratch/reproduce.py train --archive ... --out ...` |
| Resume your own run | Not supported by the current CPU workflow | Yes | Matching source and environment, data, numbered checkpoints and run provenance | The same training command with `--resume` |
| Evaluate the 16 development recordings | Only with CUDA and the required files | Yes | Checkpoint and verified development cache; the archive is needed to build the cache | `exp10_scratch/reproduce.py evaluate --checkpoint ... --output ...` |
| Separate your own song | Only with CUDA: the current CLI has no CPU fallback | Yes | Checkpoint and 44.1 kHz stereo WAV/FLAC; MoisesDB is not needed | `exp10_scratch/infer.py` |
| CUDA inference smoke test | Only with CUDA | Yes | Selected checkpoint; the test generates its own audio | `exp10_scratch/smoke_inference.py --checkpoint ...` |
| Presentation demo | Only with CUDA | Yes | Frozen Exp10 Raw checkpoint and your audio; MoisesDB is not needed | `PRESENTATION_2026-11-23/run_demo.ps1` |
| Analyze metrics and listen to exported results | Yes | Yes | Exported JSON/tables and audio examples authorized for sharing | Outputs from the corresponding run |
| Reproduce historical experiments in full | Depends on the experiment | Depends on retained artifacts | Historical manifests, data, caches, checkpoints and sometimes external implementations | See the source map in the root README |
| External final evaluation | Outside routine development | Outside routine training | Separate external holdout and its evaluation protocol | Do not substitute development evaluation |

CPU unit tests do not establish model quality or numerical equivalence with CUDA training. The repository's historical tests should not be treated as one self-contained suite: some depend on historical data and artifacts.

## Local Exp10 checks

From the repository root in PowerShell, using an existing configured environment:

```powershell
$Python = (Resolve-Path .venv/Scripts/python.exe).Path
Push-Location exp10_scratch
try {
    & $Python -B -m unittest discover -s tests -v
} finally {
    Pop-Location
}
```

`requirements-exp10.txt` pins the CUDA reproduction environment. CPU development requires a separate compatible environment with PyTorch, NumPy and SoundFile; it is not considered identical to the original training environment. `requirements.txt` belongs to the historical MUSDB/U-Net branch.

## Handoff between computers

1. The developer commits the change to Git and shares the commit, configuration, experiment objective and launch command.
2. The teammate runs the experiment from that source revision in a separate run directory. That run's source files remain unchanged during training.
3. Results are returned with the commit, configuration, checkpoint index, metrics, weight arm (Raw/EMA) and step. Small reports are sufficient for analysis; transferring every intermediate checkpoint is unnecessary.
4. Local inference additionally requires the selected checkpoint and its trusted SHA-256 digest. Separating your own song does not require the dataset, but the current CLI requires CUDA.

Record the experiment number, Git commit and run directory separately: they are different identifiers. New Exp10 training runs must be stored directly under `exp10_scratch/runs/`, as required by the current runner.

Resuming requires more than `best_overall.pt`: retain the numbered checkpoint bundles and provenance for that run. Source, configuration and environment are checked during recovery. Do not bypass these checks by moving or substituting files.

## Commands and limitations

Full commands for data preparation, training, evaluation and CLI inference are in [EXP10_REPRODUCTION.md](../EXP10_REPRODUCTION.md). Supply machine-specific data and weight paths through the supported arguments rather than copying a teammate's local paths into shared code.

For the demo, explicitly pass `-Checkpoint`: otherwise, `run_demo.ps1` looks for `exp10_scratch/runs/exp10_seed909/best_overall.pt`, which may not exist in a local checkout. For checkpoints from your own runs, use the main `infer.py` with a trusted `--checkpoint-sha256`; the presentation wrapper expects the frozen checkpoint.

`best_overall` means the highest vocal SDR, not that every additional threshold was met. Frozen Exp10 Raw at step 24,000 is the selected reference even though no `best_qualified` candidate was found. Scores from different datasets and protocols must not be interpreted as a direct quality improvement.

The original `provenance/`, `SOURCE_ORIGINS.json` and `SOURCE_SHA256SUMS.txt` preserve the recorded run/handoff history. Subsequent documentation edits are tracked by Git and do not rewrite that history.
