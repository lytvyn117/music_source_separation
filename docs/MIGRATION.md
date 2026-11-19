# Layout migration

The original clean Git state is tagged `pre-layout-refactor-2026-11-19`. No model weights or dataset were moved.

| Previous path | Active replacement |
| --- | --- |
| `exp10_scratch/exp09_core/scratch_model.py` | `src/music_separation/models/bs_roformer.py` |
| `exp10_scratch/infer.py` | `python -m music_separation.inference.separate` |
| `exp10_scratch/reproduce.py` | `python -m music_separation.cli` |
| `exp10_scratch/config.json`, `roster.json` | `configs/training/exp10/` |
| `exp10_scratch/development_cache/` | `data/development_cache/` |
| `exp10_scratch/runs/` | `runs/` |
| `PRESENTATION_2026-11-23/run_demo.ps1` | `demos/run_demo.ps1` |
| Original source, tests and reports | `archive/research_snapshot/` with their relative layout preserved |

Use an editable installation from this checkout. Configuration files remain repository resources; distributing a standalone wheel is not currently supported.

New training runs hash all shared package sources plus the frozen configuration and roster. Their provenance schema is 3, so they are intentionally distinct from archived schema-2 runs. Do not resume an archived run with refactored source. Use the original Git tag and matching environment for historical recovery. Existing frozen inference weights retain their original file and tensor digest checks.

Generated caches and runs, if present in the historical snapshot, are not automatically migrated. Prepare a verified cache for the new layout; retain original runs for historical recovery. The final checkpoint under `models/` remains where it was.

The historical audit entry points may require omitted manifests and original paths. Active training uses the frozen roster, not regenerated audit decisions. Original checksum/provenance files are evidence and were not regenerated to describe the new tree.

The original introductory notebooks remain in the top-level `notebooks/` directory as part of the project narrative, rather than in the research archive.
