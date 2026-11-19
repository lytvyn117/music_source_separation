# Experiment index

The selected reference is Exp10 Raw at update 24,000. Later experiment numbers do not imply adoption.

| Experiment | Intervention | Outcome | Historical source |
| --- | --- | --- | --- |
| Baseline–Exp06 | MUSDB18 U-Net and objective/mask variants | Exp04 retained as legacy reference | [History](../archive/research_snapshot/history/EXPERIMENTS.md) |
| Exp07–Exp08 | Phase-aware residual and subband models | Not selected as the final standalone model | [Source](../archive/research_snapshot/src/experiments/) |
| Exp09 | Independent band-split Transformer on MoisesDB | Established the current architecture | [Snapshot](../archive/research_snapshot/exp09_scratch/) |
| Exp10 | Raw/EMA checkpoint evaluation | Selected reference; no fully qualified candidate | [Active configuration](../configs/training/exp10/config.json) |
| Exp11 | Cross-song remixing pilot | Not adopted | [Snapshot](../archive/research_snapshot/exp11_scratch/) |
| Exp12 | Stereo-loss pilot | Not adopted | [Snapshot](../archive/research_snapshot/exp12_stereo_loss/) |
| Exp13 | Data expansion audit | No training launch | [Snapshot](../archive/research_snapshot/exp13_data_expansion/) |
| Exp14 | One-second versus two-second inference hop | Retained two seconds | [Snapshot](../archive/research_snapshot/exp14_inference_hop/) |

New experiments should import the shared package and record their configuration, Git commit, hypothesis, metrics and decision. Avoid copying the complete model or trainer into each experiment folder.
