# Exp13 dataset expansion decision

**NOT_READY**

Zero additional recordings are eligible under the current protocol. Seven are technically recoverable by a reviewed common-span policy. Their identity checks remain incomplete against the active external final, which has no identity manifest or independently adjudicated comparison record in the project. A controlled pilot is premature. [Dataset audit](DATASET_AUDIT.md)

## Potential gain

| Cohort | Recordings | Additional usable hours | Increase over 3.62 h | New training artists |
| --- | ---: | ---: | ---: | ---: |
| Fully eligible additions | 0 | 0 | 0% | 0 |
| Conditionally eligible, all seven cleared | 7 | 0.49562 | 13.69% | 2 |
| Ineligible under current evidence, outside the protected group | 137 | 0 approved | 0% | 0 approved |
| Ineligible because of frozen development or reserve assignment | 36 | 0 approved | 0% | 0 approved |

The baseline has 574,143,200 frames, or 3.61642 h, across 60 recordings and 26 normalized artist groups. The seven sum to 78,684,776 common frames, or 29.74 minutes. With all seven, the total would be 67 recordings, 4.11204 h and 28 groups. The gain against the unrounded baseline is 13.70%. The candidates span six groups, of which Sour Cherry and The Tainted are new. Five are rock, one rap and one singer-songwriter. Those genres are already present. Firefly alone would add 0.07034 h, about 1.95% by unrounded duration, and zero new groups. This is a modest and uneven diversity change with no measured effect on separation. [Recording ledger](RECOVERABLE_RECORDINGS.csv)

## Activity and stereo

| Full shared-span RMS per recording, dBFS | Exp10 strict min / median / max | Seven candidates min / median / max |
| --- | ---: | ---: |
| Vocal | −36.86 / −31.27 / −23.82 | −36.02 / −30.48 / −29.20 |
| Instrumental | −30.21 / −24.65 / −20.68 | −31.23 / −27.22 / −23.61 |

For the seven, the span from first to last sample exceeding an absolute amplitude of 1e−5 covers 78.83% / 92.14% / 100.00% of the vocal reference at min / median / max. The instrumental values are 99.98% / 100.00% / 100.00%. These envelopes can contain quiet gaps. Full-span RMS measures level. Neither yields the percentage of seconds above a frame-activity threshold.

In the live Firefly check, 172/253 complete seconds (68.0%) were vocal-active, 233/253 (92.1%) instrumental-active and 161/253 (63.6%) jointly active at the development threshold of −45 dBFS stereo RMS over one second. Vocal, instrumental and mixture Side-to-Mid power ratios were −12.68, −8.36 and −8.90 dB respectively, with corresponding left/right normalized correlations of 0.898, 0.753 and 0.777. Side is (L−R)/2 and Mid is (L+R)/2. The six archive-only candidates are all native stereo. Their complete-second activity and quantitative stereo width or correlation were unavailable in the saved records. No cohort-wide stereo or sampled activity shift can be estimated responsibly from one song. [Recording ledger](RECOVERABLE_RECORDINGS.csv)

## What would change the decision by November 23

1. Obtain an identity-only clearance report from the final-panel custodian. Compare the existing 60, the proposed additions and their linked credits with the sealed panel via UUIDs or equivalent recording IDs, recording aliases, artist and contributor groups, and pre-existing approved fingerprints where available. Keep final audio outside Exp13. If the comparison requires access to the sealed material and no authorized custodian can supply the result, the candidates remain conditional.
2. Complete the development near-duplicate screen for the six archive-only candidates using cleared fingerprint evidence. Record reserve metadata coverage, any remaining transformed or partial-stem duplicate risk, and adjudication of aliases. Confirm noncommercial research terms and acquisition provenance.
3. Approve a bounded, native common-span reference policy for the cleared IDs. Freeze the final roster, verify loading and sampling, and lock thresholds before any pilot result is available. Quantify one-second vocal, instrumental and joint activity and stereo profile for the final proposed roster before authorization. Compute resources fit within the deadline if these evidence gaps close. [Conditional design and resource budget](EXP13_DESIGN.md)

## Highest-information experiment available now

Use the archived final Exp10 Raw and EMA checkpoints for a paired inference-only overlap experiment on the original 16 development recordings. Reproduce the four-second, two-second-hop reference, then evaluate a one-second hop with the same four-second context, checkpoint, positive Hann weighting, references and metric implementation. Before scoring, fix the two hops and report per-song and per-artist vocal and instrumental SDR, both Side SDRs, bleed, overlap-region versus song-interior Side error, and inference time. This tests whether greater coverage across chunk positions reduces residual Side error and exposes an inference trade-off with no optimizer updates. Improvement would justify further evaluation of inference overlap. A flat result would lower the priority of overlap changes and leave data expansion dependent on independent clearance. The full panel costs about two minutes per weight set at the baseline hop, so budget approximately 12–15 minutes for both hops and weight sets on the archived RTX 3080 Ti setup, plus analysis. This experiment is a recommendation only. [Exp10 inference and performance](../exp10_scratch/README.md) · [Available checkpoint provenance](../exp12_stereo_loss/EXP12_DIAGNOSTICS.md)

