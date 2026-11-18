# EXP14 decision

**KEEP_2S_HOP**

The 1-second hop did not meet the preset benefit rule, and the 2-second hop ran faster with both weight sets.

- **RAW** changed vocal SDR by +0.0058 dB (3/7 groups improved), instrumental SDR by +0.0046 dB, and absolute bleed by -0.0058 percentage points. Side SDR changed by +0.0102 dB for vocals and +0.0102 dB for instrumental. Inference took 106.56 s at 2 s and 212.48 s at 1 s. All regression limits passed.
- **EMA** changed vocal SDR by +0.0108 dB (4/7 groups improved), instrumental SDR by +0.0422 dB, and absolute bleed by -0.0074 percentage points. Side SDR changed by +0.0062 dB for vocals and +0.0062 dB for instrumental. Inference took 107.33 s at 2 s and 213.12 s at 1 s. All regression limits passed.

All paired predictions and metrics were finite. Every song passed the archived mixture-consistency check. The external final holdout remained sealed. [Design and decision criteria](EXP14_DESIGN.md) · [Full measurements](EXP14_RESULTS.md)
