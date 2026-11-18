# Exp12 pilot decision

**NO_GO**

## Pre-registered rules

The following criteria are from [EXP12_DESIGN.md](EXP12_DESIGN.md).

> For **each** arm, require vocal and instrumental SDR deltas ≥ −0.10 dB and absolute vocal bleed delta ≤ +1.00 percentage point. Predeclare meaningful Side improvement as **≥ +0.20 dB on each Side SDR**, with both Side deltas strictly positive in at least **four of seven** artist groups.

> Verify all 6,000 steps are present, finite and have the exact Exp10 crop addresses and shard order. Any unstable step, failed validation, or mixture-consistency regression blocks GO.

> Apply the existing inference residual check per song. Compare every song and the panel maximum against archived residuals with one FP32 machine epsilon at full scale (1.1921e−7) tolerance for addition/subtraction rounding.

> Require both arms to clear the screening thresholds, then review the overall trade-off.

The tolerances apply to artist macros. The integrity review also checked the frozen split, settings, validation and serialized state. Both Side deltas track the same error power under complementary reconstruction.

## Gate outcomes

| Pre-registered gate | Raw | EMA |
| --- | --- | --- |
| Vocal SDR ≥ −0.10 dB | Pass | Pass |
| Instrumental SDR ≥ −0.10 dB | Pass | Pass |
| Absolute vocal bleed ≤ +1.00 pp | **Fail** | Pass |
| Vocal Side SDR ≥ +0.20 dB | Pass | **Fail** |
| Instrumental Side SDR ≥ +0.20 dB | Pass | **Fail** |
| Both Side deltas strictly positive in ≥ 4/7 groups | Pass | **Fail** |
| Mixture consistency per song and panel maximum | Pass | Pass |
| Finite computation, valid checkpoints and matching protocol | Pass | Pass |

Raw improved Vocal Side SDR by +0.3973 dB and Instrumental Side SDR by +0.3973 dB, with both positive in 7/7 groups. Absolute vocal bleed rose by +5.9757 pp, exceeding the maximum tolerated increase by 4.9757 pp. EMA improved bleed by 0.5268 pp and passed both whole-stem SDR tolerances. Its Side scores fell by 0.1788 dB and both improved in only 3/7 groups. Each arm therefore failed the screen for a different reason.

The joint Raw pattern supports a measured trade-off on this panel. A conflict between Side and leakage gradients was observed in earlier local diagnostics and is a plausible contributor. The pilot provides no direct evidence that this conflict caused the bleed regression, or that EMA caused the loss of Side gain through any specific mechanism. Seed and artist coverage are limited. A 6,000-update result does not establish the effect at 24,000.

Retain the verified Exp10 Raw step-24,000 best-overall checkpoint as the reference before November 23, 2026. Its hash matches Exp10's `latest.pt`. Exp10 recorded no checkpoint that passed its separate qualification guardrails. This checkpoint is the retention recommendation and the comparison in this report remains at update 6,000. [Matched results](EXP12_PILOT_RESULTS.md)
