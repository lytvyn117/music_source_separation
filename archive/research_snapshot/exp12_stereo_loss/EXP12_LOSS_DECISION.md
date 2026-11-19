# Exp12 loss decision

**Selected option A.** Replace the denominator in the existing waveform Side L1 term. The hypothesis is that giving relative Side errors more gradient weight on Mid-heavy mixtures improves Side SDR while meeting separation and bleed guardrails. The diagnostics support testing that hypothesis. They do not establish the cause of the development Side scores or predict the size of any gain.

## Formulation

For each crop, let x be the stereo mixture and e the vocal estimate minus target. Use `S(a) = (aL − aR)/2`, `r² = max(mean over channel and time of x², 1e−8)` and `s² = mean over time of S(x)²`.

```text
d = sqrt(0.25 r² + 0.75 s²)
side_l1 = mean over crop and time of |S(e)| / d
L = 1.0 complex_mse + 0.5 logmag_l1 + 2.0 wave_l1 + 0.25 side_l1 + 0.25 leakage
```

The fixed 0.25 power floor gives `0.5r ≤ d ≤ r`. The Side gradient of each example in output space is its baseline gradient multiplied by `r/d ∈ [1, 2]`. The twofold cap limits the perturbation on mono-like crops and bounds the weighted Side L1 output-gradient norm at 35.4% of the weighted waveform L1 norm for the same batch. The bound is 25% when both channel errors are nonzero wherever Side error is nonzero. These bounds follow from the 0.25 and 2.0 coefficients, the two-channel L1 gradient, and r/d ≤ 2. No development score was used to choose the cap. Across 1,044 approved crops, the gain was 1.52× / 1.80× / 1.96× at p05 / p50 / p95. The intervention mainly raises the overall Side weight in this roster and modestly redistributes it across crops. The pilot tests this joint reweighting of one term and cannot attribute a gain specifically to the redistribution.

Mixture Side power needs no target-energy denominator. For silent mixtures the baseline 1e−8 FP32 mean-square floor gives `r ≥ 1e−4` and `d ≥ 5e−5` in full-scale sample units. This retains bounded arithmetic and applies at most twice the existing inverse-scale factor. Silent vocal, instrumental, or Side targets require no masking. They still penalize erroneous Side output. Anti-phase input has `s = r` above the floor, so it receives baseline weighting. A unilateral input receives `r/d ≈ 1.265` above the floor. A common gain cancels between numerator and denominator above the baseline RMS floor. Between or below floor regimes, exact gain invariance is impossible with this absolute floor. Both power and division are FP32 with autocast disabled, and the mixture scale has no dependence on separator parameters. The L1 subgradient is zero at exact Side reconstruction.

## Alternatives considered

| Option | Mathematical and gradient effect | Stability and bleed | Compute and compatibility |
| --- | --- | --- | --- |
| **A. Bounded Side-aware normalization** | Changes the per-example weighting of the existing L1 gradient. The measured batch Side gradient remains nearly collinear with baseline. This is consistent with the diagnosed mechanism. | Common relative floor bounds amplification, including mixture cancellation or nearly silent targets. Mixed signs for Side versus leakage parameter cosines make bleed monitoring necessary. | One Side waveform and reduction, `O(BT)`. A full-size B=4 Side tensor is about 2.7 MiB in FP32. No new FFT or model state. With complementary outputs either stem supplies the same Side error magnitude. |
| **B. Mid/Side multi-resolution STFT** | A reasonable design is mean over resolutions and both M/S components of separate normalized complex-error or magnitude terms. Complex error penalizes phase and waveform mismatch. Magnitude-only loss permits wrong Side phase and can optimize a different direction from Side SDR. Multiple windows also alter frequency and time weighting, widening the hypothesis. | Separate denominators require relative and absolute floors, especially for silent or cancellation-heavy Side. Increased pressure on particular spectral bins could shift vocal leakage. The probe provides no evidence for any particular resolution or spectral weighting. | Extra forward FFTs, saved complex tensors and backward FFTs for multiple resolutions. Existing complex MSE and two-stem log-magnitude already supply related spectral gradients. At one resolution, linearly rotating equally weighted complex L2 from L/R into M/S merely changes scale. Mixture consistency still holds. |
| **C. Existing Side L1 plus low-weight M/S spectral term** | Preserves the waveform proxy while adding resolution-dependent gradients, which can double-count Side and increase Mid pressure. A small coefficient cannot guarantee a small parameter gradient. | Needs the same safeguards as B and separate gradient calibration. Conflict with leakage would depend on the extra term. | Similar FFT and activation overhead to B even with a small coefficient. Additive outputs remain enforced. The experiment would give weaker evidence on whether the denominator was the operative mechanism. |

Multi-resolution STFT has precedent in waveform generation. Its effectiveness there does not establish a controlled improvement for this separator. [Parallel WaveGAN, Yamamoto et al.](https://arxiv.org/abs/1910.11480)

