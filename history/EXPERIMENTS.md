# Experiment history

The project moved from MUSDB18 U-Net fine-tuning to a separately initialized band-split Transformer on the strict MoisesDB roster. The table records the intervention or configuration that distinguished each experiment. Historical development scores use their respective cohorts and protocols.

| Experiment | Relevant configuration and outcome |
| --- | --- |
| Baseline | Stereo STFT U-Net trained on MUSDB18. Learned magnitude masks and reconstructed with mixture phase. |
| Exp01 | Continued the baseline using a composite loss and training remixes. The architecture stayed in the U-Net family. |
| Exp02 | Continued Exp01 with a lower-rate schedule from 1e−4 to 1e−6 and 1,500 crops per epoch. |
| Exp03 | Continued Exp02 with 2,500 crops per epoch and a 5e−5 to 5e−6 schedule. |
| Exp04 | Branched from Exp03, used 4,000 crops per epoch and a 7e−5 to 7e−6 schedule, and added asymmetric vocal leakage weight 1.0. Epoch 19 became the legacy reference. |
| Exp05 | Used the same Exp03 initialization and Exp04 schedule but increased leakage weight to 3.0. The evaluated candidate underperformed Exp04 in vocal SDR. |
| Exp06 | Warm-started from Exp04 and switched from coupled softmax masks to independent sigmoid masks. Used learning-rate endpoints 3e−5 and 3e−6. Exp04 remained preferred on the legacy panel. |
| Exp07 | Added a zero-initialized phase-aware residual CNN to frozen Exp04. The pilot stopped in its first epoch when active-bin p99 instrumental amplification exceeded the +3 dB guard. |
| Exp08 | Trained a compact three-subband complex-mask model from seed 42 on MUSDB18 for 60 epochs. Exp04 remained the preferred standalone model. A fixed 50/50 Exp04/Exp08 vocal-waveform blend was retained as a comparison, with backing reconstructed by subtraction. |
| Exp09 scratch | Introduced the 62-band, six-pair Transformer and the 60-recording strict MoisesDB training roster. Seed 909, four four-second crops per update, normalized complex MSE composite loss, and 24,000 updates of AdamW with cosine decay. Scored the final Raw endpoint on 16 development songs across seven artists. |
| Exp10 | Repeated Exp09's training recipe from scratch with EMA decay 0.999, Raw/EMA evaluation and immutable checkpoint retention every 1,000 updates. Step 24,000 Raw won best overall at 7.0884 dB vocal SDR. No candidate passed all guardrails. The archived Exp09 and Exp10 Raw states matched tensor-for-tensor at this boundary. |
| Exp11 | Matched 6,000-update pilot, with deterministic cross-song accompaniment on 50% of crops and a 24,000-update learning-rate horizon. Raw bleed increased 5.0218 percentage points. EMA vocal and instrumental SDR dropped 0.1203 and 0.2553 dB. The pilot did not advance. |
| Exp12 | Matched 6,000-update stereo-loss pilot. Only the Side L1 denominator changed, from mixture RMS to `sqrt(0.25 * mixture_power + 0.75 * mixture_side_power)`. Raw vocal Side SDR gained 0.3973 dB with a 5.9757 percentage-point bleed increase. EMA lost 0.1788 dB of Side SDR. The pilot did not advance. |
| Exp13 | Audited a data-only expansion. Seven technically recoverable recordings represented 29.74 minutes. Zero were approved. |
| Exp14 | Held the step-24,000 checkpoint, four-second window, references and reconstruction fixed while comparing one-second and two-second inference hops. The shorter hop improved Raw/EMA artist-macro vocal SDR by 0.0058/0.0108 dB at roughly twice the runtime. Retained the two-second hop. |

The executed Exp09 hyperparameters are embedded in [the Exp10 configuration](../exp10_scratch/config.json). The [frozen run excerpt](../provenance/frozen_run_excerpt.json) records Exp10's runtime and production source hashes. [Completion](../provenance/completion.json) and [endpoint metrics](../provenance/development_endpoint.json) record the final selection and the matched development results. The historical results establish development performance on the stated panels. The external final holdout remained sealed.
