# Current architecture

Exp10 uses a shared `Separator` implementation in `src/music_separation/models/bs_roformer.py`.

Stereo waveform at 44.1 kHz → complex STFT → 62 band projections → six time/band attention pairs with RoPE → complex vocal mask → inverse STFT → vocal waveform. Instrumental is mixture minus vocals.

The model has 11,662,596 trainable parameters, width 256 and eight attention heads. Training crops are four seconds. Inference uses four-second windows with a two-second hop and positive Hann overlap-add. The complex mask magnitude is bounded by 2. STFT/iSTFT are part of the model; there is no separate learned waveform branch.

The training objective combines complex MSE, log-magnitude L1, waveform L1, Side L1 and vocal leakage. Configuration and frozen roster are in `configs/training/exp10/`. The training set has 60 MoisesDB recordings; development has 16 recordings across seven disjoint artist groups.

This layout refactor preserves model computation and checkpoint tensor keys. It changes Python import paths, artifact locations and the source identity of new training runs.
