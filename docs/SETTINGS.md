# Settings ownership

| Location | Purpose | Change policy |
| --- | --- | --- |
| `src/music_separation/settings.py`: `MODEL` | Fixed Exp10 architecture and audio profile: sample rate, STFT, bands, width, heads, depth and segment size | Not an inference override. Changes define a different model and require compatibility checks or retraining. |
| `configs/training/exp10/config.json` | Frozen optimizer, scheduler, loss, EMA, selection and evaluation recipe | The trainer validates the recorded recipe; this is not an unrestricted hyperparameter interface. |
| `configs/training/exp10/roster.json` | Data membership, alignment and reference identities | Hash-bound experimental data contract. |
| `configs/inference/exp10.json` | Inference hop, thread count, seed and attention backends | Validated runtime options; use `--config` for an alternative profile. |
| `src/music_separation/paths.py` | Repository resource paths | No machine-specific dataset or checkpoint paths. |
| CLI arguments | Input, output, archive and checkpoint paths; Raw/EMA choice | Set per invocation. |
| `inference/separate.py` checkpoint digests | Identity of the trusted frozen checkpoint | Integrity metadata, not a model tuning option. |

Example after editable installation:

```powershell
python -m music_separation.inference.separate --config configs/inference/exp10.json --checkpoint models/final/best_overall.pt --input-audio song.wav --output-dir outputs/example
```

All default values and model tensor keys are unchanged. Runtime settings are written into the inference manifest. Unknown fields, invalid hops, invalid thread counts and unsupported attention backend names are rejected. CPU device selection is not implemented by this change; the inference CLI still requires CUDA.

The model profile is deliberately immutable Python data instead of a freely editable architecture JSON: even changes that preserve tensor shapes (for example the STFT hop) can invalidate the meaning of trained weights. Internal initialization and numerical formulas remain implementation details.

Training and development evaluation retain their fixed recipe and guardrails. An alternative inference configuration is not an alternative training configuration. The default inference configuration is included in new-run provenance; changing shared settings prevents accidental continuation under a different source identity. Historical snapshots and hashes remain unchanged.
