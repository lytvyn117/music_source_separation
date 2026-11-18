# Exp10 implementation plan

**Goal** Prepare a random-initialization controlled replay of Exp09 with development-selected EMA checkpoints and a 24,000-update cap.

**Architecture** Vendor the exact Exp09 training primitives and derive a minimal runtime roster from the completed run and frozen panel. Keep Exp09 and the backup outside the training data path. Reuse the Exp09 positive Hann overlap-add and frame SDR definitions on all 16 admitted development recordings.

- [x] Preserve all 156 Exp09 files and verify the backup against the source, published checkpoint and scoring provenance.
- [x] Build the audio allowlist from the hash-bound 60-song roster and 16 admitted development recordings. Check role, artist and stem disjointness. Verify frozen frames, BWF evidence and expected reference hashes. Publish a validated, reusable development cache.
- [x] Add tests for artist weighting, metric conventions, guardrails, early stopping, EMA isolation, tensor/RNG resume and recoverable publication.
- [x] Implement the Exp10 launcher and bind configuration, code, dataset and GPU runtime in the run identity. Keep Exp09 raw optimizer and crop addressing identical.
- [x] Commit immutable checkpoint bundles at update zero and every 1,000 updates. Enforce one writer and derive latest, best and validation history from committed bundles.
- [x] Run the CPU correctness suite and a capped real-data GPU preflight, then report timings, space, reproducibility, seal limits and the exact command for a later authorized run.
- [x] Audit potential Exp11 additions in parallel against the frozen split and technical QC evidence.
