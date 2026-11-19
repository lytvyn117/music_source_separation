# Exp11 controlled remix pilot

1. Bind the archived eligibility and development evidence to the actual 60-song roster. Audit all 240 metadata rows, the exact Exp10 trajectory and its 6,000-update reference.
2. Test a deterministic, separately seeded remix address before adding the sampler. Keep Exp10's shard/UUID/offset as the vocal address. On the remix branch, draw an instrumental UUID uniformly from the other 59 and draw its offset independently. Stage the 60 approved float32 stereo references once using the strict Exp10 loader, with a calculated storage bound. The trainer remains single-process.
3. Keep the seed, model, objective, optimizer, 24,000-update scheduler, EMA, validation and selection. Give the experiment a new provenance identity. End the pilot at the committed update 6,000. Reuse the existing development cache read-only and report supplementary metrics through the same development inference pass.
4. Run synthetic CPU checks, compare data tensors on actual training crops, and measure amplitude and stereo statistics on approved audio. Verify matched-step evaluation and preservation evidence. The launch and resume commands require explicit later authorization.
