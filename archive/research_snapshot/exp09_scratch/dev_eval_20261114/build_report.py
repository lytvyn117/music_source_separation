"""Render a report from the frozen panel and completed paired measurements."""
from collections import Counter
from math import log10
from pathlib import Path
import json
import statistics

P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text(encoding="utf-8"))
d=read("results.json")
f=read("frozen_panel.json")
provenance=read("checkpoint_provenance.json")
models=("exp04","hybrid","exp09")
rows=d["rows"]
a=d["aggregate"]
p=d["paired"]
total=sum(r["samples"] for r in rows)/44100
title=lambda x:x.replace("|","\\|")
labels={"exp04":"Exp04","hybrid":"Hybrid","exp09":"Exp09"}
q=[]
q.append("""# Exp09 first full development evaluation

14 November 2026

**Decision.** Exp09 reduces measured vocal bleed consistently. The paired vocal SDR intervals do not establish superiority over either baseline. The hybrid has better stereo side fidelity on this panel, and artist-weighted instrumental SDR favors it on the point estimate. Keep Exp09 as a research candidate. The single next step is a blind, matched listening review of the ten exported examples, focused on audible vocal bleed, stereo and backing quality.""")
q.append(f"""## Checkpoint and dataset

The final Exp09 checkpoint was copied to [exp09_step24000.pt](exp09_step24000.pt), marked read-only in Windows and verified against the original before and after scoring. This is a local read-only copy with a recorded hash, without hardware-enforced write-once storage. Its internal update is **24,000**, and the strict model load used **11,662,596 trainable parameters**. The optimizer state was not used for inference.

- Exp09 SHA-256 `{provenance["exp09"]["sha256"]}`
- Run identity `{provenance["training_identity"]}`
- Exp04 epoch 19 SHA-256 `{provenance["baselines"]["exp04"]["sha256"]}`
- Exp08 epoch 53 SHA-256 `{provenance["baselines"]["exp08"]["sha256"]}`

[Checkpoint provenance](checkpoint_provenance.json) records the full run signature. [Execution freeze](execution_freeze.json) binds the panel, protocol, code and inputs.

**16 of 20 songs** passed the audit, from **seven normalized artist groups**, five genres, and **{total/60:.2f} minutes** of audio. Seven of the 16 songs are by Sunny Side Up. There were zero matching song UUIDs and zero normalized artist matches between the frozen panel and Exp09's 60 training recordings. There were also zero normalized artist matches against the MUSDB18 roster. The recording screen compared the panel internally and with the 60 Exp09 training recordings and 100 MUSDB18 official-train recordings, covering the baseline fit and legacy selection sets. It used exact source PCM hashes against Exp09 training, normalized artist and title comparisons, and offset-aware spectral fingerprints on the mixtures. It found no matching training stems or acoustic candidates requiring follow-up. Heavily altered versions can evade the screen, and artist aliases are limited to available credits. The official MUSDB18 test roster was checked by metadata. No official test or external final audio was opened.

The audit decoded 213 WAVs on the 20-song development roster, checked their ZIP CRCs, classified vocal and instrumental tracks using the source labels, and inspected BWF starts, rate, channels, lengths, and nonzero source energy. All 16 admitted songs use their strict native spans, with stereo 44.1 kHz audio and matching timestamps within each song. Independently read source sums matched the reference mixture at the tested positions with zero measured residual. MoisesDB supplies multitracks. The mixture is their sum, and the audit did not have a separate mastered mix or an independent stem annotation to compare against. [Frozen per-song audit](frozen_panel.json)

| Excluded before inference | Artist | Reason |
|---|---|---|""")
for row in f["songs"].values():
    if row["reasons"]:
        q.append(f"| {title(row['title'])} | {title(row['artist'])} | {'; '.join(row['reasons']).replace('; ', ' and ')} |")
q.append("""No replacement songs were selected. Public terms provide MoisesDB under CC BY-NC-SA 4.0 for noncommercial research. The licensed audio and derived listening clips are stored locally with attribution and modification notes. No separate acquisition agreement was available for this audit. Any broader use or redistribution should be checked against the applicable terms. [Provider terms](https://music.ai/research/#datasets) · [MoisesDB license](https://github.com/moises-ai/moises-db#licensing) · [CC legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) · [Audio attribution](ATTRIBUTION.md)

## Scoring protocol

Every arm received the same original-gain stereo input. Exp04 used the pinned native 6-second adapter and original outputs. The hybrid averaged Exp04 and Exp08 vocals at fixed 0.5/0.5 weights, with instrumental obtained by subtraction. Exp08 used its pinned 10-second adapter. Exp09 used four-second segments, 50% overlap, positive Hann overlap-add, and a complementary instrumental. Inference was deterministic FP32 on the RTX 3080 Ti. The model-specific chunk sizes are part of the frozen inference paths.

SDR and SI-SDR use the same repository implementation on complete one-second frames with reference RMS ≥ −45 dBFS. Each song contributes its median active-frame score. The artist macro averages song scores within each artist and then weights artists equally. Paired 95% intervals resample the seven artists with replacement, retaining the paired observations, over 10,000 draws with seed 42. Positive deltas favor Exp09 for SDR. Lower absolute bleed favors Exp09. No gain fitting for SDR, wave alignment adjustment, clipping, or loudness normalization was applied. SI-SDR uses its standard scale projection.

Bleed is the projection of instrumental error onto reference vocals, divided by vocal reference energy. Its squared coefficient gives the percentage of vocal reference energy in that projected component. Vocal retention is projection of the vocal output onto its reference. For complementary estimates, retention and signed bleed are nearly algebraic complements, so they provide closely related evidence. Side metrics use (L−R)/2, with −50 dBFS reference-side eligibility. Buried vocal frames have vocals at or below −10 dB relative to the backing while meeting the −45 dBFS vocal activity gate.""")
metrics=[
 ("Vocal SDR, dB","vocal_sdr_db",1),("Instrumental SDR, dB","instrumental_sdr_db",1),
 ("Vocal SI-SDR, dB","vocal_si_sdr_db",1),("Signed vocal bleed coefficient, %","vocal_bleed_coeff",100),
 ("Absolute vocal bleed coefficient, %","abs_vocal_bleed_coeff",100),
 ("Projected vocal bleed energy, %","vocal_bleed_energy_pct",1),("Vocal retention, %","vocal_retention",100),
 ("Vocal side SDR, dB","v_side_sdr_db",1),("Instrumental side SDR, dB","i_side_sdr_db",1),
 ("Vocal RMS level error, dB","v_level_error_db",1),("Instrumental RMS level error, dB","i_level_error_db",1),
 ("Vocal side level error, dB","v_side_level_error_db",1),
 ("Instrumental side level error, dB","i_side_level_error_db",1),
 ("Buried vocal SDR, dB","buried_vocal_sdr_db",1)]
q.append("""## Full-song results

Each cell gives the song median / artist macro.

| Metric | Exp04 | Hybrid | Exp09 |
|---|---:|---:|---:|""")
for lab,k,scale in metrics:
    vals=[f"{a[n][k]['median']*scale:.3f} / {a[n][k]['artist_macro']*scale:.3f}" for n in models]
    q.append(f"| {lab} | {' | '.join(vals)} |")
q.append("""### Paired Exp09 differences

Wins, ties and losses use a 0.01 dB tolerance for SDR and 0.01 percentage point for bleed. A regression below −1 dB is reported for SDR metrics.

| Metric and comparator | Artist-macro delta [95% CI] | Paired song median | W–T–L | >1 dB regressions |
|---|---:|---:|---:|---:|""")
for lab,k,scale in (("Vocal SDR","vocal_sdr_db",1),("Instrumental SDR","instrumental_sdr_db",1),
                    ("Vocal SI-SDR","vocal_si_sdr_db",1),("Absolute bleed, pp","abs_vocal_bleed_coeff",100),
                    ("Bleed energy, pp","vocal_bleed_energy_pct",1),("Vocal side SDR","v_side_sdr_db",1)):
    for b in models[:2]:
        t=p[b][k]
        lo,hi=[x*scale for x in t["ci95_artist_macro_delta"]]
        regress=t["regressions_gt_1db"] if t["regressions_gt_1db"] is not None else "—"
        q.append(f"| {lab} vs {labels[b]} | {t['artist_macro_delta']*scale:+.3f} [{lo:+.3f}, {hi:+.3f}] | {t['paired_median']*scale:+.3f} | {t['wins']}–{t['ties']}–{t['losses']} | {regress} |")
q.append("""### Song-level SDR and regressions

| Song | Artist | Exp04 vocal | Hybrid vocal | Exp09 vocal | Δ vocal vs Exp04 | Δ vocal vs Hybrid | Δ instrumental vs Exp04 | Δ instrumental vs Hybrid |
|---|---|---:|---:|---:|---:|---:|---:|---:|""")
for r in rows:
    v=[r[m]["vocal_sdr_db"] for m in models]
    i=[r[m]["instrumental_sdr_db"] for m in models]
    q.append(f"| {title(r['title'])} | {title(r['artist'])} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} | {v[2]-v[0]:+.2f} | {v[2]-v[1]:+.2f} | {i[2]-i[0]:+.2f} | {i[2]-i[1]:+.2f} |")
q.append("""[Complete song CSV](per_song.csv) · [All metrics, artist aggregates and paired bootstrap results](results.json)

### Signal integrity and artifact checks""")
for n in models:
    maxres=max(r[n]["mixture_max_abs"] for r in rows)
    interior=max(r[n]["mixture_interior_max_abs"] for r in rows)
    peak=max(max(r[n][k] for k in ("v_peak","i_peak")) for r in rows)
    timing=sum(r["exp04_seconds"]+r["exp08_seconds"] if n=="hybrid" else r[f"{n}_seconds"] for r in rows)
    q.append(f"- **{labels[n]}** had peak output {peak:.4f} and maximum sample-wise mixture residual {maxres:.3g}, falling to {interior:.3g} after excluding the first and last second. Synchronized inference took {timing:.2f} s for {total:.1f} s of audio, excluding staging, metrics and file writes.")
quiet=[20*log10(a[n]["below_threshold_vocal_rms"]["median"]) for n in models]
q.append(f"""
No output samples reached full scale in any arm. Exp04's larger maximum full-track residual is confined to its native boundaries. The hybrid and Exp09 use a complementary sum. Median first-difference energy errors in dB for vocals were −0.79 / −1.18 / −0.59 (Exp04 / Hybrid / Exp09). For backing they were −0.38 / −0.52 / −0.61. These describe high-frequency energy differences and do not establish audible artifact rates.

On below-threshold vocal frames, median vocal-output RMS was **{quiet[0]:.1f} / {quiet[1]:.1f} / {quiet[2]:.1f} dBFS**. Exp09 leaves more low-activity output and scores lower on buried vocals and stereo side detail. This can coexist with improved retention and reduced whole-song leakage into the backing. The exports support a controlled perceptual check. No listening preference is reported as measured. Peak PyTorch allocation/reservation for the evaluation process was **{d["gpu_peak_allocated_gib"]:.2f} / {d["gpu_peak_reserved_gib"]:.2f} GiB**.

## Ten matched listening examples

Each folder contains nine 12-second stereo float WAVs at 44.1 kHz. Files include the original additive mixture, both references, and both outputs per model. Every clip uses the same sample range and native gain. Offsets were selected before inference based on the reference activity, with all 12 seconds jointly active in each clip. These exports emphasize overlapping vocals and accompaniment.

| Example | Song | Artist | Start |
|---|---|---|---:|""")
for j,r in enumerate(d["listening"],1):
    q.append(f"| [{j:02d}]({r['directory']}/mixture.wav) | {title(r['title'])} | {title(r['artist'])} | {r['start_frame']/44100:.0f} s |")
q.append(f"""
## What limited Exp09?

The 12-song in-sample diagnostic used the prior probe's optimizer recordings and the final Exp09 weights. Artist-macro vocal SDR was **{d["fit_aggregate"]["vocal_sdr_db"]["artist_macro"]:.2f} dB** in sample versus **{a["exp09"]["vocal_sdr_db"]["artist_macro"]:.2f} dB** on development. The instrumental gap was larger, **{d["fit_aggregate"]["instrumental_sdr_db"]["artist_macro"]:.2f} versus {a["exp09"]["instrumental_sdr_db"]["artist_macro"]:.2f} dB**, but backing-to-vocal ratios and song difficulty differ across the sets. The vocal gap gives little support for a broad memorization explanation. Training used 60 songs across 26 artists and about 3.62 hours. Limited coverage is plausible, particularly with gains concentrated in Strugzy and Horizon and regressions on some other artists. The measured stereo and quiet-vocal regressions are more direct findings. One final checkpoint and no validation curve cannot distinguish late overfitting from training distribution, target behavior or other causes.

The run performed inference only. No additional training, parameter tuning or final test evaluation took place.""")
(P/"REPORT.md").write_text("\n".join(q)+"\n",encoding="utf-8")
