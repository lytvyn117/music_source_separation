"""Render the Exp12 audit artifacts from the recorded evidence."""
from collections import Counter, defaultdict
import csv
import json
from statistics import mean

from review import ARMS, BASELINE, EXP14, ROOT, RUN, OUT, file_hash, read_json, require

METRICS = {
    "vocal_sdr_db": "Vocal SDR",
    "instrumental_sdr_db": "Instrumental SDR",
    "v_side_sdr_db": "Vocal Side SDR",
    "i_side_sdr_db": "Instrumental Side SDR",
    "abs_vocal_bleed_coeff": "Absolute vocal bleed",
}
BLEED = "abs_vocal_bleed_coeff"
SIDE = "v_side_sdr_db"
STATUS = "NO_GO"
GATE_NAMES = {
    "vocal_sdr": "Vocal SDR ≥ −0.10 dB",
    "instrumental_sdr": "Instrumental SDR ≥ −0.10 dB",
    "absolute_vocal_bleed": "Absolute vocal bleed ≤ +1.00 pp",
    "vocal_side_sdr": "Vocal Side SDR ≥ +0.20 dB",
    "instrumental_side_sdr": "Instrumental Side SDR ≥ +0.20 dB",
    "artist_spread": "Both Side deltas strictly positive in ≥ 4/7 groups",
    "mixture_consistency": "Mixture consistency per song and panel maximum",
}
COMMANDS = {
    "from": str(ROOT),
    "test": r"& '..\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v",
    "documented_launch": r"& '..\.venv\Scripts\python.exe' -B -u .\train_exp12.py --authorize-pilot --out .\runs\exp12_pilot_seed909",
    "documented_matched_evaluator": r"& '..\.venv\Scripts\python.exe' -B -u .\evaluate_pilot.py --run .\runs\exp12_pilot_seed909 --output .\runs\exp12_pilot_seed909\matched_step6000.json",
    "fresh_matched_evaluation_and_integrity_audit": r"& '..\.venv\Scripts\python.exe' -B -u .\pilot_review_20261117\review.py audit",
    "two_second_replay": r"& '..\.venv\Scripts\python.exe' -B -u .\pilot_review_20261117\review.py profile",
}


def table(headings, rows):
    return "\n".join(
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in [headings, ["---"] * len(headings), *rows]
    )


def unit(key):
    return "pp" if key == BLEED else "dB"


def value(key, number, signed=False):
    if key == BLEED:
        number *= 100
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def duration(seconds):
    seconds = round(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min {seconds:02d} s" if hours else f"{minutes} min {seconds:02d} s"


def delta_stats(arm):
    result = {}
    for metric in METRICS:
        sign = -1 if metric == BLEED else 1
        pairs = {artist: row[metric]["delta"] for artist, row in arm["artists"].items()}
        counts = Counter((sign * delta > 0) - (sign * delta < 0) for delta in pairs.values())
        best = max(pairs, key=lambda artist: sign * pairs[artist])
        worst = min(pairs, key=lambda artist: sign * pairs[artist])
        result[metric] = {
            "improved": counts[1], "unchanged": counts[0], "regressed": counts[-1],
            "largest_improvement": {"artist": best, "delta": pairs[best]} if counts[1] else None,
            "largest_regression": {"artist": worst, "delta": pairs[worst]} if counts[-1] else None,
        }
    return result


def diagnostic_macro(rows, stem):
    groups = defaultdict(list)
    for row in rows:
        groups[row["artist_key"]].append(row[stem])
    return {component: mean(mean(s[component] for s in songs) for songs in groups.values())
            for component in ("left", "right", "mid", "side")}


def ensure_provenance(audit, profile):
    for name, sha in audit["protected_artifact_sha256"].items():
        require(file_hash(ROOT.parent / name) == sha, "Protected artifact changed since the audit")
    for name, sha in audit["evidence_sha256"].items():
        require(file_hash(RUN / name) == sha, "Archived run evidence changed since the audit")
    require(len(audit["training"]["checkpoints"]) == 7 and profile["hop_seconds"] == 2
            and profile["checkpoint_sha256"] == audit["matched"]["exp12_checkpoint"]["sha256"],
            "Review evidence mismatch")
    tests = (OUT / "unittest.log").read_text(encoding="utf-8-sig")
    require(tests.count(" ... ok") == 18 and "Ran 18 tests" in tests and tests.rstrip().endswith("OK"),
            "Incomplete test record")
    require(all(len(profile["arms"][arm]["songs"]) == 16 for arm in ARMS), "Incomplete replay")


def matchup(arms):
    rows = []
    for key, label in METRICS.items():
        score = []
        for arm in ARMS:
            r = arms[arm]["artist_macro"][key]
            score.extend([value(key, r["exp10"]), value(key, r["exp12"]), value(key, r["delta"], True)])
        rows.append([label, "%" if key == BLEED else "dB", *score])
    return table(["Artist-macro metric", "Unit", "Raw Exp10", "Raw Exp12", "Raw Δ",
                  "EMA Exp10", "EMA Exp12", "EMA Δ"], rows)


def artist_and_song_tables(arm, songs):
    counts = Counter(row["artist_key"] for row in arm["songs"].values())
    headers = ["Vocal SDR (dB)", "Instrumental SDR (dB)", "Vocal Side (dB)",
               "Instrumental Side (dB)", "Bleed (pp)"]
    artist_rows = [[artist, counts[artist], *(value(key, row[key]["delta"], True) for key in METRICS)]
                   for artist, row in arm["artists"].items()]
    song_rows = [[row["artist_key"], songs[key].replace("|", r"\|"),
                  *(value(metric, row[metric]["delta"], True) for metric in METRICS),
                  f'{row["mixture_max_abs"]["delta"]:+.2e}']
                 for key, row in sorted(arm["songs"].items(),
                                        key=lambda pair: (pair[1]["artist_key"], songs[pair[0]].lower()))]
    return "\n\n".join([
        "Artist deltas", table(["Artist group", "Songs", *headers], artist_rows),
        "Song deltas", table(["Artist group", "Song", *headers, "Mixture residual Δ"], song_rows),
    ])


def direction_table(stats):
    def extremum(key, row):
        return f'{row["artist"]} ({value(key, row["delta"], True)} {unit(key)})' if row else "—"

    return table(["Metric", "Improved / tied / regressed",
                  "Largest improvement", "Largest regression"], [
        [METRICS[key], f'{row["improved"]} / {row["unchanged"]} / {row["regressed"]}',
         extremum(key, row["largest_improvement"]), extremum(key, row["largest_regression"])]
        for key, row in stats.items()
    ])


def results_markdown(audit, profile, stats, songs, diagnostics):
    matched = audit["matched"]
    train = audit["training"]
    peak_grad = matched["stability_and_sampler"]["maximum_preclip_gradient_l2"]
    inference = profile["arms"]
    exp12_arms = matched["arms"]
    steps_seconds = train["update_compute_seconds"]
    validation_seconds = train["validation_total_seconds"]
    side = audit["archive_side_differences"]
    synthetic = audit["synthetic"]
    common_gain = synthetic["predicted_delta_db"]
    commands = "\n".join(COMMANDS[key] for key in (
        "test", "fresh_matched_evaluation_and_integrity_audit", "two_second_replay",
    ))
    main = f"""# Exp12 matched pilot results

**NO_GO.** This review verified the existing seed-909 pilot at update 6,000. The matched comparison uses Exp10 at update 6,000, the frozen 16-song development panel and seven artist groups. Song scores are averaged within artist and the seven artist means receive equal weight. Positive SDR deltas and negative bleed deltas favor Exp12. Bleed deltas use percentage points. Paired vocal SI-SDR is unavailable because the archived Exp10 matched-step history has no SI-SDR and its numbered checkpoint is ACL-restricted.

## Matched artist macros

{matchup(exp12_arms)}

Raw improved both whole-stem SDRs and gained {value(SIDE, exp12_arms['raw']['artist_macro'][SIDE]['delta'], True)} dB on the shared Side measure across all seven groups. Bleed rose in all seven. EMA also passed both whole-stem SDR tolerances and its bleed fell slightly. Its mean Side score declined, and only three groups improved. Raw gained the most Side SDR on sunnysideup and incurred the largest bleed penalty on evolvinghabits. The largest EMA Side loss was on andybennett. These are observed extrema on the panel.

## Side SDR metric dependence

The separator returns the instrumental estimate as the mixture minus vocals. The evaluator reconstructs vocals with positive Hann overlap-add and applies that same subtraction again to the full-song mixture, which is built from the two cached references. In exact arithmetic the stereo errors are opposites sample by sample, so their Side transforms are opposites and share one squared-error sum. In `validation.side_sdr`, each stem has its own reference Side energy in the numerator, a fixed reference activity gate and the same additive epsilon. With fixed references the numerator cancels from the change in dB. Both stem deltas therefore equal `10 log10((old Side error energy + ε)/(new Side error energy + ε))` in exact arithmetic. Artist and panel averaging preserve this equality because both metrics use the same complete set of songs.

The actual evaluator gives {synthetic['a'][SIDE]:.6f} and {synthetic['a']['i_side_sdr_db']:.6f} dB for synthetic prediction A and {synthetic['b'][SIDE]:.6f} and {synthetic['b']['i_side_sdr_db']:.6f} dB for prediction B. Both change by {common_gain:+.6f} dB with zero mixture residual. Both predictions have channel RMS errors 0.046875 and 0.015625. Between A and B, Side RMS halves from 0.03125 to 0.015625 while Mid RMS doubles from 0.015625 to 0.03125. This shows how Side error can change through the relationship between channel errors even when both channel RMS values stay fixed.

{table(['Maximum absolute difference between the two Side deltas (dB)', 'Raw', 'EMA'], [
    ['Exp11 across seven artist groups', *(f"{side[a]['exp11_max_artist_delta_difference_db']:.2e}" for a in ARMS)],
    ['Exp14 across 16 songs', *(f"{side[a]['exp14_max_song_delta_difference_db']:.2e}" for a in ARMS)],
    ['Exp14 across seven artist groups', *(f"{side[a]['exp14_max_artist_delta_difference_db']:.2e}" for a in ARMS)],
    ['Exp12 across seven artist groups', *(f"{side[a]['exp12_max_artist_delta_difference_db']:.2e}" for a in ARMS)],
])}

The small differences are consistent with independent FP32 arithmetic on the two complements. The equality is expected, and the audit found no metric blocker. Both preregistered endpoints supply one underlying reconstruction-error signal. Whole-stem SDR can differ because it takes medians over source-specific active frames. Bleed measures the projection of instrumental error onto the reference vocal.

## Per-channel and Mid/Side diagnostics

{table(['Arm', 'Stem', 'Left RMS', 'Right RMS', 'Mid RMS', 'Side RMS'], [
    [arm.upper(), label, *(f'{x:.6f}' for x in diagnostics[arm][stem].values())]
    for arm in ARMS for stem, label in (
        ('vocal_error_rms', 'Vocal'), ('instrumental_error_rms', 'Instrumental'))
])}

These are artist-macro averages of per-song absolute reconstruction RMS on the Exp12 replay. For each arm the complementary stems have equal channel and Mid/Side errors at the printed precision. Mid error is larger than Side in RMS units in both arms. The largest absolute sample of the sum of the complementary errors was {max(row['max_abs_sum_of_stem_errors'] for arm in ARMS for row in inference[arm]['songs']):.2e}. The archived Exp10 6,000-step metrics give paired Side error-power changes. Paired left, right and Mid absolute errors are unavailable. The earlier Exp12 diagnostics measured Mid/Side errors on 16 training crops at Exp10 update 24,000, a different panel and training boundary.

## Training and inference telemetry

- **Training.** {duration(train['file_timestamp_elapsed_seconds'])} from the first run record to the completion record, based on filesystem timestamps. Timed update computation totaled {duration(steps_seconds)}. Six paired validations totaled {duration(validation_seconds)}. The remaining elapsed time includes staging, checkpoint publication and uninstrumented overhead.
- **Training GPU.** RTX 3080 Ti, {train['peak_logged_allocated_gib']:.3f} GiB peak allocated and {train['peak_logged_reserved_gib']:.3f} GiB peak reserved in update logs. These PyTorch counters are sampled after updates and exclude a separate observation of the final validation peak.
- **Training stability.** All 6,000 update records and crop addresses match the frozen order. Logged losses and gradients are finite. The highest preclip gradient norm was {peak_grad:.3f}. The trainer clipped gradients to 1.0 and saved valid state at every boundary. The final learning rate was {train['checkpoints']['6000']['lr']:.11f} with a cosine horizon of 24,000. No compute-failure record exists.

{table(['Step-6,000 arm', 'Original full validation (s)', 'Replay inference (s)',
        'Chunks', 'Peak allocated / reserved (MiB)', 'Largest score difference'],
       [[arm.upper(), f"{train['step6000_validation_seconds_including_reference_io_and_metrics'][arm]:.2f}",
         f"{inference[arm]['inference_seconds']:.2f}", inference[arm]['chunks'],
         f"{inference[arm]['peak_gpu_allocated_mib']:.2f} / {inference[arm]['peak_gpu_reserved_mib']:.2f}",
         f"{inference[arm]['max_archived_score_difference']:.2e}"] for arm in ARMS])}

Original validation timing includes reference I/O and metric computation. The instrumented replay times model inference and overlap-add, with the 4-second window, 2-second hop, FP32 precision and original reconstruction. GPU peaks are PyTorch inference peaks on the replay. The pilot and baseline panel maximum mixture residual was 2.98e−8 for each arm. Every matched per-song residual passed the preregistered comparison.

## Artist and song deltas

Improvement counts use strict direction at full precision. There is no extra exploratory threshold. Extrema are descriptive. All baseline values, Exp12 values, deltas, song UUIDs and residuals are in [the results JSON](EXP12_PILOT_RESULTS.json).

### Raw

{direction_table(stats['raw'])}

{artist_and_song_tables(exp12_arms['raw'], songs)}

### EMA

{direction_table(stats['ema'])}

{artist_and_song_tables(exp12_arms['ema'], songs)}

## Protocol and reproducibility

- Exp10 and Exp12 have a byte-identical roster and identical model, sampler, strict WAV loader, checkpoint, validation and metric modules. The development cache passed reference-array and file hash verification. The active split has 60 approved train songs and 16 frozen development songs, disjoint by artist, UUID and file path. The external holdout stayed sealed.
- Step zero strictly loads the Exp10 seed-909 parameter digest for the 11,662,596-parameter model. Four 4-second crops per batch, accumulation one, AdamW, gradient clipping, EMA 0.999 and the original cosine schedule all match the established code, configuration and checkpoints. There is no remix, gain/EQ augmentation, external data addition or weight transfer in the audited path.
- The only training-objective change is replacing the Side L1 denominator with `sqrt(0.25 * max(mean(x²), 1e-8) + 0.75 * mean(Side(x)²))` in FP32. Since `mean(Side(x)²) ≤ mean(x²)`, the scale is between half and all of the inherited floored mixture RMS. Amplification is at most 2×, including zero Side. The denominator is at least 5e−5 for a silent mixture. Side L1 remains weighted at 0.25 and the other loss terms and gradients passed exact parity tests on the fixture.
- The RTX 3080 Ti had about 10.5 GiB free during review and the workspace had {audit['free_disk_bytes']/1024**3:.1f} GiB free against a 4 GiB shard budget. The available process inventory showed no conflicting Python trainer. The 18 tests all passed, including native CUDA BF16, serialization and deterministic numbered-checkpoint resume on a synthetic fixture.
- Numbered Exp12 checkpoints at 0 through 6,000 passed SHA-256 and state verification. The step-6,000 hash is `{matched['exp12_checkpoint']['sha256']}`. The two Exp10 matched archives and Exp14 arm files passed their recorded hashes. Windows denied access to the Exp10 numbered step-6,000 checkpoint. Existing aliases, checkpoints and run logs were preserved. The existing completion record is dated November 16, 2026, Bucharest time. This review reused that run and performed zero training updates.

Commands run from the Exp12 directory. The audit calls the prepared matched evaluator's `report()` and verifies its output equals the saved JSON. Replay used the Exp14 measurement helper at 2 s and the identical Exp12 scores, with separate exploratory RMS calculations.

```powershell
{commands}
```

The README documents the trainer and evaluator commands for the existing run path. Its saved run directory contains structured update and validation logs. The original console transcript was unavailable there. [README](README.md) · [Audit log](pilot_review_20261117/audit.log) · [Test log](pilot_review_20261117/unittest.log) · [Replay log](pilot_review_20261117/inference_replay.log) · [Decision](EXP12_PILOT_DECISION.md) · [Completion manifest](EXP12_PILOT_COMPLETION.json)
"""
    return main


def decision_markdown(arms):
    rows = [[rule, *("Pass" if arms[arm]["screening_gates"][key] else "**Fail**" for arm in ARMS)]
            for key, rule in GATE_NAMES.items()]
    rows.append(["Finite computation, valid checkpoints and matching protocol", "Pass", "Pass"])
    return f"""# Exp12 pilot decision

**NO_GO**

## Pre-registered rules

The following criteria are from [EXP12_DESIGN.md](EXP12_DESIGN.md).

> For **each** arm, require vocal and instrumental SDR deltas ≥ −0.10 dB and absolute vocal bleed delta ≤ +1.00 percentage point. Predeclare meaningful Side improvement as **≥ +0.20 dB on each Side SDR**, with both Side deltas strictly positive in at least **four of seven** artist groups.

> Verify all 6,000 steps are present, finite and have the exact Exp10 crop addresses and shard order. Any unstable step, failed validation, or mixture-consistency regression blocks GO.

> Apply the existing inference residual check per song. Compare every song and the panel maximum against archived residuals with one FP32 machine epsilon at full scale (1.1921e−7) tolerance for addition/subtraction rounding.

> Require both arms to clear the screening thresholds, then review the overall trade-off.

The tolerances apply to artist macros. The integrity review also checked the frozen split, settings, validation and serialized state. Both Side deltas track the same error power under complementary reconstruction.

## Gate outcomes

{table(['Pre-registered gate', 'Raw', 'EMA'], rows)}

Raw improved Vocal Side SDR by +0.3973 dB and Instrumental Side SDR by +0.3973 dB, with both positive in 7/7 groups. Absolute vocal bleed rose by +5.9757 pp, exceeding the maximum tolerated increase by 4.9757 pp. EMA improved bleed by 0.5268 pp and passed both whole-stem SDR tolerances. Its Side scores fell by 0.1788 dB and both improved in only 3/7 groups. Each arm therefore failed the screen for a different reason.

The joint Raw pattern supports a measured trade-off on this panel. A conflict between Side and leakage gradients was observed in earlier local diagnostics and is a plausible contributor. The pilot provides no direct evidence that this conflict caused the bleed regression, or that EMA caused the loss of Side gain through any specific mechanism. Seed and artist coverage are limited. A 6,000-update result does not establish the effect at 24,000.

Retain the verified Exp10 Raw step-24,000 best-overall checkpoint as the reference before November 23, 2026. Its hash matches Exp10's `latest.pt`. Exp10 recorded no checkpoint that passed its separate qualification guardrails. This checkpoint is the retention recommendation and the comparison in this report remains at update 6,000. [Matched results](EXP12_PILOT_RESULTS.md)
"""


def main():
    audit, profile = (read_json(OUT / file) for file in ("audit.json", "profile.json"))
    ensure_provenance(audit, profile)
    arms = audit["matched"]["arms"]
    stats = {arm: delta_stats(arms[arm]) for arm in ARMS}
    with (EXP14 / "per_song_paired.csv").open(encoding="utf-8", newline="") as handle:
        songs = {row["uuid"]: row["title"] for row in csv.DictReader(handle)}
    require(all(songs.keys() == arms[arm]["songs"].keys() for arm in ARMS), "Song-label roster mismatch")
    diagnostics = {
        arm: {stem: diagnostic_macro(profile["arms"][arm]["songs"], stem) for stem in
              ("vocal_error_rms", "instrumental_error_rms")} for arm in ARMS
    }
    require([key for key, v in arms["raw"]["screening_gates"].items() if not v] == ["absolute_vocal_bleed"]
            and set(key for key, v in arms["ema"]["screening_gates"].items() if not v)
            == {"vocal_side_sdr", "instrumental_side_sdr", "artist_spread"}
            and audit["matched"]["stability_and_sampler"]["clean_compute_history"],
            "Review judgments do not match verified evidence")
    results = {
        "status": STATUS,
        "review_of_existing_completed_pilot": True,
        "training_updates_in_this_review": 0,
        "decision_gates": GATE_NAMES,
        "commands": COMMANDS,
        "audit": audit,
        "artist_direction_and_extrema": stats,
        "development_song_titles_by_uuid": songs,
        "inference_replay": profile,
        "artist_macro_exploratory_error_rms": diagnostics,
        "side_sdr_conclusion": "Two reported deltas track one underlying complementary Side error energy.",
        "matched_per_channel_and_mid_error_change": None,
    }
    completion = {
        "status": STATUS, "review_date_bucharest": "2026-11-17",
        "pilot_steps": 6000, "training_updates_in_this_review": 0,
        "completed_existing_run": str(RUN),
        "run_identity": audit["run"]["identity"],
        "checkpoint_sha256": audit["matched"]["exp12_checkpoint"]["sha256"],
        "baseline": str(BASELINE),
        "baseline_step": 6000, "inference_hop_seconds": 2,
        "pilot_source_hashes": audit["evidence_sha256"],
        "checks": {"tests_run": 18, "tests_passed": 18, "tests_skipped": 0,
                   "verified_serialized_boundaries": 7, "matched_validation_arms": list(ARMS),
                   "matched_crops": 24000, "matched_songs": 16, "matched_artist_groups": 7,
                   "metric_blocker": False, "compute_failure": None, "sealed_holdout_accessed": False},
        "failed_gates": {arm: [key for key, ok in arms[arm]["screening_gates"].items() if not ok]
                         for arm in ARMS},
        "evidence_sha256": {
            f"{OUT.name}/{entry}": file_hash(OUT / entry)
            for entry in ("audit.json", "profile.json", "review.py", "write_reports.py", "unittest.log",
                          "audit.log", "inference_replay.log", "gpu_snapshot.txt")
        },
    }
    files = {
        "EXP12_PILOT_RESULTS.json": json.dumps(results, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        "EXP12_PILOT_RESULTS.md": results_markdown(audit, profile, stats, songs, diagnostics),
        "EXP12_PILOT_DECISION.md": decision_markdown(arms),
    }
    for filename, output in files.items():
        with (ROOT / filename).open("x", encoding="utf-8") as handle:
            handle.write(output)
    completion["deliverables_sha256"] = {filename: file_hash(ROOT / filename) for filename in files}
    with (ROOT / "EXP12_PILOT_COMPLETION.json").open("x", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Wrote all four verified Exp12 pilot deliverables", flush=True)


if __name__ == "__main__":
    main()
