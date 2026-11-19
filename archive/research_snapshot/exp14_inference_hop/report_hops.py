"""Pair the frozen Exp14 arms and apply the recorded decision rule."""
import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np

from evaluate_hops import OUT, SCHEDULE, file_hash, read_json, require, write_json

# A positive direction means higher values are better.
METRICS = {
    "vocal_sdr_db": ("Vocal SDR (dB)", 1),
    "vocal_si_sdr_db": ("Vocal SI-SDR (dB)", 1),
    "instrumental_sdr_db": ("Instrumental SDR (dB)", 1),
    "v_side_sdr_db": ("Vocal Side SDR (dB)", 1),
    "i_side_sdr_db": ("Instrumental Side SDR (dB)", 1),
    "abs_vocal_bleed_pct": ("Absolute vocal bleed (%)", -1),
    "mixture_max_abs": ("Mixture max error (full-scale units)", -1),
    "inference_seconds": ("Inference time (s)", -1),
    "peak_gpu_allocated_mib": ("Peak GPU allocated (MiB)", -1),
    "peak_gpu_reserved_mib": ("Peak GPU reserved (MiB)", -1),
}
QUALITY = tuple(METRICS)[:6]
MARGINS = {"vocal_sdr_db": .05, "instrumental_sdr_db": .05, "abs_vocal_bleed_pct": .5}
GUARDS = {**MARGINS, "v_side_sdr_db": .05, "i_side_sdr_db": .05}


def pair(identity, a, b):
    return {**identity, "2s": a, "1s": b, "delta_1s_minus_2s": {k: b[k] - a[k] for k in METRICS}}


def group_pair(identity, pairs):
    a, b = ({k: mean(row[hop][k] for row in pairs) for k in METRICS} for hop in ("2s", "1s"))
    return pair(identity, a, b)


def operational_totals(rows):
    return {"sum_inference_seconds": sum(r["inference_seconds"] for r in rows),
            "chunks": sum(r["chunks"] for r in rows),
            "peak_gpu_allocated_mib": max(r["peak_gpu_allocated_mib"] for r in rows),
            "peak_gpu_reserved_mib": max(r["peak_gpu_reserved_mib"] for r in rows),
            "worst_song_mixture_max_abs": max(r["mixture_max_abs"] for r in rows),
            "largest_fraction_of_mixture_error_limit": max(r["mixture_max_abs"] / r["mixture_tolerance"]
                                                            for r in rows)}


def summarize(name, arms, provenance):
    a, b = arms[name, 2], arms[name, 1]
    parity = a["baseline_parity_max_abs"]
    require(parity and max(parity.values()) <= 1e-6 and b["baseline_parity_max_abs"] is None,
            f"Missing or failed archive parity for {name}")
    songs = []
    for song, row_a, row_b in zip(provenance["development"], a["songs"], b["songs"], strict=True):
        for row in (row_a, row_b):
            require(row["uuid"] == song["uuid"] and row["artist_key"] == song["artist_key"]
                    and all(math.isfinite(row[k]) for k in METRICS)
                    and row["mixture_max_abs"] <= row["mixture_tolerance"], "Invalid song pair")
        require(all(row_a[k] == row_b[k] for k in ("vocal_active_frames", "instrumental_active_frames",
                                                   "mixture_tolerance")), "Reference settings differ")
        songs.append(pair(song, row_a, row_b))
    artists = []
    for artist in sorted({row["artist_key"] for row in songs}):
        members = [row for row in songs if row["artist_key"] == artist]
        artists.append(group_pair({"artist_key": artist, "songs": len(members)}, members))
    require(len(songs) == 16 and len(artists) == 7, "Incomplete pairing")
    macro = group_pair({}, artists)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(artists), size=(10000, len(artists)))
    intervals, counts = {}, {}
    for k, (_, direction) in METRICS.items():
        deltas = np.asarray([row["delta_1s_minus_2s"][k] for row in artists])
        intervals[k] = np.quantile(deltas[draws].mean(axis=1), [.025, .975]).tolist()
        tie = .01 if k.endswith("_db") else 0
        counts[k] = {"improved": int((deltas * direction > tie).sum()),
                     "tied": int((abs(deltas) <= tie).sum()),
                     "regressed": int((deltas * direction < -tie).sum())}
    totals = {"2s": operational_totals(a["songs"]), "1s": operational_totals(b["songs"])}
    totals["delta_1s_minus_2s"] = {k: totals["1s"][k] - totals["2s"][k] for k in totals["2s"]}
    return {"artist_macro": macro, "artist_counts_by_metric": counts, "paired_artist_bootstrap_95": intervals,
            "per_song": songs, "per_artist": artists, "panel_runtime_and_memory": totals,
            "archived_2s_baseline_parity_max_abs": parity,
            "arm_times_utc": {f"{arm['hop_seconds']}s": {k: arm[k] for k in ("started_utc", "ended_utc")}
                              for arm in (a, b)}}


def decide(results):
    def improvement(name, k):
        return results[name]["artist_macro"]["delta_1s_minus_2s"][k] * METRICS[k][1]

    def artist_count(name, k):
        return results[name]["artist_counts_by_metric"][k]["improved"]

    guards = {name: {k: improvement(name, k) >= -limit for k, limit in GUARDS.items()}
              for name in results}
    qualifying = [k for k, margin in MARGINS.items() if improvement("raw", k) > margin
                  and artist_count("raw", k) >= 5 and improvement("ema", k) > 0 and artist_count("ema", k) >= 4]
    faster_2s = all(v["panel_runtime_and_memory"]["delta_1s_minus_2s"]["sum_inference_seconds"] > 0
                   for v in results.values())
    if not all(all(row.values()) for row in guards.values()):
        decision = "KEEP_2S_HOP"
        reason = "At least one weight set exceeded a quality regression limit."
    elif qualifying:
        decision = "ADOPT_1S_HOP"
        reason = "The 1-second hop met the quality limits and the preset practical improvement and artist-consistency rules."
    elif faster_2s:
        decision = "KEEP_2S_HOP"
        reason = "The 1-second hop did not meet the preset benefit rule, and the 2-second hop ran faster with both weight sets."
    else:
        decision = "INCONCLUSIVE"
        reason = "The quality and runtime measurements gave no supported choice under the recorded rules."
    return {"status": decision, "reason": reason, "guardrails_pass_by_weight": guards,
            "guardrail_regression_limits": GUARDS, "qualifying_benefits": qualifying,
            "numerical_checks_pass": True, "faster_2s_both_weights": faster_2s}


def write_pairs(path, results, group, identity_fields):
    fields = ["weights", *identity_fields, *(f"{hop}_{k}" for k in METRICS for hop in ("2s", "1s", "delta"))]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            for row in result[group]:
                values = {f"{label}_{k}": row[hop][k] for k in METRICS for label, hop in
                          (("2s", "2s"), ("1s", "1s"), ("delta", "delta_1s_minus_2s"))}
                writer.writerow({"weights": name, **{k: row[k] for k in identity_fields}, **values})


def shown(key, value, signed=False):
    spec = ".2e" if key == "mixture_max_abs" else ".4f"
    return format(value, ("+" if signed else "") + spec)


def render(result):
    p, weights, decision = (result[k] for k in ("provenance", "weights", "decision"))
    c = p["checkpoint"]
    lines = ["# EXP14 results", "",
             f"Paired inference on **{len(p['development'])} songs across {p['artist_groups']} artist groups**. "
             f"Raw and EMA came from the same step {c['step']:,} Exp10 checkpoint. "
             "The 2-second results were reproduced from the archived Exp10 evaluation on both weight sets.", "",
             f"Checkpoint SHA-256 `{c['sha256']}`.", "",
             "Delta is 1 s minus 2 s. Higher SDR and lower bleed, mixture error, time and memory are favorable. "
             "A group counts as improved on a dB metric if its favorable change exceeds 0.01 dB. "
             "The 95% percentile intervals describe variation across paired artist groups on this development panel.", ""]
    for name, data in weights.items():
        macro = data["artist_macro"]
        ops = data["panel_runtime_and_memory"]
        a, b = ops["2s"], ops["1s"]
        lines += [f"## {name.upper()}", "",
                  "| Artist-macro metric | 2 s | 1 s | Delta | 95% interval | Groups improved / tied / regressed |",
                  "|---|---:|---:|---:|---:|---:|"]
        for k, (label, _) in METRICS.items():
            ci = data["paired_artist_bootstrap_95"][k]
            counts = data["artist_counts_by_metric"][k]
            groups = f"{counts['improved']} / {counts['tied']} / {counts['regressed']}"
            lines.append(f"| {label} | {shown(k, macro['2s'][k])} | {shown(k, macro['1s'][k])} | "
                         f"{shown(k, macro['delta_1s_minus_2s'][k], True)} | "
                         f"[{shown(k, ci[0], True)}, {shown(k, ci[1], True)}] | {groups} |")
        parity = max(data["archived_2s_baseline_parity_max_abs"].values())
        lines += ["", f"Total inference took **{a['sum_inference_seconds']:.2f} s → {b['sum_inference_seconds']:.2f} s** "
                  f"({b['sum_inference_seconds'] / a['sum_inference_seconds']:.2f}×). Chunk count was "
                  f"{a['chunks']:,} → {b['chunks']:,}. Peak GPU allocated was {a['peak_gpu_allocated_mib']:.2f} → "
                  f"{b['peak_gpu_allocated_mib']:.2f} MiB and reserved was {a['peak_gpu_reserved_mib']:.2f} → "
                  f"{b['peak_gpu_reserved_mib']:.2f} MiB. Worst song mixture error was "
                  f"{a['worst_song_mixture_max_abs']:.2e} → {b['worst_song_mixture_max_abs']:.2e}. "
                  f"The largest absolute difference from the archived 2 s row metrics was {parity:.2e}.", "",
                  "| Artist group | Songs | Vocal SDR Δ | Vocal SI-SDR Δ | Instrumental SDR Δ | Vocal Side SDR Δ | Instrumental Side SDR Δ | Bleed Δ (pp) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in data["per_artist"]:
            cells = " | ".join(shown(k, row["delta_1s_minus_2s"][k], True) for k in QUALITY)
            lines.append(f"| {row['artist_key']} | {row['songs']} | {cells} |")
        lines.append("")
    lines += ["## Files", "", "- [Full results, hashes, per-song and per-artist pairs](EXP14_RESULTS.json)",
              "- [Per-song pairs](per_song_paired.csv) · [Per-artist pairs](per_artist_paired.csv)",
              "- [Frozen design](EXP14_DESIGN.md) · [Exact commands](EXP14_COMMANDS.md) · [Decision](EXP14_DECISION.md)",
              "", f"**{decision['status']}**", ""]
    decision_lines = ["# EXP14 decision", "", f"**{decision['status']}**", "", decision["reason"], ""]
    for k in decision["qualifying_benefits"]:
        raw_count, ema_count = (weights[name]["artist_counts_by_metric"][k]["improved"] for name in ("raw", "ema"))
        decision_lines.append(f"{METRICS[k][0]} met the preset benefit rule. "
                              f"{raw_count}/7 Raw groups and {ema_count}/7 EMA groups improved.")
        decision_lines.append("")
    for name, data in weights.items():
        delta = data["artist_macro"]["delta_1s_minus_2s"]
        counts = data["artist_counts_by_metric"]
        a, b = (data["panel_runtime_and_memory"][hop]["sum_inference_seconds"] for hop in ("2s", "1s"))
        failures = [METRICS[k][0] for k, passed in decision["guardrails_pass_by_weight"][name].items() if not passed]
        guard_text = "All regression limits passed." if not failures else f"Limits exceeded for {', '.join(failures)}."
        decision_lines.append(f"- **{name.upper()}** changed vocal SDR by {delta['vocal_sdr_db']:+.4f} dB "
                              f"({counts['vocal_sdr_db']['improved']}/7 groups improved), instrumental SDR by "
                              f"{delta['instrumental_sdr_db']:+.4f} dB, and absolute bleed by "
                              f"{delta['abs_vocal_bleed_pct']:+.4f} percentage points. Side SDR changed by "
                              f"{delta['v_side_sdr_db']:+.4f} dB for vocals and {delta['i_side_sdr_db']:+.4f} dB "
                              f"for instrumental. Inference took {a:.2f} s at 2 s and {b:.2f} s at 1 s. {guard_text}")
    decision_lines += ["", "All paired predictions and metrics were finite. Every song passed the archived mixture-consistency check. "
                       "The external final holdout remained sealed. [Design and decision criteria](EXP14_DESIGN.md) · "
                       "[Full measurements](EXP14_RESULTS.md)", ""]
    return "\n".join(lines), "\n".join(decision_lines)


def main():
    provenance = read_json(OUT / "run_provenance.json")
    require(file_hash(OUT / "EXP14_DESIGN.md") == provenance["exp14_design_sha256"], "Design changed")
    require(file_hash(OUT / "evaluate_hops.py") == provenance["runner_sha256"], "Runner changed")
    require(file_hash(Path(provenance["checkpoint"]["path"])) == provenance["checkpoint"]["sha256"],
            "Checkpoint changed after inference")
    arms, hashes = {}, {}
    for name, hop in SCHEDULE:
        path = OUT / f"arm_{name}_{hop}s.json"
        arm = read_json(path)
        require(arm["weights"] == name and arm["hop_seconds"] == hop, f"Unexpected arm metadata in {path.name}")
        arms[name, hop] = arm
        hashes[path.name] = file_hash(path)
    weights = {name: summarize(name, arms, provenance) for name in ("raw", "ema")}
    result = {"provenance": provenance, "arm_files_sha256": hashes, "weights": weights,
              "delta_definition": "1s minus 2s", "decision": decide(weights)}
    results_md, decision_md = render(result)
    write_json(OUT / "EXP14_RESULTS.json", result)
    write_pairs(OUT / "per_song_paired.csv", weights, "per_song", ("uuid", "title", "artist", "artist_key"))
    write_pairs(OUT / "per_artist_paired.csv", weights, "per_artist", ("artist_key", "songs"))
    (OUT / "EXP14_RESULTS.md").write_text(results_md, encoding="utf-8")
    (OUT / "EXP14_DECISION.md").write_text(decision_md, encoding="utf-8")
    print(result["decision"]["status"])


if __name__ == "__main__":
    main()
