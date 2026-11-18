"""Generate the human-readable and CSV summaries from this run's saved artifacts."""
import csv
import json
import math
from pathlib import Path
from statistics import median

RUN = "full_20261111_seed42"
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LOGS = ROOT / "outputs/logs/exp08_subband" / RUN
MODELS = ROOT / "models/experiments/exp08_subband" / RUN


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def med(rows, key, index=None, transform=lambda value: value):
    vals = [row[key] if index is None else row[key][index] for row in rows]
    return median(transform(value) for value in vals if value is not None and math.isfinite(value))


def pair(rows_a, rows_b):
    b = {row["track"]: row for row in rows_b}
    for a in rows_a:
        yield a, b[a["track"]]


def format_duration(seconds):
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def build():
    training = read(HERE / "training_result.json")
    finished = read(HERE / "finished.json")
    results = read(HERE / "evaluation/metrics.json")
    manifest = read(HERE / "manifest.json")
    epoch_logs = {row["epoch"]: row for row in read_lines(LOGS / "epochs.jsonl")}
    progress = []
    for index, history in enumerate(training["history"]):
        epoch = history.get("epoch", index)
        log = epoch_logs.get(epoch, {})
        row = {"epoch": epoch, "training_total": history.get("train_total_loss"),
               "validation_total": history["val_total_loss"]}
        for name in ("complex_mse", "log_magnitude", "waveform", "stereo_side", "vocal_leakage"):
            row["validation_" + name] = history[name]
        row.update({"diagnostic_" + name: value
                    for name, value in log.get("diagnostics", {}).get("medians", {}).items()})
        row.update({"best_epoch": log.get("selection", {}).get("best_epoch"),
                    "clipped_updates": history.get("clipped_updates"),
                    "preclip_max": history.get("preclip_max"),
                    "gpu_temperature_c": log.get("gpu", {}).get("temperature_c"),
                    "elapsed_training_s": log.get("elapsed_training_s")})
        progress.append(row)
    write_csv(HERE / "loss_history.csv", progress)
    keys = ("vocal_sdr", "instrumental_sdr", "vocal_si_sdr", "bleed", "bleed_energy_pct", "consistency")
    by_track = []
    baseline, candidate = results["baseline"], results["native"]
    for a, b in pair(baseline, candidate):
        row = {"track": a["track"]}
        for key in keys:
            row[f"exp04_{key}"] = a[key]
            row[f"exp08_{key}"] = b[key]
            row[f"delta_{key}"] = b[key] - a[key] if a[key] is not None and b[key] is not None else None
        for key in ("side_error", "clipped", "rms_gain_db", "output_peak"):
            for i, source in enumerate(("vocal", "instrumental")):
                row[f"exp04_{source}_{key}"] = a[key][i]
                row[f"exp08_{source}_{key}"] = b[key][i]
        by_track.append(row)
    write_csv(HERE / "per_track_comparison.csv", by_track)
    keys_audio = read(HERE / "evaluation/blind_key.json")
    audio = ["# Exp08 blind listening clips", "",
             "Stereo FLOAT WAV at original amplitudes. Each directory contains the mixture, reference stems, and model A/B stems.", "",
             "| Clip | Track | Start | Duration |", "|---|---|---:|---:|"]
    for clip in keys_audio:
        name = clip["clip"]
        audio.append(f"| [{name}](evaluation/listen/{name}) | {clip['track']} | {clip['start_sample']/44100:.2f} s | {clip['samples']/44100:.2f} s |")
    with (HERE / "listening.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(audio) + "\n")
    report = ["# Exp08 full-run results", "",
              f"Training ended by {training['reason'].replace('_',' ')} after {training['completed_epochs']} completed epochs.",
              f"Elapsed training time was **{format_duration(training['training_seconds'])}**. The selected checkpoint was epoch **{results['selected_epoch']}**.",
              f"Selection used the approved full validation objective. Final validation loss was {progress[-1]['validation_total']:.6f} and best was {progress[results['selected_epoch']]['validation_total']:.6f}.",
              "", "[Loss history](loss_history.csv) · [Per-track comparison](per_track_comparison.csv) · [Blind listening index](listening.md)", "",
              "## Frozen 15-song comparison", "", "| Metric | Exp04 | Exp08 |", "|---|---:|---:|"]
    scalars = [
        ("Vocal SDR", "vocal_sdr", None, 1, " dB"),
        ("Instrumental SDR", "instrumental_sdr", None, 1, " dB"),
        ("Vocal SI-SDR", "vocal_si_sdr", None, 1, " dB"),
        ("Signed vocal bleed coefficient", "bleed", None, 100, "%"),
        ("Vocal bleed energy", "bleed_energy_pct", None, 1, "%"),
        ("Vocal absolute side RMS error", "side_error", 0, 1, " dB"),
        ("Instrumental absolute side RMS error", "side_error", 1, 1, " dB"),
        ("Vocal RMS gain", "rms_gain_db", 0, 1, " dB"),
        ("Instrumental RMS gain", "rms_gain_db", 1, 1, " dB"),
        ("Vocal clipped samples", "clipped", 0, 100, "%"),
        ("Instrumental clipped samples", "clipped", 1, 100, "%"),
    ]
    for label, key, idx, scale, unit in scalars:
        report.append(f"| Median {label} | {med(baseline,key,idx)*scale:.4f}{unit} | {med(candidate,key,idx)*scale:.4f}{unit} |")
    report.append(f"| Worst mixture reconstruction residual | {max(x['consistency'] for x in baseline):.3g} | {max(x['consistency'] for x in candidate):.3g} |")
    grade = results["grade"]
    report.extend(["", f"Vocal SDR wins on **{grade['vocal_wins']}/15** tracks. The paired 95% interval for median vocal SDR change is {grade['paired_95_ci']['vocal_sdr']}.",
                   "", f"**Promotion decision — {grade['decision'].replace('_',' ')}.** Exp04 remains the selected baseline pending a promotion decision.",
                   "", "| Gate | Pass |", "|---|---|"])
    for name, passed in grade["gates"].items():
        report.append(f"| {name.replace('_',' ')} | {'Yes' if passed else 'No'} |")
    report.extend(["", "## Training safety", "",
                   f"- {training['updates']} updates and {training['clipped_updates']} gradient clips.",
                   f"- Peak allocated {training['peak_allocated_gib']:.3f} GiB and reserved {training['peak_reserved_gib']:.3f} GiB. Minimum sampled free memory {training['minimum_free_gib']:.3f} GiB.",
                   f"- Peak sampled GPU temperature {training['peak_sampled_temperature_c']}°C.",
                   f"- Verified {finished['protected_model_files']} protected model files and {finished['protected_files_verified']} protected source and experiment artifacts.",
                   "", "[Full numerical evaluation](evaluation/metrics.json) · [Training state and safety totals](training_result.json) · [Run configuration](start.json) · [Frozen manifest](manifest.json)"])
    with (HERE / "summary.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    build()
