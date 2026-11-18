"""Replay only the archived 2-second development arms using the Exp14 evaluator."""

import argparse
import importlib.util
import json
from pathlib import Path


def load_evaluator(repo):
    path = repo / "exp14_inference_hop" / "evaluate_hops.py"
    spec = importlib.util.spec_from_file_location("frozen_exp14_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON file outside the experiment directories")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    protected = [repo / "data", *(path for path in repo.glob("exp[0-9][0-9]*"))]
    if output.exists() or any(output.is_relative_to(path) for path in protected):
        parser.error("Choose a fresh JSON output path outside experiment and data directories.")
    evaluator = load_evaluator(repo)
    record, weights, development, archived, device = evaluator.preflight()
    old_provenance = evaluator.read_json(evaluator.OUT / "run_provenance.json")
    for field in ("runner_sha256", "exp10_files_sha256", "metadata_sha256", "cache", "runtime"):
        if record[field] != old_provenance[field]:
            raise ValueError(f"Archived Exp14 provenance mismatch for {field}")
    print(f"Verified checkpoint and references for {len(development)} development songs "
          f"and {record['artist_groups']} artist groups.", flush=True)
    model = evaluator.Separator(checkpoint_blocks=True).to(device).eval().requires_grad_(False)
    from validation import artist_macro

    comparison = {}
    for name in ("raw", "ema"):
        model.load_state_dict(weights[name], strict=True)
        arm = evaluator.evaluate_arm(model, name, 2, development, archived[name]["songs"], device)
        prior = evaluator.read_json(evaluator.OUT / f"arm_{name}_2s.json")
        maxima = {key: 0.0 for key in arm["songs"][0]
                  if isinstance(arm["songs"][0][key], float) and key != "inference_seconds"}
        for row, baseline in zip(arm["songs"], prior["songs"], strict=True):
            for key, value in row.items():
                if key in maxima:
                    maxima[key] = max(maxima[key], abs(value - baseline[key]))
                elif key != "inference_seconds" and value != baseline[key]:
                    raise ValueError(f"Exp14 mismatch for {name} {row['uuid']} {key}")
        macro = artist_macro(arm["songs"])
        macro_deltas = {key: val - archived[name]["artist_macro"][key] for key, val in macro.items()}
        comparison[name] = {
            "artist_macro": macro,
            "exp10_artist_macro_delta": macro_deltas,
            "exp14_per_song_max_abs_delta_excluding_time": maxima,
            **arm,
        }

    result = {
        "tolerance_absolute": 1e-6,
        "provenance": record,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "comparison": comparison,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for name, arm in comparison.items():
        maximal_delta = max(max(arm["exp14_per_song_max_abs_delta_excluding_time"].values()),
                            max(map(abs, arm["exp10_artist_macro_delta"].values())))
        print(f"{name.upper()}: largest archived delta {maximal_delta:.12g}", flush=True)
        if maximal_delta > result["tolerance_absolute"]:
            raise ValueError(f"{name} exceeds the archived tolerance. See {output}")


if __name__ == "__main__":
    main()
