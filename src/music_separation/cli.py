"""Prepare approved Exp10 data, reproduce training, or score the frozen development panel."""

import argparse
import json
from pathlib import Path

from music_separation.inference.separate import BACKENDS, CHECKPOINT_SHA, inference_device, load_model
from music_separation.training import trainer
import torch
from torch.nn.attention import sdpa_kernel
from music_separation.evaluation.evaluator import evaluate


def evaluate_checkpoint(checkpoint, digest, arms):
    _, roster = trainer.bind()
    development = roster["development"]
    trainer.verify_cache(development)
    device = inference_device()
    results = {}
    with sdpa_kernel(BACKENDS), torch.autocast("cuda", enabled=False):
        for weights in ("raw", "ema") if arms == "both" else (arms,):
            model, provenance = load_model(checkpoint, device, weights, digest)
            results[weights] = {**provenance, **evaluate(model, development, device)}
            del model
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build or verify development references")
    prepare.add_argument("--archive", required=True, type=Path)
    train = commands.add_parser("train", help="Run the Exp10 recipe on the approved archive")
    train.add_argument("--archive", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--resume", action="store_true")
    score = commands.add_parser("evaluate", help="Score the verified development references")
    score.add_argument("--checkpoint", required=True, type=Path)
    score.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA)
    score.add_argument("--weights", choices=("raw", "ema", "both"), default="both")
    score.add_argument("--output", required=True, type=Path, help="New JSON file")
    args = parser.parse_args()

    if args.command == "evaluate":
        if args.output.exists() or not args.output.parent.is_dir():
            parser.error("Choose a new evaluation JSON file in an existing directory.")
        report = evaluate_checkpoint(args.checkpoint, args.checkpoint_sha256, args.weights)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        for arm, row in report.items():
            print(f"{arm}: {row['artist_macro']}")
        return

    archive = args.archive.resolve()
    if not archive.is_file():
        parser.error(f"Approved archive file does not exist: {archive}")
    if args.command == "prepare":
        development = trainer.load_roster()["development"]
        if not trainer.CACHE.is_dir():
            trainer.make_cache(archive, development)
        trainer.verify_cache(development)
        print(f"Verified {len(development)} development references.")
        return

    out = args.out.resolve()
    if out.parent != (trainer.ROOT / "runs").resolve():
        parser.error("Full-run outputs must be direct children of runs.")
    # The trainer binds the default archive path at import time. Override it in this
    # process so a different archive location preserves the run's source hashes.
    trainer.ARCHIVE = archive
    trainer.train(out, args.resume)


if __name__ == "__main__":
    main()
