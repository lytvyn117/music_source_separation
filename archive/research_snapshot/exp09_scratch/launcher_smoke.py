"""Exercise the final entry point for one update plus resume, in a temporary run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch

from audit import write_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(dir=ROOT) as tmp:
        run = Path(tmp) / "run"
        cmd = [sys.executable, "-B", str(ROOT / "train_exp09.py"),
               "--archive", str(args.archive), "--out", str(run)]
        subprocess.run(cmd + ["--smoke-stop", "1"], check=True, cwd=ROOT)
        first = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        if first["index"] != 1:
            raise ValueError("First stop did not checkpoint update one")
        subprocess.run(cmd + ["--smoke-stop", "2", "--resume"], check=True, cwd=ROOT)
        state = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        events = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
        if state["index"] != 2 or state["signature"] != first["signature"]:
            raise ValueError("Resume changed final checkpoint identity")
        if [r["step"] for r in events] != [1, 2] or events[0]["crops"] == events[1]["crops"]:
            raise ValueError("Launcher stream failed to advance")
        write_json(ROOT / "launcher_smoke.json", {
            "random_model_sha256": json.loads((run / "run.json").read_text())["initial_model_sha256"],
            "signature": state["signature"], "resumed_checkpoint_step": state["index"],
            "launcher_updates": events,
        })
    print("Launcher saved update one and resumed to update two", flush=True)


if __name__ == "__main__":
    main()
