"""Replay only through the original first rejected batch, without writing checkpoints."""

from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import numpy as np
import torch

from src.experiments.exp07_phase_aware_residual import mse_pilot as pilot


ROOT = Path(__file__).resolve().parent
LIMIT = 368


class TracedDataset(pilot.AugmentedMUSDBDataset):
    def __getitem__(self, index):
        self.current_reads = []
        data = super().__getitem__(index)
        self.batch_reads.append(self.current_reads)
        return data

    def _read_numpy_segment(self, path, start_frame):
        self.current_reads.append([path.parent.name, path.name, start_frame])
        return super()._read_numpy_segment(path, start_frame)


def write_json(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def detached_cpu(data):
    return {key.name: getattr(data, key.name).detach().cpu() for key in fields(data)}


def main():
    mode = pilot.configure_determinism()
    paths = pilot.Exp07Paths()
    original = paths.reports / "mse_pilot"
    launch = json.loads((original / "launch.json").read_text())
    status = json.loads((original / "training_status.json").read_text())
    log = (paths.reports / "mse_pilot_launcher/stdout.log").read_text()
    assert status["last_stage"] == {"epoch": 1, "batch": LIMIT, "phase": "training"}
    assert asdict(pilot.CONFIG) == launch["configuration"] and asdict(pilot.WEIGHTS) == launch["weights"]
    assert mode == launch["settings"]
    assert shutil.disk_usage(paths.root).free / 2**30 >= pilot.CONFIG.min_free_disk_gib
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    assert not any("python" in row.lower() and int(row.split(",")[0]) != os.getpid() for row in apps.splitlines())
    assert torch.cuda.mem_get_info()[0] / 2**30 >= status["peak_reserved_gib"] + .5
    protected = pilot.hash_protected_checkpoints(paths.root)
    assert protected == launch["protected_checkpoints"] and len(protected) == 36
    initial = paths.checkpoints / "initial.pt"
    originals = [initial, paths.checkpoints / "best.pt", paths.checkpoints / "latest.pt",
                 original / "launch.json", original / "training_status.json", original / "completion.json",
                 original / "evaluation/comparison.json"]
    fingerprints = {str(file): pilot.sha256(file) for file in originals}
    state = torch.load(initial, map_location="cpu", weights_only=False)
    assert state["epoch"] == 0 and state["global_step"] == 0
    assert state["config"] == {key: launch[key] for key in ("settings", "configuration", "weights")} | {
        "loss": "normalized_complex_mse", "head_bf16": True, "num_workers": 0,
        "exp04_sha256": pilot.sha256(paths.backbone)}
    assert f"{state['validation_objective']:.7f}" == re.search(r"objective (\d+\.\d+)", log)[1]

    config = pilot.CONFIG
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., config.max_reserved_gib * 2**30 / total_memory))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = pilot.PhaseAwareResidualSeparator.from_exp04(paths.backbone, "cuda")
    model.refinement.to(memory_format=torch.channels_last)
    model.use_bf16 = True
    optimizer, scheduler = pilot.configure_optimizer(model, config)
    rng = torch.Generator().manual_seed(config.seed)
    train_files, _ = pilot.create_train_validation_split()
    dataset = TracedDataset(track_files=train_files, samples_per_epoch=config.samples_per_epoch, augment=True)
    train_loader = pilot.loader(dataset, rng)
    pilot.restore_state(state, model, optimizer, scheduler, rng)
    del state
    model.train()
    criterion = pilot.MSEProbeLoss()
    forward, project, guard = model.forward_stft, pilot.make_projected_targets, pilot.enforce_safety
    captured = {}

    def trace_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        captured["prediction"] = result
        return result

    def trace_project(*args, **kwargs):
        result = project(*args, **kwargs)
        captured["targets"] = result
        return result

    def trace_guard(stats, reserved):
        captured["stats"] = stats.copy()
        return guard(stats, reserved)

    model.forward_stft = trace_forward
    pilot.make_projected_targets = trace_project
    pilot.enforce_safety = trace_guard
    sentinels = {int(step): loss for step, loss in re.findall(r"step (\d+)/500 loss (\d+\.\d+)", log)}
    checks, history = {}, []
    exception = None
    start = time.perf_counter()
    data = iter(train_loader)
    print(f"Restored epoch 0; bounded replay through batch {LIMIT}", flush=True)
    for step in range(1, LIMIT + 1):
        dataset.batch_reads = []
        batch = next(data)
        captured.clear()
        try:
            stats = pilot._output(model, criterion, batch, optimizer)
        except (FloatingPointError, RuntimeError) as error:
            exception = f"{type(error).__name__}: {error}"
            break
        history.append({"batch": step, **stats})
        if step in sentinels:
            checks[step] = {"original": sentinels[step], "replay": f"{stats['total']:.6f}"}
            print(f"Batch {step}: {checks[step]}", flush=True)
    match = (step == LIMIT and exception == status["message"] and len(checks) == len(sentinels)
             and all(row["original"] == row["replay"] for row in checks.values()))
    if "prediction" in captured and "targets" in captured:
        with torch.no_grad():
            torch.save({"prediction": detached_cpu(captured["prediction"]), "targets": detached_cpu(captured["targets"]),
                        "batch": batch, "source_reads": dataset.batch_reads, "step": step, "match": match}, ROOT / "capture.pt")
    write_json("history.json", history)
    result = {"batch": step, "exception": exception, "expected_exception": status["message"], "match_to_logged_precision": match,
              "sentinels": checks, "stats": captured.get("stats"), "source_reads": dataset.batch_reads,
              "seconds": time.perf_counter() - start, "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "originals_unchanged": all(pilot.sha256(Path(name)) == digest
                  for name, digest in fingerprints.items()), "protected_unchanged": pilot.hash_protected_checkpoints(paths.root) == protected}
    write_json("replay.json", result)
    print(json.dumps({k: result[k] for k in ("batch", "exception", "match_to_logged_precision", "seconds", "protected_unchanged")}), flush=True)
    return 0 if match and result["originals_unchanged"] and result["protected_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
