"""Isolate CUDA replay differences and finish timing after the formal STOP."""

import copy
from pathlib import Path
import tempfile

import torch

from run_formal import (
    ROOT, WEIGHTS, CheckpointManager, MSEProbeLoss, _fixed_batch, configure_optimizer,
    hash_protected_checkpoints, json, make_projected_targets, probe, restore_training_state,
    training_state, training_step,
)


def tensors(state, prefix=""):
    if isinstance(state, torch.Tensor):
        yield prefix, state.detach().cpu().clone()
    elif isinstance(state, dict):
        for key, value in state.items():
            yield from tensors(value, f"{prefix}/{key}")
    elif isinstance(state, (list, tuple)):
        for index, value in enumerate(state):
            yield from tensors(value, f"{prefix}/{index}")


def diff(a, b):
    a, b = dict(tensors(a)), dict(tensors(b))
    if a.keys() != b.keys():
        raise ValueError("Tensor state keys differ")
    rows = {k: float((a[k].double() - b[k].double()).abs().max())
            for k in a if not torch.equal(a[k], b[k])}
    return {"different_tensors": len(rows), "max_abs": max(rows.values(), default=0.), "by_tensor": rows}


def replay(criterion, batch, deterministic):
    torch.use_deterministic_algorithms(deterministic)
    model = probe.new_model()
    optimizer, scheduler = configure_optimizer(model, probe.CONFIG)
    generator = torch.Generator().manual_seed(42)
    mix, *refs = [wave[:1, :, :20003] for wave in batch]
    targets = make_projected_targets(mix, *refs)
    for _ in range(6):
        training_step(model, optimizer, criterion, WEIGHTS, mix, targets)
        scheduler.step()
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        manager = CheckpointManager(Path(folder))
        fields = {"config": {"diagnostic": True, "deterministic": deterministic}, "epoch": 0,
                  "validation_objective": 1.0}
        manager.initialize(training_state(model, optimizer, scheduler, generator, **fields))
        initial_parameters = dict(tensors(model.state_dict()))
        initial_optimizer = dict(tensors(optimizer.state_dict()))

        def step():
            result = training_step(model, optimizer, criterion, WEIGHTS, mix, targets)
            scheduler.step()
            return {"model": dict(tensors(model.state_dict())), "optimizer": dict(tensors(optimizer.state_dict())),
                    "gradients": dict(tensors({name: p.grad for name, p in model.refinement.named_parameters()})),
                    "objective": result["total"]}

        direct = step()
        runs = []
        for _ in range(2):
            restored = manager.load_latest(fields["config"])
            restore_training_state(restored, model, optimizer, scheduler, generator)
            runs.append({"restore_model": diff(initial_parameters, dict(tensors(model.state_dict()))),
                         "restore_optimizer": diff(initial_optimizer, dict(tensors(optimizer.state_dict()))),
                         "step": step()})
        def compare(a, b):
            return {k: diff(a[k], b[k]) for k in ("model", "optimizer", "gradients")} | {
                "objective_delta": abs(a["objective"] - b["objective"])}
        return {"deterministic_algorithms": deterministic,
                "restore": [{k: r[k] for k in ("restore_model", "restore_optimizer")} for r in runs],
                "direct_vs_reload": compare(direct, runs[0]["step"]),
                "reload_vs_reload": compare(runs[0]["step"], runs[1]["step"])}


def main():
    before = hash_protected_checkpoints(probe.PATHS.root)
    prior = json.loads((ROOT.parent / "preflight.json").read_text())
    assert before == prior["protected_checkpoints"]
    torch.cuda.set_per_process_memory_fraction(min(1., 10.5 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    batch = _fixed_batch(prior["fixed_segments"][:4])
    criterion = MSEProbeLoss()
    findings = {}
    for deterministic in (False, True):
        try:
            findings[str(deterministic)] = replay(criterion, batch, deterministic)
            print(f"Resume diagnostic deterministic={deterministic}: "
                  f"{findings[str(deterministic)]['direct_vs_reload']['model']['max_abs']}", flush=True)
        except RuntimeError as error:
            findings[str(deterministic)] = {"error": str(error)}
    probe.write_json(ROOT / "resume_diagnostics.json", findings)
    torch.use_deterministic_algorithms(False)
    del batch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timing = probe.time_disposable_epoch(probe.new_model(), criterion, WEIGHTS)
    probe.write_json(ROOT / "timing_followup.json", timing | {"protected_checkpoints_unchanged":
                     before == hash_protected_checkpoints(probe.PATHS.root)})


if __name__ == "__main__":
    main()
