"""CPU audit of approved training audio and final-Exp10 output gradients."""
import argparse
from dataclasses import asdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from checkpoint import file_hash, read_json, write_json
from data import ARCHIVE, ROOT, RestrictedArchive, Shard
from loss_candidates import get_loss
from remix import RemixSource, distribution
from scratch_model import SEGMENT, Separator
from train_exp11 import bind

BASELINE = ROOT.parent / "exp10_scratch"


def stats(values):
    return dict(zip(("min", "p05", "p50", "p95", "p99", "max"),
                    (float(v) for v in np.quantile(values, (0, .05, .5, .95, .99, 1)))))


def energy(values, axis=None):
    return np.square(values, dtype=np.float64).mean(axis=axis)


def mid_side(wave):
    return (wave[0] + wave[1]) * .5, (wave[0] - wave[1]) * .5


def stereo(wave):
    mid, side = mid_side(wave)
    mid_power, side_power = energy(mid), energy(side)
    return {
        "mid_rms": float(np.sqrt(mid_power)), "side_rms": float(np.sqrt(side_power)),
        "mid_energy": float(mid_power * mid.size), "side_energy": float(side_power * side.size),
        "side_to_mid_energy_ratio": float(side_power / mid_power) if mid_power > 0 else None,
    }


def summarize(rows):
    return {k: stats([r[k] for r in rows if r[k] is not None]) for k in rows[0]
            if any(r[k] is not None for r in rows)}


def check_real_equivalence(source, folder, roster):
    spec = spec_from_file_location("exp10_audio_reference", BASELINE / "train_exp10.py")
    baseline = module_from_spec(spec)
    spec.loader.exec_module(baseline)
    with baseline.BatchSource(ARCHIVE, folder, roster, 909, limit=1) as original:
        with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
            ref_group, addresses, (ref_x, ref_v) = original.batch(0)
    source.enabled = False
    try:
        group, crops, x, v, _ = source.arrays(0)
    finally:
        source.enabled = True
    if (group != ref_group or addresses != [(r.vocal_uuid, r.vocal_start) for r in crops]
            or not np.array_equal(x.view(np.uint32), ref_x.numpy().view(np.uint32))
            or not np.array_equal(v.view(np.uint32), ref_v.numpy().view(np.uint32))):
        raise RuntimeError("Disabled remix differs from Exp10 on actual approved audio")
    return {"update_index": 0, "batch_size": len(crops), "bitwise_equal": True,
            "addresses": [asdict(r) for r in crops]}


def gradient_probe(config, source):
    checkpoint = BASELINE / "runs/exp10_seed909/latest.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Separator().eval()
    model.load_state_dict(state["raw_model"], strict=True)
    step = state["step"]
    del state
    settings = config["exp09_settings"]
    objective = get_loss(settings["objective"])
    weights = settings["loss"]
    rows = []
    for update in (0, 499, 5999):
        _, original = source.address(update)
        for address in original[:2]:
            x, v, i = source.readers[address.vocal_uuid](address.vocal_start, SEGMENT)
            mix, target = (torch.from_numpy(np.ascontiguousarray(w.T[None])) for w in (x, v))
            with torch.no_grad():
                estimate, i_est = model(mix)
            prediction = estimate.detach().requires_grad_()
            parts = objective(model, mix, prediction, target)
            gradients = {k: torch.autograd.grad(value, prediction, retain_graph=True)[0].flatten()
                         for k, value in parts.items()}
            norms = {k: float(g.norm()) for k, g in gradients.items()}
            cosines = {}
            for a, ga in gradients.items():
                for b, gb in gradients.items():
                    if a < b:
                        denominator = norms[a] * norms[b]
                        cosines[f"{a}|{b}"] = float(ga.dot(gb) / denominator) if denominator else None
            error = estimate[0].numpy() - v.T
            inst_error = i_est[0].numpy() - i.T
            rows.append({"update_index": update, "vocal_uuid": address.vocal_uuid,
                         "start": address.vocal_start, "vocal_error": stereo(error),
                         "instrumental_error": stereo(inst_error),
                         "values": {k: float(value.detach()) for k, value in parts.items()},
                         "output_gradient_l2": norms,
                         "weighted_output_gradient_l2": {k: weights[k] * n for k, n in norms.items()},
                         "gradient_cosines": cosines})
    return {"checkpoint_step": step, "checkpoint_sha256": file_hash(checkpoint), "weights": "raw",
            "domain": "CPU FP32 d(loss)/d(vocal estimate), batch size 1",
            "values": summarize([r["values"] for r in rows]),
            "output_gradient_l2": summarize([r["output_gradient_l2"] for r in rows]),
            "weighted_output_gradient_l2": summarize([r["weighted_output_gradient_l2"] for r in rows]),
            "gradient_cosines": summarize([r["gradient_cosines"] for r in rows]),
            "vocal_error": summarize([r["vocal_error"] for r in rows]),
            "instrumental_error": summarize([r["instrumental_error"] for r in rows]),
            "examples": rows}


def parameter_gradients(config, roster):
    checkpoint = BASELINE / "runs/exp10_seed909/latest.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Separator(checkpoint_blocks=True).train()
    model.load_state_dict(state["raw_model"], strict=True)
    del state
    settings = config["exp09_settings"]
    objective = get_loss(settings["objective"])
    weights = settings["loss"]
    sampler = RemixSource(None, None, roster, settings["seed"], enabled=False)
    addresses = [sampler.address(update)[1][0] for update in (0, 5999)]
    records = {r.vocal_uuid: roster["train"][r.vocal_uuid] for r in addresses}
    args = tuple(model.parameters())
    rows = []
    with TemporaryDirectory(dir=ROOT) as name, RestrictedArchive(ARCHIVE, records) as archive:
        with Shard(Path(name), 4 * 1024**3) as shard:
            for record in records.values():
                staged = shard.stage(archive, record)
                if staged["frames"] != sampler.frames[record["uuid"]][0]:
                    raise ValueError("Diagnostic source length changed")
            for address in addresses:
                x, v, _ = shard.window(address.vocal_uuid, address.vocal_start, SEGMENT)
                mix, target = (torch.from_numpy(np.ascontiguousarray(w.T[None])) for w in (x, v))
                estimate, _ = model(mix)
                parts = objective(model, mix, estimate, target)
                keys = list(parts)
                vectors = {
                    k: torch.cat([g.detach().flatten() for g in torch.autograd.grad(
                        parts[k], args, retain_graph=j < len(keys) - 1)])
                    for j, k in enumerate(keys)
                }
                norms = {k: float(g.norm()) for k, g in vectors.items()}
                if not all(np.isfinite(n) for n in norms.values()):
                    raise ValueError("Nonfinite per-component model gradient")
                weighted = {k: n * weights[k] for k, n in norms.items()}
                cosines = {f"{a}|{b}": float(ga.dot(gb) / (norms[a] * norms[b]))
                           if norms[a] > 0 and norms[b] > 0 else None
                           for a, ga in vectors.items() for b, gb in vectors.items() if a < b}
                rows.append({"crop": asdict(address), "parameter_gradient_l2": norms,
                             "weighted_parameter_gradient_l2": weighted, "gradient_cosines": cosines})
                del vectors, parts, estimate
                print("Measured parameter gradients at update address", address.update, flush=True)
    return {"domain": "CPU FP32 d(loss)/d(all model parameters), batch size 1",
            "checkpoint_sha256": file_hash(checkpoint), "examples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-gradients-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config, roster = bind()
    if args.parameter_gradients_only:
        result = read_json(args.output)
        result["parameter_gradient_probe"] = parameter_gradients(config, roster)
        write_json(args.output, result)
        return
    with TemporaryDirectory(dir=ROOT) as name:
        folder = Path(name)
        with RemixSource(ARCHIVE, folder, roster, 909) as source:
            print("Training references staged", source.staging, flush=True)
            parity = check_real_equivalence(source, folder, roster)
            observations = []
            for update in range(0, 6000, 23):
                _, crops, mix, _, _ = source.arrays(update)
                for addr, exp11_mix in zip(crops, mix):
                    x, v, i = source.readers[addr.vocal_uuid](addr.vocal_start, SEGMENT)
                    x, v, i = x.T, v.T, i.T
                    residual = x - (v + i)
                    rounding = x.astype(np.float64) - (v.astype(np.float64) + i)
                    power = float(energy(x))
                    observations.append({
                        "crop": asdict(addr), "exp10": distribution(x), "exp11": distribution(exp11_mix),
                        "signals": {k: stereo(w) for k, w in (("mixture", x), ("vocals", v), ("instrumental", i))},
                        "max_abs": float(np.max(np.abs(residual))),
                        "channel_rms": np.sqrt(energy(residual, axis=1)).tolist(),
                        "rms": float(np.sqrt(energy(residual))), "relative_energy": float(energy(residual) / power)
                        if power else 0., "float64_rounding_max_abs": float(np.max(np.abs(rounding))),
                    })
            groups = {
                "exp10_matched_originals": (observations, "exp10"),
                "exp11_all": (observations, "exp11"),
                "exp11_original_branch": ([r for r in observations if not r["crop"]["remixed"]], "exp11"),
                "exp11_remix_branch": ([r for r in observations if r["crop"]["remixed"]], "exp11"),
                "exp10_originals_paired_to_remixes": ([r for r in observations if r["crop"]["remixed"]], "exp10"),
            }
            out = {"staging": source.staging, "disabled_equivalence": parity,
                   "unique_vocal_tracks": len({r["crop"]["vocal_uuid"] for r in observations}),
                   "unique_instrumental_tracks": len({r["crop"]["instrumental_uuid"] for r in observations}),
                   "distributions": {k: {"crops": len(rows), **summarize([r[arm] for r in rows]),
                                         "crops_over_full_scale": sum(r[arm]["peak"] > 1 for r in rows)}
                                     for k, (rows, arm) in groups.items()},
                   "stereo": {k: summarize([r["signals"][k] for r in observations])
                              for k in ("mixture", "vocals", "instrumental")},
                   "mix_residual": {k: stats([r[k] for r in observations])
                                    for k in ("max_abs", "rms", "relative_energy", "float64_rounding_max_abs")},
                   "mix_residual_channel_rms": [stats([r["channel_rms"][c] for r in observations])
                                                for c in range(2)],
                   "observations": observations}
            write_json(args.output, {**out, "output_gradient_probe": "pending"})
            print("Running final-weight CPU output gradient audit", flush=True)
            out["output_gradient_probe"] = gradient_probe(config, source)
            write_json(args.output, out)


if __name__ == "__main__":
    main()
