"""Read-only Exp10 diagnostic. Run from the project root with -B."""

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

ROOT = Path.cwd()
BASE = ROOT / "exp10_scratch"
sys.path.insert(0, str(BASE))
from train_exp10 import BatchSource, bind, setup, BACKENDS
from data import ARCHIVE, RestrictedArchive, Shard
from loss_candidates import normalized_mse_loss
from scratch_model import HOP, N_FFT, SEGMENT, Separator
from scratch_train import model_digest
from checkpoint import file_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def quantile(values):
    values = [v for v in values if v is not None]
    return (
        dict(
            zip(
                ("min", "p05", "p25", "p50", "p75", "p95", "p99", "max"),
                np.quantile(values, (0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1)).tolist(),
            )
        )
        if values
        else None
    )


def stereo(wave):
    a, b = wave.astype(np.float64)
    pm, ps = (((a + b) * 0.5) ** 2).mean(), (((a - b) * 0.5) ** 2).mean()
    return dict(
        mid_rms=float(np.sqrt(pm)),
        side_rms=float(np.sqrt(ps)),
        side_to_mid_energy_ratio=float(ps / pm) if pm > 0 else None,
    )


def distribution(rows):
    return {key: quantile([row[key] for row in rows]) for key in rows[0]}


def vector_summary(vectors, components):
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    total = sum(vectors[k] for k in components)
    norm = float(total.norm())
    summed = sum(norms[k] for k in components)
    cos = {
        f"{a}|{b}": float(x.dot(y)) / (norms[a] * norms[b]) if norms[a] * norms[b] else None
        for (a, x), (b, y) in itertools.combinations(vectors.items(), 2)
    }
    return dict(
        norm=norms,
        total_norm=norm,
        relative_to_total={k: n / norm for k, n in norms.items()},
        relative_to_sum_of_component_norms={k: n / summed for k, n in norms.items()},
        cosine=cos,
    )


def run(output):
    config, roster = bind()
    runtime = setup(909)
    settings = config["exp09_settings"]
    weights = settings["loss"]
    device = torch.device("cuda")
    run_path = BASE / "runs" / "exp10_seed909"
    cp = run_path / "latest.pt"
    idx = read(run_path / "checkpoint_index.json")["latest"]
    cp_hash = file_hash(cp)
    assert cp_hash == idx["sha256"] and cp.stat().st_size == idx["bytes"]
    state = torch.load(cp, map_location="cpu", weights_only=True)
    assert (state["step"], state["identity"]) == (24000, idx["identity"])
    history = read(run_path / "validation_history.json")
    assert history == state["history"]
    model = Separator(checkpoint_blocks=True).to(device).train()
    parameters = tuple(model.parameters())
    archived = read(ROOT / "exp11_scratch" / "audio_audit.json")
    sampler = BatchSource(ARCHIVE, None, roster, 909)
    observations = defaultdict(list)
    for row in archived["observations"]:
        addr = row["crop"]
        key = addr["vocal_uuid"]
        assert (key, addr["vocal_start"]) in sampler.address(addr["update"])[1]
        observations[key].append(row["signals"])
    assert observations.keys() == roster["train"].keys()
    assert sum(map(len, observations.values())) == len(archived["observations"])
    signals = ("mixture", "vocals", "instrumental")
    ranged = {
        key: dict(
            n=len(rows),
            artist=roster["train"][key]["artist_key"],
            signals={s: distribution([row[s] for row in rows]) for s in signals},
        )
        for key, rows in observations.items()
    }
    pooled = {
        s: distribution([row[s] for group in observations.values() for row in group])
        for s in signals
    }
    fractions = {
        s: {
            str(threshold): sum(
                row[s]["side_to_mid_energy_ratio"] < threshold
                for group in observations.values()
                for row in group
                if row[s]["side_to_mid_energy_ratio"] is not None
            )
            for threshold in (0.01, 0.1, 0.25, 1.0)
        }
        for s in signals
    }
    archive_mix_gains = []
    for group in observations.values():
        for item in group:
            m, s = (item["mixture"][k] for k in ("mid_rms", "side_rms"))
            r2 = max(m * m + s * s, 1e-8)
            archive_mix_gains.append((r2 / ((r2 + 3 * s * s) * 0.25)) ** 0.5)
    with (run_path / "steps.jsonl").open(encoding="utf-8") as handle:
        logs = [json.loads(line) for line in handle]
    assert [r["step"] for r in logs] == list(range(1, 24001))
    keys = [*weights, "loss", "gradient_l2"]
    logs_summary = {
        label: {
            k: dict(
                raw=quantile([r[k] for r in rows]),
                weighted=quantile([r[k] * weights.get(k, 1) for r in rows]),
            )
            for k in keys
        }
        for label, rows in (
            ("full", logs),
            ("first_6000", logs[:6000]),
            ("last_1000", logs[-1000:]),
        )
    }
    steps = [0, 999, 2999, 5999]
    addresses = {step: sampler.address(step)[1] for step in steps}
    needed = defaultdict(set)
    for step, crops in addresses.items():
        assert logs[step]["crops"] == [list(addr) for addr in crops]
        for key, offset in crops:
            needed[key].add(offset)
    print(f"Staging {len(needed)} approved songs one at a time", flush=True)
    batches = {}
    with TemporaryDirectory(prefix="exp12_probe_", dir=ROOT) as temp, RestrictedArchive(
        ARCHIVE, roster["train"]
    ) as zip_data:
        for key, offsets in needed.items():
            with Shard(Path(temp), roster["shard_plan"]["budget_bytes"]) as shard:
                staged = shard.stage(zip_data, roster["train"][key])
                assert staged["frames"] == roster["shard_plan"]["frames_by_uuid"][key]
                for offset in sorted(offsets):
                    batches[key, offset] = shard.window(key, offset, SEGMENT)
    print("Approved crops staged", flush=True)

    errors, gradients, digests, performance = [], [], {}, []
    arms = {"raw": state["raw_model"], "ema": state["ema"]["weights"]}
    for arm, tensors in arms.items():
        model.load_state_dict(tensors, strict=True)
        digests[arm] = model_digest(model)
        for step, crops in addresses.items():
            base = [
                torch.from_numpy(np.stack([batches[c][i].T for c in crops])).to(device)
                for i in range(3)
            ]
            x, v, i = base
            with sdpa_kernel(BACKENDS), torch.autocast("cuda", dtype=torch.bfloat16):
                y, i_est = model(x)
            for j, (uuid, start) in enumerate(crops):
                y0 = y[j : j + 1]
                with torch.no_grad():
                    parts = normalized_mse_loss(model, x[j : j + 1], y0, v[j : j + 1])
                    errors.append(
                        dict(
                            arm=arm,
                            step_index=step,
                            uuid=uuid,
                            start=start,
                            mixture=stereo(x[j].cpu().numpy()),
                            vocals=stereo(v[j].cpu().numpy()),
                            instrumental=stereo(i[j].cpu().numpy()),
                            vocal_error=stereo((y[j] - v[j]).cpu().numpy()),
                            instrumental_error=stereo((i_est[j] - i[j]).cpu().numpy()),
                            mixture_residual=float((y[j] + i_est[j] - x[j]).abs().max()),
                            loss={k: float(t) for k, t in parts.items()},
                        )
                    )
            z = y.detach().requires_grad_()
            parts = normalized_mse_loss(model, x, z, v)
            components = {k: t * weights[k] for k, t in parts.items()}
            zs = [
                torch.stft(
                    t.flatten(0, 1),
                    N_FFT,
                    HOP,
                    window=model.window.float(),
                    return_complex=True,
                    pad_mode="constant",
                ).unflatten(0, (x.size(0), 2))
                for t in (x, v, z)
            ]
            zx, zv, zy = zs
            scale = zx.abs().square().mean((1, 2, 3), keepdim=True).clamp_min(1e-8).sqrt()

            def mag(a, b):
                return (torch.log1p(a.abs() / scale) - torch.log1p(b.abs() / scale)).abs().mean()

            vocal_log, inst_log = 0.25 * mag(zy, zv), 0.25 * mag(zx - zy, zx - zv)
            wave_scale2 = x.square().mean((1, 2)).clamp_min(1e-8)
            side_scale2 = ((x[:, 0] - x[:, 1]) * 0.5).square().mean(-1)
            trial_denom = ((wave_scale2 + 3 * side_scale2) * 0.25).sqrt()
            error = z - v
            side = (error[:, 0] - error[:, 1]) * 0.5
            alternative = 0.25 * (side.abs() / trial_denom[:, None]).mean()
            probes = {
                **components,
                "vocal_logmag_split": vocal_log,
                "instrumental_logmag_split": inst_log,
                "side_capped_counterfactual": alternative,
            }
            out = {
                k: torch.autograd.grad(value, z, retain_graph=True)[0]
                for k, value in probes.items()
            }
            torch.testing.assert_close(
                out["vocal_logmag_split"] + out["instrumental_logmag_split"],
                out["logmag_l1"],
                atol=1e-7,
                rtol=1e-5,
            )
            pvec = {}
            start = perf_counter()
            for j, (k, g) in enumerate(out.items()):
                with sdpa_kernel(BACKENDS):
                    pvec[k] = torch.cat(
                        [
                            t.detach().reshape(-1)
                            for t in torch.autograd.grad(
                                y, parameters, g, retain_graph=j + 1 < len(out)
                            )
                        ]
                    )
            pvec["spectral_combined"] = pvec["complex_mse"] + pvec["logmag_l1"]
            out["spectral_combined"] = out["complex_mse"] + out["logmag_l1"]
            for vectors in (pvec, out):
                assert all(t.isfinite().all() for t in vectors.values())
            assert all(p.grad is None for p in parameters)
            performance.append(
                dict(
                    seconds=perf_counter() - start,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                )
            )
            gradients.append(
                dict(
                    arm=arm,
                    step_index=step,
                    crops=crops,
                    values={k: float(t.detach()) for k, t in probes.items()},
                    scale_gain=(wave_scale2.sqrt() / trial_denom).tolist(),
                    output=vector_summary({k: g.flatten() for k, g in out.items()}, components),
                    parameters=vector_summary(pvec, components),
                )
            )
            del (
                y,
                i_est,
                z,
                out,
                pvec,
                probes,
                parts,
                components,
                alternative,
                side,
                error,
                zs,
                zx,
                zv,
                zy,
                vocal_log,
                inst_log,
                y0,
            )
            print(arm, step, "measured", round(performance[-1]["seconds"], 2), "s", flush=True)
        assert model_digest(model) == digests[arm]
    assert file_hash(cp) == cp_hash
    result = dict(
        checkpoint_sha256=cp_hash,
        checkpoint_step=state["step"],
        checkpoint_identity=state["identity"],
        model_sha256=digests,
        runtime=runtime,
        archived_audio_sha256=file_hash(ROOT / "exp11_scratch" / "audio_audit.json"),
        step_log_sha256=file_hash(run_path / "steps.jsonl"),
        validation_sha256=file_hash(run_path / "validation_history.json"),
        pooled_stereo=pooled,
        sample_size=sum(map(len, observations.values())),
        ratio_counts_below=fractions,
        song_median_ratios={
            s: quantile(
                [v["signals"][s]["side_to_mid_energy_ratio"]["p50"] for v in ranged.values()]
            )
            for s in signals
        },
        archived_counterfactual_gain=quantile(archive_mix_gains),
        per_song=ranged,
        logged=logs_summary,
        crop_errors=errors,
        gradients=gradients,
        diagnostic_gpu=performance,
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
