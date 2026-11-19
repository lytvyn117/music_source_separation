import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import soundfile as sf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR
from src.audio.stft import compute_istft, compute_stft
from src.data.splits import create_train_validation_split
from src.inference.separate_track import load_separation_model, separate_waveform
from probe_core import bounded_mask, bounded_outputs, box_oracle, features, spectral_scale

LENGTH = 441000
BANDS = ((0, 80), (80, 250), (250, 1000), (1000, 4000), (4000, 8000), (8000, 22051))
BASE = ROOT / "models/experiments/exp04_asymmetric_leakage/best.pt"


def read_clip(track, source, start):
    audio, sr = sf.read(track / f"{source}.wav", start=start, frames=LENGTH,
                        dtype="float32", always_2d=True)
    assert sr == 44100 and audio.shape == (LENGTH, 2)
    return torch.from_numpy(audio.T.copy())


def manifest():
    train, validation = create_train_validation_split()
    assert len(train) == 85 and len(validation) == 15 and not set(train) & set(validation)
    assert all(track.parent == PROCESSED_DIR / "train" for track in train + validation)
    rng = random.Random(808)
    records = []
    for index, track in enumerate(rng.sample(sorted(train), 16)):
        max_start = sf.info(track / "vocals.wav").frames - LENGTH
        starts = [rng.randrange(max_start + 1) for _ in range(3)]
        start = starts[0] if index % 3 == 0 else max(
            starts, key=lambda s: float(read_clip(track, "vocals", s).square().mean()))
        records.append((track, start))
    return train, validation, records


def samples(records, device):
    for track, start in records:
        wav = torch.stack([read_clip(track, stem, start) for stem in
                           ("mixture", "vocals", "instrumental")]).to(device)
        x, v, i = compute_stft(wav)
        vp = v + .5 * (x - v - i)
        yield wav, x.unsqueeze(0), vp.unsqueeze(0)


def ratio_db(signal, error):
    return 10 * math.log10((signal + 1e-8) / (error + 1e-8))


def score(wav, estimate):
    target = wav[1:].double()
    prediction = estimate.double()
    t = target.reshape(2, 2, 10, 44100)
    e = (target - prediction).reshape_as(t)
    energy = t.square().sum(dim=(1, 3))
    error = e.square().sum(dim=(1, 3))
    scores = []
    for ref, err in zip(energy, error):
        active = ref >= 2 * 44100 * 10 ** (-45 / 10)
        frame_sdr = 10 * ((ref[active] + 1e-8) / (err[active] + 1e-8)).log10()
        scores.append(float(frame_sdr.quantile(.5)) if active.any() else None)
    bleed = ((prediction[1] - target[1]) * target[0]).sum() / target[0].square().sum().clamp_min(1e-8)
    return {"vocal_sdr": scores[0], "instrumental_sdr": scores[1], "bleed": float(bleed),
            "abs_bleed": abs(float(bleed)),
            "instrumental_mae": float((prediction[1] - target[1]).abs().mean())}


def invert(x, v):
    return compute_istft(torch.cat((v, x - v)), length=LENGTH)


def spectral_stats(x, target, oracle, baseline):
    inst = x - target
    power = target.abs().square() + inst.abs().square()
    err = (target - oracle).abs().square()
    base_err = (target - baseline).abs().square()
    near_zero = x.abs() <= .01 * x.abs().amax(dim=(-2, -1), keepdim=True)
    strong = (x.abs() <= .25 * (target.abs() + inst.abs())) & (power > 0)
    mask = near_zero & strong
    f = (torch.arange(1025, device=x.device) * 44100 / 2048)[None, None, :, None]
    bands = []
    for lo, hi in BANDS:
        band = ((f >= lo) & (f < hi)).expand_as(power)
        bands.append([float(a[band].sum()) for a in
                      (target.abs().square(), inst.abs().square(), err, base_err)])
    real_range = torch.where(x != 0, target / torch.where(x != 0, x, 1), 0)
    outside = ((real_range.real < -.15) | (real_range.real > 1.15)
               | (real_range.imag.abs() > .25))
    return {"bands": bands, "quiet_cancellation_bins": int(mask.sum()),
            "total_bins": x.numel(), "source_energy": float(power.sum()),
            "quiet_cancellation_source_energy": float(power[mask].sum()),
            "quiet_cancellation_oracle_error": float(err[mask].sum()),
            "quiet_cancellation_exp04_error": float(base_err[mask].sum()),
            "oracle_total_error": float(err.sum()), "outside_box_bins": int(outside.sum()),
            "outside_box_source_energy": float(power[outside].sum())}


def median(rows, key):
    return statistics.median(row[key] for row in rows if row[key] is not None)


def aggregate_oracle(rows):
    methods = ("exp04", "bounded", "unbounded")
    metrics = ("vocal_sdr", "instrumental_sdr", "bleed", "abs_bleed", "instrumental_mae")
    summary = {method: {key: median([row[method] for row in rows], key) for key in metrics}
               for method in methods}
    for method in methods[1:]:
        summary[method]["paired_delta"] = {
            key: statistics.median(row[method][key] - row["exp04"][key] for row in rows
                                   if row[method][key] is not None and row["exp04"][key] is not None)
            for key in metrics
        }
    spectral = [row["spectral"] for row in rows]
    summary["spectral"] = {key: sum(s[key] for s in spectral) for key in spectral[0] if key != "bands"}
    band_results = []
    for index, bounds in enumerate(BANDS):
        v, i, err, base_err = [sum(s["bands"][index][j] for s in spectral) for j in range(4)]
        band_results.append({"hz": bounds, "vocal_bounded_db": ratio_db(v, err),
                             "vocal_exp04_db": ratio_db(v, base_err),
                             "instrumental_bounded_db": ratio_db(i, err),
                             "instrumental_exp04_db": ratio_db(i, base_err),
                             "bounded_vocal_error": err})
    summary["bands"] = band_results
    return summary


def pack_sample(wav, x, target):
    freq = torch.linspace(0, 1, 1025, device=x.device)[:, None].expand(1025, x.shape[-1])
    feat = torch.cat((features(x)[0], freq[None])).flatten(1).T.contiguous()
    edge = ((freq == 0) | (freq == 1)).flatten()
    return wav, feat, x, target, edge, spectral_scale(x)


def point_model():
    return nn.Sequential(nn.Linear(7, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 4))


@torch.no_grad()
def assess(model, dataset):
    metrics = []
    for wav, feat, x, target, _, scale in dataset:
        logits = torch.cat([model(block) for block in feat.split(65536)])
        logits = logits.T.reshape(1, 4, *x.shape[-2:])
        v, _ = bounded_outputs(logits, x)
        err = ((v - target) / scale).abs()
        row = score(wav, invert(x, v))
        row.update(l1=float(err.mean()), mse=float(err.square().mean()))
        metrics.append(row)
    return {key: median(metrics, key) for key in metrics[0]}


def fit_arm(dataset, seed, power, device):
    torch.manual_seed(seed)
    model = point_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = random.Random(8080 + seed)
    gen = torch.Generator(device=device).manual_seed(8080 + seed)
    before = {"fit": assess(model, dataset[:10]), "held": assess(model, dataset[10:])}
    norms, losses = [], []
    max_bound_excess, max_additivity_ratio = 0., 0.
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        losses_step = []
        for index in rng.sample(range(10), 2):
            _, feat, x, target, edge, scale = dataset[index]
            ids = torch.randint(feat.shape[0], (16384,), generator=gen, device=device)
            z = x.reshape(1, 2, -1)[:, :, ids]
            logits = model(feat[ids]).T.unsqueeze(0)
            v = bounded_mask(logits, edge[ids]) * z
            i = z - v
            distance = ((v - target.reshape(1, 2, -1)[:, :, ids]) / scale.flatten()[0]).abs()
            losses_step.append((distance if power == 1 else distance.square()).mean())
            with torch.no_grad():
                max_bound_excess = max(max_bound_excess, float(
                    (torch.maximum(v.abs(), i.abs()) - 1.1768603 * z.abs()).max()))
                tol = 1e-5 + 4 * 2**-24 * (z.abs() + v.abs() + i.abs())
                max_additivity_ratio = max(max_additivity_ratio, float(((v+i-z).abs()/tol).max()))
        loss = torch.stack(losses_step).mean()
        assert torch.isfinite(loss)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        assert all(torch.isfinite(param).all() for param in model.parameters())
        norms.append(float(norm))
        losses.append(float(loss))
    after = {"fit": assess(model, dataset[:10]), "held": assess(model, dataset[10:])}
    return {"seed": seed, "power": power, "before": before, "after": after,
            "gradient_norm": {"first": norms[0], "min": min(norms),
                              "median": statistics.median(norms), "max": max(norms),
                              "clipped_steps": sum(n > 1 for n in norms)},
            "max_bound_excess": max_bound_excess, "max_additivity_ratio": max_additivity_ratio,
            "nonfinite_steps": 0, "step_loss": losses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, records = manifest()
    base, meta, _ = load_separation_model(BASE, device)
    dataset, rows = [], []
    with torch.no_grad():
        for (track, start), (wav, x, target) in zip(records, samples(records, device)):
            pred = separate_waveform(base, wav[0].cpu().numpy(), device=device,
                                     segment_length_seconds=6., overlap_ratio=.5)
            pair = torch.stack([torch.from_numpy(pred[key]) for key in ("vocals", "instrumental")]).to(device)
            v = box_oracle(x, target)
            row = {"track": track.name, "start_sample": start,
                   "exp04": score(wav, pair), "bounded": score(wav, invert(x, v)),
                   "unbounded": score(wav, invert(x, target)),
                   "spectral": spectral_stats(x, target, v, compute_stft(pair[:1]))}
            rows.append(row)
            dataset.append(pack_sample(wav, x, target))
    summary = aggregate_oracle(rows)
    print(json.dumps(summary, indent=2), flush=True)
    results = []
    for seed in (42, 43, 44):
        for power in (1, 2):
            result = fit_arm(dataset, seed, power, device)
            results.append(result)
            print({k: v for k, v in result.items() if k != "step_loss"}, flush=True)
    with BASE.open("rb") as handle:
        baseline_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    with args.output.open("x", encoding="utf-8") as out:
        json.dump({"torch": torch.__version__, "device": str(device),
                   "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                   "baseline_epoch": meta["epoch"], "baseline_sha256": baseline_hash,
                   "train": [p.name for p in train], "validation": [p.name for p in val],
                   "fit_tracks": [p.name for p, _ in records[:10]],
                   "held_tracks": [p.name for p, _ in records[10:]],
                   "oracle": summary, "clips": rows, "fit_arms": results}, out, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
