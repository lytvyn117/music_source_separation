"""Deterministic development inference and the Exp09 artist-macro metrics."""
from collections import defaultdict
import math

import numpy as np
import torch

from music_separation.data.moisesdb import open_references
from music_separation.audio.overlap_add import OverlapAdd, chunk_batch, chunk_starts
from music_separation.evaluation.metrics import compute_framewise_metrics
from music_separation.models.bs_roformer import RATE, SEGMENT

METRICS = ("vocal_sdr_db", "instrumental_sdr_db", "v_side_sdr_db", "i_side_sdr_db",
           "abs_vocal_bleed_coeff")
EPS = 1e-8


def energy(samples):
    return float(np.einsum("ij,ij->", samples, samples, dtype=np.float64))


def side_sdr(ref, pred):
    ref_side = (ref[0] - ref[1]) * .5
    pred_side = (pred[0] - pred[1]) * .5
    power = float(np.dot(ref_side, ref_side))
    if power < len(ref_side) * 1e-5:
        return None
    delta = ref_side - pred_side
    return 10 * math.log10((power + EPS) / (float(np.dot(delta, delta)) + EPS))


def song_metrics(mix, vocals, inst, pred_v, pred_i):
    if any(a.shape != mix.shape or not np.isfinite(a).all()
           for a in (mix, vocals, inst, pred_v, pred_i)):
        raise ValueError("Invalid reference or prediction")
    residual = float(np.abs(pred_v + pred_i - mix).max())
    if residual > 2e-5 * max(1., float(np.abs(mix).max())):
        raise ValueError("Predictions fail mixture consistency")
    v_stats = compute_framewise_metrics(vocals, pred_v, silence_threshold_db=-45)
    i_stats = compute_framewise_metrics(inst, pred_i, silence_threshold_db=-45)
    projection = 0.
    # Accumulate matching full-song projection in float64 without full FP64 copies.
    for start in range(0, mix.shape[1], 262144):
        end = start + 262144
        error = pred_i[:, start:end].astype(np.float64) - inst[:, start:end]
        projection += float(np.einsum("ij,ij->", error, vocals[:, start:end], dtype=np.float64))
    bleed = projection / (energy(vocals) + EPS)
    return {
        "vocal_sdr_db": v_stats["median_sdr_db"],
        "instrumental_sdr_db": i_stats["median_sdr_db"],
        "v_side_sdr_db": side_sdr(vocals, pred_v),
        "i_side_sdr_db": side_sdr(inst, pred_i),
        "abs_vocal_bleed_coeff": abs(bleed),
        "signed_vocal_bleed_coeff": bleed,
        "vocal_active_frames": v_stats["active_frames"],
        "instrumental_active_frames": i_stats["active_frames"],
        "mixture_max_abs": residual,
    }


def artist_macro(rows):
    by_artist = defaultdict(list)
    for row in rows:
        by_artist[row["artist_key"]].append(row)
    result = {}
    for key in METRICS:
        means = []
        for artist in sorted(by_artist):
            values = [row[key] for row in by_artist[artist]
                      if row[key] is not None and math.isfinite(row[key])]
            if len(values) != len(by_artist[artist]):
                raise ValueError(f"Missing metric {key} in artist group {artist}")
            means.append(float(np.mean(values)))
        if not means:
            raise ValueError("Empty development panel")
        result[key] = float(np.mean(means))
    return result


@torch.inference_mode()
def evaluate(model, development, device, ids=None):
    if model.training:
        raise ValueError("Evaluation requires an evaluation-mode model")
    rows = []
    for key in development if ids is None else ids:
        with open_references(development, key) as refs:
            vocals, instrumental = (np.ascontiguousarray(r.T) for r in refs)
        mix = vocals + instrumental
        starts = chunk_starts(mix.shape[1], SEGMENT, .5)
        accumulator = OverlapAdd(2, mix.shape[1], SEGMENT, len(starts) > 1)
        for start in starts:
            wave = torch.from_numpy(chunk_batch(mix, [start], SEGMENT))
            pred, _ = model(wave.to(device))
            accumulator.add(pred[0].cpu().numpy(), start)
        pred_v = accumulator.finish()
        rows.append({
            "uuid": key, "artist_key": development[key]["artist_key"],
            "chunks": len(starts), **song_metrics(mix, vocals, instrumental, pred_v, mix - pred_v),
        })
    return {"songs": rows, "artist_macro": artist_macro(rows)}
