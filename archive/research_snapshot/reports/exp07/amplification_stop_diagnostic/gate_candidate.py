"""Offline candidate guard and controls. The production guard is unchanged."""

import json
from pathlib import Path

import numpy as np
import torch

from analyze import db, weighted_p99


ROOT = Path(__file__).resolve().parent
LIMIT_DB = 3.0


def active_scores(x, mask, estimate):
    magnitude = np.abs(x)
    active = (magnitude > 0) & (magnitude >= .001 * magnitude.max(axis=(-2, -1), keepdims=True))
    result = []
    for m, a, y, keep in zip(magnitude, mask, np.abs(estimate), active):
        if not keep.any():
            result.append(0.)
            continue
        weight = m[keep].astype(np.float64) ** 2
        old = weighted_p99(a[keep], weight)
        new = weighted_p99(y[keep] / np.maximum(m[keep], 1e-12), weight)
        result.append(db(max(new, 1e-12) / max(old, 1e-12)))
    return result


def check_candidate():
    x = np.ones((2, 2, 1, 1000), dtype=np.complex64)
    x[..., 100:] = .002
    mask = np.full(x.shape, .5, dtype=np.float32)
    baseline = x * mask
    quiet = baseline.copy()
    quiet[..., 100:] *= 2
    broadband = baseline.copy()
    broadband[0, ..., :100] *= 2
    sparse = baseline.copy()
    sparse[1, ..., :2] *= 2
    controls = {"identity": active_scores(x, mask, baseline), "quiet_bin_doubling": active_scores(x, mask, quiet),
                "one_segment_high_energy_doubling": active_scores(x, mask, broadband),
                "sparse_high_energy_doubling": active_scores(x, mask, sparse),
                "silent_identity": active_scores(x * 0, mask, x * 0)}
    assert np.quantile(np.abs(quiet) / np.abs(x), .99) / .5 > 10 ** (LIMIT_DB / 20)
    assert np.quantile(np.abs(sparse) / np.abs(x), .99) == .5
    for label in ("identity", "quiet_bin_doubling", "silent_identity"):
        assert max(controls[label]) <= LIMIT_DB
    for label in ("one_segment_high_energy_doubling", "sparse_high_energy_doubling"):
        assert max(controls[label]) > LIMIT_DB
    for gain in (.001, 10.):
        assert np.allclose(active_scores(x * gain, mask, sparse * gain), controls["sparse_high_energy_doubling"])
    return controls


if __name__ == "__main__":
    controls = check_candidate()
    capture = torch.load(ROOT / "capture.pt", map_location="cpu", weights_only=True)
    assert capture["match"]
    p = {name: tensor.numpy() for name, tensor in capture["prediction"].items()}
    scores = {name: active_scores(p["mixture_stft"], p["masks"][:, i], p[name + "_stft"])
              for i, name in enumerate(("vocals", "instrumental"))}
    result = {"synthetic_controls": controls, "reproduced_batch_scores_by_segment_db": scores,
              "reproduced_batch_passed": all(max(s) <= LIMIT_DB for s in scores.values())}
    (ROOT / "candidate_controls.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
