"""Indexed, float32 cross-song mixing on the frozen Exp10 training roster."""
from contextlib import ExitStack
from dataclasses import dataclass
import random
import shutil
from time import perf_counter

import numpy as np

from data import RestrictedArchive, Shard
from scratch_model import SEGMENT
from shard_plan import sample_position

BATCH = 4
RECIPE = "exp11-remix-v1"


@dataclass(frozen=True)
class Crop:
    seed: int
    update: int
    position: int
    remixed: bool
    vocal_uuid: str
    instrumental_uuid: str
    vocal_start: int
    instrumental_start: int


def distribution(wave):
    magnitude = np.abs(wave)
    return {
        "peak": float(magnitude.max()),
        "rms": float(np.sqrt(np.einsum("ct,ct->", wave, wave, dtype=np.float64) / wave.size)),
        "fraction_over_full_scale": float(np.count_nonzero(magnitude > 1) / wave.size),
    }


class RemixSource:
    def __init__(self, archive_path, output, roster, seed, *, enabled=True):
        self.archive_path, self.output, self.seed, self.enabled = archive_path, output, seed, enabled
        self.training, self.plan = roster["train"], roster["shard_plan"]
        self.frames = {k: (n, None) for k, n in self.plan["frames_by_uuid"].items()}
        if (self.training.keys() != self.frames.keys() or len(self.training) < 2
                or any(n < SEGMENT for n, _ in self.frames.values())):
            raise ValueError("The remix pool must consist of approved four-second training tracks")
        self.keys = sorted(self.training)
        self.partners = {key: tuple(other for other in self.keys if other != key) for key in self.keys}
        payload = sum(16 * self.frames[k][0] + 256 for k in self.keys)
        transient = max(48 * self.frames[k][0] + 8192 + max(t["bytes"] for t in self.training[k]["tracks"])
                        for k in self.keys)
        self.budget_bytes = payload + transient

    def address(self, update):
        group, originals = sample_position(self.plan["shards"], self.frames, SEGMENT, self.seed,
                                           update, BATCH, self.plan["updates_per_shard"])
        crops = []
        for position, (vocal_uuid, vocal_start) in enumerate(originals):
            rng = random.Random(f"{RECIPE}:{self.seed}:{update}:{position}")
            remixed = self.enabled and rng.random() < .5
            inst_uuid, inst_start = vocal_uuid, vocal_start
            if remixed:
                inst_uuid = rng.choice(self.partners[vocal_uuid])
                inst_start = rng.randrange(self.frames[inst_uuid][0] - SEGMENT + 1)
            crops.append(Crop(self.seed, update, position, remixed, vocal_uuid, inst_uuid,
                              vocal_start, inst_start))
        return group, crops

    def __enter__(self):
        started = perf_counter()
        if shutil.disk_usage(self.output).free < self.budget_bytes:
            raise OSError(f"Staging requires {self.budget_bytes} free bytes")
        with ExitStack() as stack:
            self.archive = stack.enter_context(RestrictedArchive(self.archive_path, self.training))
            shard = stack.enter_context(Shard(self.output, self.budget_bytes))
            # Follow ZIP member order to keep archive I/O sequential where possible.
            ordered = sorted(self.keys, key=lambda k: min(self.archive.getinfo(t["wav"]).header_offset
                                                          for t in self.training[k]["tracks"]))
            for key in ordered:
                staged = shard.stage(self.archive, self.training[key])
                if staged["frames"] != self.frames[key][0]:
                    raise ValueError(f"Training reference length changed: {key}")
            self.readers = {k: stack.enter_context(shard.reader(k)) for k in self.keys}
            self.staging = {"songs": len(self.keys), "seconds": perf_counter() - started,
                            "peak_bytes": shard.peak_reserved_bytes, "bound_bytes": self.budget_bytes}
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def arrays(self, update):
        group, crops = self.address(update)
        mixtures, vocals, instruments = [], [], []
        for crop in crops:
            mix, vocal, instrumental = self.readers[crop.vocal_uuid](crop.vocal_start, SEGMENT)
            if crop.remixed:
                _, _, instrumental = self.readers[crop.instrumental_uuid](crop.instrumental_start,
                                                                         SEGMENT)
                mix = vocal + instrumental
            if (not np.isfinite(vocal).all() or not np.isfinite(instrumental).all()
                    or not np.isfinite(mix).all()):
                raise ValueError(f"Nonfinite training audio at {crop}")
            if not np.array_equal(mix, vocal + instrumental):
                raise ValueError(f"Targets fail exact float32 additivity at {crop}")
            mixtures.append(mix.T)
            vocals.append(vocal.T)
            instruments.append(instrumental.T)
        return group, crops, np.stack(mixtures), np.stack(vocals), np.stack(instruments)
