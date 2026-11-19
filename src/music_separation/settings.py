"""Fixed checkpoint architecture and validated runtime settings.

Changing the model profile creates a different model; it is not an inference option.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass(frozen=True)
class ModelSettings:
    sample_rate: int = 44100
    n_fft: int = 2048
    stft_hop: int = 441
    segment_samples: int = 176400
    bands: tuple[int, ...] = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
    width: int = 256
    heads: int = 8
    pairs: int = 6

MODEL = ModelSettings()

@dataclass(frozen=True)
class InferenceSettings:
    hop_samples: int
    threads: int
    seed: int
    attention_backends: tuple[str, ...]
    cpu_group_chunk: int = 8

    def __post_init__(self):
        for name in ("hop_samples", "threads", "seed", "cpu_group_chunk"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 < self.hop_samples <= MODEL.segment_samples:
            raise ValueError("hop_samples must be between 1 and segment_samples")
        if self.threads < 1 or self.seed < 0 or self.cpu_group_chunk < 1:
            raise ValueError("threads and cpu_group_chunk must be positive; seed must be nonnegative")
        if not self.attention_backends or any(x not in ("EFFICIENT_ATTENTION", "MATH") for x in self.attention_backends):
            raise ValueError("Unsupported attention backend")

    def to_dict(self):
        return asdict(self)


def load_inference_settings(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"hop_samples", "threads", "seed", "attention_backends"}
    if not isinstance(values, dict) or not expected <= set(values) or set(values) - expected - {"cpu_group_chunk"}:
        raise ValueError(f"Inference configuration requires these fields (cpu_group_chunk is optional): {sorted(expected)}")
    if not isinstance(values["attention_backends"], list):
        raise ValueError("attention_backends must be a list")
    values["attention_backends"] = tuple(values["attention_backends"])
    return InferenceSettings(**values)
