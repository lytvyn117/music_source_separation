import numpy as np


def chunk_starts(total_samples: int, chunk_samples: int, overlap_ratio: float) -> list[int]:
    if min(total_samples, chunk_samples) <= 0 or not 0 <= overlap_ratio < 0.95:
        raise ValueError("Positive lengths and overlap in [0, 0.95) are required")
    if total_samples <= chunk_samples:
        return [0]
    last = total_samples - chunk_samples
    hop = max(1, round(chunk_samples * (1 - overlap_ratio)))
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return starts


def chunk_batch(waveform: np.ndarray, starts: list[int], chunk_samples: int) -> np.ndarray:
    if waveform.shape[-1] < chunk_samples:
        waveform = np.pad(waveform, ((0, 0), (0, chunk_samples - waveform.shape[-1])))
    return np.stack([waveform[:, start:start + chunk_samples] for start in starts])


class OverlapAdd:
    """Stream chunks and divide by the actual sum of positive Hann weights."""

    def __init__(self, channels: int, total_samples: int, chunk_samples: int, windowed: bool):
        self.audio = np.zeros((channels, total_samples), dtype=np.float32)
        self.weight = np.zeros(total_samples, dtype=np.float32)
        self.window = (np.hanning(chunk_samples + 2)[1:-1].astype(np.float32)
                       if windowed else np.ones(chunk_samples, dtype=np.float32))

    def add(self, chunk: np.ndarray, start: int) -> None:
        length = min(self.window.size, self.weight.size - start)
        end = start + length
        self.audio[:, start:end] += chunk[:, :length] * self.window[:length]
        self.weight[start:end] += self.window[:length]

    def finish(self) -> np.ndarray:
        if np.any(self.weight <= 0):
            raise ValueError("Chunks do not cover the waveform")
        self.audio /= self.weight
        return self.audio


def overlap_add_chunks(chunks, starts, total_samples: int) -> np.ndarray:
    if len(chunks) != len(starts) or not len(starts):
        raise ValueError("One start position per chunk is required")
    channels, length = chunks[0].shape
    accumulator = OverlapAdd(channels, total_samples, length, len(starts) > 1 and starts[1] < length)
    for chunk, start in zip(chunks, starts):
        accumulator.add(chunk, start)
    return accumulator.finish()
