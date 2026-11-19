"""Bounded, temporary reference shards for sample-aligned stereo loading."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import soundfile as sf

RATE = 44100
CHANNELS = 2
BLOCK = 262144


def blocks(frames):
    for start in range(0, frames, BLOCK):
        yield start, min(start + BLOCK, frames)


def bwf_reference(path):
    """Read the optional BWF sample timestamp, without reading audio data."""
    with path.open("rb") as handle:
        header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return None
        while len(chunk := handle.read(8)) == 8:
            tag, size = struct.unpack("<4sI", chunk)
            if tag == b"bext" and size >= 346:
                handle.seek(338, 1)
                return struct.unpack("<Q", handle.read(8))[0]
            handle.seek(size + size % 2, 1)
    return None


class SignalStats:
    def __init__(self, pcm_bits=None):
        self.peak = 0.0
        self.samples = 0
        self.samples_over_full_scale = 0
        self.samples_at_or_over_full_scale = 0
        self.pcm_bits = pcm_bits
        self.pcm_rail_samples = 0
        self.first_active_frame = None
        self.last_active_frame = None

    def update(self, signal, start):
        if not np.isfinite(signal).all():
            raise ValueError("Nonfinite samples")
        magnitude = np.abs(signal)
        self.peak = max(self.peak, float(magnitude.max()))
        self.samples += signal.size
        self.samples_over_full_scale += int(np.count_nonzero(magnitude > 1))
        self.samples_at_or_over_full_scale += int(np.count_nonzero(magnitude >= 1))
        if self.pcm_bits:
            high = 1 - 2.0 ** (1 - self.pcm_bits)
            self.pcm_rail_samples += int(np.count_nonzero((signal >= high) | (signal <= -1)))
        active = np.flatnonzero(np.any(magnitude > 1e-5, axis=1))
        if len(active):
            if self.first_active_frame is None:
                self.first_active_frame = int(start + active[0])
            self.last_active_frame = int(start + active[-1])

    def report(self):
        return {k: v for k, v in vars(self).items() if k != "pcm_bits"}


@contextmanager
def mapped(path, mode, **kwargs):
    array = np.lib.format.open_memmap(path, mode=mode, **kwargs)
    try:
        yield array
    finally:
        if mode != "r":
            array.flush()
        # Release file handles before temporary file cleanup on Windows.
        array._mmap.close()


class Shard:
    """Single-writer, process-local shard. Context exit removes its own scratch files."""
    def __init__(self, root, budget_bytes):
        self.root = Path(root)
        self.budget = budget_bytes
        self.songs = {}
        self.payload_bytes = 0
        self.peak_reserved_bytes = 0

    def __enter__(self):
        self.temp = TemporaryDirectory(prefix="exp09_shard_", dir=self.root)
        self.path = Path(self.temp.name)
        return self

    def __exit__(self, *exception):
        self.temp.cleanup()

    def _source(self, archive, track, path):
        info = archive.getinfo(track["wav"])
        if (not track["available"] or info.file_size != track["bytes"]
                or f"{info.CRC:08x}" != track["crc32"]):
            raise ValueError(f'Archive does not match manifest for {track["wav"]}')
        start = perf_counter()
        with archive.open(info) as source, path.open("wb") as handle:
            shutil.copyfileobj(source, handle, 1024 * 1024)
        return perf_counter() - start

    def stage(self, archive, song, *, verify=False, common_span=False):
        """Stage all approved tracks. Default requires equal native waveform lengths."""
        key = song["uuid"]
        if key in self.songs:
            raise ValueError("Song already staged")
        tracks = sorted(song["tracks"], key=lambda t: archive.getinfo(t["wav"]).header_offset)
        if (not tracks or len({t["wav"] for t in tracks}) != len(tracks)
                or {t["stem"] == "vocals" for t in tracks} != {True, False}):
            raise ValueError("Require distinct vocal and instrumental source tracks")
        signatures = {(t["bytes"], t["crc32"]) for t in tracks}
        if len(signatures) != len(tracks):
            raise ValueError("Repeated WAV signature")
        max_source = max(t["bytes"] for t in tracks)
        if self.payload_bytes + max_source > self.budget:
            raise ValueError("Shard byte budget cannot fit largest WAV")
        supported = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": None}
        stat_rows, reference_times = [], []
        unpack_seconds = 0.0
        started = perf_counter()
        with TemporaryDirectory(prefix="build_", dir=self.path) as folder:
            work = Path(folder)
            raw = work / "track.wav"
            accum = work / "accum.npy"
            cached = work / "refs.npy"
            unpack_seconds += self._source(archive, tracks[0], raw)
            with sf.SoundFile(raw) as audio:
                first_frames = len(audio)
            # No source exceeds the first allocation. For common-span mode, the
            # final reference is reduced to the minimum successfully read length.
            limit = self.payload_bytes + max_source + 48 * first_frames + 8192
            if first_frames < 1 or limit > self.budget:
                raise ValueError("Shard byte budget cannot fit accumulator and output")
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, limit)
            common = first_frames
            check_ranges = [
                (int(f * max(0, first_frames - 4096)), min(4096, first_frames))
                for f in (0.0, 0.5, 0.95)
            ] if verify else []
            direct = [np.zeros((n, CHANNELS), dtype="float64") for _, n in check_ranges]

            with mapped(accum, "w+", shape=(2, first_frames, CHANNELS), dtype="float64") as totals:
                totals[:] = 0
                for index, track in enumerate(tracks):
                    if index:
                        unpack_seconds += self._source(archive, track, raw)
                    reference_times.append(bwf_reference(raw))
                    with sf.SoundFile(raw) as audio:
                        frames = len(audio)
                        if (audio.samplerate != RATE or audio.channels != CHANNELS or
                                frames < 1 or audio.subtype not in supported or
                                (frames != first_frames and not common_span)):
                            raise ValueError(f'Unsupported format or length in {track["wav"]}: '
                                             f'{audio.samplerate} Hz, {audio.channels} ch, '
                                             f'{frames} frames, {audio.subtype}; first {first_frames}')
                        common = min(common, frames)
                        stats = SignalStats(supported[audio.subtype])
                        channel = 0 if track["stem"] == "vocals" else 1
                        read_frames = frames if verify else common
                        for begin, end in blocks(read_frames):
                            signal = audio.read(end - begin, dtype="float64", always_2d=True)
                            if len(signal) != end - begin:
                                raise ValueError("Truncated audio data")
                            if verify:
                                stats.update(signal, begin)
                            elif not np.isfinite(signal).all():
                                raise ValueError("Nonfinite samples")
                            stop = min(end, common)
                            if stop > begin:
                                totals[channel, begin:stop] += signal[:stop - begin]
                        for (begin, size), total in zip(check_ranges, direct):
                            stop = min(begin + size, common)
                            if stop > begin:
                                audio.seek(begin)
                                total[:stop - begin] += audio.read(stop - begin, dtype="float64",
                                                                  always_2d=True)
                        stat_rows.append({"wav": track["wav"], "stem": track["stem"],
                                          "frames": frames, "rate": audio.samplerate,
                                          "channels": audio.channels, "subtype": audio.subtype,
                                          "bwf_start": reference_times[-1],
                                          **(stats.report() if verify else {})})
                    raw.unlink()
                if len({ref for ref in reference_times if ref is not None}) > 1:
                    raise ValueError("Conflicting BWF start timestamps")
                combined = [SignalStats() for _ in range(3)]
                with mapped(cached, "w+", shape=(2, common, CHANNELS), dtype="float32") as refs:
                    for begin, end in blocks(common):
                        refs[:, begin:end] = totals[:, begin:end]
                        if verify:
                            v, i = refs[:, begin:end]
                            for stat, data in zip(combined, (v, i, v + i)):
                                stat.update(data, begin)
                    residuals = []
                    for (begin, size), total in zip(check_ranges, direct):
                        size = min(size, common - begin)
                        if size > 0:
                            v, i = refs[:, begin:begin + size]
                            residuals.append(float(np.max(np.abs((v + i) - total[:size]))))
            destination = self.path / (work.name + ".npy")
            cached.replace(destination)
        bytes_written = destination.stat().st_size
        self.payload_bytes += bytes_written
        self.songs[key] = (destination, common)
        return {
            "uuid": key, "frames": common, "duration_seconds": common / RATE,
            "tracks": stat_rows, "length_policy": "common_span" if common_span else "strict",
            "frame_length_spread": max(row["frames"] for row in stat_rows)
                                   - min(row["frames"] for row in stat_rows),
            "timing": {"stage_seconds": perf_counter() - started,
                       "zip_to_temporary_wav_seconds": unpack_seconds},
            "cache_bytes": bytes_written, "peak_reserved_bytes": self.peak_reserved_bytes,
            **({"reconstruction_max_abs": max(residuals, default=None),
                "reconstruction_windows": [r for r in check_ranges if r[0] < common],
                "signals": dict(zip(("vocals", "instrumental", "mixture"),
                                    [s.report() for s in combined]))} if verify else {}),
        }

    @contextmanager
    def reader(self, key):
        """Open once per worker/song for repeated random reads within the shard lifetime."""
        path, frames = self.songs[key]
        with mapped(path, "r") as refs:
            def window(start, size):
                if start < 0 or size < 1 or start + size > frames:
                    raise ValueError("Requested crop falls outside the shared frame span")
                vocal, instrumental = refs[:, start:start + size].copy()
                return vocal + instrumental, vocal, instrumental
            yield window

    def window(self, key, start, size):
        with self.reader(key) as read:
            return read(start, size)
