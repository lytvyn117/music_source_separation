"""Freeze development eligibility and excerpt addresses before inference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import itertools
import json
import os
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
SCRATCH = OUT.parent
BASE = SCRATCH.with_name("music_source_separation")
ARCHIVE = Path(r"C:\Users\YOUR_USER\Downloads\moisesdb.zip")
sys.path.insert(0, str(SCRATCH))

from audit import artist_key, norm, write_json
from eligibility import scan_song, signature, fingerprint_candidates, wave_match, FP_RATE
from loader import RATE, Shard
from musdb_eligibility import moises_mix, musdb_mix, track_fields

GIB = 1024**3
ACTIVE = 10**(-45/20)
EXPECTED = "f4b8888aa68720778c447b926e185a9114f37f3c8f73e44746eb94a8ad230f4d"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def preserve():
    source = SCRATCH / "runs/exp09_mse_seed909/latest.pt"
    target = OUT / "exp09_step24000.pt"
    assert hash_file(source) == EXPECTED
    if not target.exists():
        with target.open("xb") as dst, source.open("rb") as src:
            while block := src.read(2**22):
                dst.write(block)
        os.chmod(target, 0o444)
    assert hash_file(target) == hash_file(source) == EXPECTED
    with ZipFile(target) as z:
        assert z.testzip() is None
    state = torch.load(target, map_location="cpu", weights_only=True)
    run = read(source.with_name("run.json"))
    assert state["index"] == 24000 and state["signature"]["data"] == run["identity"]
    identity = hashlib.sha256()
    for n in ("scratch_model.py", "scratch_train.py", "loss_candidates.py", "shard_plan.py",
              "loader.py", "train_exp09.py"):
        identity.update((SCRATCH/n).read_bytes())
    identity.update(json.dumps(run["decision"], sort_keys=True).encode())
    assert identity.hexdigest() == run["identity"]
    assert all(hash_file(SCRATCH / name) == sha for name, sha in run["decision"]["data_sha256"].items())
    bases = read(BASE / "reports/exp04_exp08_hybrid_review_20261111/freeze_manifest.json")
    checked = {}
    for name, entry in bases["checkpoints"].items():
        path = Path(entry["path"])
        assert hash_file(path) == entry["sha256"]
        checked[name] = entry
    verified_code = {}
    for path, sha in bases["source_files"].items():
        if path.startswith("src/") or path == "config.py":
            assert hash_file(BASE/path) == sha, path
            verified_code[path] = sha
    evidence = {"exp09": {"path": str(target), "sha256": EXPECTED, "index": state["index"],
                          "signature": state["signature"], "readonly": not bool(target.stat().st_mode & 0o222)},
                "baselines": checked, "baseline_code": verified_code,
                "training_identity": run["identity"], "inputs": run["decision"]["data_sha256"]}
    assert evidence["exp09"]["readonly"]
    write_json(OUT / "checkpoint_provenance.json", evidence)
    print("Verified read-only checkpoint copy and pinned baselines", flush=True)


def inspect(archive, row):
    audio, proxy = scan_song(archive, row)
    errors = []
    if row["flags"] or not row["vocal_wavs"] or not row["instrumental_wavs"]:
        errors.append("metadata flags or missing source group")
    if audio["missing_bwf"] or audio["time_reference_count"] != 1:
        errors.append("BWF start timestamps missing or inconsistent")
    if audio["frames_min"] != audio["frames_max"]:
        if any(t["tail_active_samples"] for t in audio["tracks"]):
            errors.append("unequal lengths with active source tail")
    if any(t["rail_samples"] for t in audio["tracks"]):
        errors.append("source PCM saturation")
    hashes = [t["pcm_sha256"] for t in audio["tracks"]]
    if len(hashes) != len(set(hashes)):
        errors.append("repeated source PCM")
    if min(audio["rms"]["vocals"], audio["rms"]["instrumental"]) < ACTIVE:
        errors.append("inactive source")
    return audio, proxy, errors


def choose_excerpt(v, i):
    """12 s, maximizes jointly active seconds with central-song tie break."""
    n = len(v)//RATE
    if n < 12:
        raise ValueError("No full 12-second example")
    def energy(x):
        return np.sqrt(np.mean(x[:n*RATE].reshape(n,RATE,2)**2,axis=(1,2)))
    vr, ir = energy(v), energy(i)
    valid = (vr >= ACTIVE) & (ir >= ACTIVE)
    count = np.convolve(valid, np.ones(12,dtype=int), "valid")
    options = np.where(count == count.max())[0]
    start = min(options, key=lambda s: (abs(int(s) + 6 - n/2), int(s)))
    return int(start * RATE), int(count[start])


def baseline_proxy(path):
    # Include official training only (fit and historical model selection).
    return musdb_mix(path)[0]


def main():
    preserve()
    sample = read(SCRATCH/"split_external_final.json")
    songs = {r["uuid"]:r for r in read(SCRATCH/"manifest.json")}
    strict = read(SCRATCH/"training_eligibility.json")
    training = set(strict["strict_train"])
    dev = [a["uuid"] for a in sample["assignments"] if a["role"] == "validation"]
    dev_set = set(dev)
    assert len(dev_set) == len(dev) == 20 and not dev_set & training
    assert not {songs[k]["artist_key"] for k in dev} & {songs[k]["artist_key"] for k in training}
    baseline_paths = sorted((BASE/"data/processed/train").iterdir())
    assert len(baseline_paths) == 100 and all(p.is_dir() for p in baseline_paths)
    baseline_names = {p.name: track_fields(p.name) for p in baseline_paths}
    musmeta = read(SCRATCH/"integrity.json")["musdb_roster"]
    assert {p.name+".mp4" for p in baseline_paths} == {
        p.name for p in (BASE/"data/raw/train").glob("*.stem.mp4")}
    blocked_artists = {artist_key(r["artist"]) for r in musmeta}
    decoded = read(SCRATCH/"eligibility_complete.json")["rows"]
    fit_stems = {t["pcm_sha256"]:k for k in training for t in decoded[k]["tracks"]}
    good = {}
    proxies = {}
    with ZipFile(ARCHIVE) as archive:
        for k in dev:
            row = songs[k]
            try:
                audio, proxy, errors = inspect(archive, row)
            except (ValueError, OSError, RuntimeError) as exc:
                good[k] = {"reasons":[str(exc)]}
                continue
            if row["artist_key"] in blocked_artists:
                errors.append("shared MUSDB18 artist")
            if any(t["pcm_sha256"] in fit_stems for t in audio["tracks"]):
                errors.append("identical source PCM in Exp09 training")
            proxies[k] = proxy
            good[k] = {**audio,"reasons":errors}
            print("Reference audit", k, len(errors), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k, wave in zip(sorted(training), pool.map(
                lambda k: moises_mix(ARCHIVE, songs[k], strict["rows"][k]["frames_min"]),
                sorted(training))):
            proxies[k] = wave
        print("Screened Exp09 training audio", flush=True)
        for p,wave in zip(baseline_paths,pool.map(baseline_proxy,baseline_paths)):
            proxies[p.name] = wave
        print("Screened MUSDB18 official train audio", flush=True)

    marks = {k:signature(v) for k,v in proxies.items()}
    pairs = {tuple(sorted((a,b))):f"{votes} aligned spectral hashes" for a,b,votes in
             fingerprint_candidates(marks) if a in dev_set or b in dev_set}
    for k in dev_set & proxies.keys():
        title = norm(songs[k]["title"])
        for other in proxies.keys() - {k}:
            name = songs[other]["title"] if other in songs else baseline_names[other][1]
            if norm(name) == title:
                pairs[tuple(sorted((k,other)))] = "normalized title"
    comparisons = []
    for (a,b), reason in sorted(pairs.items()):
        hits = wave_match(proxies[a], proxies[b])
        strong = [h for h in hits if h["correlation"]>=.70]
        likely = len(strong)>=2 and max(h["lag_seconds"] for h in strong)-min(h["lag_seconds"] for h in strong)<.25
        comparisons.append({"a":a,"b":b,"reason":reason,"likely":likely,"measurements":hits})
        if likely:
            for key in {a,b} & dev_set:
                good[key]["reasons"].append("probable duplicate or alternate release")

    admitted = []
    with ZipFile(ARCHIVE) as archive:
        for k in dev:
            g = good[k]
            g.update({f:songs[k][f] for f in ("uuid","artist","artist_key","title","genre")})
            if g["reasons"]:
                print("Excluded",k,g["reasons"],flush=True)
                continue
            with Shard(OUT,4*GIB) as shard:
                common = g["frames_min"] != g["frames_max"]
                verify = shard.stage(archive,songs[k],verify=True,common_span=common)
                if verify["reconstruction_max_abs"] > 1e-6:
                    g["reasons"].append("direct stem sum differs from reconstructed mixture")
                    continue
                with shard.reader(k) as win:
                    x,v,i = win(0,verify["frames"])
                if max(float(np.max(np.abs(x-v-i))),0) > 1e-6:
                    raise ValueError("Shared reference sum exceeds tolerance")
                g["excerpt_frame"],g["excerpt_active_seconds"] = choose_excerpt(v,i)
                g["mixed_sum_residual"] = verify["reconstruction_max_abs"]
                g["policy"] = "shared span with tail below -100 dBFS" if common else "strict native span"
                g["reference_hashes"] = {stem:hashlib.sha256(y.data).hexdigest()
                                          for stem,y in zip(("mix","vocals","instrumental"),(x,v,i))}
                admitted.append(k)
    # Purely reference-based excerpts. Equal artist representation first, then stronger
    # simultaneous activity, with stable UUID tie breaks.
    by_artist = {}
    for k in admitted:
        by_artist.setdefault(good[k]["artist_key"],[]).append(k)
    for items in by_artist.values():
        items.sort(key=lambda k:(-good[k]["excerpt_active_seconds"],k))
    listen = []
    while len(listen) < min(10,len(admitted)):
        for a in sorted(by_artist):
            if by_artist[a] and len(listen)<10:
                listen.append(by_artist[a].pop(0))
    result = {"songs":good,"admitted":admitted,"listening":listen,"candidates":comparisons,
              "screen":{"exp09_train":60,"musdb_fit_and_selection":100,
                        "musdb_official_test":"metadata only","official_test_audio":0,
                        "moises_development":20,"moises_reserve_or_final_audio":0}}
    write_json(OUT/"frozen_panel.json",result)
    print("Panel frozen",len(admitted),"songs",len({good[k]["artist_key"] for k in admitted}),"artists",
          len(comparisons),"acoustic candidates",flush=True)


if __name__ == "__main__":
    main()
