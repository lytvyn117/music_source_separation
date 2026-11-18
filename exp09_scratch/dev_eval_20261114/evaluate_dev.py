"""Deterministic inference, paired reporting and native-level development listening."""
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import csv
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from panel import OUT, SCRATCH, BASE, ARCHIVE, RATE, ACTIVE, choose_excerpt, read, hash_file, write_json
from loader import Shard
from scratch_model import Separator, SEGMENT

sys.path.insert(0,str(BASE))
from src.evaluation.metrics import compute_framewise_metrics, compute_leakage_diagnostics
from src.experiments.exp04_exp08_hybrid.inference import HybridSeparator, fixed_blend, separate_exp04
from src.experiments.exp07_phase_aware_residual.chunking import OverlapAdd, chunk_starts, chunk_batch
from src.experiments.exp08_subband.inference import separate as exp08_output
from src.experiments.exp08_subband.train import configure_determinism

NAMES=("exp04","hybrid","exp09")
GIB=1024**3
BOOTSTRAP=10000


@torch.inference_mode()
def exp09_output(model, x, device):
    starts=chunk_starts(x.shape[-1],SEGMENT,.5)
    acc=OverlapAdd(2,x.shape[-1],SEGMENT,len(starts)>1)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        for start in starts:
            wave=torch.from_numpy(chunk_batch(x,[start],SEGMENT)).to(device)
            v,_=model(wave)
            acc.add(v[0].float().cpu().numpy(),start)
    v=acc.finish()
    return v,x-v


def energy(x):
    return float(np.einsum("ij,ij->",x,x,dtype=np.float64))


def db_ratio(a,b):
    return float(10*np.log10((a+1e-8)/(b+1e-8)))


def source_stats(r,y):
    s=compute_framewise_metrics(r,y,silence_threshold_db=-45)
    side=lambda x:(x[0]-x[1])*.5
    sr,sy=side(r),side(y)
    side_ref=float(np.dot(sr,sr))
    side_err=sr-sy
    p=float(np.abs(y).max())
    # Sample peak and first-difference level are measured artifact proxies.
    return {"sdr_db":s["median_sdr_db"],"si_sdr_db":s["median_si_sdr_db"],
            "active_frames":s["active_frames"],
            "level_error_db":db_ratio(energy(y),energy(r)),
            "gain_coeff":float(np.einsum("ij,ij->",r,y,dtype=np.float64)/(energy(r)+1e-8)),
            "side_sdr_db":db_ratio(side_ref,float(np.dot(side_err,side_err))) if side_ref >= len(sr)*1e-5 else None,
            "side_level_error_db":db_ratio(float(np.dot(sy,sy)),side_ref) if side_ref >= len(sr)*1e-5 else None,
            "peak":p, "over_full_scale_pct":float(100*np.count_nonzero(np.abs(y)>1)/y.size),
            "first_difference_level_error_db":db_ratio(energy(np.diff(y)),energy(np.diff(r)))}


def measure(x,v,i,y,z):
    if any(s.shape!=x.shape or not np.isfinite(s).all() for s in (v,i,y,z)):
        raise ValueError("Invalid stereo prediction or target")
    voc,inst=source_stats(v,y),source_stats(i,z)
    bleed=compute_leakage_diagnostics(v,i,y,z)
    residual=(y+z)-x
    seconds=v.shape[-1]//RATE
    def rms(w):
        return np.sqrt(np.mean(w[:,:seconds*RATE].reshape(2,seconds,RATE)**2,axis=(0,2)))
    vr,ir=rms(v),rms(i)
    quiet=vr<ACTIVE
    buried=(vr>=ACTIVE)&(vr<=ir*.31622777)
    def slice_scores(flags):
        values=[]
        for n in np.flatnonzero(flags):
            a,b=v[:,n*RATE:(n+1)*RATE],y[:,n*RATE:(n+1)*RATE]
            values.append(db_ratio(energy(a),energy(a-b)))
        return float(np.median(values)) if values else None
    return {"vocal_sdr_db":voc["sdr_db"],"vocal_si_sdr_db":voc["si_sdr_db"],
            "instrumental_sdr_db":inst["sdr_db"],"instrumental_si_sdr_db":inst["si_sdr_db"],
            "vocal_bleed_coeff":bleed["vocal_to_inst_bleed_coeff"],
            "abs_vocal_bleed_coeff":abs(bleed["vocal_to_inst_bleed_coeff"]),
            "vocal_bleed_energy_pct":bleed["vocal_to_inst_bleed_energy_pct"],
            "instrumental_bleed_coeff":bleed["inst_to_vocal_bleed_coeff"],
            "vocal_retention":voc["gain_coeff"],
            "mixture_max_abs":float(np.abs(residual).max()),
            "mixture_nrmse":float(np.sqrt(energy(residual)/(energy(x)+1e-8))),
            "mixture_interior_max_abs":float(np.abs(residual[:,RATE:-RATE]).max()) if x.shape[-1]>2*RATE else None,
            "below_threshold_vocal_rms":float(np.sqrt(np.mean(rms(y)[quiet]**2))) if quiet.any() else None,
            "buried_vocal_sdr_db":slice_scores(buried),"buried_frames":int(buried.sum()),
            **{f"{prefix}_{key}":val for prefix,source in (("v",voc),("i",inst))
               for key,val in source.items() if key not in {"sdr_db","si_sdr_db"}}}


def paired(rows,baseline,key,threshold, direction=1):
    groups=defaultdict(list)
    data=[]
    for row in rows:
        x,y=row["exp09"].get(key),row[baseline].get(key)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            groups[row["artist_key"]].append(x-y)
            data.append(x-y)
    if not groups:
        return None
    g=np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng=np.random.default_rng(42)
    sampled=rng.choice(g,(BOOTSTRAP,len(g)),replace=True).mean(axis=1)
    wins=sum(direction*x>threshold for x in data)
    losses=sum(direction*x< -threshold for x in data)
    return {"songs":len(data),"artists":len(g),"artist_macro_delta":float(g.mean()),
            "ci95_artist_macro_delta":np.quantile(sampled,[.025,.975]).tolist(),
            "paired_median":float(np.median(data)),"wins":wins,"losses":losses,"ties":len(data)-wins-losses,
            "regressions_gt_1db":sum(direction*x< -1 for x in data) if key.endswith("_db") else None}


def aggregate(rows,arm):
    by_artist=defaultdict(list)
    for r in rows:
        by_artist[r["artist_key"]].append(r[arm])
    result={}
    for key in rows[0][arm]:
        vals=[r[arm][key] for r in rows if r[arm][key] is not None and np.isfinite(r[arm][key])]
        if not vals:
            continue
        groups=[[r[key] for r in vals if r[key] is not None and np.isfinite(r[key])]
                for vals in by_artist.values()]
        result[key]={"median":float(np.median(vals)),"artist_macro":float(np.mean([np.mean(g) for g in groups if g])),
                     "songs":len(vals),"artists":sum(bool(g) for g in groups)}
    return result


def sync(device):
    if device.type=="cuda":
        torch.cuda.synchronize()


def infer(fn,device):
    sync(device)
    t=perf_counter()
    result=fn()
    sync(device)
    return result,perf_counter()-t


def export(number,key,offset,x,v,i,predictions):
    root=OUT/"listening"/f"{number:02d}_{key[:8]}"
    root.mkdir(parents=True,exist_ok=False)
    starts=offset
    ends=offset+12*RATE
    files={"mixture":x,"ref_vocals":v,"ref_instrumental":i}
    for name,(a,b) in predictions.items():
        files[f"{name}_vocals"]=a
        files[f"{name}_instrumental"]=b
    for name,w in files.items():
        if w[:,starts:ends].shape!=(2,12*RATE):
            raise ValueError("Incomplete listening excerpt")
        sf.write(root/f"{name}.wav",w[:,starts:ends].T,RATE,subtype="FLOAT")
    return str(root.relative_to(OUT))


def write_metrics(rows):
    keys=["uuid","artist","artist_key","title","genre","samples",
          "exp04_seconds","exp08_seconds","exp09_seconds"]
    measures=list(rows[0]["exp04"])
    with (OUT/"per_song.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=keys+[f"{m}_{k}" for m in NAMES for k in measures]+[
            f"delta_exp09_vs_{m}_{k}" for m in NAMES[:2] for k in measures]
        w=csv.DictWriter(f,fields)
        w.writeheader()
        for r in rows:
            d={k:r[k] for k in keys}
            d.update({f"{m}_{k}":r[m][k] for m in NAMES for k in measures})
            for m in NAMES[:2]:
                for k in measures:
                    a,b=r["exp09"][k],r[m][k]
                    d[f"delta_exp09_vs_{m}_{k}"]=a-b if a is not None and b is not None else None
            w.writerow(d)


def main():
    configure_determinism(42)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    panel=read(OUT/"frozen_panel.json")
    assert panel["admitted"] and not (OUT/"listening").exists()
    splits=read(SCRATCH/"split_external_final.json")
    allowed={x["uuid"] for x in splits["assignments"] if x["role"]=="validation"}
    probe=read(SCRATCH/"multisong_probe.json")
    diagnostic=[r["uuid"] for r in probe["train"]]
    assert len(diagnostic)==12
    assert set(diagnostic)<=set(read(SCRATCH/"training_eligibility.json")["strict_train"])
    assert set(panel["admitted"])<=allowed and set(panel["listening"])<=set(panel["admitted"])
    provenance=read(OUT/"checkpoint_provenance.json")
    assert all(hash_file(BASE/n)==v for n,v in provenance["baseline_code"].items())
    checkpoint=Path(provenance["exp09"]["path"])
    assert hash_file(checkpoint)==provenance["exp09"]["sha256"]
    # Freeze evaluator, panel, settings and diagnostic fit roster before model inference.
    write_json(OUT/"execution_freeze.json",{
        "files_sha256":{n:hash_file(OUT/n) for n in
                        ("evaluate_dev.py","panel.py","frozen_panel.json","PROTOCOL.md","checkpoint_provenance.json")},
        "scratch_model_sha256":hash_file(SCRATCH/"scratch_model.py"),
        "fit_diagnostic_ids":diagnostic,"dtype":"FP32","torch":str(torch.__version__),
        "device":torch.cuda.get_device_name() if dev.type=="cuda" else "cpu",
        "stft_activity_dbfs":-45,"frame_seconds":1,"bootstrap_seed":42,"bootstrap_replicates":BOOTSTRAP
    })
    model=Separator().eval()
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert ck["index"]==24000 and ck["signature"]==provenance["exp09"]["signature"]
    assert sum(p.numel() for p in model.parameters())==11662596
    model.load_state_dict(ck["model"],strict=True)
    del ck
    model.to(dev)
    baselines=HybridSeparator(dev)
    rows=[]
    files=[]
    songs={r["uuid"]:r for r in read(SCRATCH/"manifest.json")}
    with ZipFile(ARCHIVE) as archive:
        for key in panel["admitted"]:
            g=panel["songs"][key]
            with Shard(OUT,4*GIB) as shard:
                staged=shard.stage(archive,songs[key],common_span=g["frames_min"]!=g["frames_max"])
                with shard.reader(key) as window:
                    x,v,i=window(0,staged["frames"])
            assert all(hashlib.sha256(y.data).hexdigest()==g["reference_hashes"][name]
                       for name,y in zip(("mix","vocals","instrumental"),(x,v,i)))
            x,v,i=(np.ascontiguousarray(y.T) for y in (x,v,i))
            raw,sec04=infer(lambda:separate_exp04(baselines.exp04,x,dev),dev)
            p04=raw["vocals"],raw["instrumental"]
            raw,sec08=infer(lambda:exp08_output(baselines.exp08,x,dev,chunk_samples=441000,batch_size=2),dev)
            ph=tuple(fixed_blend(x,p04[0],raw[0]))
            del raw
            p09,sec09=infer(lambda:exp09_output(model,x,dev),dev)
            estimates=dict(zip(NAMES,(p04,ph,p09)))
            row={k:g[k] for k in ("uuid","artist","artist_key","title","genre")}
            row.update(samples=x.shape[1],exp04_seconds=sec04,exp08_seconds=sec08,exp09_seconds=sec09)
            row.update({name:measure(x,v,i,*p) for name,p in estimates.items()})
            for name in ("hybrid","exp09"):
                assert row[name]["mixture_max_abs"]<=2e-5*max(1,float(np.abs(x).max()))
            if key in panel["listening"]:
                files.append({"uuid":key,"artist":g["artist"],"title":g["title"],
                              "start_frame":g["excerpt_frame"],"seconds":12,
                              "directory":export(panel["listening"].index(key)+1,key,g["excerpt_frame"],
                                                 x,v,i,estimates)})
            rows.append(row)
            write_json(OUT/"progress.json",rows)
            print(len(rows),g["artist"],g["title"],[(n,round(row[n]["vocal_sdr_db"],3)) for n in NAMES],flush=True)
        # Fit-domain sensitivity uses a fixed previously used set of recordings.
        fit=[]
        for key in diagnostic:
            with Shard(OUT,4*GIB) as shard:
                staged=shard.stage(archive,songs[key])
                with shard.reader(key) as window:
                    x,v,i=window(0,staged["frames"])
            x,v,i=(np.ascontiguousarray(y.T) for y in (x,v,i))
            p09=exp09_output(model,x,dev)
            n,active=choose_excerpt(v.T,i.T)
            sl=slice(n,n+12*RATE)
            fit.append({"uuid":key,"artist_key":songs[key]["artist_key"],"title":songs[key]["title"],
                        "excerpt_frame":n,"exp09":measure(x,v,i,*p09),
                        "matched_active_excerpt":measure(*(a[:,sl] for a in (x,v,i,*p09)))})
            print("Fit diagnostic",len(fit),flush=True)
    write_metrics(rows)
    compare=("vocal_sdr_db","instrumental_sdr_db","vocal_si_sdr_db",
             "abs_vocal_bleed_coeff","vocal_bleed_energy_pct","vocal_retention",
             "v_side_sdr_db","i_side_sdr_db","v_over_full_scale_pct","i_over_full_scale_pct",
             "buried_vocal_sdr_db")
    result={"rows":rows,"fit_in_sample":fit,"fit_aggregate":aggregate(fit,"exp09"),
            "aggregate":{n:aggregate(rows,n) for n in NAMES},
            "paired":{b:{k:paired(rows,b,k,.01 if k.endswith("_db") else .0001,
                                  -1 if "bleed" in k or "over_full" in k else 1)
                         for k in compare} for b in NAMES[:2]},
            "listening":sorted(files,key=lambda r:r["directory"]),
            "gpu_peak_reserved_gib":torch.cuda.max_memory_reserved()/GIB if dev.type=="cuda" else None,
            "gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/GIB if dev.type=="cuda" else None}
    write_json(OUT/"results.json",result)
    assert hash_file(checkpoint)==hash_file(SCRATCH/"runs/exp09_mse_seed909/latest.pt")==provenance["exp09"]["sha256"]


if __name__=="__main__":
    main()
