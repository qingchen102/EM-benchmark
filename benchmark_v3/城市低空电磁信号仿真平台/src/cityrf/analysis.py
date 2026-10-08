"""Deterministic public-data inspection, not an LLM or ground-truth evaluator."""
import json
import re
from pathlib import Path
import numpy as np
from scipy.signal import welch, find_peaks
from .export import digest


def load_public(public_dir):
    root=Path(public_dir).resolve()
    document=json.loads((root/"observations.json").read_text(encoding="utf-8"))
    def safe_id(value):
        if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}",value):
            raise ValueError("invalid public ID")
    ids=[]
    for obs in document["receivers"]:
        safe_id(obs["receiver_id"])
        ids.append(obs["receiver_id"])
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate receiver ID")
    task_ids=[]
    for task in document.get("tasks",[]):
        safe_id(task["id"])
        task_ids.append(task["id"])
        if not set(task["receiver_ids"])<=set(ids):
            raise ValueError("task references unknown receiver ID")
    if len(task_ids)!=len(set(task_ids)):
        raise ValueError("duplicate task ID")
    for obs in document["receivers"]:
        path=(root/obs["iq_file"]).resolve()
        if not path.is_relative_to(root) or path.suffix != ".npy":
            raise ValueError("IQ path outside public observation directory")
        if digest(path) != obs["sha256"]:
            raise ValueError("IQ hash mismatch")
    return document


def analyze_observations(public_dir):
    root=Path(public_dir)
    obs=load_public(root)
    stations=[]
    for r in obs["receivers"]:
        x=np.load(root/r["iq_file"],allow_pickle=False,mmap_mode="r")
        if list(x.shape) != r["shape"] or x.dtype != np.complex64 or not np.isfinite(x).all():
            raise ValueError("IQ differs from declared shape/dtype or contains nonfinite values")
        fs=r["sample_rate_hz"]
        nfft=min(4096,x.shape[1])
        f,p=welch(x,fs=fs,nperseg=nfft,noverlap=nfft//2,return_onesided=False,detrend=False,axis=-1)
        f,p=np.fft.fftshift(f),np.fft.fftshift(p.mean(axis=0))
        peaks,_=find_peaks(10*np.log10(np.maximum(p,1e-30)),prominence=6,distance=3)
        peaks=sorted(peaks,key=lambda k:p[k],reverse=True)[:12]
        block=max(1,round(fs*.001))
        envelope=[float(np.mean(abs(x[:,i:i+block])**2)) for i in range(0,x.shape[1],block)]
        stations.append({
            "receiver_id":r["receiver_id"],
            "mean_port_power_w":np.mean(abs(x)**2,axis=1).tolist(),
            "spectral_peaks":[{"offset_hz":float(f[k]),"psd_dbw_per_hz":float(10*np.log10(p[k]))} for k in peaks],
            "power_envelope":{"block_duration_s":block/fs,"power_w":envelope}})
    return {"analysis_type":"public IQ deterministic overview",
            "not_a_source_count_estimator":True,"stations":stations}
