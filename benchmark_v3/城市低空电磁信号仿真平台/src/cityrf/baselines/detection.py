"""Robust Welch energy thresholding; not a calibrated Pfa-CFAR detector."""
import numpy as np
from scipy.ndimage import binary_closing, label, uniform_filter1d
from scipy.signal import welch


def detect_bands(iq, fs, receiver_bandwidth, threshold_db=8., nfft=4096, merge_gap_hz=8000.):
    nfft=min(nfft,iq.shape[1])
    f,p=welch(iq,fs=fs,nperseg=nfft,noverlap=nfft//2,detrend=False,
              return_onesided=False,axis=-1,scaling="density")
    f=np.fft.fftshift(f)
    p=np.fft.fftshift(p.mean(axis=0)).astype(float)
    smooth=uniform_filter1d(p,3,mode="nearest")
    valid=abs(f)<=receiver_bandwidth*.45  # exclude frontend transition band
    floor=float(np.quantile(smooth[valid],.3))
    floor=max(floor,1e-30)
    mask=valid & (smooth>floor*10**(threshold_db/10))
    gap=max(1,int(merge_gap_hz/(fs/nfft)))
    mask=binary_closing(mask,structure=np.ones(gap+1)) & valid
    labels,count=label(mask)
    bands=[]
    df=fs/nfft
    for i in range(1,count+1):
        indices=np.flatnonzero(labels==i)
        if len(indices)<2:
            continue
        k=int(indices[np.argmax(p[indices])])
        bands.append({"low_hz":float(f[indices[0]]-df/2),"high_hz":float(f[indices[-1]]+df/2),
                      "peak_hz":float(f[k]),"power_w":float(p[indices].sum()*df),
                      "peak_excess_db":float(10*np.log10(p[k]/floor))})
    bands=sorted(sorted(bands,key=lambda b:b["power_w"],reverse=True)[:12],key=lambda b:b["low_hz"])
    return {"bands":bands,"noise_floor_dbw_per_hz":float(10*np.log10(floor)),
            "threshold_db":threshold_db,"frequency_resolution_hz":df,
            "frequency_hz":f[::2].tolist(),"psd_dbw_per_hz":(10*np.log10(np.maximum(p[::2],1e-30))).tolist()}


def frequency_groups(stations):
    """Connected frequency intervals across sites, with no source-count claim."""
    intervals=[]
    for s in stations:
        for b in s["detection"]["bands"]:
            intervals.append((b["low_hz"]+s["center_frequency_hz"],
                              b["high_hz"]+s["center_frequency_hz"],s["receiver_id"]))
    groups=[]
    for low,high,station in sorted(intervals):
        if groups and low<=groups[-1]["high_rf_hz"]+8000:
            g=groups[-1]
            g["high_rf_hz"]=max(g["high_rf_hz"],high)
            g["receiver_ids"]=sorted(set(g["receiver_ids"]+[station]))
        else:
            groups.append({"low_rf_hz":low,"high_rf_hz":high,"receiver_ids":[station]})
    for i,g in enumerate(groups):
        g["group_id"]=f"band_{i:02d}"
    return groups
