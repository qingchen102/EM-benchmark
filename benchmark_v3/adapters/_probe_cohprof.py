# -*- coding: utf-8 -*-
# 一次性探针 5：相干度剖面 c(f) 到底长什么样、峰在哪、为什么 _support_center 常返回 0。
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rad_cls"))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402
import _diag_d2_extract as dx       # noqa: E402
import _diag_d2_off as off          # noqa: E402

D2 = os.path.join(os.path.dirname(HERE), "dataset_d2")
SID = sys.argv[1] if len(sys.argv) > 1 else "d2_00001"

cap = os.path.join(D2, SID, "capture")
obs = json.loads(Path(cap, "public", "observations.json").read_text(encoding="utf-8"))
private = Path(cap) / "private"
ns = obs["receivers"][0]["shape"][1]
allowed = [r["receiver_id"] for r in obs["receivers"]]
aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
fs = obs["receivers"][0]["sample_rate_hz"]
ct.set_context(ct.ToolContext(
    capture_dir=os.path.join(cap, "public"),
    cache_dir=os.path.join(D2, "_agent_cache", SID),
    allowed_stations=allowed,
    fs=fs, fc=obs["receivers"][0]["center_frequency_hz"],
    target_bw_normalized=0.25, aoi_center=aoi,
    target_reference=rca._build_target_reference(private, ns)))
gt = json.loads(Path(private, "d1_truth.json").read_text(encoding="utf-8"))
print(SID, "GT:", [(g["kind"], round(float(g["freq_offset_hz"]) / 1e3, 1),
                    round(float(g["bandwidth_hz"]) / 1e3, 1)) for g in gt["interferers"]])

clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
cands = clu.get("candidates") or []
grid = np.arange(-1800, 1801, 200) * 1e3

for k, c in enumerate(cands[:3]):
    px = c["position_xy_m"]
    fc = obs["receivers"][0]["center_frequency_hz"]
    print(f"cand{k} pos={np.round(px,1).tolist()} band=({c['band_hz'][0]/1e3:.0f},{c['band_hz'][1]/1e3:.0f})kHz")
    for label, kw in (("noco ", {}), ("comode", {"common_mode": True})):
        # 单站
        r = ct._station_iq_raw("station_a", on_residual=True).astype(np.complex64)
        X = np.fft.fft(r, axis=1)
        if kw.get("common_mode"):
            X = X - X.mean(axis=0, keepdims=True)
        a = dx.steering(obs["receivers"][0], [px], True, fc)[:, 0]
        a = a / (np.linalg.norm(a) + 1e-12)
        num = np.abs(np.conj(a) @ X) ** 2
        den = np.sum(np.abs(X) ** 2, axis=0) / r.shape[0]
        cprof = np.fft.fftshift(num / (den + 1e-30))
        f = np.fft.fftshift(np.fft.fftfreq(r.shape[1], 1 / fs))
        kk = max(1, int(round(15e3 / (fs / r.shape[1]))))
        ker = np.ones(2 * kk + 1) / (2 * kk + 1)
        cs = np.convolve(np.r_[cprof[-kk:], cprof, cprof[:kk]], ker, mode="same")[kk:-kk]
        vals = [float(cs[int(np.argmin(np.abs(f - g)))]) for g in grid]
        imax = int(np.argmax(cs))
        print(f"   {label} c(f): " + " ".join(f"{g/1e3:6.0f}:{v:7.2f}" for g, v in zip(grid, vals)))
        print(f"   {label} argmax@{f[imax]/1e3:8.1f}kHz val={cs[imax]:7.2f}  c(0)={cs[int(np.argmin(np.abs(f)))]:7.2f}  "
              f"support={off._support_center(cs, f, fs, len(f), 450e3, 15e3, 0.6)}")
