# -*- coding: utf-8 -*-
# 一次性探针 4：D2 残差里究竟还有没有可识别的源？
# 对照：原始 IQ（含合作目标）vs 剥离后残差 vs 工具落盘的 bfw（CNN 实际输入）。
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

D2 = os.path.join(os.path.dirname(HERE), "dataset_d2")

SID = sys.argv[1] if len(sys.argv) > 1 else "d2_00001"
ST = sys.argv[2] if len(sys.argv) > 2 else "station_a"

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
                    round(float(g.get("bandwidth_hz", 0)) / 1e3, 1)) for g in gt["interferers"]])

grid = np.array([-1700, -1200, -800, -500, -300, -176, -100, 0, 100, 200, 330, 500, 800, 1200, 1700]) * 1e3


def smooth_psd(x, half_hz=20e3):
    n = x.shape[-1]
    P = np.abs(np.fft.fft(x)) ** 2
    k = max(1, int(round(half_hz / (fs / n))))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    if P.ndim == 1:
        P = np.convolve(np.r_[P[-k:], P, P[:k]], ker, mode="same")[k:-k]
    else:
        P = np.stack([np.convolve(np.r_[p[-k:], p, p[:k]], ker, mode="same")[k:-k] for p in P])
    f = np.fft.fftfreq(n, 1 / fs)
    return f, P


for label, kw in (("raw  ", {"on_residual": False}), ("resid", {"on_residual": True})):
    r = ct._station_iq_raw(ST, **kw).astype(np.complex64)
    f, P = smooth_psd(r)
    Ptot = P.sum(axis=0) if P.ndim == 2 else P
    ref = float(np.median(Ptot))
    vals = []
    for g_hz in grid:
        i = int(np.argmin(np.abs(f - g_hz)))
        vals.append(10 * np.log10(Ptot[i] / ref + 1e-30))
    p95 = 10 * np.log10(np.percentile(Ptot, 99.9) / ref)
    print(f"{label} n={r.shape[-1]} 总功率={np.mean(np.abs(r)**2):.3e} 中位底=0dB "
          f"p99.9={p95:5.1f}dB  动态范围={np.max(10*np.log10(Ptot/ref)):.1f}dB")
    print("       " + " ".join(f"{g/1e3:6.0f}:{v:5.1f}" for g, v in zip(grid, vals)))

# 工具落盘的 CNN 输入
for k in range(6):
    p = Path(D2, "_agent_cache", SID, f"bfw_{ST}_{k}.npy")
    if not p.is_file():
        continue
    w = np.load(p)
    f, P = smooth_psd(w)
    ref = float(np.median(P))
    vals = [10 * np.log10(P[int(np.argmin(np.abs(f - g)))] / ref + 1e-30) for g in grid]
    print(f"bfw{k}  n={w.shape[0]} 动态范围={np.max(10*np.log10(P/ref)):.1f}dB")
    print("       " + " ".join(f"{g/1e3:6.0f}:{v:5.1f}" for g, v in zip(grid, vals)))
