# -*- coding: utf-8 -*-
# 一次性探针 3：候选的位置已知，那么"该方向上的频谱"就是每源频带的直接测量。
# 目的：验证候选 band_hz（旧宽带判据）是否真的盖住了该源自己的载频。
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

D2 = os.path.join(os.path.dirname(HERE), "dataset_d2")
MATCH_M = 40.0


def dir_spectrum(obs, px, win_hz=450e3, smooth_hz=15e3):
    """候选方向上的非相干方向谱（四站求和），返回 (f, P) 只在 |f|<=win_hz 内。"""
    fc = obs["receivers"][0]["center_frequency_hz"]
    fs = obs["receivers"][0]["sample_rate_hz"]
    tot, f = None, None
    for rr in obs["receivers"]:
        try:
            r = ct._station_iq_raw(rr["receiver_id"], on_residual=True).astype(np.complex64)
        except Exception:
            continue
        X = np.fft.fft(r, axis=1)
        a = dx.steering(rr, [px], True, fc)[:, 0]
        a = (a / (np.linalg.norm(a) + 1e-12)).astype(np.complex64)
        p = np.fft.fftshift(np.abs(np.conj(a) @ X) ** 2)
        tot = p if tot is None else tot + p
        f = np.fft.fftshift(np.fft.fftfreq(r.shape[1], 1 / fs))
    n = len(f)
    k = max(1, int(round(smooth_hz / (fs / n))))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    tot = np.convolve(np.r_[tot[-k:], tot, tot[:k]], ker, mode="same")[k:-k]
    m = np.abs(f) <= win_hz
    return f[m], tot[m]


def peaks(f, P, sep_hz=80e3, top=4):
    order = np.argsort(P)[::-1]
    out, picks = [], []
    for i in order:
        if P[i] < P.max() * 10 ** (-12 / 10):
            break
        if all(abs(f[i] - f[j]) > sep_hz for j in picks):
            picks.append(i)
            out.append((float(f[i]), float(10 * np.log10(P[i] / P.max()))))
        if len(out) >= top:
            break
    return out


idx = json.loads(Path(D2, "index.json").read_text(encoding="utf-8"))
for s in idx["samples"][:8]:
    sid = s["id"]
    cap = os.path.join(D2, sid, "capture")
    obs = json.loads(Path(cap, "public", "observations.json").read_text(encoding="utf-8"))
    private = Path(cap) / "private"
    ns = obs["receivers"][0]["shape"][1]
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    fs = obs["receivers"][0]["sample_rate_hz"]
    ct.set_context(ct.ToolContext(
        capture_dir=os.path.join(cap, "public"),
        cache_dir=os.path.join(D2, "_agent_cache", sid),
        allowed_stations=allowed,
        fs=fs, fc=obs["receivers"][0]["center_frequency_hz"],
        target_bw_normalized=0.25, aoi_center=aoi,
        target_reference=rca._build_target_reference(private, ns)))
    gt = json.loads(Path(private, "d1_truth.json").read_text(encoding="utf-8"))
    vg = [(g, None, None) for g in gt["interferers"]]
    clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
    cands = clu.get("candidates") or []
    print(f"--- {sid}  candidates={len(cands)}  GT={len(vg)}")
    for k, c in enumerate(cands):
        px = c["position_xy_m"]
        blo, bhi = c["band_hz"]
        near = [(float(np.linalg.norm(np.asarray(px) - np.asarray(g["position_m"][:2]))), g)
                for g, _n, _p in vg]
        near.sort(key=lambda t: t[0])
        d, g = near[0]
        f0 = float(g["freq_offset_hz"])
        f, P = dir_spectrum(obs, px)
        pk = peaks(f, P)
        inside = "IN " if blo <= f0 <= bhi else "OUT"
        print(f"   cand{k} band=({blo/1e3:8.1f},{bhi/1e3:8.1f})kHz  band_c={0.5*(blo+bhi)/1e3:8.1f}  "
              f"GT<{d:5.1f}m> {g['kind']:14s} f0={f0/1e3:8.1f}kHz [{inside}]")
        print("       dirpeaks: " + "  ".join(f"{x/1e3:7.1f}kHz({y:5.1f}dB)" for x, y in pk))
