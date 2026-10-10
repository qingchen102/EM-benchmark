"""只读：底估计方案对比（残留虚警/漏报定位，零 token）。

问题：300 kHz 中位滤波底会被**宽带平台**和被自身抬高，在通带边缘造出假带
（750/810/830k、±1090~1490k），同时把真源（D2 中等功率、宽带的共存体）漏掉。

候选方案：
  A 300k 中位（现状）
  B 1200k 中位（宽窗，不被宽带平台抬高）
  C min(300k 中位, 1200k 中位)（自适应：宽带用外圈底，窄带用内圈底）
  D 全局中位（=旧判据的底）+ 相干门控（**只加相干门控，不动底**）
在每个候选带上打标（与 GT 同带=真），统计 真/假 数与可分性。

用法：python _diag_floor_ab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

D1 = _V3 / "dataset_d1"
D2 = _V3 / "dataset_d2"


def bands_with(resid, fs, rx_half, mode, thr_db=8.0, coh_thr=0.6,
               local_hz=300e3, wide_hz=1200e3, win_hz=20e3, step_hz=10e3,
               with_coh=True):
    n = resid.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    inb = np.where((f >= -rx_half) & (f <= rx_half))[0]
    psd = (np.abs(np.fft.fft(resid, axis=-1)) ** 2).mean(axis=0)
    db = 10 * np.log10(psd[inb] + 1e-30)
    fs_bin = fs / n

    def mf(hz):
        k = max(int(round(hz / fs_bin)) | 1, 5)
        k = min(k, (db.size // 2) | 1)
        return median_filter(db, size=k, mode="nearest")

    if mode == "A":
        floor = mf(local_hz)
    elif mode == "B":
        floor = mf(wide_hz)
    elif mode == "C":
        floor = np.minimum(mf(local_hz), mf(wide_hz))
    else:
        floor = np.full_like(db, float(np.median(db)))
    over = db - floor
    step = max(int(round(step_hz / fs_bin)), 1)
    marks = []
    for ci in range(0, inb.size, step):
        fc = float(f[inb[ci]])
        o = float(over[ci])
        if o < thr_db:
            continue
        if with_coh:
            c = ct._band_coherence(resid, fs, fc, win_hz)
            if not np.isfinite(c) or c < coh_thr:
                continue
            marks.append((fc, o, c))
        else:
            marks.append((fc, o, 1.0))
    if not marks:
        return []
    groups, cur = [], [marks[0]]
    for prev, item in zip(marks, marks[1:]):
        if abs(item[0] - prev[0]) <= 2 * step * fs_bin + 1.0:
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]
    groups.append(cur)
    out = []
    for g in groups:
        best = max(g, key=lambda t: t[1])
        lo = min(t[0] for t in g) - win_hz
        hi = max(t[0] for t in g) + win_hz
        out.append({"f": best[0], "over": best[1], "coh": best[2],
                    "band": (lo, hi)})
    out.sort(key=lambda p: -p["over"])
    return out[:6]


def evaluate(root: Path, sids, mode, with_coh=True):
    real = fake = missed = 0
    clean_fa = 0
    for sid in sids:
        cap = root / sid / "capture" / "public"
        priv = root / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        if root is D2:
            nz = tr["noise"]["noise_floor_dbm"]
            ach = tr.get("achieved") or {}
            vis_any = False
            for j in range(len(ifw)):
                if any((ach.get(f"interferer_{j+1}", {}).get(f"s{i}", {})
                        .get("pr_dbm", -999) - nz) >= 3.0 for i in range(len(allowed))):
                    vis_any = True
            if not vis_any:
                continue
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                     # noqa: BLE001
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=root / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        station = "station_a" if root is D1 else allowed[1]
        try:
            ct.estimate_band_occupancy_by_correlation(station)
            resid = np.load(ctx.cache_dir / f"resid_{station}.npy")
        except Exception:                                     # noqa: BLE001
            continue
        bands = bands_with(resid, fs, 0.45 * fs, mode, with_coh=with_coh)
        if not ifw:
            if bands:
                clean_fa += 1
            continue
        matched = set()
        for b in bands:
            ctr = 0.5 * (b["band"][0] + b["band"][1])
            hit = False
            for gi, g in enumerate(ifw):
                if abs(ctr - g["freq_offset_hz"]) <= 250e3:
                    matched.add(gi)
                    hit = True
            if hit:
                real += 1
            else:
                fake += 1
        missed += len(ifw) - len(matched)
    return real, fake, missed, clean_fa


def main() -> None:
    idx1 = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    idx2 = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    s1 = [s["id"] for s in idx1["samples"][:12]]
    s2 = [s["id"] for s in idx2["samples"][:12]]
    print(f"{'方案':<28}{'D1干净虚警':>11}{'D1真峰':>8}{'D1假峰':>8}"
          f"{'D2真峰':>8}{'D2假峰':>8}{'D2漏':>7}")
    for label, mode, coh in (("A 300k中位+相干", "A", True),
                             ("A' 300k中位 无相干", "A", False),
                             ("B 1200k中位+相干", "B", True),
                             ("B' 1200k中位 无相干", "B", False),
                             ("C min(300k,1200k)+相干", "C", True),
                             ("C' min(300k,1200k) 无相干", "C", False),
                             ("D 全局中位+相干", "D", True),
                             ("E 全局中位 无相干=旧判据", "D", False)):
        r1, f1, m1, cfa = evaluate(D1, s1, mode, coh)
        r2, f2, m2, _ = evaluate(D2, s2, mode, coh)
        print(f"{label:<28}{cfa:>11}{r1:>8}{f1:>8}{r2:>8}{f2:>8}{m2:>7}")


if __name__ == "__main__":
    main()
