"""只读：重设计峰判据——"局部底 + 窄窗空间相干"门控（零 token）。

已确认的两个事实（本会话实测）：
  1) 剥目标后的残差在通带内是**平坦 ~-88 dB** 的失配能量（真热噪声底 -131 dB），
     即参考回放里那 ~0.15% 的能量，宽频带、跨阵元不相干；
  2) 真干扰是点源：其 ±5~40k 窄窗内子带协方差第一特征值占比 lambda1/sum 高（>=0.9），
     而失配能量稳定在 ~0.15。
现有判据的缺陷：用**全局中位**当底 + >=5 bin 连段，会把 [400k,810k] 的目标滚降段
并成一个 820 kHz 宽的"强峰"（46~49 dB），真源只是其中最高点 —— 带宽/高度全失真，
这正是 agent 把干净样本报出 2~5 个干扰的直接来源。

本脚本换成：
  - 局部底：PSD 中位滤波（~300 kHz 窗），不被目标滚降带偏；
  - 相干门控：中心 +-20k 的 lambda1/sum >= 0.6 才算 occupied（失配能量整段滤掉）；
  - 连通段 -> band（峰频 / 带宽 / over 局部底 / coh@+-10k）。

用法：python _diag_bands_v2.py [n_samples]
"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
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


def coh(resid, fs, center, win):
    n = resid.shape[1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = np.abs(f - center) <= win
    X = np.fft.fft(resid, axis=1)[:, m]
    if X.shape[1] < 2:
        return float("nan")
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-30)
    R = X @ X.conj().T / X.shape[1]
    w = np.linalg.eigvalsh(R)[::-1].real
    return float(w[0] / (w.sum() + 1e-30))


def find_bands(resid, fs, rx_half, local_hz=300e3, win_hz=20e3,
               thr_db=8.0, coh_thr=0.6):
    n = resid.shape[1]
    f = np.fft.fftfreq(n, 1 / fs)
    inb = np.where((f >= -rx_half) & (f <= rx_half))[0]
    psd = (np.abs(np.fft.fft(resid, axis=1)) ** 2).mean(axis=0)
    db = 10 * np.log10(psd[inb] + 1e-30)
    fs_bin = fs / n
    k = max(int(round(local_hz / fs_bin)) | 1, 5)
    floor = median_filter(db, size=k, mode="nearest")
    over = db - floor
    step = max(int(round(10e3 / fs_bin)), 1)
    curve = []
    for ci in range(0, len(inb), step):
        fc = float(f[inb[ci]])
        o = float(over[ci])
        c = coh(resid, fs, fc, win_hz)
        curve.append((fc, o, c, bool(o >= thr_db and c >= coh_thr)))
    bands, cur = [], []
    for i, item in enumerate(curve):
        if item[3]:
            if cur and (i - cur[-1][0]) > 2:
                bands.append(cur)
                cur = []
            cur.append((i, item))
    if cur:
        bands.append(cur)
    out = []
    for b in bands:
        items = [x[1] for x in b]
        best = max(items, key=lambda t: t[1])
        f_lo = min(t[0] for t in items) - win_hz
        f_hi = max(t[0] for t in items) + win_hz
        out.append({"f_khz": best[0] / 1e3,
                    "bw_khz": (f_hi - f_lo) / 1e3,
                    "over_db": round(best[1], 1),
                    "coh10": round(coh(resid, fs, best[0], 10e3), 3),
                    "coh20": round(best[2], 3)})
    out.sort(key=lambda p: -p["over_db"])
    return out[:8]


def main(n_samples: int) -> None:
    idx = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    real, fake, misses = [], [], []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        rx_half = 0.45 * fs
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        refs = rca._build_target_reference(priv, n)
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        ct.estimate_band_occupancy_by_correlation("station_a")
        resid = np.load(ctx.cache_dir / "resid_station_a.npy")
        bands = find_bands(resid, fs, rx_half)
        matched = set()
        gts = [(g["kind"], round(g["freq_offset_hz"] / 1e3), round(g["bandwidth_hz"] / 1e3))
               for g in ifw]
        print(f"\n{sid}  GT={gts}")
        for p in bands:
            hit = None
            for gi, g in enumerate(ifw):
                if abs(p["f_khz"] * 1e3 - g["freq_offset_hz"]) <= 200e3:
                    hit = gi
                    break
            tag = "REAL" if hit is not None else "fake"
            if hit is not None:
                matched.add(hit)
                real.append((p["coh10"], p["coh20"], p["over_db"], p["bw_khz"], sid))
            else:
                fake.append((p["coh10"], p["coh20"], p["over_db"], p["bw_khz"], sid))
            print(f"   {tag}  f={p['f_khz']:8.1f}k bw={p['bw_khz']:7.1f}k"
                  f" over={p['over_db']:5.1f}dB coh10={p['coh10']:.2f} coh20={p['coh20']:.2f}")
        for gi, g in enumerate(ifw):
            if gi not in matched:
                misses.append((sid, g["kind"], round(g["freq_offset_hz"] / 1e3)))
    print(f"\n===== 汇总：真 band {len(real)} 假 band {len(fake)} 未覆盖 GT {len(misses)} =====")
    for nm, rows in (("真", real), ("假", fake)):
        if not rows:
            print(f"  {nm}: 无")
            continue
        c10 = np.array([r[0] for r in rows])
        c20 = np.array([r[1] for r in rows])
        o = np.array([r[2] for r in rows])
        bw = np.array([r[3] for r in rows])
        print(f"  {nm}: coh10 中位 {np.median(c10):.2f} | coh20 中位 {np.median(c20):.2f}"
              f" | over 中位 {np.median(o):.1f}dB | bw 中位 {np.median(bw):.1f}k")
    for m in misses:
        print("   未覆盖:", m)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
