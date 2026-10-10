"""临时只读：脊线显著性判据重设计 —— 端到端验证。

现行判据：λ1 / median(λ1 over 全带子带) >= 6      ← D2 上恒不满足（能量地毯）
候选判据：λ1 / mean(λ2..λ8) >= K  （子带内自带噪声参考；= 1 + M·SNR_elem，与
          带宽/滤波器/跨子带无关，物理可解释）

对 D1（一半无干扰）与 D2 各测：子带保留数 / 脊数 / 脊命中真源数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(HERE / "rad_cls"), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cityrf.baselines.spatial import band_covariance  # noqa: E402

import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402


def link(pts, hop, sub_bw, du_link=0.05, min_span=3):
    ridges: list[dict] = []
    for f_c, u, w in sorted(pts, key=lambda t: t[0]):
        best = None
        for r in ridges:
            if abs(r["u_last"] - u) <= du_link and r["f_last"] < f_c - hop / 2:
                if best is None or r["f_last"] > best["f_last"]:
                    best = r
        if best is None:
            ridges.append({"u_last": u, "f_last": f_c, "fcs": [f_c], "us": [u], "ws": [w]})
        else:
            best["u_last"] = u
            best["f_last"] = f_c
            best["fcs"].append(f_c)
            best["us"].append(u)
            best["ws"].append(w)
    out = []
    for r in ridges:
        if len(r["fcs"]) < min_span:
            continue
        ws = np.asarray(r["ws"])
        us = np.asarray(r["us"])
        out.append({"u": float(np.sum(us * ws) / np.sum(ws)),
                    "f_lo": min(r["fcs"]) - sub_bw / 2,
                    "f_hi": max(r["fcs"]) + sub_bw / 2,
                    "span": len(r["fcs"]), "strength": float(len(r["fcs"]) * np.mean(ws))})
    out.sort(key=lambda r: -r["strength"])
    return out


def run(root: Path, sid: str, station: str, sub=200e3, hop=100e3):
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    try:
        refs = rca._build_target_reference(priv, n)
    except Exception:                                        # noqa: BLE001
        refs = {}
    ctx = ct.ToolContext(capture_dir=cap, cache_dir=root / "_agent_cache" / sid,
                         allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                         fs=float(obs["receivers"][0]["sample_rate_hz"]),
                         fc=float(obs["receivers"][0]["center_frequency_hz"]),
                         target_bw_normalized=0.25, aoi_center=[0, 0], target_reference=refs)
    ct.set_context(ctx)
    try:
        ct.strip_target_all_stations(-ctx.fs * 0.45, ctx.fs * 0.45)
    except Exception:                                        # noqa: BLE001
        pass
    tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    gts = list(tr.get("interferers", []))                    # D1 的 achieved 为空，直接用全部
    iq = ct._station_iq_raw(station, on_residual=True)
    rx = ctx.fs * 0.45
    ns = int(2 * (rx - sub / 2) // hop)
    raw = []                                                 # (f_c, u, rel, lam1, lamN_mean)
    for i in range(ns):
        f_c = -rx + sub / 2 + i * hop
        try:
            sc = ct._u_scan_cached(station, f_c, sub, on_residual=True)
            lo, hi = ct._band_edges(f_c, sub)
            R, _ = band_covariance(iq, ctx.fs, lo, hi)
        except Exception:                                    # noqa: BLE001
            continue
        ev = np.linalg.eigvalsh(R)[::-1]
        for u, rel in sc["u_peaks"]:
            raw.append((f_c, float(u), float(rel), float(ev[0]), float(ev[1:].mean())))
    if not raw:
        return None
    med = float(np.median([r[3] for r in raw]))
    pos, axis = ct._station_geometry(station)
    u_true = []
    for g in gts:
        d = np.asarray(g["position_m"], float)[:3] - pos
        u_true.append((float((d / np.linalg.norm(d)) @ axis), g["freq_offset_hz"]))
    res = {}
    for tag, key, K in (("A 现行 λ1/中位 K=6", lambda r, m: r[3] / (m + 1e-30), 6.0),
                        ("B λ1/mean(λ2..8) K=3", lambda r, m: r[3] / (r[4] + 1e-30), 3.0),
                        ("C λ1/mean(λ2..8) K=8", lambda r, m: r[3] / (r[4] + 1e-30), 8.0)):
        pts = [(f, u, 10 * np.log10(key((f, u, rl, a, b), med)))
               for (f, u, rl, a, b) in raw if key((f, u, rl, a, b), med) >= K]
        rgs = link(pts, hop, sub)
        hit = 0
        for r in rgs[:6]:
            for (u_t, f_t) in u_true:
                if abs(r["u"] - u_t) < 0.1 and r["f_lo"] - 2e5 <= f_t <= r["f_hi"] + 2e5:
                    hit += 1
                    break
        res[tag] = (len(pts), len(rgs), hit)
    return res, len(gts)


if __name__ == "__main__":
    for tag, root in (("D2", _V3 / "dataset_d2"), ("D1", _V3 / "dataset_d1")):
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        print(f"\n== {tag} station_a ==")
        for s in idx["samples"][:4]:
            r = run(root, s["id"], "station_a")
            if r is None:
                print(f"  {s['id']} 无数据")
                continue
            res, ngt = r
            print(f"  {s['id']} (GT {ngt}): " +
                  " | ".join(f"{k}: 点{v[0]} 脊{v[1]} 命中{v[2]}" for k, v in res.items()))
