"""临时只读诊断（不参与生产链路）：空间聚类过聚类 / 漏召回根因。

用法：python _diag_cluster.py d2_00008 [d2_00002 ...]
输出：每站 残差峰 / MDL rank / u 峰清单 / 真值源的真 u 是否落在峰内 / 聚类结果。
"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct  # noqa: E402

D2 = _V3 / "dataset_d2"


def setup(sid: str, target_bw_norm: float = 0.25):
    cap = D2 / sid / "capture" / "public"
    priv = D2 / sid / "capture" / "private"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    fc = float(obs["receivers"][0]["center_frequency_hz"])
    n = int(obs["receivers"][0]["shape"][1])
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    sts = [{"id": r["receiver_id"], "position_m": r["receiver_position_m"]}
           for r in obs["receivers"]]
    aoi = np.mean([s["position_m"] for s in sts], axis=0).tolist()
    ref = {}
    try:
        import run_capture_agent as rca
        ref = rca._build_target_reference(priv, n)
    except Exception as exc:                       # noqa: BLE001
        print(f"  [warn] target ref 构造失败: {str(exc)[:140]}")
    ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                         allowed_stations=allowed, fs=fs, fc=fc,
                         target_bw_normalized=target_bw_norm,
                         aoi_center=aoi, target_reference=ref)
    ct.set_context(ctx)
    return ctx, obs, fs, fc


def diag(sid: str) -> None:
    print(f"\n===== {sid} =====")
    ctx, obs, fs, fc = setup(sid)
    priv = D2 / sid / "capture" / "private"
    truth = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    scen = json.loads((priv / "scenario.json").read_text(encoding="utf-8"))
    tgt_pos = np.asarray([e for e in scen["emitters"] if e["id"] == "target"][0]
                         ["position_m"], float)[:3]
    rx_half = fs * 0.45
    nz = truth["noise"]["noise_floor_dbm"]

    print("-- 逐站残差峰（estimate_band_occupancy_by_correlation）--")
    for st in ctx.allowed_stations:
        occ = ct.estimate_band_occupancy_by_correlation(st)
        pk = [(round(p["freq_offset_hz"] / 1e3), round(p["bandwidth_hz"] / 1e3),
               p["over_floor_db"]) for p in (occ.get("residual_peaks") or [])]
        print(f"  {st:10s} err={str(occ.get('error'))[:40]:40s} matched={occ.get('matched_energy_fraction')} "
              f"peaks(kHz,kHz,dB)={pk}")

    print("-- 逐站全带 u 扫描（宽带回退口径 f0=0 bw=2*0.45fs*0.98）--")
    for st in ctx.allowed_stations:
        scan = ct._u_scan(st, 0.0, 2 * rx_half * 0.98, on_residual=True)
        print(f"  {st:10s} rank={scan['rank']} u_peaks={scan['u_peaks']}")

    print("-- 真值源在每站的真 u（问：真峰在不在峰表里？）--")
    srcs = [("target", tgt_pos, truth["target"]["emitter"])]
    for j, g in enumerate(truth.get("interferers", [])):
        srcs.append((f"ifw{j+1}", np.asarray(g["position_m"], float)[:3], g["kind"]))
    peaks_cache = {st: ct._u_scan_cached(st, 0.0, 2 * rx_half * 0.98, on_residual=True)
                   for st in ctx.allowed_stations}
    for st in ctx.allowed_stations:
        pos, axis = ct._station_geometry(st)
        ups = [u for u, _ in peaks_cache[st]["u_peaks"]]
        line = []
        for name, p3, kind in srcs:
            d = p3 - pos
            u_t = float(axis @ d) / (np.linalg.norm(d) + 1e-12)
            dmin = min((abs(u_t - up), up) for up in ups) if ups else (9, None)
            line.append(f"{name}/{kind[:9]} u={u_t:+.3f} 最近峰Δ={dmin[0]:.3f}")
        print(f"  {st:10s} " + " | ".join(line))
    for name, p3, kind in srcs:
        eid = name if name == "target" else f"interferer_{name[3:]}"
        ach = (truth.get("achieved") or {}).get(eid, {})
        inr = {k: round(ach[k]["pr_dbm"] - nz, 1) for k in ach}
        print(f"    {name:8s} {kind:16s} INR/站(dB)={inr}")

    print("-- 聚类结果 --")
    res = ct.estimate_sources_by_spatial_clustering(on_residual=True)
    print("  n_intersections=", res.get("n_intersections"), " bands=", res.get("bands"))
    for s in res.get("sources", []):
        p = s["position_xy_m"]
        dd = sorted(np.linalg.norm(np.array([p[0], p[1]]) - np.asarray(q[1], float)[:2])
                    for q in srcs)
        print(f"    src xy={p} n_triads={s['n_agreeing_triads']} conf={s['confidence']} "
              f"→ 距最近真源 {dd[0]:.0f} m")
    if res.get("error"):
        print("    error:", res["error"])


if __name__ == "__main__":
    for sid in (sys.argv[1:] or ["d2_00008"]):
        diag(sid)
