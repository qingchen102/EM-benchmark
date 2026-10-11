"""临时只读：候选级诊断——真簇 vs 假簇 的"被解释站数 / 各站残差"。"""
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
GATE = 50.0


def run(n_samples: int) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    import run_capture_agent as rca
    rows = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        try:
            ref = rca._build_target_reference(priv, n)
        except Exception:                                  # noqa: BLE001
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=ref)
        ct.set_context(ctx)
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        gt = []
        for j, g in enumerate(tr.get("interferers", [])):
            eid = f"interferer_{j+1}"
            if any((ach.get(eid, {}).get(k, {}).get("pr_dbm", -999) - nz) >= 3.0
                   for k in [f"s{i}" for i in range(len(allowed))]):
                gt.append(np.asarray(g["position_m"], float)[:2])
        # 各站全带峰表（宽带回退口径，与聚类一致）
        peaks, geoms = {}, {}
        for st in allowed:
            try:
                sc = ct._u_scan_cached(st, 0.0, 0.9 * fs, on_residual=True)
            except Exception:                              # noqa: BLE001
                continue
            peaks[st] = [u for u, _ in sc["u_peaks"]]
            geoms[st] = ct._station_geometry(st)
        res = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        for x in res.get("sources", []):
            p = np.asarray(x["position_xy_m"], float)
            resid, expl = [], 0
            for st, ups in peaks.items():
                pos, axis = geoms[st]
                d = np.array([p[0] - pos[0], p[1] - pos[1], 0.0])
                d = d / (np.linalg.norm(d) + 1e-12)
                u = float(axis @ d)
                best = min((abs(u - up) for up in ups), default=9.0)
                resid.append(best)
                expl += int(best < 0.05)
            is_true = any(np.linalg.norm(p - g) <= GATE for g in gt)
            rows.append((sid, is_true, expl, len(peaks), float(np.median(resid)),
                         float(np.max(resid)), x.get("deflation_db")))
    t = [r for r in rows if r[1]]
    f = [r for r in rows if not r[1]]
    print(f"\n== 候选级：真簇 {len(t)} / 假簇 {len(f)}（前 {n_samples} 样本）==")
    for tag, g in (("真簇", t), ("假簇", f)):
        if not g:
            continue
        print(f"  {tag}: 被解释站数 中位 {np.median([r[2] for r in g]):.1f} "
              f"| 可用站数中位 {np.median([r[3] for r in g]):.1f} "
              f"| 最大Δu 中位 {np.median([r[5] for r in g]):.3f} "
              f"| 中位Δu 中位 {np.median([r[4] for r in g]):.3f} "
              f"| deflation 中位 {np.median([r[6] for r in g if r[6] is not None] or [0]):.1f}")
    print("\n  '最大Δu' 阈值扫描（真簇保留 / 假簇剩余）：")
    for th in (0.05, 0.10, 0.20, 0.30, 0.50):
        kg = sum(1 for r in t if r[5] <= th)
        kb = sum(1 for r in f if r[5] <= th)
        print(f"    ≤{th:4.2f} : 真 {kg:3d}/{len(t):3d} ({kg/max(len(t),1):4.0%})   "
              f"假 {kb:3d}/{len(f):3d} ({kb/max(len(f),1):4.0%})")
    print("\n  '被解释站数 == 可用站数' 严格全解释：")
    kg = sum(1 for r in t if r[2] == r[3])
    kb = sum(1 for r in f if r[2] == r[3])
    print(f"    真 {kg}/{len(t)} ({kg/max(len(t),1):.0%})   假 {kb}/{len(f)} ({kb/max(len(f),1):.0%})")

    # 排序取 Top-K（K = 该样本 GT 可见干扰数）的精度/召回——即"改成候选级打分
    # + NMS 取前 K"后的可达指标
    by: dict[str, list] = {}
    for sid, is_true, expl, navail, med, mx, dfl in rows:
        by.setdefault(sid, []).append((is_true, expl, mx))
    import json as _j
    for key_name, key in (("(被解释站数↓, 最大Δu↑)", lambda r: (-r[1], r[2])),
                          ("(最大Δu↑ 单键)", lambda r: (r[2],))):
        tp = fp = fn = 0
        for sid, lst in by.items():
            idx2 = _j.loads((D2 / "index.json").read_text(encoding="utf-8"))
            k = sum(1 for x in lst if x[0])          # 以真簇数为 K（上界口径）
            sel = sorted(lst, key=key)[:max(k, 1)]
            tp += sum(1 for x in sel if x[0])
            fp += sum(1 for x in sel if not x[0])
            fn += sum(1 for x in lst if x[0]) - sum(1 for x in sel if x[0])
        print(f"\n  排序键 {key_name}: 选中真 {tp} 假 {fp} 漏 {fn} → 精度 {tp/max(tp+fp,1):.0%}  "
              f"召回 {tp/max(tp+fn,1):.0%}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
