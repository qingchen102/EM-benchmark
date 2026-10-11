"""临时只读：QA2 上的候选精度专项 —— 幻影形态 + 排序键区分力。

对每个候选：真/假（距最近 GT ≤50m 为真），假候选再算它与最近真源的
  · 2D 距离（判"近失幻影"还是"远处乱点"）
  · 各站 |Δu| 最大值（判是否与真源同方向——若是，则 u 空间 NMS 是解法）
并逐字段做阈值扫描，看谁真能分离真/假。
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
import run_capture_agent as rca  # noqa: E402

D2 = _V3 / "dataset_d2"
FIELDS = ("n_explained_stations", "max_du", "perp_rms_m", "deflation_db",
          "n_agreeing_triads", "confidence")


def main(n_samples: int) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    rows = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                    # noqa: BLE001
            refs = {}
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=float(obs["receivers"][0]["sample_rate_hz"]),
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        vis = [g for j, g in enumerate(tr.get("interferers", []))
               if any((ach.get(f"interferer_{j+1}", {}).get(f"s{i}", {}).get("pr_dbm", -999)
                       - nz) >= 3.0 for i in range(4))]
        if not vis:
            continue
        try:
            clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        except Exception:                                    # noqa: BLE001
            continue
        # 各站阵几何（算 u）
        geo = {r["receiver_id"]: (np.asarray(r["receiver_position_m"], float),
                                  (lambda a: a / np.linalg.norm(a))(
                                      np.asarray(r["element_offsets_world_m"], float)[-1]
                                      - np.asarray(r["element_offsets_world_m"], float)[0]))
               for r in obs["receivers"]}

        def us(p):
            out = {}
            for k, (rp, ax) in geo.items():
                d = np.array([p[0] - rp[0], p[1] - rp[1], 0.0])
                out[k] = float((d / (np.linalg.norm(d) + 1e-12)) @ ax)
            return out

        gt_pos = [np.asarray(g["position_m"], float)[:2] for g in vis]
        for c in clu.get("candidates") or []:
            p = np.asarray(c["position_xy_m"], float)
            dd = [float(np.linalg.norm(p - g)) for g in gt_pos]
            j = int(np.argmin(dd))
            is_true = dd[j] <= 50.0
            du_max = 0.0
            if not is_true:
                up, ug = us(p), us(gt_pos[j])
                du_max = max(abs(up[k] - ug[k]) for k in geo)
            rows.append({"sid": sid, "true": is_true, "dnn": dd[j], "du_max": du_max,
                         **{f: c.get(f, 0.0) for f in FIELDS}})
    T = [r for r in rows if r["true"]]
    F = [r for r in rows if not r["true"]]
    print(f"\n== QA2 候选精度专项（{n_samples} 样本）==")
    print(f"  候选 {len(rows)}  真 {len(T)}  假 {len(F)}  → 精度 {100*len(T)/max(len(rows),1):.0f}%")
    if F:
        d = np.array([r["dnn"] for r in F])
        du = np.array([r["du_max"] for r in F])
        print(f"  假候选距最近真源: 中位 {np.median(d):.0f} m  ≤100m {100*np.mean(d<=100):.0f}%"
              f"  ≤300m {100*np.mean(d<=300):.0f}%")
        print(f"  假候选与最近真源的 max|Δu|（同方向性）: 中位 {np.median(du):.2f}"
              f"  <0.10 占 {100*np.mean(du<0.10):.0f}%  <0.25 占 {100*np.mean(du<0.25):.0f}%")
    print("\n  字段区分力（真/假中位 + 阈值扫描保留真/剩余假）：")
    for f in FIELDS:
        tv = np.array([r[f] for r in T], float)
        fv = np.array([r[f] for r in F], float)
        if not len(tv) or not len(fv):
            continue
        lower_better = f in ("max_du", "perp_rms_m")
        th = np.median(fv)
        kt = np.mean(tv <= th) if lower_better else np.mean(tv >= th)
        kf = np.mean(fv <= th) if lower_better else np.mean(fv >= th)
        print(f"    {f:22s} 真中位 {np.median(tv):7.2f} 假中位 {np.median(fv):7.2f}"
              f"  | 中位阈: 留真 {100*kt:3.0f}% 留假 {100*kf:3.0f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
