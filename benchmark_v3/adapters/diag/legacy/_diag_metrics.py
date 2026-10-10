"""临时只读：聚类级三指标（真源召回 / 假簇数 / 最近真源距离）批量统计。"""
from __future__ import annotations

import json
import sys
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


def main(n_samples: int) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    import run_capture_agent as rca
    n_true = n_hit = n_src = n_false = 0
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
        except Exception:                                    # noqa: BLE001
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
        res = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        srcs = [np.asarray(x["position_xy_m"], float) for x in res.get("sources", [])]
        hit = 0
        for g in gt:
            if any(np.linalg.norm(g - p) <= GATE for p in srcs):
                hit += 1
        used = set()
        for p in srcs:
            for i, g in enumerate(gt):
                if i not in used and np.linalg.norm(g - p) <= GATE:
                    used.add(i)
        n_true += len(gt); n_hit += hit
        n_src += len(srcs)
        n_false += len(srcs) - len(used)
        nd = min((float(np.min([np.linalg.norm(p - g) for g in gt])) for p in srcs), default=-1)
        rows.append((sid, len(gt), len(srcs), len(used), round(nd, 0)))
    print(f"\n== D2 前 {len(rows)} 样本（跨站可见干扰为准，门限 {GATE:.0f} m）==")
    print(f"  真源召回     : {n_hit}/{n_true} = {n_hit/max(n_true,1):.0%}")
    print(f"  输出簇总数   : {n_src}   其中假簇 {n_false} ({n_false/max(n_src,1):.0%})")
    print(f"  {'样本':10s} {'GT':>3s} {'簇':>3s} {'命中':>4s} {'最近真源m':>9s}")
    for r in rows:
        print(f"  {r[0]:10s} {r[1]:>3d} {r[2]:>3d} {r[3]:>4d} {r[4]:>9}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
