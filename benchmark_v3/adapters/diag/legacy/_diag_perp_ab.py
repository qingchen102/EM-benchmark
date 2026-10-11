"""临时只读 A/B：给交会增加 perpendicular_rms 残差门限，看三指标变化。

不改生产代码——通过包装 cityrf.intersect_rays，超门限直接返回 None（等价于在
capture_tools 的接受条件里加一条）。
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

from cityrf.baselines import localization as loc  # noqa: E402

import capture_tools as ct  # noqa: E402

D2 = _V3 / "dataset_d2"
GATE = 50.0
_orig = loc.intersect_rays


def install(perp_tol: float | None) -> None:
    if perp_tol is None:
        loc.intersect_rays = _orig
        return

    def _p(origins, directions):
        r = _orig(origins, directions)
        if r is None or r["perpendicular_rms_m"] > perp_tol:
            return None
        return r

    loc.intersect_rays = _p


def metrics(n_samples: int, perp_tol: float | None) -> None:
    install(perp_tol)
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    import run_capture_agent as rca
    n_true = n_hit = n_src = n_false = n_zero = 0
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
        except Exception:                                   # noqa: BLE001
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
        if not srcs:
            n_zero += 1
        used = set()
        for p in srcs:
            for i, g in enumerate(gt):
                if i not in used and np.linalg.norm(g - p) <= GATE:
                    used.add(i)
        n_true += len(gt); n_hit += len(used); n_src += len(srcs)
        n_false += len(srcs) - len(used)
    tag = "基线（无残差门限）" if perp_tol is None else f"perpendicular ≤ {perp_tol:.0f} m"
    print(f"  {tag:24s} 召回 {n_hit:3d}/{n_true:3d} = {n_hit/max(n_true,1):4.0%}   "
          f"簇 {n_src:3d} 假 {n_false:3d} ({n_false/max(n_src,1):4.0%})   零输出样本 {n_zero}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print(f"\n== D2 前 {n} 样本：交会残差门限 A/B ==")
    for tol in (None, 200.0, 100.0, 50.0, 20.0):
        metrics(n, tol)
