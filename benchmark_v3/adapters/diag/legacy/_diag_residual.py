"""临时只读：交会残差（perpendicular_rms / angular_rms）对真/假交会的判别力。

做法：包装 cityrf 的 intersect_rays，记录聚类里每个**通过当前接受条件**的交会点的
残差与它到最近真源的距离，看残差能否把真交会与假交会分开。
"""
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

from cityrf.baselines import localization as loc  # noqa: E402

import capture_tools as ct  # noqa: E402

D2 = _V3 / "dataset_d2"
GATE = 50.0

_records: list[dict] = []
_orig = loc.intersect_rays


def _patched(origins, directions):
    r = _orig(origins, directions)
    if r is not None:
        _records.append({"pos": np.asarray(r["position_m"], float),
                         "perp": float(r["perpendicular_rms_m"]),
                         "ang": float(r["angular_rms_deg"]),
                         "rng": float(max(r["ranges_m"]))})
    return r


loc.intersect_rays = _patched


def run(n_samples: int) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    import run_capture_agent as rca
    all_rows = []
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
        scen = json.loads((priv / "scenario.json").read_text(encoding="utf-8"))
        gts = [np.asarray([e for e in scen["emitters"] if e["id"] == "target"][0]
                          ["position_m"], float)[:2]]
        gts += [np.asarray(g["position_m"], float)[:2] for g in tr.get("interferers", [])]
        _records.clear()
        ct.estimate_sources_by_spatial_clustering(on_residual=True)
        # 只统计"通过当前接受条件"的交会
        for r in _records:
            if not (0 < r["pos"][2] <= 300 and r["rng"] < 2000):
                continue
            d = min(float(np.linalg.norm(r["pos"][:2] - g)) for g in gts)
            all_rows.append((sid, r["perp"], r["ang"], d, d <= GATE))
    good = [r for r in all_rows if r[4]]
    bad = [r for r in all_rows if not r[4]]
    print(f"\n== 通过当前接受条件的交会点 {len(all_rows)} 个（前 {n_samples} 样本）==")
    print(f"  真源交会(≤{GATE:.0f}m) {len(good)}   假交会 {len(bad)}")
    if good:
        p = np.array([r[1] for r in good])
        print(f"  真源 perpendicular_rms : 中位 {np.median(p):8.1f} m  p90 {np.percentile(p,90):9.1f} m")
        print(f"  真源 angular_rms       : 中位 {np.median([r[2] for r in good]):8.3f}°")
    if bad:
        p = np.array([r[1] for r in bad])
        print(f"  假交会 perpendicular_rms : 中位 {np.median(p):8.1f} m  p90 {np.percentile(p,90):9.1f} m")
        print(f"  假交会 angular_rms     : 中位 {np.median([r[2] for r in bad]):8.3f}°")
    print("\n  perpendicular_rms 阈值扫描（保留真 / 剩余假）：")
    for th in (20, 50, 100, 200, 500, 1000):
        kg = sum(1 for r in good if r[1] <= th)
        kb = sum(1 for r in bad if r[1] <= th)
        print(f"    ≤{th:5d} m : 真 {kg:4d}/{len(good):4d} ({kg/max(len(good),1):4.0%})   "
              f"假 {kb:5d}/{len(bad):5d} ({kb/max(len(bad),1):4.0%})")
    print("\n  angular_rms 阈值扫描：")
    for th in (0.5, 1.0, 2.0, 5.0, 10.0):
        kg = sum(1 for r in good if r[2] <= th)
        kb = sum(1 for r in bad if r[2] <= th)
        print(f"    ≤{th:4.1f}° : 真 {kg:4d}/{len(good):4d} ({kg/max(len(good),1):4.0%})   "
              f"假 {kb:5d}/{len(bad):5d} ({kb/max(len(bad),1):4.0%})")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
