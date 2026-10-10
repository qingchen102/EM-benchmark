"""只读：d2_00000 聚类失败归因（零 token）。

smoke_s2_tools 在 d2_00000 报 "no valid intersections in any residual band"。
本脚本打印该样本的：逐站占用峰（对外/legacy）、定带结果、聚类返回键，
用于判断 (a) 真 bug、(b) 候选为空触发全通带回退后的正常失败。

用法：python _diag_d2_00000.py [sid]
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

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

ROOT = _V3 / "dataset_d2"


def main(sid: str) -> None:
    cap = ROOT / sid / "capture" / "public"
    priv = ROOT / sid / "capture" / "private"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    refs = rca._build_target_reference(priv, n)
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    ctx = ct.ToolContext(capture_dir=cap, cache_dir=ROOT / "_agent_cache" / sid,
                         allowed_stations=allowed, fs=fs,
                         fc=float(obs["receivers"][0]["center_frequency_hz"]),
                         target_bw_normalized=0.25, aoi_center=aoi, target_reference=refs)
    ct.set_context(ctx)
    for st in allowed:
        try:
            occ = ct.estimate_band_occupancy_by_correlation(st)
        except Exception as exc:                              # noqa: BLE001
            print(f"  {st}: occupancy EXC {type(exc).__name__}: {str(exc)[:70]}")
            continue
        if "error" in occ:
            print(f"  {st}: occupancy error {occ['error']}")
            continue
        pk = occ.get("residual_peaks") or []
        lg = occ.get("residual_peaks_legacy") or []
        print(f"  {st}: 对外峰 {len(pk)} {[round(p['freq_offset_hz']/1e3) for p in pk]}")
        print(f"  {st}: legacy {len(lg)} {[round(p['freq_offset_hz']/1e3) for p in lg]}"
              f"  occupied={occ.get('band_occupied_beyond_target')}"
              f" matched={occ.get('matched_energy_fraction')}")
    res = ct.estimate_sources_by_spatial_clustering(on_residual=True)
    print("  聚类返回键:", list(res.keys()))
    if "error" in res:
        print("  error:", res["error"])
    print("  candidates:", len(res.get("candidates") or []),
          " sources:", len(res.get("sources") or []))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d2_00000")
