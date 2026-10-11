"""只读：全通带（0 Hz, 1.76 MHz）下 d2_00000 的 u_peaks 与 3D 交会可行性（零 token）。

用法：python _diag_d2_fullband.py d2_00000
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

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402
from cityrf.baselines.localization import intersect_rays  # noqa: E402

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
    rx_half = fs * 0.45
    bw = 2 * rx_half * 0.98
    rays = {}
    for st in allowed:
        scan = ct._u_scan_cached(st, 0.0, bw, on_residual=True)
        pos_s, axis_s = ct._station_geometry(st)
        ups = [u for u, _ in scan["u_peaks"][:ct.U_PEAK_MAX]]
        rays[st] = [(pos_s, d) for u in ups for d in ct._u_to_world_rays(u, pos_s, axis_s)]
        print(f"  {st}: u_peaks {len(scan['u_peaks'])} "
              f"{[round(float(u), 3) for u in ups]}")
    print("\n--- trio 交会 (前 12 组随机) ---")
    rng = np.random.default_rng(7)
    n_ok = 0
    for i in range(60):
        pair = rng.choice(allowed, 3, replace=False)
        sel = [rays[s][int(rng.integers(len(rays[s])))] for s in pair]
        hit = intersect_rays([r[0] for r in sel], [r[1] for r in sel])
        if not hit:
            continue
        z = hit["position_m"][2]
        rng_max = max(hit["ranges_m"])
        ok = (0 < z <= 300) and rng_max < 2000
        n_ok += int(ok)
        if i < 12:
            print(f"   {list(pair)} z={z:8.1f} maxrange={rng_max:8.1f} -> "
                  f"{'OK' if ok else 'REJECT'}")
    print(f"  60 次里 OK {n_ok} 次")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d2_00000")
