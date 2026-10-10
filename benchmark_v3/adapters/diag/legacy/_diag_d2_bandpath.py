"""只读：d2_00000 定带与 u 扫描链路逐带打印（零 token）。

目标：确认 "no valid intersections" 的直接原因是不是某个带的所有站 u_scan 都空，
以及它是否由我改动的 occupancy 输出导致。

用法：python _diag_d2_bandpath.py d2_00000
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
    rx_half = fs * 0.45

    # ---- 复刻聚类定带（当前实现）----
    band_map: dict[int, dict] = {}
    for st in allowed:
        occ = ct.estimate_band_occupancy_by_correlation(st)
        for p in occ.get("residual_peaks") or []:
            key = int(round(p["freq_offset_hz"] / 50e3))
            if key not in band_map or p["over_floor_db"] > band_map[key]["over_floor_db"]:
                band_map[key] = p
    with_new = sorted(band_map.values(), key=lambda p: -p["over_floor_db"])[:4]

    band_map2: dict[int, dict] = {}
    for st in allowed:
        occ = ct.estimate_band_occupancy_by_correlation(st)
        for p in occ.get("residual_peaks_legacy") or []:
            key = int(round(p["freq_offset_hz"] / 50e3))
            if key not in band_map2 or p["over_floor_db"] > band_map2[key]["over_floor_db"]:
                band_map2[key] = p
    with_leg = sorted(band_map2.values(), key=lambda p: -p["over_floor_db"])[:4]

    for label, bands in (("对外峰定带", with_new), ("legacy定带", with_leg)):
        print(f"\n===== {label} =====")
        for pb in bands:
            f0 = pb["freq_offset_hz"]
            bw = pb.get("bandwidth_hz")
            print(f"  带 f0={f0/1e3:8.1f}k  bw={bw/1e3 if bw else 0:7.1f}k"
                  f"  over={pb['over_floor_db']:.1f}")
            for st in allowed:
                try:
                    scan = ct._u_scan_cached(st, f0, bw, on_residual=True)
                    up = scan.get("u_peaks") or []
                    print(f"      {st}: u_peaks {len(up)}"
                          + (f" {[round(float(x[0]),3) for x in up[:4]]}" if up else ""))
                except Exception as exc:                      # noqa: BLE001
                    print(f"      {st}: u_scan EXC {type(exc).__name__}: {str(exc)[:50]}")
    print("\n  rx_half =", rx_half)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d2_00000")
