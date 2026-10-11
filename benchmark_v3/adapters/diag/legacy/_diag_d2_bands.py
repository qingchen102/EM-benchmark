"""只读：D2 逐站 band 清单 新旧对比（定位召回损失来源，零 token）。

用法：python _diag_d2_bands.py d2_00008 d2_00009
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

D2 = _V3 / "dataset_d2"


def main(sids: list[str]) -> None:
    for sid in sids:
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        refs = rca._build_target_reference(priv, n)
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        print(f"\n===== {sid} =====")
        for j, g in enumerate(tr.get("interferers", [])):
            eid = f"interferer_{j+1}"
            vis = [f"station_{chr(ord('a') + i)}" for i in range(len(allowed))
                   if (ach.get(eid, {}).get(f"s{i}", {}).get("pr_dbm", -999) - nz) >= 3.0]
            print(f"  GT {eid} {g['kind']} f={g['freq_offset_hz']/1e3:8.1f}k"
                  f" bw={g['bandwidth_hz']/1e3:6.1f}k vis={vis}")
        for st in allowed:
            try:
                ct.estimate_band_occupancy_by_correlation(st)
                resid = np.load(ctx.cache_dir / f"resid_{st}.npy")
            except Exception as exc:                          # noqa: BLE001
                print(f"  {st}: FAIL {str(exc)[:50]}")
                continue
            for tag, fn in (("OLD", ct._residual_peaks_legacy), ("NEW", ct._residual_bands)):
                bands = fn(resid, fs, 0.45 * fs)
                desc = "  ".join(
                    f"{b['freq_offset_hz']/1e3:.0f}k/{b['bandwidth_hz']/1e3:.0f}k"
                    + (f"(c{b['coherence']:.2f})" if b.get("coherence") is not None else "")
                    for b in bands)
                print(f"  {st:10s} {tag}: {len(bands):2d} -> {desc}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["d2_00008"])
