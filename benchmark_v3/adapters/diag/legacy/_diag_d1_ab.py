"""只读 A/B：新旧峰判据在 D1 有干扰样本上的候选对比（零 token）。

D1 工具级回归换判据后"频偏(带中心)误差 中位 10k -> 639k"需要归因：
本脚本把同一残差分别喂给新旧判据，打印候选带清单、GT、以及"最近真源候选"的
带中心误差，直接看出是**判据退化**还是**候选选择/带宽口径**变了。

用法：python _diag_d1_ab.py
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

D1 = _V3 / "dataset_d1"


def main() -> None:
    idx = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    old_err, new_err = [], []
    for s in idx["samples"][:20]:
        sid = s["id"]
        priv = D1 / sid / "capture" / "private"
        cap = D1 / sid / "capture" / "public"
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        if not ifw:
            continue
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        refs = rca._build_target_reference(priv, n)
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        ct.estimate_band_occupancy_by_correlation("station_a")
        resid = np.load(ctx.cache_dir / "resid_station_a.npy")
        rx_half = 0.45 * fs
        print(f"\n===== {sid} =====")
        for g in ifw:
            print(f"  GT   f={g['freq_offset_hz']/1e3:9.1f}k bw={g['bandwidth_hz']/1e3:7.1f}k"
                  f" kind={g['kind']}")
        for tag, fn in (("OLD", ct._residual_peaks_legacy), ("NEW", ct._residual_bands)):
            bands = fn(resid, fs, rx_half)
            print(f"  -- {tag} 候选 {len(bands)} --")
            best = None
            for b in bands:
                bh = b.get("band_hz") or [b["freq_offset_hz"] - b["bandwidth_hz"] / 2,
                                          b["freq_offset_hz"] + b["bandwidth_hz"] / 2]
                ctr = 0.5 * (bh[0] + bh[1])
                for g in ifw:
                    d = abs(ctr - g["freq_offset_hz"]) / 1e3
                    if best is None or d < best[0]:
                        best = (d, ctr, g["freq_offset_hz"], b)
                print(f"     f={b['freq_offset_hz']/1e3:9.1f}k bw={b['bandwidth_hz']/1e3:7.1f}k"
                      f" band=[{bh[0]/1e3:7.1f},{bh[1]/1e3:7.1f}]k"
                      f" ctr={ctr/1e3:7.1f}k over={b['over_floor_db']:5.1f}"
                      + (f" coh={b.get('coherence')}" if b.get("coherence") is not None else ""))
            if best:
                print(f"     -> 最近真源候选: 带中心误差 {best[0]:.1f} kHz "
                      f"(ctr {best[1]/1e3:.1f}k vs GT {best[2]/1e3:.1f}k)")
                (old_err if tag == "OLD" else new_err).append(best[0])
    print("\n===== 带中心误差 中位 =====")
    print(f"  OLD: {np.median(old_err):.1f} kHz   NEW: {np.median(new_err):.1f} kHz")


if __name__ == "__main__":
    main()
