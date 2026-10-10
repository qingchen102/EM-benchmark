"""只读诊断：D1/S0 的"残差峰清单"口径审查（零 token）。

D1×S0 的真实链路是：analyze_spectrum → estimate_band_occupancy_by_correlation
（剥目标）→ 读 residual_peaks → 逐峰作答。agent 到底看到什么峰、这些峰是真是假，
本脚本一次打全，用于给 prompt 门槛收口提供实测依据。

输出三部分：
  A 逐样本：GT 干扰 vs 工具 residual_peaks（含 over_floor_db / bandwidth）
  B 带外/过渡带嫌疑：|f| 落在 [nominal passband 810k, tool rx_half 1800k) 的峰
  C 汇总：真源峰是否 ≥ 8dB（工具门槛内）、虚警峰来源分布

用法：python _diag_d1_peaks.py [n_samples]
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
NOMINAL_PASSBAND_HZ = 0.405 * 4e6   # 生成器口径 ±1620k? 见下：标称 0.45*0.9*fs
RX_FRAC = 0.45 * 0.9


def main(n_samples: int) -> None:
    idx = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    rows = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                   # noqa: BLE001
            refs = {}
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        peaks = occ.get("residual_peaks") or []
        nominal = RX_FRAC * fs          # ±1620k（生成器"通带"）
        tool_half = 0.45 * fs           # ±1800k（工具搜索范围）

        print(f"\n=== {sid}  GT 干扰 {len(ifw)} 个 | 工具峰 {len(peaks)} 个"
              f" | nominal±{nominal/1e3:.0f}k tool±{tool_half/1e3:.0f}k ===")
        for g in ifw:
            print(f"   GT  {g.get('kind'):>8s} f={g.get('freq_offset_hz')/1e3:9.1f}k"
                  f" bw={g.get('bandwidth_hz', 0)/1e3:7.1f}k"
                  f" intent={g.get('intent')}")
        for p in peaks:
            f = p["freq_offset_hz"]
            tag = "过渡带" if abs(f) > nominal else "通带  "
            print(f"   PK{f:9.1f}k over={p['over_floor_db']:5.1f}dB"
                  f" bw={p['bandwidth_hz']/1e3:7.1f}k  [{tag}]")
        if not peaks:
            print("   PK (无峰)")
        rows.append({"sid": sid, "n_gt": len(ifw), "peaks": peaks,
                     "gt": [g.get("freq_offset_hz") for g in ifw],
                     "nominal": nominal})
        _ = NOMINAL_PASSBAND_HZ

    # ---------- 汇总 ----------
    print("\n\n========== 汇总 ==========")
    n_gt_hit_ge8 = n_gt_tot = 0
    trans_fa = inband_fa = 0
    clean_samples = clean_with_peak = 0
    for r in rows:
        peaks = r["peaks"]
        trans = [p for p in r["peaks"] if abs(p["freq_offset_hz"]) > r["nominal"]]
        inb = [p for p in r["peaks"] if abs(p["freq_offset_hz"]) <= r["nominal"]]
        if r["n_gt"] == 0:
            clean_samples += 1
            if peaks:
                clean_with_peak += 1
            trans_fa += len(trans)
            inband_fa += len(inb)
        else:
            for g in r["gt"]:
                n_gt_tot += 1
                if any(abs(p["freq_offset_hz"] - g) <= 120e3 for p in peaks):
                    n_gt_hit_ge8 += 1
    print(f"  有干扰样本的真源被峰清单覆盖: {n_gt_hit_ge8}/{n_gt_tot}")
    print(f"  干净样本 {clean_samples} 个，其中报出峰 {clean_with_peak} 个"
          f"（虚警样本率 {100*clean_with_peak/max(clean_samples,1):.0f}%）")
    print(f"  干净样本的虚警峰：过渡带 {trans_fa} 个 / 通带内 {inband_fa} 个")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
