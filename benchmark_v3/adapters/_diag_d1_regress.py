"""临时只读：D1 工具级回归（零 token）。

D1 = 视距 + 频率分离，约半数样本无干扰、半数为单个 tone/noise。
检查两件事：
  A 无干扰样本上工具是否**报虚警**（应报 0 候选）
  B 有干扰样本上工具能否报出（应报 ≥1 且位置/频偏接近真值）
这是"工具大改是否破坏简单场景"的最便宜检查，不需要 API。
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

import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402

D1 = _V3 / "dataset_d1"


def main(n_samples: int) -> None:
    idx = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    clean = {"n": 0, "fa_samples": 0, "cands": 0}
    dirty = {"n": 0, "found": 0, "cands": 0, "pos_err": [], "freq_err": []}
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                    # noqa: BLE001
            refs = {}
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        try:
            clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        except Exception as exc:                             # noqa: BLE001
            print(f"  {sid} EXC {str(exc)[:60]}")
            continue
        cands = clu.get("candidates") or []
        if not ifw:
            clean["n"] += 1
            clean["cands"] += len(cands)
            if cands:
                clean["fa_samples"] += 1
        else:
            dirty["n"] += 1
            dirty["cands"] += len(cands)
            hit = None
            for c in cands:
                p = np.asarray(c["position_xy_m"], float)
                for g in ifw:
                    d = float(np.linalg.norm(p - np.asarray(g["position_m"], float)[:2]))
                    if d <= 50 and (hit is None or d < hit[0]):
                        hit = (d, g, c)
            if hit:
                dirty["found"] += 1
                dirty["pos_err"].append(hit[0])
                fe = (abs(0.5 * (hit[2]["band_hz"][0] + hit[2]["band_hz"][1])
                          - hit[1]["freq_offset_hz"]) / 1e3)
                dirty["freq_err"].append(fe)
                print(f"    {sid}: 命中 GT {hit[1].get('kind')}"
                      f" @{hit[1]['freq_offset_hz']/1e3:8.1f}k  pos_err {hit[0]:.0f}m"
                      f"  候选 band=[{hit[2]['band_hz'][0]/1e3:.1f},"
                      f"{hit[2]['band_hz'][1]/1e3:.1f}]k"
                      f" 带中心误差 {fe:.1f}k")
            else:
                print(f"    {sid}: 未命中（候选 {len(cands)} 个）"
                      f" GT {[round(g['freq_offset_hz']/1e3) for g in ifw]}")
    print(f"\n== D1 工具级回归（{n_samples} 样本）==")
    print(f"  A 无干扰样本 {clean['n']} 个：报出候选的样本 {clean['fa_samples']}"
          f"（虚警率 {100*clean['fa_samples']/max(clean['n'],1):.0f}%），候选合计 {clean['cands']}")
    print(f"  B 有干扰样本 {dirty['n']} 个：找到 {dirty['found']}"
          f"（召回 {100*dirty['found']/max(dirty['n'],1):.0f}%），候选合计 {dirty['cands']}")
    if dirty["pos_err"]:
        print(f"     位置误差 中位 {np.median(dirty['pos_err']):.0f} m"
              f" | 频偏(带中心)误差 中位 {np.median(dirty['freq_err']):.0f} kHz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
