"""临时只读：条件准确率的剩余差距归因 —— 位置误差 / 源功率 / 共道伴随源。

对每个"认领到真源"的候选（gate 同口径），记录：
  pos_err  候选位置 vs 真源的 2D 误差
  inr      该真源在 station_a 的 INR
  companion_ratio  同带（共道）内最强伴随源功率 / 本源功率（=1 表示共道等强）
  correct  CNN 是否判对
然后按正确/错误分组对比。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(HERE / "rad_cls"), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct  # noqa: E402
from data_gen16 import CLS_IDX  # noqa: E402
from oracle16 import load_model  # noqa: E402

D2 = _V3 / "dataset_d2"


def main(n_samples: int) -> None:
    import torch
    model, _ = load_model()
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    rows = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        try:
            refs = __import__("run_capture_agent")._build_target_reference(priv, n)
        except Exception:                                   # noqa: BLE001
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=float(obs["receivers"][0]["sample_rate_hz"]),
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        vis = [(j, g) for j, g in enumerate(tr.get("interferers", []))
               if any((ach.get(f"interferer_{j+1}", {}).get(f"s{i}", {}).get("pr_dbm", -999)
                       - nz) >= 3.0 for i in range(4))]
        try:
            clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        except Exception:                                   # noqa: BLE001
            continue
        cands = clu.get("candidates") or []
        if not cands:
            continue
        ct.extract_source_waveforms()          # 必须先跑提取，否则 bfw_* 是上一轮残留
        for k, c in enumerate(cands):
            p = np.asarray(c["position_xy_m"], float)
            best = None
            for j, g in vis:
                d = float(np.linalg.norm(p - np.asarray(g["position_m"], float)[:2]))
                if d < 50 and (best is None or d < best[0]):
                    best = (d, j, g)
            if best is None:
                continue
            d, j, g = best
            w = np.load(ctx.cache_dir / f"bfw_station_a_{k}.npy").astype(np.complex128)
            # 与 gate 同口径：按候选频带中心变频 + RMS 归一化（缺任一步都会把
            # 条件准确率从 ~46% 压到 ~7%——2026-10-04 踩坑）
            _t = np.arange(len(w))
            _fc = 0.5 * (c["band_hz"][0] + c["band_hz"][1])
            w = w * np.exp(-1j * 2 * np.pi * _fc * _t / ctx.fs)
            w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)
            xb = torch.from_numpy(__import__("data_gen16").to_input(w.astype(np.complex64))[None])
            with torch.no_grad():
                pred = int(model(xb).argmax(dim=1)[0])
            # 共道伴随：与本源频带重叠的其他可见源中，功率最高者的功率比
            pw = {}
            for j2, g2 in vis:
                pw[j2] = ach.get(f"interferer_{j2+1}", {}).get("s0", {}).get("pr_dbm", -999)
            lo1 = g["freq_offset_hz"] - g["bandwidth_hz"] / 2
            hi1 = g["freq_offset_hz"] + g["bandwidth_hz"] / 2
            comp = 0.0
            for j2, g2 in vis:
                if j2 == j:
                    continue
                lo2 = g2["freq_offset_hz"] - g2["bandwidth_hz"] / 2
                hi2 = g2["freq_offset_hz"] + g2["bandwidth_hz"] / 2
                if not (hi2 < lo1 or lo2 > hi1):
                    comp = max(comp, 10 ** ((pw.get(j2, -999) - pw.get(j, -999)) / 10))
            rows.append((sid, k, d, pw.get(j, -999) - nz, comp,
                         int(pred == CLS_IDX[g["kind"]])))
    ok = [r for r in rows if r[5]]
    bad = [r for r in rows if not r[5]]
    print(f"\n== 认领总数 {len(rows)}  判对 {len(ok)}  判错 {len(bad)} ==")
    for tag, g in (("判对", ok), ("判错", bad)):
        if not g:
            continue
        pe = np.array([r[2] for r in g])
        inr = np.array([r[3] for r in g])
        cm = np.array([r[4] for r in g])
        print(f"  {tag}: 位置误差 中位 {np.median(pe):5.1f} m (p90 {np.percentile(pe,90):6.1f})"
              f" | INR 中位 {np.median(inr):5.1f} dB | 共道伴随功率比 中位 {np.median(cm):5.2f}"
              f" | 有共道伴随者 {int((cm>0.1).sum())}/{len(g)}")
    print("\n  分位对比（判对 vs 判错 的中位数比）：")
    if ok and bad:
        print(f"    位置误差  {np.median([r[2] for r in ok]):.1f} vs {np.median([r[2] for r in bad]):.1f} m")
        print(f"    INR       {np.median([r[3] for r in ok]):.1f} vs {np.median([r[3] for r in bad]):.1f} dB")
        print(f"    共道功率比 {np.median([r[4] for r in ok]):.2f} vs {np.median([r[4] for r in bad]):.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
