"""临时只读：位置精化的收益上限（oracle 位置 A/B）。

对每个"认领到真源"的候选，比较两种提取导向：
  A 现行   = 估计位置（工具落盘的 CNN-ready bfw_*）
  B oracle = 用**真值位置**重建导向矢量再提取（其余口径完全一致：带中心变频 + RMS 归一）
B 就是"位置精化做到完美"能达到的条件准确率上限。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(HERE / "rad_cls"), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen16 import CLS_IDX, to_input  # noqa: E402
from oracle16 import load_model  # noqa: E402

import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402

D2 = _V3 / "dataset_d2"
C = 299792458.0


def main(n_samples: int) -> None:
    model, _ = load_model()
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    acc = {"A": [0, 0], "B": [0, 0]}
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                    # noqa: BLE001
            refs = {}
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        vis = [g for j, g in enumerate(tr.get("interferers", []))
               if any((ach.get(f"interferer_{j+1}", {}).get(f"s{i}", {}).get("pr_dbm", -999)
                       - nz) >= 3.0 for i in range(4))]
        try:
            ct.estimate_sources_by_spatial_clustering(on_residual=True)
            ct.extract_source_waveforms()
        except Exception:                                    # noqa: BLE001
            continue
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cands = clu.get("candidates") or []
        if not cands:
            continue
        rec = next(r for r in obs["receivers"] if r["receiver_id"] == "station_a")
        off = np.asarray(rec["element_offsets_world_m"], float)
        rp = np.asarray(rec["receiver_position_m"], float)
        resid = ct._station_iq_raw("station_a", on_residual=True)
        t = np.arange(resid.shape[1])
        used = set()
        for k, c in enumerate(cands):
            g = None
            for i, gg in enumerate(vis):
                if i in used:
                    continue
                if np.linalg.norm(np.asarray(c["position_xy_m"], float)
                                  - np.asarray(gg["position_m"], float)[:2]) < 50:
                    g = (i, gg)
                    break
            if g is None:
                continue
            used.add(g[0])
            gt = g[1]
            f_c = 0.5 * (c["band_hz"][0] + c["band_hz"][1])
            # A 现行：读工具落盘的 CNN-ready 缓存
            wf = ctx.cache_dir / f"bfw_station_a_{k}.npy"
            if wf.exists():
                wk = np.load(wf).astype(np.complex64)
                with torch.no_grad():
                    pr = int(model(torch.from_numpy(np.stack([to_input(wk)]))).argmax(dim=1)[0])
                acc["A"][0] += int(pr == CLS_IDX[gt["kind"]])
                acc["A"][1] += 1
            # B oracle 位置：用真值位置重建导向再提取
            p = np.asarray(gt["position_m"], float)[:3]
            d = p - rp
            d[2] = 0
            u = d / (np.linalg.norm(d) + 1e-12)
            a = np.exp(2j * np.pi * 2.4e9 / C * (off @ u))
            w = (a.conj() @ resid) / (a.conj() @ a)
            w = w * np.exp(-1j * 2 * np.pi * f_c * t / fs)
            w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)
            with torch.no_grad():
                pr = int(model(torch.from_numpy(np.stack([to_input(w.astype(np.complex64))])))
                         .argmax(dim=1)[0])
            acc["B"][0] += int(pr == CLS_IDX[gt["kind"]])
            acc["B"][1] += 1
    a = acc["A"]
    b = acc["B"]
    print(f"\n== 位置精化上限（{n_samples} 样本）==")
    print(f"  A 现行（估计位置）      : {a[0]}/{a[1]} = {100*a[0]/max(a[1],1):.1f}%")
    print(f"  B oracle（真值位置）    : {b[0]}/{b[1]} = {100*b[0]/max(b[1],1):.1f}%")
    print(f"  → 位置精化可拿的上限     : {100*(b[0]-a[0])/max(a[1],1):+.1f}pp")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
