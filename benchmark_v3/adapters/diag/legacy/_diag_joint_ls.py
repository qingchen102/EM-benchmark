"""临时只读：K=1 vs 共道联合提取（oracle 共道伙伴）。

背景：FORCE_SINGLE（K=1）是因为"近共线候选让 LS 病态"才开的；
但新数据集已强制**共道对 |Δu| ≥ 2/M**，导向矢量应当已去相关 → 联合 LS 可能重新可行。

对比（同一批"认领到真源"的候选，估计位置、同一带中心变频+RMS 归一）：
  A K=1：只用该候选自己的导向                    （现行 FORCE_SINGLE）
  B K=2：加**真值共道伙伴**的导向做联合 LS        （oracle 分组，测上限）
共道伙伴 = 频带与该源重叠、且在 station_a 可见的其他真源。
"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
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


def steer(off, rp, p2):
    d = np.asarray(p2, float)[:2] - rp[:2]
    d3 = np.array([d[0], d[1], 0.0])
    u = d3 / (np.linalg.norm(d3) + 1e-12)
    return np.exp(2j * np.pi * 2.4e9 / C * (off @ u))


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
            a_own = steer(off, rp, c["position_xy_m"])
            # A: K=1
            w = (a_own.conj() @ resid) / (a_own.conj() @ a_own)
            w = w * np.exp(-1j * 2 * np.pi * f_c * t / fs)
            w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)
            with torch.no_grad():
                pr = int(model(torch.from_numpy(np.stack([to_input(w.astype(np.complex64))])))
                         .argmax(dim=1)[0])
            pa = int(pr == CLS_IDX[gt["kind"]])
            # B: K=2（加真值共道伙伴）
            p1 = np.asarray(gt["position_m"], float)[:2]
            lo1, hi1 = gt["freq_offset_hz"] - gt["bandwidth_hz"] / 2, gt["freq_offset_hz"] + gt["bandwidth_hz"] / 2
            partner = None
            for gg in vis:
                if gg is gt:
                    continue
                lo2, hi2 = gg["freq_offset_hz"] - gg["bandwidth_hz"] / 2, gg["freq_offset_hz"] + gg["bandwidth_hz"] / 2
                if not (hi2 < lo1 or lo2 > hi1):
                    partner = gg
                    break
            if partner is None:
                continue
            A = np.stack([a_own, steer(off, rp, np.asarray(partner["position_m"], float)[:2])], axis=1)
            if float(np.linalg.cond(A)) > 1e6:
                continue
            S = np.linalg.pinv(A) @ resid
            w2 = S[0] * np.exp(-1j * 2 * np.pi * f_c * t / fs)
            w2 = w2 / (np.sqrt(np.mean(np.abs(w2) ** 2)) + 1e-12)
            with torch.no_grad():
                pr = int(model(torch.from_numpy(np.stack([to_input(w2.astype(np.complex64))])))
                         .argmax(dim=1)[0])
            acc["A"][0] += pa                      # 同集对照：只在 B 也算得出时才记 A
            acc["A"][1] += 1
            acc["B"][0] += int(pr == CLS_IDX[gt["kind"]])
            acc["B"][1] += 1
    a, b = acc["A"], acc["B"]
    print(f"\n== K=1 vs 共道联合提取（{n_samples} 样本）==")
    print(f"  A K=1 单源投影（现行）        : {a[0]}/{a[1]} = {100*a[0]/max(a[1],1):.1f}%")
    print(f"  B K=2 联合 LS（含真值共道伙伴）: {b[0]}/{b[1]} = {100*b[0]/max(b[1],1):.1f}%")
    print(f"  → 联合提取可拿的上限          : {100*(b[0]-a[0])/max(a[1],1):+.1f}pp")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
