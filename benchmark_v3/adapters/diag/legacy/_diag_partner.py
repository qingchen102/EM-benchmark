"""临时只读：oracle +12.5pp 的漏损分解。

对每个"认领到真源"的候选（真源有可视频带重叠伙伴），逐环节检查：
  1 伙伴是否进了候选集（有候选落在伙伴真值 50m 内）
  2 该伙伴候选是否满足入组双条件（频带重叠 AND 站均导向互相关<0.8）
  3 二者是否真的落进同一个组
  4 组内成分：组里有几个候选、其中几个是真源
输出每一环的通过率 → 定位 +12.5pp 漏在哪。
"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from collections import Counter
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

D2 = _V3 / "dataset_d2"
C = 299792458.0
CORR_TOL = 0.8


def steer(off, rp, p2):
    d = np.array([float(p2[0]) - rp[0], float(p2[1]) - rp[1], 0.0])
    u = d / (np.linalg.norm(d) + 1e-12)
    return np.exp(2j * np.pi * 2.4e9 / C * (off @ u))


def main(n_samples: int) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    st = Counter()
    gsize = Counter()
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
            clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        except Exception:                                    # noqa: BLE001
            continue
        cands = clu.get("candidates") or []
        if not cands:
            continue
        recs = obs["receivers"]
        steers = {}
        for r in recs:
            off = np.asarray(r["element_offsets_world_m"], float)
            rp = np.asarray(r["receiver_position_m"], float)
            steers[r["receiver_id"]] = [steer(off, rp, c["position_xy_m"]) for c in cands]

        def corr(i, j):
            return float(np.mean([abs(np.vdot(steers[r["receiver_id"]][i],
                                              steers[r["receiver_id"]][j]))
                                  / off_n for r in recs for off_n in [8]]))

        # 生产分组复刻
        groups: list[dict] = []
        for k, c in enumerate(cands):
            blo, bhi = c["band_hz"]
            placed = False
            for g in groups:
                if bhi < g["lo"] or blo > g["hi"]:
                    continue
                if all(corr(k, m) < CORR_TOL for m in g["idx"]):
                    g["idx"].append(k)
                    g["lo"], g["hi"] = min(g["lo"], blo), max(g["hi"], bhi)
                    placed = True
                    break
            if not placed:
                groups.append({"idx": [k], "lo": blo, "hi": bhi})
        gid = {}
        for gi, g in enumerate(groups):
            for m in g["idx"]:
                gid[m] = gi

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
                st["无共道伙伴(不在分母)"] += 1
                continue
            st["分母:有共道伙伴"] += 1
            # 1 伙伴是否进了候选集
            pk = [m for m in range(len(cands))
                  if np.linalg.norm(np.asarray(cands[m]["position_xy_m"], float)
                                    - np.asarray(partner["position_m"], float)[:2]) < 50]
            if not pk:
                st["① 伙伴未进候选集"] += 1
                continue
            st["① 伙伴在候选集"] += 1
            # 2 是否满足双条件（对该认领 k）
            ok = [m for m in pk
                  if not (cands[m]["band_hz"][1] < c["band_hz"][0]
                          or cands[m]["band_hz"][0] > c["band_hz"][1])
                  and corr(k, m) < CORR_TOL]
            if not ok:
                st["② 双条件不过(频带或去相关)"] += 1
                continue
            st["② 双条件通过"] += 1
            # 3 是否同组
            same = [m for m in ok if gid.get(m) == gid.get(k)]
            if not same:
                st["③ 未落进同组"] += 1
                continue
            st["③ 同组 ✓"] += 1
            gs = len(groups[gid[k]]["idx"])
            gsize[gs] += 1
            # 组内真源数
            ntrue = 0
            for m in groups[gid[k]]["idx"]:
                pm = np.asarray(cands[m]["position_xy_m"], float)
                if any(np.linalg.norm(pm - np.asarray(gg["position_m"], float)[:2]) < 50 for gg in vis):
                    ntrue += 1
            st[f"   组内真源数={ntrue}"] += 1
    tot = st["分母:有共道伙伴"]
    print(f"\n== oracle +12.5pp 的漏损分解（{n_samples} 样本）==")
    print(f"  分母（认领中真源有共道伙伴的）: {tot}")
    for kk in sorted(st):
        if kk != "分母:有共道伙伴":
            print(f"  {kk:30s} {st[kk]:4d}"
                  + (f"  ({100*st[kk]/max(tot,1):.0f}% of 分母)" if kk[0] in "①②③" else ""))
    print(f"  组规模分布（对③通过的）: {dict(sorted(gsize.items()))}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
