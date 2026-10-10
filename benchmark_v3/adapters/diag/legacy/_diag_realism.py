"""QA 前后现实性自检：新约束有没有把场景推歪？

对照 dataset_d2（QA 后）与 dataset_d2_preqa（QA 前），五项：
  1 可见站数分布（全 4 站 = 源被赶到网心）
  2 频谱深重叠比例（D2 的"挤"是否保住）
  3 源 2D 位置散布（std / 面积 / 最近邻距离）
  4 INR 分布（是否靠抬功率才可见）
  5 每样本干扰数
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parent.parent
NEW = V3 / "dataset_d2"
OLD = V3 / "dataset_d2_preqa"


def load(root: Path):
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    vis_c: Counter = Counter()
    ich_c: Counter = Counter()
    pos: list[np.ndarray] = []
    inrs: list[float] = []
    pairs = deep = 0
    for s in idx["samples"]:
        t = json.loads((root / s["id"] / "capture" / "private" / "d1_truth.json")
                       .read_text(encoding="utf-8"))
        obs = json.loads((root / s["id"] / "capture" / "public" / "observations.json")
                         .read_text(encoding="utf-8"))
        rp = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0)
        nz = t["noise"]["noise_floor_dbm"]
        ach = t.get("achieved") or {}
        ifw = t.get("interferers", [])
        ich_c[len(ifw)] += 1
        for i, g in enumerate(ifw):
            eid = f"interferer_{i+1}"
            v = sum(1 for k in range(4)
                    if (ach.get(eid, {}).get(f"s{k}", {}).get("pr_dbm", -999) - nz) >= 3.0)
            vis_c[v] += 1
            p = np.asarray(g["position_m"], float)[:2]
            pos.append(p)
            prs = [ach.get(eid, {}).get(f"s{k}", {}).get("pr_dbm", -999) for k in range(4)]
            inrs.append(float(np.median(prs)) - nz)
        for i in range(len(ifw)):
            for j in range(i + 1, len(ifw)):
                pairs += 1
                a, b = ifw[i], ifw[j]
                lo1, hi1 = a["freq_offset_hz"] - a["bandwidth_hz"] / 2, a["freq_offset_hz"] + a["bandwidth_hz"] / 2
                lo2, hi2 = b["freq_offset_hz"] - b["bandwidth_hz"] / 2, b["freq_offset_hz"] + b["bandwidth_hz"] / 2
                ov = max(0.0, min(hi1, hi2) - max(lo1, lo2)) / max(min(a["bandwidth_hz"], b["bandwidth_hz"]), 1e-9)
                if ov > 0.5:
                    deep += 1
    P = np.array(pos)
    # AOI 半径内 90% 分位散布
    r = np.linalg.norm(P - P.mean(axis=0), axis=1)
    nn = []
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = 1e9
        nn.append(d.min())
    return {
        "n": len(P), "vis": vis_c, "ich": ich_c,
        "deep_ratio": deep / max(pairs, 1), "pairs": pairs,
        "pos_std_km": (P.std(axis=0) / 1000.0).round(2),
        "r90_km": round(float(np.percentile(r, 90)) / 1000, 2),
        "nn_med_m": round(float(np.median(nn)), 0),
        "inr_med": round(float(np.median(inrs)), 1),
        "inr_p10": round(float(np.percentile(inrs, 10)), 1),
    }


if __name__ == "__main__":
    o = load(OLD)
    w = load(NEW)
    print(f"{'检查项':<28}{'QA 前':>18}{'QA 后':>18}")
    print(f"{'干扰总数':<28}{o['n']:>18}{w['n']:>18}")
    print(f"{'每样本干扰数分布':<28}{str(dict(sorted(o['ich'].items()))):>18}{str(dict(sorted(w['ich'].items()))):>18}")
    print(f"{'可见站数分布 {0,1,2,3,4}':<28}{str([o['vis'].get(k,0) for k in range(5)]):>18}{str([w['vis'].get(k,0) for k in range(5)]):>18}")
    print(f"{'  └ 4 站占比':<28}{o['vis'].get(4,0)/max(o['n'],1):>17.0%}{w['vis'].get(4,0)/max(w['n'],1):>18.0%}")
    print(f"{'深重叠对占比 >50%':<28}{o['deep_ratio']:>17.0%}{w['deep_ratio']:>18.0%}")
    print(f"{'源位置 std (x,y) km':<28}{str(o['pos_std_km']):>18}{str(w['pos_std_km']):>18}")
    print(f"{'源离质心 r90 km':<28}{o['r90_km']:>18}{w['r90_km']:>18}")
    print(f"{'源最近邻距离 中位 m':<28}{o['nn_med_m']:>18}{w['nn_med_m']:>18}")
    print(f"{'INR 中位 dB':<28}{o['inr_med']:>18}{w['inr_med']:>18}")
    print(f"{'INR p10 dB':<28}{o['inr_p10']:>18}{w['inr_p10']:>18}")
