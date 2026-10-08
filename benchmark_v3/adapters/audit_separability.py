"""D2–D4 数据集可辨识性审计（只读）：逐源对量化"干扰不可分"到底发生在哪一维。

背景（2026-09-28）：D2 考生成绩全轴塌方（S0 检测 0.06 / 溯源 0.027 / 频偏 0.16），
需要区分"模型能力不足"与"题目物理上不可辨识"。本脚本对每个样本的每个源对
（目标×干扰、干扰×干扰）计算：

  频率维  overlap = 两源频谱重叠量 / 较窄带宽（>0 视为重叠，>0.5 视为深度重叠）
  角度维  ΔDOA 逐站（公式 GT），报 s0（station_a，S0 轴）与全站最优
  时域维  粗类别（cont/hop/tdma/burst/sweep/pulse/tone）相同 → 时域不可分
  截断    源带外缘 |f|+bw/2 超出接收通带半宽（D2–D4 rx_bw=0.405·fs=810 kHz，
          而目标 bw=1 MHz 恒截断——即 D1 2026-09-27 修掉的坑在 D2–D4 仍在）

判定：
  仪器可分(S0)   = 频率基本不重叠 或 ΔDOA(s0)≥15°（现有 7 件工具能走通的路径）
  理论可分(S0)   = 仪器可分 或 时域粗类别不同（需要时频仪器，当前工具集没有）
  不可辨识(S0)   = 深度重叠 + ΔDOA(s0)<15° + 时域同类 —— 答案键在 S0 上无物理答案

用法：
  python audit_separability.py ../dataset_d2 [--slevel S0|S2]
输出聚合统计 + 不可辨识源对清单（样本号、源对、overlap、ΔDOA）。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# 时域粗类别（emitters.py 波形结构归纳；同类 => 时域维不可分）
COARSE = {
    "LTE_QPSK": "cont", "UAV_video_OFDM": "cont", "TV_broadcast": "cont",
    "GNSS_spoof": "cont", "GNSS_jam": "cont", "Noise_jam": "cont",
    "Bluetooth_FHSS": "hop", "UAV_RC_hopping": "hop",
    "DECT": "tdma",
    "WiFi_burst": "burst", "Zigbee": "burst", "Microwave": "burst",
    "Welder": "burst", "VFD_noise": "burst", "Corona": "burst",
    "Chirp_jam": "sweep", "Radar_pulsed_LFM": "pulse",
    "Tone_jam": "tone", "tone": "tone", "noise": "noise",
}

ANG_SEP_DEG = 15.0      # 与 v3 核心仿真分离约束同口径
FREQ_SEP_MAX = 0.10     # 重叠占较窄带宽比例低于此值 = 频率基本可分


def wrap_deg(d: float) -> float:
    d = abs(d) % 360.0
    return 360.0 - d if d > 180.0 else d


def overlap_frac(f1: float, b1: float, f2: float, b2: float) -> float:
    lo1, hi1 = f1 - b1 / 2, f1 + b1 / 2
    lo2, hi2 = f2 - b2 / 2, f2 + b2 / 2
    ov = min(hi1, hi2) - max(lo1, lo2)
    return max(0.0, ov) / max(min(b1, b2), 1e-9)


def audit(root: str | Path, slevel: str = "S0") -> dict:
    root = Path(root)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    subsets = idx["observation_subsets"]
    st_keys = [f"s{i}" for i in range(len(subsets[slevel]))]   # S0→[s0] S2→[s0..s3]

    n_samples = n_pairs = n_if = 0
    stat = {"freq_overlap_pairs": 0, "deep_overlap_pairs": 0,
            "ddoa_s0_lt15": 0, "ddoa_best_lt15": 0,
            "inst_sep_s0": 0, "theo_sep_s0": 0, "unident_s0": 0,
            "inst_sep_multi": 0, "unident_multi": 0, "tgt_pair_s0_hard": 0,
            "ifw_edge_clip": 0, "tgt_clip_samples": 0,
            "ifw_links_achieved": 0, "ifw_links_invisible": 0}
    bad_pairs: list[dict] = []

    for s in idx["samples"]:
        sid = s["id"]
        cap = root / sid / "capture"
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        cfg = json.loads((cap / "private" / "scenario.json").read_text(encoding="utf-8"))
        fs = float(cfg["sample_rate_hz"])
        rx_half = float(cfg["receivers"][0]["bandwidth_hz"]) / 2
        nz = truth["noise"]["noise_floor_dbm"]

        srcs = [{"name": "target", "kind": truth["target"]["emitter"], "f": 0.0,
                 "bw": float(cfg["emitters"][0]["components"][0]["bandwidth_hz"]),
                 "doa": {k: truth["target"]["per_station"][k]["doa_deg"]
                         for k in truth["target"]["per_station"]}}]
        for j, g in enumerate(truth.get("interferers", [])):
            srcs.append({"name": f"ifw{j+1}", "kind": g["kind"], "f": g["freq_offset_hz"],
                         "bw": g["bandwidth_hz"],
                         "doa": {k: v["doa_deg"] for k, v in g["per_station"].items()}})
            n_if += 1
            if abs(g["freq_offset_hz"]) + g["bandwidth_hz"] / 2 > rx_half:
                stat["ifw_edge_clip"] += 1
            ach = (truth.get("achieved") or {}).get(f"interferer_{j+1}", {})
            for k in st_keys:
                if k in ach:
                    stat["ifw_links_achieved"] += 1
                    if ach[k]["pr_dbm"] - nz < 3.0:
                        stat["ifw_links_invisible"] += 1
        if srcs[0]["bw"] > 2 * rx_half:
            stat["tgt_clip_samples"] += 1

        n_samples += 1
        for i in range(len(srcs)):
            for j in range(i + 1, len(srcs)):
                a, b = srcs[i], srcs[j]
                n_pairs += 1
                ov = overlap_frac(a["f"], a["bw"], b["f"], b["bw"])
                if ov > 0:
                    stat["freq_overlap_pairs"] += 1
                if ov > 0.5:
                    stat["deep_overlap_pairs"] += 1
                d_s0 = wrap_deg(a["doa"].get("s0", 0) - b["doa"].get("s0", 0))
                d_best = max(wrap_deg(a["doa"].get(k, 0) - b["doa"].get(k, 0))
                             for k in {**a["doa"], **b["doa"]})
                if d_s0 < ANG_SEP_DEG:
                    stat["ddoa_s0_lt15"] += 1
                if d_best < ANG_SEP_DEG:
                    stat["ddoa_best_lt15"] += 1
                freq_sep = ov < FREQ_SEP_MAX
                inst_s0 = freq_sep or d_s0 >= ANG_SEP_DEG
                theo_s0 = inst_s0 or COARSE.get(a["kind"]) != COARSE.get(b["kind"])
                inst_multi = freq_sep or d_best >= ANG_SEP_DEG
                stat["inst_sep_s0"] += int(inst_s0)
                stat["theo_sep_s0"] += int(theo_s0)
                stat["inst_sep_multi"] += int(inst_multi)
                if not inst_s0:
                    rec = {"sample": sid, "pair": f"{a['kind']}({a['name']}) x {b['kind']}({b['name']})",
                           "overlap": round(ov, 2), "ddoa_s0": round(d_s0, 1),
                           "ddoa_best": round(d_best, 1),
                           "time_sep_theory": COARSE.get(a["kind"]) != COARSE.get(b["kind"])}
                    coop = (a["name"] == "target") or (b["name"] == "target")
                    if coop:
                        # 目标层共道：合作参考剥离（第四维）可解，不计无解
                        # （2026-09-28 口径升级：与生成约束一致——约束只管干扰×干扰对）
                        stat["tgt_pair_s0_hard"] += 1
                        rec["verdict"] = "目标层共道(合作剥离可解)"
                        bad_pairs.append(rec)
                    elif not inst_multi and COARSE.get(a["kind"]) == COARSE.get(b["kind"]):
                        # 干扰×干扰 且全站最优角度也分不开 + 时域同类 = 真无解
                        # （与生成约束 _unidentifiable_pair 同口径）
                        stat["unident_s0"] += 1
                        rec["verdict"] = "不可辨识(干扰对:频/角/时全不可分)"
                        bad_pairs.append(rec)
                    elif not inst_multi:
                        stat["unident_multi"] += 1
                        rec["verdict"] = "需时频仪器(全站角度也分不开,时域理论可分)"
                        bad_pairs.append(rec)
                    else:
                        rec["verdict"] = "S0不可分(多站角度可分)"
                        bad_pairs.append(rec)
    return {"root": str(root), "slevel": slevel, "samples": n_samples,
            "interferers": n_if, "pairs": n_pairs, "stat": stat, "bad_pairs": bad_pairs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--slevel", default="S0", choices=["S0", "S1", "S2"])
    ap.add_argument("--list", type=int, default=12, help="打印前 N 条不可分源对")
    args = ap.parse_args()
    r = audit(args.root, args.slevel)
    st = r["stat"]
    n = max(r["pairs"], 1)
    print(f"== {r['root']}  ({r['slevel']}, {r['samples']} 样本 / {r['interferers']} 干扰 / "
          f"{r['pairs']} 源对) ==")
    print(f"  频谱重叠源对          : {st['freq_overlap_pairs']}/{n}"
          f"（深度重叠>50%: {st['deep_overlap_pairs']}/{n}）")
    print(f"  ΔDOA(s0)<15° 源对     : {st['ddoa_s0_lt15']}/{n}")
    print(f"  ΔDOA(全站最优)<15°    : {st['ddoa_best_lt15']}/{n}")
    print(f"  现有仪器可分(S0)      : {st['inst_sep_s0']}/{n} = {st['inst_sep_s0']/n:.0%}")
    print(f"  理论可分含时域(S0)    : {st['theo_sep_s0']}/{n} = {st['theo_sep_s0']/n:.0%}")
    print(f"  目标层共道(合作剥离可解): {st['tgt_pair_s0_hard']}/{n}")
    print(f"  不可辨识(干扰对三维全糊): {st['unident_s0']}/{n} = {st['unident_s0']/n:.0%}")
    print(f"  多站也分不开(需时域)  : {st['unident_multi']}/{n} = {st['unident_multi']/n:.0%}")
    print(f"  干扰带外缘超通带截断  : {st['ifw_edge_clip']}/{r['interferers']}"
          f"（目标 1MHz>通带 810kHz 截断样本: {st['tgt_clip_samples']}/{r['samples']}）")
    print(f"  RT 实测干扰-站链路    : {st['ifw_links_invisible']}/{st['ifw_links_achieved']} "
          f"低于 INR 3dB（该站实际看不到）")
    print("== 不可分源对清单（前 {} 条） ==".format(args.list))
    for rec in r["bad_pairs"][: args.list]:
        print(f"  [{rec['verdict']}] {rec['sample']}  {rec['pair']}  "
              f"overlap={rec['overlap']} dDOA_s0={rec['ddoa_s0']}° dDOA_best={rec['ddoa_best']}°")
    if len(r["bad_pairs"]) > args.list:
        print(f"  … 共 {len(r['bad_pairs'])} 条")


if __name__ == "__main__":
    main()
