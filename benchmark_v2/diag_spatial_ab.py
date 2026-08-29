"""空间候选生成 vs 旧一维路径的离线 A/B（零 token，500 冻结样本）。

对比三臂：
- legacy : tools_v2.analyze_spectrum 的 sources_candidates（现状，v5 基线口径）
- spatial: spatial_candidates.spatial_candidates（MVDR 频率×角度，本实验）
- merged : 两臂并集（freq 容差 0.02 去重；接入方案预演）

指标与 diag_offline_report.py 同口径，可直接与 diag_full500.json 对照：
- 覆盖率 hit = 存在候选 |f_cand - f_gt| < 0.03（按 类别×INR / 带宽 分格）
- 幽灵：无干扰样本上 |f|>max(tgt_bw,0.1)/2 且 power≥-6 的强候选
- 功率/频点精度：GT 干扰与最近候选（<0.03）配对后的误差分布
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import tools_v2
import spatial_candidates as sc

_ROOT = Path(__file__).resolve().parent


def inr_band(inr):
    return "<=6dB" if inr <= 6 else ("6~15dB" if inr <= 15 else ">15dB")


def bw_band(bw):
    return "<0.15" if bw < 0.15 else ("0.15~0.35" if bw < 0.35 else ">0.35")


def merge_arms(legacy, spatial, tol=0.02):
    """并集：spatial 候选若与 legacy 频点重合则视为同源（合并 angle 信息）。"""
    out = [dict(c, arm="legacy") for c in legacy]
    for s in spatial:
        near = [c for c in legacy if abs(c["freq"] - s["freq"]) < tol]
        if near:
            out[legacy.index(near[0])].setdefault("angle_deg", s.get("angle_deg"))
        else:
            out.append(dict(s, arm="spatial"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本（调试）")
    ap.add_argument("--out", default="diag_spatial_ab.json")
    args = ap.parse_args()

    root = _ROOT / args.dataset
    gt_data = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    if args.limit:
        gt_data = gt_data[:args.limit]

    arms = ("legacy", "spatial", "merged")
    cov = {a: defaultdict(lambda: [0, 0]) for a in arms}
    cov_bw = {a: defaultdict(lambda: [0, 0]) for a in arms}
    perr = {a: [] for a in arms}      # (err_db, gt_cat, gt_inr)
    ferr = {a: [] for a in arms}
    aerr = {a: [] for a in arms}
    ghosts = {a: [0, 0] for a in arms}  # [有强幽灵的干净样本, 干净样本总数]

    for rec in gt_data:
        gt = rec["ground_truth"]
        gts = gt["sources"][1:]
        tgt_bw = gt["sources"][0].get("bandwidth_normalized")
        path = str(root / rec["file"])

        legacy = tools_v2.analyze_spectrum(
            path, target_bandwidth_normalized=tgt_bw)["sources_candidates"]
        spatial = sc.spatial_candidates(
            path, target_bandwidth_normalized=tgt_bw)["sources_candidates"]
        merged = merge_arms(legacy, spatial)
        arm_cands = {"legacy": legacy, "spatial": spatial, "merged": merged}

        if not gts:
            for a, cands in arm_cands.items():
                strong = sum(1 for c in cands
                             if abs(c["freq"]) > max(tgt_bw or 0.1, 0.1) / 2
                             and c["power_ratio_approx_db"] >= -6)
                ghosts[a][0] += int(strong >= 2)
                ghosts[a][1] += 1
            continue

        for a, cands in arm_cands.items():
            for g in gts:
                gf = g["freq_offset_normalized"]
                hits = sorted(((abs(c["freq"] - gf), c) for c in cands),
                              key=lambda t: t[0])
                hit, best = (hits[0] if hits and hits[0][0] < 0.03 else (None, None))
                key = (g["v2_category"], inr_band(g.get("inr_db", 99)))
                cov[a][key][0] += int(bool(hit))
                cov[a][key][1] += 1
                kbw = bw_band(g["bandwidth_normalized"])
                cov_bw[a][kbw][0] += int(bool(hit))
                cov_bw[a][kbw][1] += 1
                if best is not None:
                    perr[a].append(best["power_ratio_approx_db"] - g["power_ratio_db"])
                    ferr[a].append(best["freq"] - gf)
                    if "angle_deg" in best and best.get("angle_deg") is not None:
                        aerr[a].append(abs(best["angle_deg"] - g["doa_degree"]))

    def cov_total(d):
        h = sum(v[0] for v in d.values())
        n = sum(v[1] for v in d.values())
        return h, n

    report = {"n_samples": len(gt_data), "arms": {}}
    for a in arms:
        h, n = cov_total(cov[a])
        p = np.array(perr[a]) if perr[a] else np.array([0.0])
        f = np.array(ferr[a]) if ferr[a] else np.array([0.0])
        cal = float(np.median(-p))
        arm = {
            "cov_total": [h, n], "cov_overall": round(h / max(n, 1), 3),
            "cov_by_cat_inr": {str(k): v for k, v in sorted(cov[a].items())},
            "cov_by_bw": {str(k): v for k, v in sorted(cov_bw[a].items())},
            "clean_samples": ghosts[a][1],
            "strong_ghost_samples": ghosts[a][0],
            "power_err_raw_median": round(float(np.median(p)), 2),
            "power_err_raw_mae": round(float(np.mean(np.abs(p))), 2),
            "power_cal_implied_db": round(cal, 2),
            "power_err_cal_mae": round(float(np.mean(np.abs(p + cal))), 2),
            "freq_err_mae": round(float(np.mean(np.abs(f))), 4),
        }
        if aerr[a]:
            arm["angle_err_median_deg"] = round(float(np.median(aerr[a])), 1)
        report["arms"][a] = arm

    print("=" * 72)
    print("离线 A/B：候选覆盖率 / 功率精度 / 幽灵（500 冻结样本，同官方口径）")
    print("=" * 72)
    hdr = f"{'指标':<28}" + "".join(f"{a:>14}" for a in arms)
    print(hdr)
    rows = []
    for a in arms:
        r = report["arms"][a]
        rows.append(r)
    print(f"{'覆盖总率':<26}" + "".join(
        f"{r['cov_overall']:>10} ({r['cov_total'][0]}/{r['cov_total'][1]})"
        for r in rows))
    for bwb in ("<0.15", "0.15~0.35", ">0.35"):
        print(f"{'覆盖 bw ' + bwb:<24}" + "".join(
            f"{r['cov_by_bw'].get(bwb, [0, 0])[0]}/{r['cov_by_bw'].get(bwb, [0, 0])[1]:<4}"
            f" = {r['cov_by_bw'].get(bwb, [0, 0])[0] / max(r['cov_by_bw'].get(bwb, [0, 0])[1], 1):.2f}"
            for r in rows))
    for cat in ("co_channel", "adjacent", "blocking", "none"):
        for inrb in ("<=6dB", "6~15dB", ">15dB"):
            k = f"('{cat}', '{inrb}')"
            vals = [r["cov_by_cat_inr"].get(k, [0, 0]) for r in rows]
            if sum(v[1] for v in vals) == 0:
                continue
            print(f"{'  ' + cat + ' INR' + inrb:<24}" + "".join(
                f"{v[0]}/{v[1]} = {v[0] / max(v[1], 1):.2f}"[:13].rjust(14)
                for v in vals))
    print(f"{'幽灵样本/干净样本':<24}" + "".join(
        f"{r['strong_ghost_samples']}/{r['clean_samples']}".rjust(14) for r in rows))
    print(f"{'功率误差 MAE(未标定)':<23}" + "".join(
        f"{r['power_err_raw_mae']:>13.2f}dB" for r in rows))
    print(f"{'功率标定常数(隐含)':<23}" + "".join(
        f"{r['power_cal_implied_db']:>13.2f}dB" for r in rows))
    print(f"{'功率误差 MAE(标定后)':<23}" + "".join(
        f"{r['power_err_cal_mae']:>13.2f}dB" for r in rows))
    print(f"{'频点误差 MAE':<26}" + "".join(f"{r['freq_err_mae']:>14.4f}" for r in rows))
    for r in rows:
        if "angle_err_median_deg" in r:
            print(f"角度误差中位数[{r is rows[1] and 'spatial' or 'merged'}]: "
                  f"{r['angle_err_median_deg']}°")

    out = _ROOT / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
