# -*- coding: utf-8 -*-
"""Step 1: 全量 500 样本离线诊断报告（零 API）。

产出四块：
A. 工具指标分层表：源数建议准确率(按真值源数×SNR)、频偏候选覆盖率(按类别×INR×带宽)、
   强幽灵样本率(按目标调制×SNR)
B. 类别规则稳健性：给 GT 参数加"工具级"测量噪声后套判定规则，看规则边界脆弱度
D. 格子计数审计：为 Step 2 定向补数据提供依据

运行: python diag_offline_report.py [--dataset dataset]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_SIM_DIR = _ROOT / "simulation"
for _p in (_ROOT, _SIM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tools_v2


def snr_band(snr):
    return "Low(<0)" if snr < 0 else ("Mid(0~8)" if snr <= 8 else "High(>8)")


def pct(h, n):
    return f"{h}/{n}={h / max(n, 1):.2f}"


def tgt_mod_of(gt):
    return gt["sources"][0].get("modulation", "?")


def rule_cat(ratio, power_db):
    if ratio < 0.5:
        return "co_channel"
    if power_db >= 10.0:
        return "blocking"
    if ratio < 2.0:
        return "adjacent"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    args = ap.parse_args()
    root = _ROOT / args.dataset
    gt_data = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    report = {"n_samples": len(gt_data)}

    # ---------- A. 工具指标分层 ----------
    src_cell = defaultdict(lambda: [0, 0])
    cov_cell = defaultdict(lambda: [0, 0])
    cov_bw = defaultdict(lambda: [0, 0])
    ph_cell = defaultdict(lambda: [0, 0])

    for rec in gt_data:
        gt = rec["ground_truth"]
        gts = gt["sources"][1:]
        tgt_bw = gt["sources"][0].get("bandwidth_normalized")
        path = str(root / rec["file"])
        band = snr_band(gt.get("snr_db", 0))
        spec = tools_v2.analyze_spectrum(path, target_bandwidth_normalized=tgt_bw)
        cands = [(c["freq"], c["power_ratio_approx_db"]) for c in spec["sources_candidates"]]

        if not gts:
            ob_strong = sum(1 for f, r in cands
                            if abs(f) > max(tgt_bw or 0.1, 0.1) / 2 and r >= -6)
            ph_cell[(tgt_mod_of(gt), band)][0] += int(ob_strong >= 2)
            ph_cell[(tgt_mod_of(gt), band)][1] += 1
            continue
        ns = len(gts)
        r = tools_v2.estimate_num_sources(path, target_bandwidth_normalized=tgt_bw)
        ok = int(r["final_suggestion"] - 1 == ns)
        src_cell[(ns, band)][0] += ok
        src_cell[(ns, band)][1] += 1
        for g in gts:
            gf = g["freq_offset_normalized"]
            hit = any(abs(f - gf) < 0.03 for f, _ in cands)
            inr_b = ("<=6dB" if g.get("inr_db", 99) <= 6 else
                     "6~15dB" if g.get("inr_db", 99) <= 15 else ">15dB")
            cov_cell[(g["v2_category"], inr_b)][0] += int(hit)
            cov_cell[(g["v2_category"], inr_b)][1] += 1
            bwb = ("<0.15" if g["bandwidth_normalized"] < 0.15 else
                   "0.15~0.35" if g["bandwidth_normalized"] < 0.35 else ">0.35")
            cov_bw[bwb][0] += int(hit)
            cov_bw[bwb][1] += 1

    print("=" * 64)
    print("A1. 源数 final_suggestion 准确率（按真值干扰数 × SNR）")
    for (ns_, band), (ok, tot) in sorted(src_cell.items()):
        print(f"    {ns_}干扰 @{band:9s}: {pct(ok, tot)}")
    print("A2. 频偏候选覆盖率（按类别 × INR）")
    for (cat, inrb), (hit, tot) in sorted(cov_cell.items()):
        print(f"    {cat:10s} INR{inrb:7s}: {pct(hit, tot)}")
    print("A3. 频偏候选覆盖率（按干扰带宽）")
    for bwb, (hit, tot) in sorted(cov_bw.items()):
        print(f"    bw {bwb:9s}: {pct(hit, tot)}")
    print("A4. 强幽灵样本率（干净样本，按目标调制 × SNR）")
    for (m, band), (ph, tot) in sorted(ph_cell.items()):
        if tot >= 3:
            print(f"    目标={m:6s} @{band:9s}: {ph}/{tot}")
    report["src_by_cell"] = {str(k): v for k, v in src_cell.items()}
    report["cov_by_cat_inr"] = {str(k): v for k, v in cov_cell.items()}
    report["cov_by_bw"] = {str(k): v for k, v in cov_bw.items()}

    # ---------- B. 类别规则稳健性 ----------
    print("\nB. 类别规则稳健性（GT 参数 + 测量噪声 -> 规则 -> 与 GT 标签比对）")
    rng = np.random.default_rng(0)
    rows = []                     # (cat_true, f_off, tgt_bw, power_db)
    for rec in gt_data:
        gt = rec["ground_truth"]
        tbw = max(gt["sources"][0].get("bandwidth_normalized") or 0.1, 1e-6)
        for g in gt["sources"][1:]:
            if g.get("is_pulse"):
                continue                      # pulse 由时域证据判，不在此模拟
            rows.append((g["v2_category"], g["freq_offset_normalized"], tbw,
                         g.get("power_ratio_db", 0.0)))
    for sig_f, sig_p in [(0.02, 1.5), (0.04, 3.0)]:
        conf = Counter()
        for cat_true, f_off, tbw, pw in rows:
            f_hat = f_off + rng.normal(0, sig_f)
            p_hat = pw + rng.normal(0, sig_p)
            conf[(cat_true, rule_cat(abs(f_hat) / tbw, p_hat))] += 1
        hits = sum(c for (a, b), c in conf.items() if a == b)
        print(f"    σf={sig_f}, σp={sig_p}dB: 规则命中 {hits}/{len(rows)} = "
              f"{hits / len(rows):.3f}")
        if sig_f == 0.04:
            print("      主要混淆:")
            for (a, b), c in sorted(conf.items(), key=lambda kv: -kv[1]):
                if a != b and c >= 5:
                    print(f"        GT={a:10s} -> rule={b:10s} x{c}")
    report["rule_rows"] = len(rows)

    # ---------- D. 格子计数审计 ----------
    print("\nD. 格子计数审计（Step 2 定向补数据的依据）")
    cell_cnt = Counter()
    for rec in gt_data:
        gt = rec["ground_truth"]
        band = snr_band(gt.get("snr_db", 0))
        for g in gt["sources"][1:]:
            mod = g.get("display_name") or g.get("waveform_type") or "?"
            cell_cnt[(g["v2_category"], mod, band)] += 1
    thin = [(k, c) for k, c in cell_cnt.items() if c < 8]
    print(f"    干扰总格子数 {len(cell_cnt)},其中样本数<8 的薄格子 {len(thin)} 个:")
    for k, c in sorted(thin, key=lambda kv: kv[1])[:15]:
        print(f"      {k}: {c}")

    root_out = _ROOT / "diag_full500.json"
    root_out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n报告 -> {root_out}")


if __name__ == "__main__":
    main()
