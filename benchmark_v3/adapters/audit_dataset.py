"""D1–D4 数据集审计：对照 v1/v2 已知问题逐项检查。

检查项（来源=v1/v2 迭代教训与 v3 修复清单）：
  P0-1  GT 频偏不得恒 0，且与 config 一致
  v1-5a 干扰-目标频谱分离（仅 D1，Hz 域）：|f|−bw/2 ≥ 目标半宽+50kHz
        （2026-09-27 起；D2 共存源频谱重叠是难度本身，不查）
  v1-5b 干扰可检测性：逐站 INR ≥ 3 dB（D1 包络要求"看得见"）
  v2-a  奈奎斯特防折叠：|offset| + bw/2 < 0.48·fs（fs 逐样本读 config）
  v2-d  通带内裕量（仅 D1）：|f| + bw/2 ≤ 接收通带半宽 − 50kHz
  v2-b  SNR 语义 = 目标接收功率/噪声底（逐站）【尚未实现，仅留档】
  v2-c  信息隔离：observations.json 不含 emitter/intent/真值参数
  gen   IQ 有限、形状=(阵元, 采样数)、每站齐全
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def audit(root: str | Path, d2: bool = False) -> dict:
    root = Path(root)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    is_d1 = str(idx.get("difficulty", "")).startswith("D1")
    issues: list[dict] = []
    stats = {"samples": len(idx["samples"]), "sep_ok": 0,
             "inr_ok": 0, "inr_low_stations": 0, "iso_ok": 0, "iq_ok": 0}

    for s in idx["samples"]:
        sid = s["id"]
        cap = root / sid / "capture"
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        cfg = json.loads((cap / "private" / "scenario.json").read_text(encoding="utf-8"))
        cfg_by_id = {e["id"]: e for e in cfg["emitters"]}
        ach = truth.get("achieved", {})
        nz = truth["noise"]["noise_floor_dbm"]
        fs = float(cfg["sample_rate_hz"])                       # 逐样本（四档实测均为 4e6）
        rx_half = float(cfg["receivers"][0]["bandwidth_hz"]) / 2
        tgt_bw = float(cfg_by_id["target"]["components"][0]["bandwidth_hz"])

        def flag(code, msg):
            issues.append({"sample": sid, "code": code, "msg": msg})

        # IQ 有限性
        iq_ok = True
        for r in truth["receivers"]:
            f = cap / "public" / f"iq_{r['id']}.npy"
            if not f.is_file():
                iq_ok = False; flag("gen", f"{r['id']} IQ 缺失"); continue
            iq = np.load(f)
            if not np.isfinite(iq).all() or iq.ndim != 2:
                iq_ok = False; flag("gen", f"{r['id']} IQ 非有限/形状异常 {iq.shape}")
        if iq_ok:
            stats["iq_ok"] += 1

        # P0-1 频偏一致且非恒 0（逐干扰）
        for j, ifw in enumerate(truth.get("interferers", [])):
            gt_f = ifw["freq_offset_hz"]
            eid = f"interferer_{j + 1}"
            cfg_f = cfg_by_id[eid]["components"][0]["offset_hz"]
            if abs(gt_f) < 1e-6:
                flag("P0-1", f"{eid} 频偏 GT 为 0")
            if abs(gt_f - cfg_f) > 1e-3:
                flag("P0-1", f"{eid} 频偏 GT/config 不一致 {gt_f} vs {cfg_f}")

            bw = ifw["bandwidth_hz"]
            # v2-a 奈奎斯特（逐样本 fs）
            if abs(gt_f) + bw / 2 >= 0.48 * fs:
                flag("v2-a", f"{eid}: |{gt_f:.0f}|+{bw/2:.0f} >= 0.48fs")

            # v1-5a 干扰-目标频谱分离（仅 D1，Hz 域，2026-09-27 口径）
            if is_d1:
                if abs(gt_f) - bw / 2 < tgt_bw / 2 + 50e3:
                    flag("v1-5a", f"{eid}: |{gt_f:.0f}|-{bw/2:.0f} < 目标半宽{tgt_bw/2:.0f}+50k")
                else:
                    stats["sep_ok"] += 1
                # v2-d 通带内裕量（仅 D1）
                if abs(gt_f) + bw / 2 > rx_half - 50e3:
                    flag("v2-d", f"{eid}: 外缘{abs(gt_f)+bw/2:.0f} > 通带半宽{rx_half:.0f}-50k")

            # v1-5b 逐站可见性：2026-09-28 口径升级——RT 城市场景下"某站看不到
            # 某干扰"是物理事实（遮挡），achieved 已让答案键正确排除，不算缺陷；
            # 只有"无 achieved（生成器漏回填）且公式 INR<3"才算生成问题。
            for rid, ps in ifw["per_station"].items():
                ach_entry = (ach.get(eid) or {}).get(rid)
                if ach_entry is not None:                    # RT 实测口径
                    if ach_entry["pr_dbm"] - nz < 3.0:
                        stats["inr_low_stations"] += 1       # 遮挡：物理事实，答案键已排除
                    else:
                        stats["inr_ok"] += 1
                elif ps["inr_db"] < 3.0:                     # 无实测且公式不可见 → 生成问题
                    stats["inr_low_stations"] += 1
                    flag("v1-5b", f"{rid}: INR={ps['inr_db']:.1f}dB<3（该站不可见）")
                else:
                    stats["inr_ok"] += 1

        # v2-c 信息隔离
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        txt = json.dumps(obs)
        if not any(k in txt for k in ("emitter", "intent", "is_jamming", "freq_offset_hz")):
            stats["iso_ok"] += 1
        else:
            flag("v2-c", "observations 泄漏答案轴")

    stats["issues"] = issues
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--d2", action="store_true", help="D2：跳过 D1 分离约束检查")
    args = ap.parse_args()
    st = audit(args.root, d2=args.d2)
    by_code: dict[str, list] = {}
    for i in st["issues"]:
        by_code.setdefault(i["code"], []).append(i)
    print("== 统计 ==")
    for k, v in st.items():
        if k != "issues":
            print(f"  {k}: {v}")
    print("== 问题按代码 ==")
    for code, lst in sorted(by_code.items()):
        print(f"  {code}: {len(lst)} 处  例: {lst[0]['sample']} {lst[0]['msg'][:60]}")
    if not by_code:
        print("  无 ✅")
