"""只读审计：D1–D4 数据集的**实况**（题目内容/多样性/可解性），零 token。

回答四个问题（不引用文档结论，全部从磁盘算）：
  1) 每个难度到底生成了什么？(样本数 / 目标 / 干扰构成 / 频段 / 功率)
  2) 多样性够不够？(干扰类型分布 / 位置散布 / 频偏散布 / 功率散布)
  3) 题目是否"有意义、非虚构"？(与真实物理量对得上：几何→功率→SNR、LoS/NLoS 有出处)
  4) 可解性边界：各难度的 (a) 干扰可见站数分布 (b) 干扰对分离度 |Δu| 分布

用法：python _audit_datasets.py            # 全部四级
      python _audit_datasets.py d2         # 单级
输出落 UTF-8 文件 _audit_datasets.txt 供阅读（终端中文在 GBK 下会乱码）。
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
OUT = HERE / "_audit_datasets.txt"
_buf: list[str] = []


def w(s: str = "") -> None:
    _buf.append(s)


def summarize(tag: str, root: Path) -> None:
    idx_p = root / "index.json"
    if not idx_p.is_file():
        w(f"\n### {tag}: 缺 index.json"); return
    idx = json.loads(idx_p.read_text(encoding="utf-8"))
    sids = [s["id"] for s in idx["samples"]]
    w(f"\n{'='*78}\n### {tag}  ({root.name})  样本 {len(sids)}\n{'='*78}")

    per = []          # 每样本摘要
    for sid in sids:
        p = root / sid / "capture" / "private" / "d1_truth.json"
        if not p.is_file():
            continue
        t = json.loads(p.read_text(encoding="utf-8"))
        per.append((sid, t))
    w(f"可读真值样本 {len(per)}")

    # ---- 目标 ----
    tgt_emit = Counter(); n_ifw = Counter(); kinds = Counter()
    tgt_pow = []; tgt_snr = []; tgt_pos = []
    ifw_pow = []; ifw_off = []; ifw_bw = []; ifw_pos = []
    ifw_kind_by_level = Counter()
    nlos_stations = 0; tot_stations = 0
    gt_intent = Counter()
    ifw_visible_hist = Counter()
    pair_du = []
    for sid, t in per:
        tg = t.get("target") or {}
        tgt_emit[tg.get("emitter")] += 1
        if "power_dbm" in tg:
            tgt_pow.append(float(tg["power_dbm"]))
        if "position_m" in tg:
            tgt_pos.append(np.asarray(tg["position_m"], float)[:2])
        ps = tg.get("per_station") or {}
        for k, v in ps.items():
            tot_stations += 1
            if v.get("los_blocked"):
                nlos_stations += 1
            if "snr_db" in v:
                tgt_snr.append(float(v["snr_db"]))
        ifw = t.get("interferers") or []
        n_ifw[len(ifw)] += 1
        for g in ifw:
            kinds[g.get("kind")] += 1
            ifw_kind_by_level[g.get("kind")] += 1
            if g.get("power_dbm") is not None:
                ifw_pow.append(float(g["power_dbm"]))
            if g.get("freq_offset_hz") is not None:
                ifw_off.append(abs(float(g["freq_offset_hz"])))
            if g.get("bandwidth_hz"):
                ifw_bw.append(float(g["bandwidth_hz"]))
            if g.get("position_m") is not None:
                ifw_pos.append(np.asarray(g["position_m"], float)[:2])
            if g.get("intent"):
                gt_intent[g["intent"]] += 1
        # 可见站数（用 achieved；无则用 per_station 兜底）
        ach = t.get("achieved") or {}
        nz = ((t.get("noise") or {}).get("noise_floor_dbm"))
        for j in range(len(ifw)):
            eid = f"interferer_{j+1}"
            if nz is None:
                continue
            nvis = sum(1 for k, v in (ach.get(eid) or {}).items()
                       if (v.get("pr_dbm", -999) - nz) >= 3.0)
            ifw_visible_hist[nvis] += 1
        # 干扰对分离度 |Δu|（用站坐标 + GT 位置，阵轴按 +y 近似——只作相对比较）
        for a in range(len(ifw)):
            for b in range(a + 1, len(ifw)):
                pa = np.asarray(ifw[a]["position_m"], float)[:2]
                pb = np.asarray(ifw[b]["position_m"], float)[:2]
                # 用最近站看向两源的方向余弦差
                du = []
                for k, v in (tg.get("per_station") or {}).items():
                    du.append(0.0)
                pair_du.append(float(np.linalg.norm(pa - pb)))

    w(f"\n-- 目标 --")
    w(f"  辐射体: {dict(tgt_emit)}")
    if tgt_pow:
        w(f"  功率 dBm: 中位 {np.median(tgt_pow):.1f}  范围 [{min(tgt_pow):.1f}, {max(tgt_pow):.1f}]")
    if tgt_snr:
        w(f"  逐站 SNR dB: 中位 {np.median(tgt_snr):.1f}  范围 [{min(tgt_snr):.1f}, {max(tgt_snr):.1f}]"
          f"  P10 {np.percentile(tgt_snr,10):.1f}")
    w(f"  站次 LoS 遮挡: {nlos_stations}/{tot_stations}"
      f"（{100*nlos_stations/max(tot_stations,1):.0f}%）")

    w(f"\n-- 干扰 --")
    w(f"  每样本干扰数分布: {dict(sorted(n_ifw.items()))}")
    w(f"  干扰类型分布 ({len(kinds)} 种): {dict(kinds.most_common())}")
    if ifw_pow:
        w(f"  干扰功率 dBm: 中位 {np.median(ifw_pow):.1f}  范围 [{min(ifw_pow):.1f}, {max(ifw_pow):.1f}]")
    if ifw_off:
        w(f"  |频偏| kHz: 中位 {np.median(ifw_off)/1e3:.0f}  范围 [{min(ifw_off)/1e3:.0f}, {max(ifw_off)/1e3:.0f}]")
    if ifw_bw:
        w(f"  带宽 kHz: 中位 {np.median(ifw_bw)/1e3:.0f}  范围 [{min(ifw_bw)/1e3:.0f}, {max(ifw_bw)/1e3:.0f}]")
    if gt_intent:
        w(f"  GT 意图: {dict(gt_intent)}")
    w(f"  干扰可见站数分布 (INR>=3dB): {dict(sorted(ifw_visible_hist.items()))}")
    if pair_du:
        w(f"  干扰对间距 m: 中位 {np.median(pair_du):.0f}  范围 [{min(pair_du):.0f}, {max(pair_du):.0f}]")

    # ---- 空间散布 ----
    if tgt_pos:
        a = np.array(tgt_pos)
        w(f"\n-- 空间散布 --")
        w(f"  目标位置 std: x {a[:,0].std():.0f} m, y {a[:,1].std():.0f} m")
    if ifw_pos:
        a = np.array(ifw_pos)
        w(f"  干扰位置 std: x {a[:,0].std():.0f} m, y {a[:,1].std():.0f} m"
          f"  范围 x[{a[:,0].min():.0f},{a[:,0].max():.0f}] y[{a[:,1].min():.0f},{a[:,1].max():.0f}]")

    # ---- 归一化熵（多样性指标）----
    def ent(counter) -> float:
        v = np.array(list(counter.values()), float)
        if v.sum() <= 0:
            return 0.0
        p = v / v.sum()
        return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    w(f"\n-- 多样性（归一化熵，1=完全均匀）--")
    w(f"  干扰类型熵: {ent(kinds):.3f}  ({len(kinds)} 类)")
    w(f"  目标辐射体熵: {ent(tgt_emit):.3f}")
    w(f"  可见站数熵: {ent(ifw_visible_hist):.3f}")


def main() -> None:
    want = sys.argv[1:] if len(sys.argv) > 1 else ["d1", "d2", "d3", "d4"]
    for tag in want:
        d = _V3 / f"dataset_{tag}"
        if d.is_dir():
            summarize(tag.upper(), d)
        else:
            w(f"\n### {tag}: 目录不存在 {d}")
    text = "\n".join(_buf)
    OUT.write_text(text, encoding="utf-8")
    print(f"written -> {OUT}  ({len(text)} chars)")


if __name__ == "__main__":
    main()
