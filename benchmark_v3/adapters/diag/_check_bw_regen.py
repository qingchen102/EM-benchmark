# -*- coding: utf-8 -*-
"""带宽重生成验收（只读，零 API）。

对比「现行 dataset_dX」与「_regen_bw/dataset_dX」（= 用带宽修好后的生成器重放同一批
冻结信道得到的数据），回答三件事：

  1) 每站 IQ 总功率是否变化（带限到底有没有动到发射/接收功率）；
  2) station_a 峰值附近 99% 连续占用带宽（旧 vs 新），以及越出接收通带 ±810 kHz 的比例；
  3) **答案键 achieved.pr_dbm（重放时从源样本拷贝，属旧波形）** vs 平台新
     truth.json 里实测的 mean_port_power_w —— 量化 achieved 是否已失效。

用法：python -u _check_bw_regen.py [d2,d3,d4]
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FS = 4e6
PASS_HZ = 810e3
NEW_ROOT = os.path.join(ROOT, "_regen_bw")


def psd(iq):
    x = np.asarray(iq)
    X = np.fft.fft(x, axis=-1)
    return np.fft.fftshift((np.abs(X) ** 2).sum(axis=0))


def psd_db(p):
    return 10.0 * np.log10(p + 1e-30)


def edge_drop(io, inw, sources, fs=FS):
    """设计带宽的「带内 vs 紧邻带外」能量变化（新 − 旧，dB）。

    带内应≈0 dB（总功率没变，平台照样归一化），紧邻设计带宽之外应显著为负
    （软带限把原来洩到带外的能量削掉了）。返回 (带内中位, 带外中位,
    |f|>810kHz 能量占比_旧, _新)。
    """
    n = io.shape[-1]
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    po = psd(io)
    pn = psd(inw)
    d = psd_db(pn) - psd_db(po)
    ins, outs = [], []
    for f0, bw in sources:
        half = bw / 2.0
        m_in = np.abs(f - f0) <= 0.5 * half
        m_out = (np.abs(f - f0) > half + 60e3) & (np.abs(f - f0) <= half + 500e3)
        if m_in.any():
            ins.append(float(np.median(d[m_in])))
        if m_out.any():
            outs.append(float(np.median(d[m_out])))
    occ_o = float(po[np.abs(f) > PASS_HZ].sum() / po.sum())
    occ_n = float(pn[np.abs(f) > PASS_HZ].sum() / pn.sum())
    return (float(np.median(ins)) if ins else float("nan"),
            float(np.median(outs)) if outs else float("nan"), occ_o, occ_n)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def audit(diff):
    old_root = os.path.join(ROOT, "dataset_%s" % diff)
    new_root = os.path.join(NEW_ROOT, "dataset_%s" % diff)
    sids = sorted(d for d in os.listdir(new_root)
                  if os.path.isdir(os.path.join(new_root, d)))
    p_ratio = []
    occ_old, occ_new = [], []
    n_rows = 0
    d_in_rows, d_out_rows = [], []
    dpr, dpr_nonzero = [], []
    miss = []
    n_ifw_rows = 0
    for sid in sids:
        o_cap = os.path.join(old_root, sid, "capture")
        n_cap = os.path.join(new_root, sid, "capture")
        key = load_json(os.path.join(n_cap, "private", "d1_truth.json"))
        if not key.get("interferers"):
            continue
        n_ifw_rows += 1
        # --- 1) 总功率 ---
        io = np.load(os.path.join(o_cap, "public", "iq_station_a.npy"))
        inw = np.load(os.path.join(n_cap, "public", "iq_station_a.npy"))
        if io.shape != inw.shape:
            miss.append("%s: shape %s -> %s" % (sid, io.shape, inw.shape))
            continue
        po = float(np.mean(np.abs(io) ** 2))
        pn = float(np.mean(np.abs(inw) ** 2))
        p_ratio.append(10.0 * np.log10((pn + 1e-30) / (po + 1e-30)))
        # --- 2) 设计带宽内外的能量变化 ---
        srcs = [(float(g["freq_offset_hz"]), float(g["bandwidth_hz"]))
                for g in (key.get("interferers") or [])]
        d_in, d_out, occ_o, occ_n = edge_drop(io, inw, srcs)
        d_in_rows.append(d_in)
        d_out_rows.append(d_out)
        occ_old.append(occ_o)
        occ_new.append(occ_n)
        # --- 3) achieved 是否失效 ---
        tp = load_json(os.path.join(n_cap, "private", "truth.json"))
        kmap = {r["id"]: "s%d" % i for i, r in enumerate(key["receivers"])}
        fresh = {}
        for st in tp["stations"]:
            sk = kmap.get(st["receiver_id"])
            for src in st["sources"]:
                pw = float(np.mean(src["mean_port_power_w"]))
                fresh[(src["emitter_id"], sk)] = 10.0 * np.log10(pw + 1e-30) + 30.0
        ach = key.get("achieved") or {}
        for eid, per in ach.items():
            for sk, a in per.items():
                if (eid, sk) not in fresh:
                    continue
                d = fresh[(eid, sk)] - float(a["pr_dbm"])
                dpr.append(abs(d))
                if abs(d) > 0.5:
                    dpr_nonzero.append((sid, eid, sk, round(d, 2)))
                n_rows += 1
    print("=" * 78)
    print("### D%s  样本 %d（含干扰 %d）" % (diff[1:], len(sids), n_ifw_rows))
    if not p_ratio:
        print("    （无可用样本）")
        return
    pr = np.array(p_ratio)
    print("    总功率变化 new/old：中位 %+.2f dB  最小 %+.2f  最大 %+.2f  （|Δ|>1dB 的样本 %d/%d）"
          % (np.median(pr), pr.min(), pr.max(), int(np.sum(np.abs(pr) > 1.0)), pr.size))
    di = np.array(d_in_rows, dtype=float)
    do = np.array(d_out_rows, dtype=float)
    print("    带限效果（station_a，新−旧 dB）：设计带内中位 %+.2f dB（功率没变） "
          "紧邻设计带宽之外中位 %+.2f dB" % (np.nanmedian(di), np.nanmedian(do)))
    print("    带外能量跌 >3 dB 的源占样本 %d/%d   |f|>810kHz 能量占比：旧 %.2f%% → 新 %.2f%%"
          % (int(np.nansum(do < -3.0)), do.size,
             100 * float(np.median(occ_old)), 100 * float(np.median(occ_new))))
    if dpr:
        a = np.array(dpr)
        print("    achieved.pr_dbm 与平台新实测的偏差：中位 %.2f dB  p90 %.2f dB  最大 %.2f dB"
              % (np.median(a), np.percentile(a, 90), a.max()))
        print("    偏差 >0.5 dB 的 (源,站) 条数：%d/%d" % (len(dpr_nonzero), a.size))
        for row in dpr_nonzero[:5]:
            print("       例 %s %s %s  Δ=%+.2f dB" % row)
    if miss:
        print("    形状不一致：%s" % miss[:5])


def main():
    diffs = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["d2", "d3", "d4"])
    for d in diffs:
        audit(d.strip())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
