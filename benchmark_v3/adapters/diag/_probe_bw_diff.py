# -*- coding: utf-8 -*-
"""带宽重生成探针（只读）：`iq_new − iq_old` 的能量落在哪。

判据：软带限削掉的是「设计带宽之外」的能量，所以差值谱应集中在各干扰源设计带宽之外；
若差值集中**在**设计带内，说明变化来自别的机制（随机实现/相位/功率重标），需换解释。

用法：python -u _probe_bw_diff.py [d2,d3,d4] [每档样本数]
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NEW_ROOT = os.path.join(ROOT, "_regen_bw")
FS = 4e6


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    diffs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["d2", "d3", "d4"]
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for diff in diffs:
        old_root = os.path.join(ROOT, "dataset_%s" % diff)
        new_root = os.path.join(NEW_ROOT, "dataset_%s" % diff)
        sids = sorted(d for d in os.listdir(new_root)
                      if os.path.isdir(os.path.join(new_root, d)))[:ns]
        print("=" * 78)
        print("### D%s  前 %d 个样本" % (diff[1:], len(sids)))
        for sid in sids:
            key = load(os.path.join(new_root, sid, "capture", "private", "d1_truth.json"))
            io = np.load(os.path.join(old_root, sid, "capture", "public", "iq_station_a.npy"))
            inw = np.load(os.path.join(new_root, sid, "capture", "public", "iq_station_a.npy"))
            d = inw - io
            n = d.shape[-1]
            f = np.fft.fftfreq(n, 1.0 / FS)
            pd = (np.abs(np.fft.fft(d, axis=-1)) ** 2).sum(axis=0)
            po = (np.abs(np.fft.fft(io, axis=-1)) ** 2).sum(axis=0)
            rel = 10 * np.log10((np.mean(np.abs(d) ** 2) + 1e-30)
                                / (np.mean(np.abs(io) ** 2) + 1e-30))
            ifws = [(g["kind"], float(g["freq_offset_hz"]), float(g["bandwidth_hz"]))
                    for g in (key.get("interferers") or [])]
            in_any = np.zeros(n, bool)
            for _k, f0, bw in ifws:
                in_any |= np.abs(f - f0) <= bw / 2.0
            fr = float(pd[in_any].sum() / (pd.sum() + 1e-300))
            fro = float(po[in_any].sum() / (po.sum() + 1e-300))
            desc = "  ".join("%s@%+.0fk/%.0fk" % (k, f0 / 1e3, bw / 1e3) for k, f0, bw in ifws)
            print("  %s  差值/原信号 %+6.1f dB   差值能量落在设计带内 %.0f%%（原信号 %.0f%%）"
                  % (sid, rel, 100 * fr, 100 * fro))
            print("      %s" % desc)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
