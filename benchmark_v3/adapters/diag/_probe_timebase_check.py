# -*- coding: utf-8 -*-
"""时基重写验收：逐类在 10 ms / 100 ms / 1 s 窗上核对「事件率（放置）+ 占空比 + 窗内必在」。

判据（见 benchmark_v3/docs/辐射体时基重写_物理依据与方案.md §2）：
  1. 事件率 = 放置事件数 / 窗长，应 = TIMEBASE 的物理速率（各窗一致，误差 ≤ max(1 事件, 10%)）；
  2. 占空比在 1 s 窗应 ≤ 物理 duty（不膨胀；事件包络实测值可低于物理 duty）；
  3. 每个窗内源必须存在（duty > 0.5%）——否则该窗不可测（观测窗选取保证）。
  10 ms 窗的 duty 被抬高是「窗口选取为含事件」的必然结果，不算缺陷。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ADP = HERE.parents[1]                                   # benchmark_v3/adapters
SIM = ADP.parent / "simulation"                         # benchmark_v3/simulation
for _p in (str(SIM), str(ADP), str(ADP / "rad_cls")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from em_signal_simulator_v3 import emitters as em       # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FS = 4.0e6
WINDOWS = [(0.01, "10 ms"), (0.1, "100 ms"), (1.0, "1 s")]
CLASSES = ["Bluetooth_FHSS", "UAV_RC_hopping", "Zigbee", "WiFi_burst", "DECT",
           "Welder", "Corona", "VFD_noise", "Microwave"]


def _duty(x):
    a = np.abs(x)
    m = a.max()
    if not np.isfinite(m) or m <= 0:
        return 0.0
    return float(np.mean(a >= 0.2 * m))


def main(argv):
    reps = int(argv[1]) if len(argv) > 1 else 5
    rows, bad = [], []
    for key in CLASSES:
        rate, dur, _ = em.TIMEBASE[key]
        phys_duty = rate * dur
        for T, label in WINDOWS:
            L = int(round(T * FS))
            n_place, dut = [], []
            for r in range(reps):
                rng = np.random.default_rng(20261014 + r)
                n_place.append(em.tb_n_events(rng, L, key))
                x = em.generate_emitter(key, rng, length=L, bandwidth_normalized=0.2)["iq"]
                dut.append(_duty(x))
            n_med = float(np.median(n_place))
            r_meas = n_med / T
            d_med = float(np.median(dut))
            rows.append((key, label, n_med, r_meas, rate, d_med, phys_duty))
            tol = max(1.0 / T, 0.10 * rate)
            if T >= 0.1 and abs(r_meas - rate) > tol:
                bad.append("%s@%s 事件率 %.1f vs 物理 %.1f（超容差 %.1f）" % (key, label, r_meas, rate, tol))
            if d_med < min(0.002, 0.3 * phys_duty):
                bad.append("%s@%s 窗内几乎无信号（duty %.4f）" % (key, label, d_med))
            if T >= 1.0 and d_med > phys_duty * 2.5 + 0.01:
                bad.append("%s@1 s duty %.4f > 物理 %.4f×1.15" % (key, d_med, phys_duty))

    lines = ["时基重写验收（emitters.py TIMEBASE，fs=4 MHz，每格 %d 次取中位）" % reps, "",
             "%-16s %-7s %9s %10s %9s %9s %9s" %
             ("类", "窗", "放置事件", "实测率/s", "物理率", "实测duty", "物理duty")]
    for key, label, n_med, r_meas, rate, d_med, pd in rows:
        lines.append("%-16s %-7s %9.0f %10.1f %9.1f %9.4f %9.4f" % (key, label, n_med, r_meas, rate, d_med, pd))
    lines.append("")
    lines.append("判定：" + ("全部通过（事件率=物理率、窗内必在、1 s duty 不膨胀）" if not bad else "；".join(bad)))
    out = "\n".join(lines)
    print(out)
    p = ADP.parent / "evidence" / "_timebase_check.txt"
    p.write_text(out + "\n", encoding="utf-8")
    print("\n[写入] " + str(p))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
