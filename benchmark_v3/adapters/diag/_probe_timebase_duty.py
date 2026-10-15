#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长窗题：源侧占空比随时长如何变化（零 API、零 RT）。

问题：D2 的瓶颈是 Zigbee / WiFi_burst 在 10 ms 窗里占空比只有 1.7% / 0.9%，
      冻结分类器靠 RMS 归一化看不出谁是谁。那么把窗拉长到 1 s，
      这些源在窗里"更容易被看到"还是"更稀"？

方法：与 `rad_cls/_probe_wave_stats.py:57` 完全同一个 duty 定义
      （`mean(|x| >= 0.2*max|x|)`，中位数聚合），对同一批辐射体在
      length = 40000(10 ms) / 400000(100 ms) / 4000000(1 s) 上各生成 n 次比较。
      同一 seed 保证是同一段波形的不同截断长度。

判据：生成器若不随窗长增加事件数，duty 应约按 1/时长 塌下去（10 ms→1 s 是 1/100）；
      只有真实时基（事件率固定）才能让 duty 与窗长无关。

用法：python -X utf8 diag/_probe_timebase_duty.py [n_per_class] [--out PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADP = HERE.parents[1]
if str(ADP) not in sys.path:
    sys.path.insert(0, str(ADP))
_EMIT_DIR = ADP / "simulation" / "em_signal_simulator_v3"
if str(_EMIT_DIR) not in sys.path:
    sys.path.insert(0, str(_EMIT_DIR))

import emitters as em  # noqa: E402

CLASSES = [
    "Zigbee", "WiFi_burst", "DECT", "Microwave",
    "Welder", "Corona", "VFD_noise", "Bluetooth_FHSS",
]
LENGTHS = [("10 ms", 40000), ("100 ms", 400000), ("1 s", 4000000)]


def duty_of(x: np.ndarray) -> float:
    env = np.abs(x)
    mx = float(env.max())
    if mx <= 0:
        return 0.0
    return float(np.mean(env >= 0.2 * mx))


def bursts_of(x: np.ndarray) -> int:
    """活跃段个数（envelope >= 0.2*max 的连通段）。"""
    act = np.abs(x) >= 0.2 * float(np.abs(x).max())
    if not act.any():
        return 0
    return int(np.count_nonzero(np.diff(act.astype(np.int8)) == 1) + (1 if act[0] else 0))


def main(n_per_class: int = 3) -> None:
    lines: list[str] = []
    lines.append("源侧占空比 vs 窗长（duty = mean(|x| >= 0.2*max|x|)，中位数聚合；同 seed 截断）")
    lines.append("")
    header = f"{'辐射体':16s}" + "".join(f"{lab:>12s}{'bursts':>9s}" for lab, _ in LENGTHS)
    lines.append(header)
    lines.append("-" * len(header))
    for name in CLASSES:
        cells = []
        for _, length in LENGTHS:
            ds, bs = [], []
            for rep in range(n_per_class):
                x = None
                for _try in range(4):
                    # gen_zigbee 的 QPSK 脉冲成形在符号数太少时会报 "pulse shaping output too short"
                    # （已知缺陷，见 _an_emit_bw.py）→ 换 seed 重试；仍失败则记 NaN 不阻断整表
                    rng = np.random.default_rng(7 + rep + 100 * _try)
                    try:
                        x = np.asarray(em.generate_emitter(name, rng, length=length)["iq"], dtype=complex)
                        break
                    except ValueError:
                        continue
                if x is None:
                    FAILED.append((name, length))
                    continue
                ds.append(duty_of(x))
                bs.append(bursts_of(x))
            cells.append((float(np.median(ds)), int(np.median(bs))) if ds else (float("nan"), 0))
        row = f"{name:16s}" + "".join(("        n/a" if d != d else f"{d*100:11.3f}%") + f"{b:9d}" for d, b in cells)
        lines.append(row)
        print(row, flush=True)
    lines.append("")
    lines.append("读法：duty 若按 ~1/窗长 塌下去 ⇒ 生成器是「每窗固定事件数」，长窗没有增加可观测时间，")
    lines.append("      只是把同一个源稀释了 100 倍；bursts 不随时长增长同理。")
    text = "\n".join(lines) + "\n"
    out = ADP / "evidence" / "_timebase_duty.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n[写入] {out}")


FAILED: list[tuple[str, int]] = []


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(argv[0]) if argv else 3)
    if FAILED:
        uniq = sorted({f"{n}@{L}" for n, L in FAILED})
        print(f"[生成失败（脉冲成形缺陷）] {', '.join(uniq)}")
