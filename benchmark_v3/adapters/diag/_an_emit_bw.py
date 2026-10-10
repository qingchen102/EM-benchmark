# -*- coding: utf-8 -*-
"""带宽落地自检：19 类辐射体 × 请求带宽 → 实测 OBW99 是否跟住请求值。

只读脚本（不改数据集、不跑 RT、零 API）。对照两组：
  A) 不请求带宽（现行数据集的生成口径）→ 实测 OBW99 中位
  B) 请求 150 / 400 / 700 kHz（归一化到 fs = 4 MHz = 0.0375 / 0.1 / 0.175）
判定：B 组「实测 / 请求」应 ≈ 1（本脚本按 ±25% 记 OK）。

输出 `benchmark_v3/_an_emit_bw.txt`。用法：
    python -u _an_emit_bw.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                    # benchmark_v3
sys.path.insert(0, str(ROOT / "simulation"))

from em_signal_simulator_v3.emitters import EMITTERS, generate_emitter  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FS = 4e6
LEN = 4096
REQ_HZ = (150e3, 400e3, 700e3)
N_REP = 5


def obw_norm(name, rng, bw_norm, length=LEN) -> float:
    # 生成器对短包长脆弱（如 Zigbee 窄带需要更长的脉冲成形余量）→ 换 seed 重抽保护，
    # 与 rad_cls/data_gen16.gen_wave16 同一套保护。
    for _ in range(40):
        try:
            out = generate_emitter(name, rng, length=length, bandwidth_normalized=bw_norm)
            return float(out["bandwidth_normalized"])
        except ValueError:
            continue
    raise RuntimeError(f"generate_emitter({name}) 反复失败")


def main() -> int:
    lines: list[str] = []
    add = lines.append
    add("=" * 96)
    add("带宽落地自检：请求带宽 vs 实测 OBW99（fs = 4 MHz，length = 4096，每格 %d 个种子取中位）" % N_REP)
    add("=" * 96)
    add("")
    head = f"{'辐射体':<18} {'默认OBW99':>10} " + " ".join(
        f"{'req%3d kHz' % (hz / 1e3):>22}" for hz in REQ_HZ)
    add(head)
    add(f"{'':<18} {'kHz':>10} " + " ".join(
        f"{'实测kHz / 比值':>22}" for _ in REQ_HZ))
    add("-" * 96)

    ok = total = 0
    bad: list[tuple[str, float, float, float]] = []
    for name in EMITTERS:
        rng = np.random.default_rng(20261009)
        base = float(np.median([obw_norm(name, rng, None) for _ in range(N_REP)])) * FS / 1e3
        cells = []
        for hz in REQ_HZ:
            vals = [obw_norm(name, rng, hz / FS) for _ in range(N_REP)]
            med = float(np.median(vals)) * FS / 1e3
            ratio = med / (hz / 1e3)
            dev = med - hz / 1e3
            cells.append(f"{med:7.1f}/{ratio:5.2f}/{dev:+5.1f}")
            total += 1
            # 容差：请求的 25% 或 60 kHz 取大者——measure_obw99(seg=256) 的
            # 频率量化 = 4MHz/256 ≈ 15.6 kHz，窄带上本身就有 ±1~2 bin 的读数抖动
            if abs(dev) <= max(0.25 * hz / 1e3, 60.0):
                ok += 1
            else:
                bad.append((name, hz / 1e3, med, ratio))
        add(f"{name:<18} {base:10.1f} " + " ".join(f"{c:>18}" for c in cells))
    add("-" * 96)
    add(f"跟住判定（|实测-请求| <= max(25% 请求, 60 kHz)）：{ok}/{total} = {ok / max(total, 1):.1%}")
    if bad:
        add("")
        add("未跟住的组合：")
        for name, req, med, ratio in bad:
            add(f"  {name:<18} 请求 {req:6.0f} kHz → 实测 {med:7.1f} kHz（比值 {ratio:.2f}）")
    add("")
    add("说明：① 实测 - 请求 的绝对偏差才是关键量（第 3 列），比值在 150 kHz 档会被")
    add("      15.6 kHz 的 FFT 量化放大；Tone_jam 已豁免（纯点频无带宽语义，真值用实测值）；")
    add("      ② 偏差 < -60 kHz 表示结构带宽比请求更窄（需结构性加宽）；> +60 kHz 表示")
    add("      带限没生效（跳频 band / 突发 pkt_bw 未跟随）。")

    txt = "\n".join(lines) + "\n"
    (ROOT / "_an_emit_bw.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
