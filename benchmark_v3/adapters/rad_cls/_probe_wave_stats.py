"""19 类波形的"时域占空 / 频谱占据 / 实测 99% 占带"三张表。

为什么：D2 的源全是 ISM 共存辐射体（窄带、突发、低占空），D3/D4 的源以
压制型干扰（连续、宽带）为主。分类器看的是"窗口里这段波形长什么样"，
所以这两个量（占空比 / 频谱占据率）比"标称带宽"更能解释
"为什么哪些类过了解析式切片链就必然错"。

用法：python -X utf8 _probe_wave_stats.py [n_per_class] [length]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data_gen16 import CLASSES16, gen_wave16  # noqa: E402

FS = 4e6


def obw99(x: np.ndarray) -> float:
    """99% 占带（Hz）：功率谱累积到 99% 的宽度。"""
    X = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(x.size, 1 / FS))
    c = np.cumsum(X)
    if c[-1] <= 0:
        return 0.0
    c = c / c[-1]
    i0 = int(np.searchsorted(c, 0.005))
    i1 = int(np.searchsorted(c, 0.995))
    return float(f[min(i1, f.size - 1)] - f[max(i0, 0)])


def main(n_per_class: int = 20, length: int = 40000) -> None:
    print(f"\n== 波形统计（每类 {n_per_class} 条，length={length} = {length / FS * 1e3:.0f} ms）==")
    print(f"{'体制':16s} {'占空比(≥20%峰值)':>16s} {'频谱占据(≥5%峰)':>16s} {'99%占带(kHz)':>13s}")
    rows = []
    for name in CLASSES16:
        duties, occs, bws = [], [], []
        for k in range(n_per_class):
            rng = np.random.default_rng(900000 + k * 131)
            try:
                x = np.asarray(gen_wave16(name, rng, length=length), dtype=np.complex128)
            except TypeError:
                x = np.asarray(gen_wave16(name, rng, length), dtype=np.complex128)
            if x.ndim != 1:
                x = x.ravel()
            env = np.abs(x)
            mx = float(env.max())
            if mx <= 0:
                continue
            duties.append(float(np.mean(env >= 0.2 * mx)))
            Xs = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
            occs.append(float(np.mean(Xs >= 0.05 * Xs.max())))
            bws.append(obw99(x))
        if not duties:
            continue
        rows.append((name, float(np.median(duties)), float(np.median(occs)), float(np.median(bws)) / 1e3))
    for name, d, o, b in sorted(rows, key=lambda r: r[1]):
        print(f"{name:16s} {d:>16.1%} {o:>16.1%} {b:>13.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 20, int(a[1]) if len(a) > 1 else 40000)
