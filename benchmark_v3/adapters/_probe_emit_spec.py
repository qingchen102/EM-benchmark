"""只读探针：逐个 v3 辐射体直接生成波形，看能量实际落在哪里。

回答两个问题：
  1) 生成器内部的"载频放置"是否随机、幅度多大（→ 真值 freq_offset_hz 是否可观测）；
  2) 哪些 kind 的实测占用带宽 ≫ 标称带宽（生成器忽略 bw 参数）。
"""
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B / "simulation"))

from em_signal_simulator_v3.emitters import EMITTERS, generate_emitter  # noqa: E402

FS = 4.0e6
N = 40000
BW = 0.10                      # 归一化带宽（= 400 kHz 标称），模拟数据集里的典型值


def band99(iq, fs=FS, frac=0.99):
    x = iq - iq.mean()
    x = x * np.hanning(len(x))
    p = np.abs(np.fft.fft(x)) ** 2
    f = np.fft.fftfreq(len(x), 1.0 / fs)
    o = np.argsort(f)
    f, p = f[o], p[o]
    c = np.cumsum(p) / (p.sum() + 1e-300)
    lo = float(f[int(np.searchsorted(c, (1 - frac) / 2))])
    hi = float(f[int(np.searchsorted(c, 1 - (1 - frac) / 2))])
    return lo, hi, float(f[int(np.argmax(p))])


print(f"每种辐射体 5 个随机种子：峰频 / 99% 带宽（标称 BW={BW*FS/1e3:.0f} kHz, N={N}）")
print(f"{'kind':<18}{'峰频(kHz)':<34}{'99%带宽中位(kHz)':<20}峰频跨度(kHz)")
for name in sorted(EMITTERS):
    peaks, wids = [], []
    for s in range(5):
        rng = np.random.default_rng(1000 + s)
        try:
            out = generate_emitter(name, rng, length=N, bw=BW)
        except TypeError:
            try:
                out = generate_emitter(name, rng, length=N)
            except Exception as exc:
                print(f"{name:<18}[生成失败] {exc}")
                break
        except Exception as exc:
            print(f"{name:<18}[生成失败] {exc}")
            break
        iq = np.asarray(out["iq"] if isinstance(out, dict) else out, dtype=complex)
        lo, hi, pk = band99(iq)
        peaks.append(pk / 1e3)
        wids.append((hi - lo) / 1e3)
    if peaks:
        print(f"{name:<18}{str([round(p) for p in peaks]):<34}"
              f"{np.median(wids):<20.0f}{max(peaks)-min(peaks):.0f}")
