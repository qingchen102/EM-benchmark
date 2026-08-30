"""训练样本生成：类名 → 复数基带波形（含增强）。

纪律：种子段与考卷/上限脚本完全隔离（见 README）。考卷干扰带宽 0.05~0.60、
SNR -5~15dB；本模块训练采样带宽 U(0.02,0.90)、SNR U(-10,20)，频偏与时移逐样本随机。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SIM_DIR = _ROOT / "simulation"
for _p in (_ROOT, _SIM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from em_signal_simulator.baseband import generate_baseband, MODULATIONS   # noqa: E402
from em_signal_simulator.channel import apply_freq_offset           # noqa: E402
from em_signal_simulator.jamming import generate_interferer_waveform  # noqa: E402
import tools_v2                                                     # noqa: E402

CLASSES = ["BPSK", "QPSK", "GFSK", "LFM", "OFDM",
           "nfm", "single_tone", "swept", "pulse", "broadband"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
MOD_CLASS_SET = set(MODULATIONS)                        # baseband 可生成的全部调制
MOD_CLASSES = {"BPSK", "QPSK", "GFSK", "LFM", "OFDM"}   # 干扰池中的调制类（答题空间）
LENGTH = 1024

# 训练分布（与考卷隔离的证据，勿改动）
SEED_BASE = 500_000          # 考卷与上限脚本用 1000~ 数千段
BW_RANGE = (0.02, 0.90)      # 考卷干扰带宽 0.05~0.60
SNR_RANGE = (-10.0, 20.0)    # 考卷 SNR -5~15
FOFF_RANGE = (-0.45, 0.45)


def make_wave(name: str, rng: np.random.Generator, bw: float | None = None) -> np.ndarray:
    """按类名生成一段干净（无噪声）复数基带波形。

    baseband 支持的名字（含 16QAM/64QAM/OOK/FHSS 等目标调制）走 generate_baseband；
    其余（nfm/single_tone/swept/pulse/broadband）走干扰波形生成器。
    """
    if bw is None:
        bw = float(rng.uniform(*BW_RANGE))
    if name in MOD_CLASS_SET:
        return generate_baseband(name, LENGTH, rng=rng, bandwidth_normalized=bw)
    kw = tools_v2._waveform_kw_for_bandwidth(name, bw)
    return generate_interferer_waveform(name, LENGTH, rng=rng, **kw)


def augment(x: np.ndarray, rng: np.random.Generator,
            snr_db: float | None = None, f_off: float | None = None,
            apply_shift: bool = True) -> np.ndarray:
    """频偏 + 加噪（SNR 相对信号功率）+ 循环时移。"""
    if f_off is None:
        f_off = float(rng.uniform(*FOFF_RANGE))
    x = apply_freq_offset(x, f_off)
    p = float(np.mean(np.abs(x) ** 2)) + 1e-30
    snr = float(rng.uniform(*SNR_RANGE)) if snr_db is None else snr_db
    amp = np.sqrt(p * 10.0 ** (-snr / 10.0) / 2.0)
    noise = amp * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    x = x + noise
    if apply_shift:
        x = np.roll(x, int(rng.integers(0, len(x))))
    return x


def to_input(x: np.ndarray) -> np.ndarray:
    """复数波形 → 网络输入 (2, L) float32，逐样本 RMS 归一化。"""
    x = np.asarray(x, dtype=np.complex64)
    rms = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-6
    x = x / rms
    return np.stack([x.real, x.imag]).astype(np.float32)


def slice_band(x: np.ndarray, f_center: float, bw: float, pad: float = 0.02):
    """Hann 掩膜频带切片 + 解旋到基带（与 eval 上限协议/tools_v2._band_slice 同口径）。

    失败（频带过窄/零功率）返回 None。
    """
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    half = bw / 2.0 + pad
    i_lo = max(int(np.searchsorted(freq, f_center - half)), 0)
    i_hi = min(int(np.searchsorted(freq, f_center + half)), n - 1)
    res = tools_v2._band_slice(x, freq, psd, i_lo, i_hi)
    if res is None or res[0] is None:
        return None
    return res[0]
