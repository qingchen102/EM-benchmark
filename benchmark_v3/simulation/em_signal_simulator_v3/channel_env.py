"""v3 低空环境信道扩展。

在 v2 信道（CFO/阵列导向/LNA 饱和/阵元失配/AWGN）之上新增三类真实环境效应，
全部为可组合函数：输入 (M, L) 阵列信号，输出同形状。设计参考低空电磁数据集
（S-ICDF 类）的场景构成：低空空地链路存在较强直射径（Rician）+ 地面/建筑散射径，
无人机机动引入多普勒；每根天线有独立下变频链（IQ 不平衡逐天线独立），
本振相位噪声为公共项。
"""
from __future__ import annotations

import numpy as np


def _steer(doa_deg: float, m: int) -> np.ndarray:
    """d=λ/2 均匀线阵导向矢量（与 v2 music_spectrum 同约定）。"""
    idx = np.arange(m)
    return np.exp(1j * 2.0 * np.pi * 0.5 * np.sin(np.deg2rad(doa_deg)) * idx)


def apply_multipath(x: np.ndarray, rng: np.random.Generator,
                    n_paths: int = 2, echo_power_db=(-18.0, -6.0),
                    delay_norm_max: float = 0.08, dopp_norm_max: float = 0.03,
                    doa_spread_deg: float = 30.0, los_doa_deg: float = 0.0) -> np.ndarray:
    """空地多径：x 的直射径（LOS，原样保留）+ n_paths 条散射回波。

    每条回波：随机 DOA（LOS ± doa_spread）、分数时延（FFT 相位斜坡实现）、
    多普勒（复指数）、逐径瑞利衰落系数、逐径衰减。
    x: (M, L) 阵列信号；返回同形状。
    """
    m, n = x.shape
    t = np.arange(n)
    y = x.copy()
    f = np.fft.fftfreq(n)
    for k in range(n_paths):
        pw = 10.0 ** (rng.uniform(*echo_power_db) / 10.0 - 0.6 * k)
        delay = rng.uniform(0.02, delay_norm_max) * n
        fd = rng.uniform(-dopp_norm_max, dopp_norm_max)
        doa = los_doa_deg + rng.uniform(-doa_spread_deg, doa_spread_deg)
        X = np.fft.fft(x[0])
        xd = np.fft.ifft(X * np.exp(-1j * 2.0 * np.pi * f * delay))
        fading = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
        dopp = np.exp(1j * 2.0 * np.pi * fd * t)
        steer = _steer(doa, m)
        y += np.sqrt(pw) * fading * steer[:, None] * xd[None, :] * dopp[None, :]
    return y


def apply_iq_imbalance(x: np.ndarray, rng: np.random.Generator,
                       gain_db_max: float = 1.0, phase_max_deg: float = 8.0) -> np.ndarray:
    """每接收链独立 IQ 不平衡：x' = K1·x + K2·conj(x)，产生镜像分量。"""
    m = x.shape[0]
    y = np.empty_like(x)
    for a in range(m):
        g = 10.0 ** (rng.uniform(0.0, gain_db_max) / 20.0)
        phi = np.deg2rad(rng.uniform(0.0, phase_max_deg))
        k1 = (1.0 + g * np.exp(1j * phi)) / 2.0
        k2 = (1.0 - g * np.exp(-1j * phi)) / 2.0
        y[a] = k1 * x[a] + k2 * np.conj(x[a])
    return y


def apply_phase_noise(x: np.ndarray, rng: np.random.Generator,
                      sigma_max_deg: float = 3.0) -> np.ndarray:
    """公共本振相位噪声：维纳过程，逐样本累积，全阵共通。"""
    sigma = np.deg2rad(rng.uniform(0.2, sigma_max_deg))
    phi = np.cumsum(sigma * rng.standard_normal(x.shape[1]))
    return x * np.exp(1j * phi)[None, :]
