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

    **功率归一化（修复 P1-4）**：每径回波功率按 echo_power_db 衰减，且为保持
    总能量守恒（总功率 = 直射径 + 各回波），回波由直射径能量"让渡"，不额外
    抬高接收总功率——避免把混合信号能量无谓抬高 10dB 造成 SNR/INR 语义错乱。

    x: (M, L) 阵列信号；返回同形状。
    """
    m, n = x.shape
    t = np.arange(n)
    f = np.fft.fftfreq(n)
    # 直射径能量占比：1 径→0.8、2 径→0.65、3 径→0.55（回波合计守恒）
    los_frac = {1: 0.80, 2: 0.65, 3: 0.55}.get(n_paths, 0.55)
    echo_total = 1.0 - los_frac
    y = np.sqrt(los_frac) * x.copy()
    # 各回波功率归一化：让 (Σ 回波功率) = echo_total，避免能量无谓抬高
    p_ref = float(np.mean(np.abs(x) ** 2)) + 1e-30
    for k in range(n_paths):
        pw = 10 ** (rng.uniform(*echo_power_db) / 10.0)          # 相对衰减
        # 归一化使总回波功率 = echo_total，逐径按相对衰减分配
        delay = rng.uniform(0.02, delay_norm_max) * n
        fd = rng.uniform(-dopp_norm_max, dopp_norm_max)
        doa = los_doa_deg + rng.uniform(-doa_spread_deg, doa_spread_deg)
        X = np.fft.fft(x[0])
        xd = np.fft.ifft(X * np.exp(-1j * 2.0 * np.pi * f * delay))
        fading = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
        dopp = np.exp(1j * 2.0 * np.pi * fd * t)
        steer = _steer(doa, m)
        echo = np.sqrt(pw) * fading * steer[:, None] * xd[None, :] * dopp[None, :]
        # 逐径功率标定到 echo_total/n_paths 相对直射径
        pe = float(np.mean(np.abs(echo) ** 2)) + 1e-30
        echo = echo * np.sqrt((echo_total / n_paths) * p_ref / pe)
        y = y + echo
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


def apply_imd3(x: np.ndarray, rng: np.random.Generator,
               ip3_db: float = 12.0, return_imd: bool = False):
    """三阶互调（IMD3）：强源经接收链非线性产生 2f1±f2 / 2f2±f1 带外杂散。

    当有 ≥2 个强干扰（尤其强阻塞）时，其非线性互调产物可落在目标带内，
    是可观测杂散的重要来源。ip3_db 越小非线性越强、互调越显著。
    x: (M, L)；返回含 IMD 杂散后的信号；return_imd=True 时额外返回 IMD 分量。
    """
    m, n = x.shape
    # 每根天线同一非线性系数（公共链路），对整阵逐样本压缩
    imd = np.empty_like(x)
    a3 = 10.0 ** (-ip3_db / 20.0)          # 三阶系数（相对线性项）
    for a in range(m):
        imd[a] = a3 * x[a] * np.abs(x[a]) ** 2
    y = x + imd
    if return_imd:
        return y, imd
    return y
