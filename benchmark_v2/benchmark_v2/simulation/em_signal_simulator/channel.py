"""Channel effects for multi-source, multi-antenna IQ signals (v2)."""
from __future__ import annotations
import numpy as np

C = 299792458.0  # 光速 m/s


def make_receiver(center_frequency_hz=2.4e9, sampling_rate_hz=20e6, antenna_spacing_wavelength=0.5):
    """构造接收机参数。返回 dict，字段与 metadata.json 中 receiver 一致。"""
    wavelength_m = C / float(center_frequency_hz)
    return {
        "center_frequency_hz": float(center_frequency_hz),
        "sampling_rate_hz": float(sampling_rate_hz),
        "wavelength_m": wavelength_m,
        "antenna_spacing_m": wavelength_m * float(antenna_spacing_wavelength),
        "num_antennas": None,  # 由调用方指定
    }


def steering_vector(
    doa_degree,
    num_antennas,
    antenna_spacing_m,
    wavelength_m,
    gain_error_std=0.0,
    phase_error_std_deg=0.0,
    rng=None,
    gain_error=None,
    phase_error_rad=None,
):
    """均匀线阵 (ULA) 转向矢量，支持可选的阵元增益/相位失配。

    理想响应：a(θ) = exp(1j * 2π * d * sin(θ) / λ * antenna_idx)
    失配后：a_m(θ) *= (1 + Δg_m) * exp(1j * Δφ_m)
        - Δg_m ~ N(0, gain_error_std^2)   （增益误差，默认 0）
        - Δφ_m ~ N(0, phase_error_std_deg^2)（相位误差，度，默认 0）
    两种失配注入方式：
        a) 传 gain_error_std / phase_error_std_deg：函数内部用 rng 现抽一组失配；
        b) 传预先算好的 gain_error / phase_error_rad 数组：物理上一个阵列只有一组
           失配，所有源应共享同一组（推荐由工厂层一次性抽取后传入）。
    返回形状 (num_antennas,)，dtype complex128。
    默认参数下退化为理想阵列，与旧版行为完全一致。
    """
    theta = np.deg2rad(float(doa_degree))
    idx = np.arange(int(num_antennas))
    phase = 2.0 * np.pi * float(antenna_spacing_m) * np.sin(theta) / float(wavelength_m)
    sv = np.exp(1j * phase * idx)

    if gain_error is not None or phase_error_rad is not None:
        # 显式失配向量（物理阵列共享同一组失配）
        dg = np.zeros(len(idx)) if gain_error is None else np.asarray(gain_error, dtype=float)
        dphi = np.zeros(len(idx)) if phase_error_rad is None else np.asarray(phase_error_rad, dtype=float)
        sv = sv * (1.0 + dg) * np.exp(1j * dphi)
    elif float(gain_error_std) > 0.0 or float(phase_error_std_deg) > 0.0:
        # 自动随机失配
        rng = np.random.default_rng() if rng is None else rng
        dg = rng.normal(0.0, float(gain_error_std), len(idx))
        dphi = np.deg2rad(rng.normal(0.0, float(phase_error_std_deg), len(idx)))
        sv = sv * (1.0 + dg) * np.exp(1j * dphi)
    return sv.astype(np.complex128)


def apply_freq_offset(signal, freq_offset_normalized):
    """对单天线信号施加归一化频偏 exp(1j*2π*f_offset*t)。"""
    x = np.asarray(signal, dtype=np.complex128)
    n = len(x)
    f = float(freq_offset_normalized)
    if f == 0.0:
        return x
    return x * np.exp(1j * 2 * np.pi * f * np.arange(n))


def expand_to_array(
    baseband,
    doa_degree,
    num_antennas,
    antenna_spacing_m,
    wavelength_m,
    gain_error_std=0.0,
    phase_error_std_deg=0.0,
    rng=None,
    gain_error=None,
    phase_error_rad=None,
):
    """将单天线基带信号扩展到 ULA 多天线。

    输入 (num_samples,)，输出 (num_antennas, num_samples)。
    阵列失配参数与 steering_vector 相同（默认关闭，向后兼容）。
    """
    sv = steering_vector(
        doa_degree,
        num_antennas,
        antenna_spacing_m,
        wavelength_m,
        gain_error_std=gain_error_std,
        phase_error_std_deg=phase_error_std_deg,
        rng=rng,
        gain_error=gain_error,
        phase_error_rad=phase_error_rad,
    )
    return np.outer(sv, np.asarray(baseband, dtype=np.complex128))


def apply_lna_saturation(signal, a_sat, p=2.0):
    """接收机 LNA 饱和模型（Rapp 软限幅 AM-AM 压缩）。

        y = x / (1 + (|x| / A_sat)^(2p))^(1/(2p))

    - |x| << A_sat 时 y ≈ x（线性区，无失真）；
    - |x| >> A_sat 时 y ≈ A_sat * x/|x|（饱和，幅度被压到 A_sat）。
    对混合阵列信号逐样本施加：强阻塞干扰使 LNA 进入压缩区，
    目标信号幅度随之被压缩/扭曲 —— 即真实的 "blocking" 效应。
    """
    x = np.asarray(signal, dtype=np.complex128)
    a_sat = float(a_sat)
    p = float(p)
    if a_sat <= 0.0:
        return x.copy()
    mag = np.abs(x)
    if not np.any(mag > 0.0):
        return x.copy()
    denom = (1.0 + (mag / a_sat) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    return (x / denom).astype(np.complex128)


def add_awgn(signal, snr_db, rng=None, ref_power=None):
    """按 SNR 注入复高斯白噪声。

    - ref_power=None：噪声功率基于信号自身总功率（v1 兼容语义，SINR 式）；
    - ref_power=目标功率：噪声功率 = ref_power / 10^(SNR/10)，
      即 SNR 定义为「目标功率 / 噪声功率」，干扰不参与 SNR 计算。
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(signal, dtype=np.complex128)
    power = float(np.mean(np.abs(x) ** 2)) if ref_power is None else float(ref_power)
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    noise = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) * np.sqrt(noise_power / 2.0)
    return x + noise


def apply_channel(signal, snr_db=10.0, cfo=0.0, rng=None):
    """兼容 v1 的单天线信道函数（多源信号叠加后可调用）。"""
    x = np.asarray(signal, dtype=np.complex128)
    x = apply_freq_offset(x, float(cfo)) if cfo else x
    y = add_awgn(x, snr_db, rng=rng)
    return y.astype(np.complex128)
