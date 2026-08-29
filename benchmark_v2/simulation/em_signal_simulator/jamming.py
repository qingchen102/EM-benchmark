"""Interference waveform generation for jammer sources (v2).

波形类型（JAMMING_TYPES）：
- single_tone: 单音连续波 (CW)
- swept:      线性调频 (LFM)，支持脉冲化（duty_cycle / pri / pulse_width）
- pulse:      随机门控噪声脉冲
- broadband:  宽带噪声（AWGN 弹幕）
- nfm:        噪声调频（带限高斯噪声积分进载波相位，主动宽带弹幕压制）
"""
from __future__ import annotations
import numpy as np
from scipy.signal import firwin, lfilter

JAMMING_TYPES = {"none", "single_tone", "swept", "pulse", "broadband", "nfm"}

# 干扰波形可选项（single_tone 已含在 JAMMING_TYPES，此处列举可作干扰源的波形）
INTERFERER_MODULATIONS = {"single_tone", "swept", "pulse", "broadband", "nfm"}


def _pulse_train_mask(n, duty_cycle=None, pri=None, rng=None):
    """周期脉冲门控掩码（PRI 固定的相干脉冲串）。

    - duty_cycle: 占空比 (0,1]，脉冲宽度 = duty_cycle * PRI，默认 0.1
    - pri: 归一化脉冲重复间隔 (0,1]，相对整段观测时长的比例，默认 1.0（单脉冲）
    - rng: 可选，用于给脉冲串引入随机起始相位
    """
    n = int(n)
    duty = float(duty_cycle) if duty_cycle is not None else 0.1
    duty = float(np.clip(duty, 1e-3, 1.0))
    pri = float(pri) if pri is not None else 1.0
    pri = float(np.clip(pri, 1e-6, 1.0))
    pri_s = max(1, int(round(pri * n)))
    pw_s = max(1, int(round(pri_s * duty)))
    t = np.arange(n)
    mask = (t % pri_s) < pw_s
    if rng is not None and pri_s > 1:
        # 随机起始相位（脉冲串与观测窗口的时延对齐）
        shift = int(rng.integers(0, pri_s))
        mask = np.roll(mask, shift)
    return mask


def _nfm_waveform(n, rng, center_frequency=0.0, freq_deviation=0.2, noise_bandwidth=0.1):
    """噪声调频 (NFM) 波形：带限高斯噪声积分进载波相位。

        phase(t) = 2π f_c t + 2π K_f ∫ n_B(t) dt

    - noise_bandwidth: 噪声带宽（归一化单边，0~0.45），用 firwin 低通实现严格带限
      （旧实现为矩形移动平均，sinc 频谱滚降慢，实际带宽比标注宽）；
    - freq_deviation:  归一化频偏（cycle/sample），即 K_f·RMS(n_B)。
    """
    n = int(n)
    rng = np.random.default_rng() if rng is None else rng
    fc = float(center_frequency)
    fdev = float(freq_deviation)
    bw = float(np.clip(noise_bandwidth, 1e-4, 0.45))
    num_taps = 65
    # firwin 截止频率以 Nyquist(0.5·fs) 为单位 → 单边噪声带宽 bw 对应 cutoff = 2·bw
    taps = firwin(num_taps, min(0.9, 2.0 * bw), window='hamming')
    # 扩展 + 中段切片，避免滤波起始暂态
    ext_len = n + 2 * num_taps
    noise = rng.normal(size=ext_len)
    noise_f = lfilter(taps, 1.0, noise)
    noise_f = noise_f[num_taps:num_taps + n]
    std = float(np.std(noise_f))
    if std > 0.0:
        noise_f = noise_f / std
    t = np.arange(n)
    phase = 2.0 * np.pi * fc * t + 2.0 * np.pi * fdev * np.cumsum(noise_f)
    return np.exp(1j * phase)


def _make_waveform(typ, n, rng, kw):
    """生成 (n,) 的干扰波形（归一化幅度 ~1，未含频偏/阵列响应）。"""
    t = np.arange(n)
    if typ == "single_tone":
        f = float(kw.get("frequency", kw.get("center_frequency", 0.0)))
        return np.exp(1j * 2 * np.pi * f * t)
    if typ == "swept":
        # 连续 LFM
        f0 = float(kw.get("f_start", -0.4))
        f1 = float(kw.get("f_stop", 0.4))
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * max(n, 1)))
        j = np.exp(1j * phase)
        # 脉冲化 LFM（ECM 脉冲串）：任一脉冲参数给出即启用
        duty = kw.get("duty_cycle")
        pri = kw.get("pri")
        pw = kw.get("pulse_width")
        if duty is not None or pri is not None or pw is not None:
            if pw is not None and duty is not None and pri is None:
                pri = float(pw) / max(float(duty), 1e-6)
            j = j * _pulse_train_mask(n, duty_cycle=duty, pri=pri, rng=rng)
        return j
    if typ == "pulse":
        duty = float(kw.get("duty_cycle", 0.1))
        pri = kw.get("pri")
        if pri is not None:
            # 周期脉冲串（PRI 固定，随机起始相位；噪声幅度）
            mask = _pulse_train_mask(n, duty_cycle=duty, pri=pri, rng=rng)
        else:
            # 旧行为：伯努利随机门控（无 PRI/脉宽控制）
            mask = rng.random(n) < duty
        return mask * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    if typ == "nfm":
        return _nfm_waveform(
            n,
            rng,
            kw.get("center_frequency", kw.get("frequency", 0.0)),
            kw.get("freq_deviation", kw.get("modulation_index", 0.2)),
            kw.get("noise_bandwidth", 0.1),
        )
    # broadband：宽带噪声（AWGN 弹幕）
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)


def inject_jamming(signal, jamming_type="none", jsr_db=3.0, rng=None, **kw):
    """把干扰按 JSR 功率比叠加到目标信号上。"""
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(signal, dtype=np.complex128)
    n = len(x)
    typ = str(jamming_type).lower()
    if typ not in JAMMING_TYPES:
        raise ValueError(f"Unsupported jamming type: {jamming_type}")
    if typ == "none":
        return x.copy()
    j = _make_waveform(typ, n, rng, kw)
    target = np.mean(np.abs(x) ** 2) * 10 ** (float(jsr_db) / 10)
    jp = np.mean(np.abs(j) ** 2)
    return x + j * np.sqrt(target / jp) if jp > 0 else x.copy()


def generate_interferer_waveform(jamming_type="single_tone", num_samples=1024, rng=None, **kw):
    """生成纯（归一化）干扰波形，不含目标信号。

    返回 (num_samples,) 的复数波形，调用方可再施加频偏 / 阵列响应。
    支持参数（**kw）：
        single_tone: frequency / center_frequency
        swept:       f_start / f_stop；脉冲化：duty_cycle、pri、pulse_width（任给其一即脉冲化）
        pulse:       duty_cycle
        nfm:         noise_bandwidth、freq_deviation（或 modulation_index）、center_frequency
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(num_samples)
    typ = str(jamming_type).lower()
    if typ not in JAMMING_TYPES:
        raise ValueError(f"Unsupported jamming type: {jamming_type}")
    j = _make_waveform(typ, n, rng, kw)
    return j.astype(np.complex128)
