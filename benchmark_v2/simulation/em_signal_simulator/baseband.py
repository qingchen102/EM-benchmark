"""Baseband waveform generation for the target signal (v2)."""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter

MODULATIONS = {"BPSK", "QPSK", "16QAM", "64QAM", "GFSK", "OOK", "OFDM", "FHSS", "LFM"}

# 默认占用带宽（未显式配置带宽时的实际值）：
#   QAM 族 / OOK: (1+rolloff)/sps = 1.35/8
#   GFSK: (0.34375 + 1.09375*BT)/sps，BT=0.35 时 = 0.72656/8（-35dB 占用标定，见 _gfsk_bw_coef）
#   OFDM: 激活子载波 102 / nfft 128
#   FHSS: 跳频范围 ±0.45
#   LFM: 扫频范围 ±0.4
BANDWIDTH_TABLE = {
    "OOK": 0.16875,
    "GFSK": 0.0908203125,
    "BPSK": 0.16875,
    "QPSK": 0.16875,
    "16QAM": 0.16875,
    "64QAM": 0.16875,
    "OFDM": 0.796875,
    "FHSS": 0.9,
    "LFM": 0.8,
}

# 调制参数配置
MODULATION_CONFIG = {
    "BPSK": {"sps": 8, "rolloff": 0.35},
    "QPSK": {"sps": 8, "rolloff": 0.35},
    "16QAM": {"sps": 8, "rolloff": 0.35},
    "64QAM": {"sps": 8, "rolloff": 0.35},
    "OOK": {"sps": 8, "rolloff": 0.35},
    "GFSK": {"sps": 8, "rolloff": 0.5},
    "OFDM": {"sps": 1, "rolloff": 0.0},    # OFDM 不做成型滤波
    "FHSS": {"sps": 1, "rolloff": 0.0},    # FHSS 不做成型滤波
    "LFM": {"sps": 1, "rolloff": 0.0},     # LFM 不做成型滤波
}


def default_bandwidth(mod_type: str) -> float:
    return float(BANDWIDTH_TABLE.get(str(mod_type).upper(), 0.16875))


def _gfsk_bw_coef(bt):
    """GFSK 占用带宽系数 c(BT)：占用带宽 ≈ c(BT)/sps。

    线性拟合自实测 -35dB 占用带宽（×sps）：
        BT=0.2 → 0.563，BT=0.35 → 0.781，BT=0.5 → 0.969，BT=1.0 → 1.438
    GFSK 不走 RC 成型，占用带宽由峰值频偏（0.25·Rs）与高斯滤波 BT 决定，
    不能用 (1+β)/sps 近似（旧值 0.1875 比实测 0.098 大一倍）。
    """
    return 0.34375 + 1.09375 * float(bt)


def actual_bandwidth(mod_type, num_samples=1024, bandwidth_normalized=None, **kw) -> float:
    """计算 generate_baseband(..., bandwidth_normalized=...) 实际产生的占用带宽（归一化）。

    定义：
        - QAM 族 / OOK：占用带宽 = (1+rolloff)/sps（含滚降，sps 由目标带宽整数化）
        - GFSK：占用带宽 = (0.34375 + 1.09375·BT)/sps（实测标定）
        - LFM：扫频范围 |f1 - f0|
        - OFDM：激活子载波数 / nfft
        - FHSS：跳频范围 (hop_high - hop_low)
    该函数与 generate_baseband 的带宽解析逻辑保持一致，供工厂层写元数据。
    """
    mod = str(mod_type).upper()
    bw = None if bandwidth_normalized is None else float(bandwidth_normalized)
    if mod == "GFSK":
        c = _gfsk_bw_coef(kw.get("bt", 0.35))
        sps = int(MODULATION_CONFIG.get(mod, {"sps": 8})["sps"])
        if bw is not None:
            sps = int(np.clip(round(c / bw), 2, 64))
        return c / sps
    if mod == "LFM":
        if bw is None:
            f0 = float(kw.get("f0", -0.4))
            f1 = float(kw.get("f1", 0.4))
        else:
            half = bw / 2.0
            f0 = float(kw.get("f0", -half))
            f1 = float(kw.get("f1", half))
        return abs(f1 - f0)
    if mod == "OFDM":
        nfft = int(kw.get("nfft", 128))
        if bw is None:
            carriers = int(kw.get("subcarriers", max(4, int(round(0.8 * nfft)))))
        else:
            carriers = max(4, min(nfft - 1, int(round(bw * nfft))))
        return min(1.0, carriers / nfft)
    if mod == "FHSS":
        if bw is not None:
            return bw
        lo = float(kw.get("hop_low", -0.45))
        hi = float(kw.get("hop_high", 0.45))
        return hi - lo
    cfg = MODULATION_CONFIG.get(mod, {"sps": 8, "rolloff": 0.35})
    sps = int(cfg["sps"])
    rolloff = float(cfg["rolloff"])
    if sps <= 1:
        return 0.0
    if bw is not None:
        sps = int(np.clip(round((1.0 + rolloff) / bw), 2, 64))
    return (1.0 + rolloff) / sps


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    p = np.sqrt(np.mean(np.abs(x) ** 2))
    return x / p if p > 0 else x


def measure_obw99(wave, seg=256, percentile=0.99):
    """测量单通道波形的 99% 占用带宽（OBW99，归一化频率，双边长）。

    测量方法（benchmark 带宽标签的统一定义）：
        Hann 窗 + 50% 重叠分段平均 PSD → 按功率累积取 [0.5%, 99.5%] 分位跨度。

    注意：这是「可测量」带宽（评估对齐用），与理论占用带宽（actual_bandwidth，
    生成参数推导）定义不同 —— 对成型调制，OBW99 通常小于理论占用带宽
    （RC 滚降区的能量占比小）；对 LFM/OFDM/FHSS 二者基本一致。
    """
    x = np.asarray(wave, dtype=np.complex128)
    n = len(x)
    if n <= 0:
        return 0.0
    seg = min(int(seg), n)
    win = np.hanning(seg)
    f = np.fft.fftfreq(seg, 1.0)
    psd = np.zeros(seg)
    cnt = 0
    for s in range(0, n - seg + 1, seg // 2):
        psd += np.abs(np.fft.fft(x[s:s + seg] * win)) ** 2
        cnt += 1
    psd /= max(cnt, 1)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    total = float(np.sum(psd))
    if total <= 0.0:
        return 0.0
    cdf = np.cumsum(psd) / total
    lo = f[int(np.searchsorted(cdf, (1.0 - float(percentile)) / 2.0))]
    hi = f[int(np.searchsorted(cdf, 1.0 - (1.0 - float(percentile)) / 2.0))]
    return float(hi - lo)


def _qam(n, order, rng):
    m = int(np.sqrt(order))
    levels = np.arange(-(m - 1), m, 2)
    z = levels[rng.integers(0, m, n)] + 1j * levels[rng.integers(0, m, n)]
    return z / np.sqrt(np.mean(np.abs(z) ** 2))


def _raised_cosine_taps(sps, rolloff, num_taps):
    """升余弦 (Raised Cosine) 脉冲成形 FIR 系数，中心抽头 h(0) = 1。

        h(t) = sinc(t/Ts) * cos(pi*beta*t/Ts) / (1 - (2*beta*t/Ts)^2)

    奇点 t = ±1/(2β) 处取极限 (π/4)·sinc(1/(2β))。rolloff=0 时退化为 sinc。
    归一化约定：保持 h(0)=1（不按 DC 增益归一），这样符号中心采样点
    恰好恢复星座点（Nyquist 无 ISI），常量符号流输出常量 1。
    """
    beta = float(rolloff)
    n = np.arange(int(num_taps)) - (int(num_taps) - 1) / 2.0
    t = n / float(sps)  # 以符号周期 Ts 为单位
    with np.errstate(divide="ignore", invalid="ignore"):
        taps = np.sinc(t) * np.cos(np.pi * beta * t) / (1.0 - (2.0 * beta * t) ** 2)
        if beta > 0.0:
            singular = np.abs(np.abs(2.0 * beta * t) - 1.0) < 1e-9
            taps = np.where(singular, (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta)), taps)
        taps = np.nan_to_num(taps, nan=0.0, posinf=0.0, neginf=0.0)
    return taps


def _apply_pulse_shaping(symbols, sps, rolloff, num_taps=65, margin_symbols=4, out_length=None):
    """
    对符号序列进行过采样 + 升余弦成型滤波（带边界伪影消除）。

    symbols: 复数符号序列 (num_symbols,)
    sps: 每个符号的采样点数（过采样因子）
    rolloff: 升余弦滚降系数 (0~1)
    num_taps: 滤波器阶数（默认 65，奇数保证中心抽头对齐符号中心）
    margin_symbols: 两端额外生成的边际符号数（循环扩展，保证滤波稳态）
    out_length: 输出长度；默认 num_symbols * sps

    原理：
    - lfilter 直接对 num_symbols 个符号滤波时，起始段存在滤波暂态，
      且旧实现用 wrap 填充把信号首尾硬接在一起，产生相位不连续 → FFT 高频毛刺。
    - 修复：符号序列两端循环扩展 margin 个符号 → 滤波并延迟补偿后，
      只取中间对应原始符号的稳定段 out_length 点，两端都是稳态波形，
      相位连续、包络恒定，无启动零点和边界突变。

    返回: 滤波后的基带信号 (out_length,)
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    if sps == 1:
        x = symbols
        if out_length is not None:
            x = x[: int(out_length)]
        return x.astype(np.complex128)

    num_symbols = len(symbols)
    n_out = int(num_symbols * sps) if out_length is None else int(out_length)
    if n_out <= 0 or n_out > num_symbols * sps:
        raise ValueError(f"out_length ({n_out}) must be in (0, num_symbols*sps] ({num_symbols * sps})")

    # 升余弦 FIR 系数（真升余弦成形，rolloff 控制滚降形状）
    taps = _raised_cosine_taps(sps, rolloff, num_taps)
    delay = num_taps // 2

    # 边际符号数至少能覆盖滤波器延迟，保证中段切片全部处于稳态
    margin = max(int(margin_symbols), int(np.ceil(delay / sps)))

    # 循环扩展：前缀接末尾 margin 个符号，后缀接开头 margin 个符号
    if margin > 0:
        ext_symbols = np.concatenate([symbols[-margin:], symbols, symbols[:margin]])
    else:
        ext_symbols = symbols

    # 过采样：每个符号插入 sps-1 个零
    upsampled = np.zeros(len(ext_symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = ext_symbols

    # 滤波 + 延迟补偿
    filtered = lfilter(taps, 1.0, upsampled)
    filtered = filtered[delay:]

    # 只取中间对应原始符号的稳定段
    start = margin * sps
    seg = filtered[start:start + n_out]
    if len(seg) < n_out:
        raise ValueError(
            f"pulse shaping output too short ({len(seg)} < {n_out}); increase margin_symbols"
        )
    return seg.astype(np.complex128)


def generate_baseband(mod_type="QPSK", num_samples=1024, rng=None, bandwidth_normalized=None, **kw):
    """生成 (num_samples,) 复基带波形。

    bandwidth_normalized: 目标占用带宽（归一化，相对采样率）。为 None 时使用
    各调制类型的默认带宽（BANDWIDTH_TABLE）。注意：受 sps 整数化/钳位影响，
    实际带宽可能与请求值略有差异，用 actual_bandwidth() 获取实际值。
    """
    rng = np.random.default_rng() if rng is None else rng
    mod = str(mod_type).upper()
    n = int(num_samples)
    if n <= 0:
        raise ValueError("num_samples must be positive")
    if mod not in MODULATIONS:
        raise ValueError(f"Unsupported modulation: {mod_type}")

    cfg = MODULATION_CONFIG.get(mod, {"sps": 8, "rolloff": 0.35})
    sps = int(cfg["sps"])
    rolloff = float(cfg["rolloff"])

    # 带宽 → sps（仅对需要成型滤波的调制生效；GFSK 用实测标定系数）
    if bandwidth_normalized is not None and sps > 1:
        if mod == "GFSK":
            sps = int(np.clip(
                round(_gfsk_bw_coef(float(kw.get("bt", 0.35))) / float(bandwidth_normalized)),
                2, 64,
            ))
        else:
            sps = int(np.clip(round((1.0 + rolloff) / float(bandwidth_normalized)), 2, 64))

    # 计算需要的符号数（向上取整，确保最终长度 >= num_samples）
    num_symbols = int(np.ceil(n / sps)) if sps > 1 else n

    # 生成符号序列（1D 复数符号）
    if mod == "BPSK":
        symbols = 2 * rng.integers(0, 2, num_symbols) - 1
    elif mod == "QPSK":
        symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, num_symbols)))
    elif mod == "16QAM":
        symbols = _qam(num_symbols, 16, rng)
    elif mod == "64QAM":
        symbols = _qam(num_symbols, 64, rng)
    elif mod == "OOK":
        symbols = rng.integers(0, 2, num_symbols).astype(float)
    elif mod == "GFSK":
        # GFSK：符号级高斯滤波 → 零阶保持上采样到 sps 样点/符号 → 相位积分
        bits = 2 * rng.integers(0, 2, num_symbols) - 1
        bt = float(kw.get("bt", 0.35))
        span = max(3, int(4 / max(bt, 0.05)))
        kernel = np.ones(span) / span
        freq = np.convolve(bits, kernel, mode="same")          # 符号级频率 ∈ [-1, 1]
        freq_up = np.repeat(freq, sps)                          # 每符号 sps 个样点
        phase = 0.5 * np.pi * np.cumsum(freq_up) / sps          # 每符号相移 ±π/2
        x = np.exp(1j * phase)
        if len(x) > n:
            x = x[:n]
        elif len(x) < n:
            x = np.pad(x, (0, n - len(x)), mode='wrap')
        return _norm(x)
    elif mod == "LFM":
        if bandwidth_normalized is None:
            f0, f1 = kw.get("f0", -0.4), kw.get("f1", 0.4)
        else:
            half = float(bandwidth_normalized) / 2.0
            f0, f1 = kw.get("f0", -half), kw.get("f1", half)
        t = np.arange(n) / n
        x = np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t) * n)
        return _norm(x)
    elif mod == "FHSS":
        hops = int(kw.get("hops", max(2, n // 128)))
        if bandwidth_normalized is None:
            lo, hi = kw.get("hop_low", -0.45), kw.get("hop_high", 0.45)
        else:
            half = float(bandwidth_normalized) / 2.0
            lo, hi = kw.get("hop_low", -half), kw.get("hop_high", half)
        continuous = bool(kw.get("continuous_phase", True))  # 默认连续相位（无跳变毛刺）
        hop_len = max(1, n // hops)
        freqs = rng.uniform(lo, hi, hops)
        x = np.zeros(n, dtype=np.complex128)
        phase_acc = 0.0
        for i in range(hops):
            sl = slice(i * hop_len, min(n, (i + 1) * hop_len))
            m = sl.stop - sl.start
            if continuous:
                ph = phase_acc + 2 * np.pi * freqs[i] * np.arange(m)
                phase_acc += 2 * np.pi * freqs[i] * hop_len
            else:
                # 旧行为：每跳相位从 0 重新开始（跳变边界相位不连续）
                ph = 2 * np.pi * freqs[i] * np.arange(m)
            x[sl] = np.exp(1j * ph)
        return _norm(x)
    elif mod == "OFDM":
        # 大点数 IFFT + 零填充子载波：占用带宽 ≈ 激活子载波数 / nfft，可精确配置
        nfft = int(kw.get("nfft", 128))
        cp = int(kw.get("cyclic_prefix", 8))
        if bandwidth_normalized is None:
            carriers = int(kw.get("subcarriers", max(4, int(round(0.8 * nfft)))))
        else:
            carriers = max(4, min(nfft - 1, int(round(float(bandwidth_normalized) * nfft))))
        k_low = carriers // 2
        k_high = carriers - k_low
        # 频谱整形窗（边缘升余弦滚降，抑制子载波 sinc 旁瓣）：
        # 不加窗时 OFDM 的 99% OBW 会显著大于激活子载波跨度（旁瓣能量），
        # 加窗后 OBW99 ≈ carriers/nfft，与理论带宽一致（防折叠约束才成立）。
        edge = int(kw.get("window_edge", max(4, nfft // 16)))
        win = np.ones(nfft)
        if edge > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
            win[:edge] = ramp
            win[-edge:] = ramp[::-1]
        out = []
        while len(out) < n:
            sym = np.zeros(nfft, dtype=np.complex128)
            sym[:k_low] = _qam(k_low, 4, rng)
            sym[nfft - k_high:] = _qam(k_high, 4, rng)
            time_sym = np.fft.ifft(sym) * win
            out.extend(np.r_[time_sym[-cp:], time_sym])
        x = np.asarray(out[:n], dtype=np.complex128)
        # 整段边缘滚降：num_samples 通常不是块长整数倍，直接截断会在观测窗
        # 边缘产生大幅跳变 → 宽带谱泄漏（OBW99 虚高）。首尾各 edge 样本
        # 乘升余弦斜坡消除跳变（真实 OFDM burst 的常见做法）。
        if edge > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
            x[:edge] *= ramp
            x[-edge:] *= ramp[::-1]
        return _norm(x)

    # 对需要成型滤波的调制（BPSK/QPSK/16QAM/64QAM/OOK），应用升余弦成型滤波
    if mod in ["BPSK", "QPSK", "16QAM", "64QAM", "OOK"]:
        x = _apply_pulse_shaping(symbols, sps, rolloff, out_length=n)
    else:
        # 其他调制类型（OFDM/FHSS/LFM）已经特殊处理过，不会走到这里
        x = np.asarray(symbols, dtype=np.complex128)
        if len(x) > n:
            x = x[:n]
        elif len(x) < n:
            x = np.pad(x, (0, n - len(x)), mode='wrap')

    return _norm(x)
