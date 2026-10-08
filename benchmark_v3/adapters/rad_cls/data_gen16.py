"""16 辐射体分类器：训练数据生成（A 路线，v8 方法论平移）。

训练分布 = 评测分布（"训练视角=评测视角"纪律）：分类器的实际输入是
"剥目标后的残差在干扰带的带限切片"，因此每个训练样本在线模拟该分布：
    干净波形（16 辐射体，随机 seed/带宽变体，短包长重抽保护）
    → 频偏（模拟测频/剥离残差的小偏移）
    → 多径（2–4 径随机时延/增益，模拟 RT）
    → 带限（Hann 软带限，模拟剥离带限）
    → 加噪（SNR U(-5,25)，覆盖 INR 5–20dB 的干扰功率谱系）
    → 接收机滤波（firwin129@810kHz，与平台 receive 一致；2026-10-02 补）
    → 循环时移 + RMS 归一化
种子段与评测完全隔离（SEED_BASE 独立段，勿改）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent.parent
for _p in (str(_V3 / "simulation"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3.parent / "benchmark_v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from em_signal_simulator_v3.emitters import generate_emitter  # noqa: E402

# 19 类（2026-09-29 +工业脉冲噪声家族 3 种；变量名保留 CLASSES16 兼容历史引用）
CLASSES16 = ["Bluetooth_FHSS", "WiFi_burst", "Microwave", "Zigbee", "DECT",
             "LTE_QPSK", "UAV_video_OFDM", "UAV_RC_hopping", "Radar_pulsed_LFM",
             "GPS_DSSS", "GNSS_jam", "GNSS_spoof", "TV_broadcast",
             "Chirp_jam", "Noise_jam", "Tone_jam",
             "Welder", "VFD_noise", "Corona"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES16)}
LENGTH = 4096                       # 训练长度（残差切片的典型量级；CNN 长度无关）

# 训练分布（与 D2–D4 干扰口径对齐：带宽 400–800k@4MHz → 归一化 0.10–0.20）
BW_RANGE = (0.06, 0.28)
SNR_RANGE = (-15.0, 25.0)           # 2026-10-02 下探到 -15：D2 弱干扰 INR 2–8dB 的样本级 SNR 可到 -10dB 以下
FOFF_RANGE_HZ = (-20e3, 20e3)
SEED_BASE = 900_000                 # 独立种子段（评测用 31000+/其他，隔离）


def gen_wave16(name: str, rng: np.random.Generator,
               length: int = LENGTH, bw: float | None = None) -> np.ndarray:
    """按辐射体名生成干净波形；生成器对短包长脆弱 → 换 seed 重抽保护。"""
    bw = float(rng.uniform(*BW_RANGE)) if bw is None else bw
    for _ in range(40):
        try:
            return generate_emitter(name, rng, length=length,
                                    **({"bw": bw} if _accepts_bw(name) else {}))["iq"]
        except ValueError:
            continue
    raise RuntimeError(f"generate_emitter({name}) 反复失败")


_ACCEPT_BW_CACHE: dict[str, bool] = {}


def _accepts_bw(name: str) -> bool:
    import inspect
    from em_signal_simulator_v3.emitters import EMITTERS
    if name not in _ACCEPT_BW_CACHE:
        gen = EMITTERS[name][0]
        _ACCEPT_BW_CACHE[name] = "bw" in inspect.signature(gen).parameters
    return _ACCEPT_BW_CACHE[name]


# 接收机滤波（2026-10-02 根因修复）：真实评测输入全部过了平台 receiver 的
# firwin129 低通（bandwidth_hz/2 截止）。此前训练分布缺这一环——CNN 对"未滤波"
# 形态过拟合，真实滤波形态上全部误判为 WiFi_burst（GT-slice 三连挂的总根因）。
# 模拟顺序与 receive() 一致：先加噪，再一起过滤波。
from scipy.signal import firwin, fftconvolve as _fftconvolve  # noqa: E402

RX_FILTER_TAPS = 129
RX_CUTOFF_HZ = 810e3            # rx.bandwidth_hz / 2（平台 receiver 配置）
_RX_TAPS = firwin(RX_FILTER_TAPS, RX_CUTOFF_HZ, fs=4e6)


def _rx_filter(x: np.ndarray) -> np.ndarray:
    """复基带接收滤波（与平台 receive 的 firwin129 一致，逐阵元/单通道均可用）。"""
    x = np.asarray(x)
    if x.ndim == 1:
        return _fftconvolve(x, _RX_TAPS, mode="same")
    return _fftconvolve(x, _RX_TAPS[None, :], mode="same", axes=1)


CROSSTALK_RANGE = (0.0, 150e3)     # 兄弟干扰相对主干扰的频偏（Hz）
CROSSTALK_BW = (400e3, 800e3)       # 兄弟干扰带宽（对齐 D2 干扰带宽）
CROSSTALK_INR = (5.0, 20.0)         # 兄弟干扰 INR（对齐 D2 干扰功率）


def _add_crosstalk_sib(x: np.ndarray, rng: np.random.Generator,
                       n_sib: int, fs: float = 4e6) -> np.ndarray:
    """叠加 n_sib 个"共道兄弟干扰"（来自其他类，模拟真实 D2 的重叠分布）。

    每个兄弟干扰：随机从 16 类挑一个（可含主类近似），随机频偏 ±150kHz、
    带宽 400–800k、功率 INR U(5,20)（相对当前主干扰均方根）。这使训练切片带内
    含兄弟能量——分类器学会在 crosstalk 下识别主干扰。
    """
    from em_signal_simulator_v3.emitters import generate_emitter
    main_power = float(np.mean(np.abs(x) ** 2)) + 1e-30
    length = len(x)
    t = np.arange(length)
    for _ in range(int(n_sib)):
        sib_name = str(rng.choice(CLASSES16))
        try:
            sib = gen_wave16(sib_name, rng, length=length).astype(np.complex128)
        except RuntimeError:
            continue
        f_c = float(rng.uniform(*CROSSTALK_RANGE)) * rng.choice([-1, 1])
        sib = sib * np.exp(1j * 2 * np.pi * f_c * t / fs)
        # 兄弟功率 = INR 相对主干扰
        sib_p = float(np.mean(np.abs(sib) ** 2)) + 1e-30
        inr = float(rng.uniform(*CROSSTALK_INR))
        g = np.sqrt(main_power * 10 ** (inr / 10) / sib_p)
        x = x + g * sib
    return x


def simulate_residual(x: np.ndarray, rng: np.random.Generator,
                      fs: float = 4e6, n_sib: int = 0) -> np.ndarray:
    """残差分布模拟：频偏 → 多径 → 共道兄弟干扰 → 带限 → 加噪 → 时移 → RMS 归一化。

    n_sib >= 1 时叠加共道兄弟干扰（来自其他类），模拟真实 D2 的重叠。训练时用
    n_sib>0 使分布对齐真实残差；oracle 可按 n_sib=0/1/2 分桶评估 crosstalk 影响。
    """
    length = len(x)
    # 1. 小频偏（测频/剥离残差）
    f_off = float(rng.uniform(*FOFF_RANGE_HZ))
    x = x * np.exp(1j * 2 * np.pi * f_off * np.arange(length) / fs)
    # 2. 多径：LoS + 1–3 条反射径（时延 ≤64 采样 = 16µs，增益递减）
    x = x.astype(np.complex128)
    for _ in range(int(rng.integers(1, 4))):
        d = int(rng.integers(2, 64))
        g = float(rng.uniform(0.1, 0.6)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        x = x + g * np.roll(x, d)
    # 3. 共道兄弟干扰（模拟真实 D2 重叠分布）
    if n_sib > 0:
        x = _add_crosstalk_sib(x, rng, n_sib, fs)
    # 4. 带限（Hann 软带限到带宽 bw，模拟剥离带限）
    bw = float(rng.uniform(*BW_RANGE))
    spec = np.fft.fft(x)
    freq = np.fft.fftfreq(length, 1 / fs)
    edge = bw * fs / 2
    soft = np.clip((edge - np.abs(freq)) / (0.05 * fs) + 1, 0, 1)
    x = np.fft.ifft(spec * soft)
    # 4. 加噪（SNR 相对信号功率）
    p = float(np.mean(np.abs(x) ** 2)) + 1e-30
    snr = float(rng.uniform(*SNR_RANGE))
    noise = np.sqrt(p * 10 ** (-snr / 10) / 2) * (
        rng.standard_normal(length) + 1j * rng.standard_normal(length))
    x = x + noise
    # 4.5 接收机滤波（信号+噪声一起过，与平台 receive 顺序一致）
    x = _rx_filter(x)
    # 5. 循环时移
    x = np.roll(x, int(rng.integers(0, length)))
    # 6. RMS 归一化
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) + 1e-12
    return (x / rms).astype(np.complex64)


def to_input(x: np.ndarray) -> np.ndarray:
    """(2, L) float32 I/Q——与 v8 同口径（输入已 RMS 归一化）。"""
    return np.stack([x.real, x.imag]).astype(np.float32)


def simulate_beamformed_train(x_main: np.ndarray, cls_main: int,
                              xs_pool: np.ndarray, ys_pool: np.ndarray,
                              rng: np.random.Generator, m: int = 8) -> np.ndarray:
    """训练分布终段：模拟"空间聚类定源 → 波束成形提取"的完整评测链路。

    主源 + 1–2 个兄弟源（随机其他类波形、随机方向/信道）在 8 元 0.5λ ULA 上
    叠加 → 已知全部方向的多源 LS 提取主源波形。训练样本 = 提取后波形——
    与评测时 CNN 的真实输入（波束成形提取产物）同分布，crosstalk 被 LS
    显式分离而非污染训练。
    """
    length = len(x_main)
    idx = np.arange(m)
    # 随机方向（主源 + 兄弟），主/兄弟角距 ≥0.15 防 LS 病态。
    # 2026-10-02：兄弟数随机 0/1/2——评测口径（全带 LS）下提取波形可能"干净"
    # （兄弟被显式分离），训练分布必须覆盖无兄弟样本。
    u_main = float(rng.uniform(-0.9, 0.9))
    u_sibs = []
    for _ in range(int(rng.integers(0, 3))):
        u = float(rng.uniform(-0.9, 0.9))
        if min(abs(u - u_main), abs(np.pi - abs(u - u_main))) < 0.15:
            u = float(np.clip(u + 0.35, -0.95, 0.95))
        u_sibs.append(u)
    a_main = np.exp(2j * np.pi * 0.5 * u_main * idx)
    a_sibs = [np.exp(2j * np.pi * 0.5 * u * idx) for u in u_sibs]
    # 信道：复增益（模拟平坦信道；多径由 CNN 泛化覆盖）
    h_main = rng.uniform(0.6, 1.4) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    X = np.outer(a_main * h_main, x_main.astype(np.complex128))
    for a_s in a_sibs:
        while True:                              # 兄弟取其他类实例
            j = int(rng.integers(len(xs_pool)))
            if ys_pool[j] != cls_main:
                break
        h_s = rng.uniform(0.4, 1.2) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        X = X + np.outer(a_s * h_s, xs_pool[j].astype(np.complex128))
    # 噪声
    p = float(np.mean(np.abs(X) ** 2)) + 1e-30
    snr = float(rng.uniform(*SNR_RANGE))
    X = X + np.sqrt(p * 10 ** (-snr / 10) / 2 / m) * (
        rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
    # 接收机滤波（信号+噪声一起过，与平台 receive 顺序一致；LS 在滤波后提取）
    X = _rx_filter(X)
    # 多源 LS 提取主源（已知全部方向——模拟聚类已定位）
    A = np.stack([a_main] + a_sibs, axis=1)
    S = np.linalg.pinv(A) @ X
    y = S[0]
    # 7. 带限切片增强（2026-10-02）：评测工具/Agent 可能对提取波形再做窄带
    # 聚焦（通带外=0 的口径），训练分布必须覆盖。切片以主源为中心（真实工具
    # 的源带聚焦保证主源在带内），带宽 = 主源带宽 × U(1.0, 1.4)，brick/软边各半。
    if rng.random() < 0.5:
        f_grid = np.fft.fftfreq(length, 1 / 4e6)
        bw_main = float(rng.uniform(*BW_RANGE)) * 4e6        # 与 gen_wave16 的带宽口径一致
        bw_c = bw_main * float(rng.uniform(1.0, 1.4))
        c_off = float(rng.uniform(-1, 1)) * max(bw_c - bw_main, 0) / 2 + \
            float(rng.uniform(-0.05, 0.05)) * 4e6            # 小随机偏移（测频误差）
        lo_c, hi_c = c_off - bw_c / 2, c_off + bw_c / 2
        if rng.random() < 0.5:                   # brick
            gain_m = ((f_grid >= lo_c) & (f_grid <= hi_c)).astype(float)
        else:                                    # 软边（50kHz Hann 滚降）
            roll = 50e3
            gain_m = np.clip(np.minimum((f_grid - (lo_c - roll)) / roll,
                                        ((hi_c + roll) - f_grid) / roll), 0, 1)
            gain_m = 0.5 * (1 - np.cos(np.pi * gain_m))
        y = np.fft.ifft(np.fft.fft(y) * gain_m)
    rms = float(np.sqrt(np.mean(np.abs(y) ** 2))) + 1e-12
    return (y / rms).astype(np.complex64)
