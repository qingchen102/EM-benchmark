"""16 辐射体分类器：训练数据生成（A 路线，v8 方法论平移）。

训练分布 = 评测分布（"训练视角=评测视角"纪律）：分类器的实际输入是
"剥目标后的残差在干扰带的带限切片"，因此每个训练样本在线模拟该分布：
    干净波形（16 辐射体，随机 seed/带宽变体，短包长重抽保护）
    → 频偏（模拟测频/剥离残差的小偏移）
    → 多径（2–4 径随机时延/增益，模拟 RT）
    → 带限（Hann 软带限，模拟剥离带限）
    → 加噪（SNR U(-5,25)，覆盖 INR 5–20dB 的干扰功率谱系）
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

CLASSES16 = ["Bluetooth_FHSS", "WiFi_burst", "Microwave", "Zigbee", "DECT",
             "LTE_QPSK", "UAV_video_OFDM", "UAV_RC_hopping", "Radar_pulsed_LFM",
             "GPS_DSSS", "GNSS_jam", "GNSS_spoof", "TV_broadcast",
             "Chirp_jam", "Noise_jam", "Tone_jam"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES16)}
LENGTH = 4096                       # 训练长度（残差切片的典型量级；CNN 长度无关）

# 训练分布（与 D2–D4 干扰口径对齐：带宽 400–800k@4MHz → 归一化 0.10–0.20）
BW_RANGE = (0.06, 0.28)
SNR_RANGE = (-5.0, 25.0)
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
    # 5. 循环时移
    x = np.roll(x, int(rng.integers(0, length)))
    # 6. RMS 归一化
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) + 1e-12
    return (x / rms).astype(np.complex64)


def to_input(x: np.ndarray) -> np.ndarray:
    """(2, L) float32 I/Q——与 v8 同口径（输入已 RMS 归一化）。"""
    return np.stack([x.real, x.imag]).astype(np.float32)
