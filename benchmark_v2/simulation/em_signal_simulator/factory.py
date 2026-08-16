"""样本工厂：多源（目标 + 干扰）多天线 IQ 信号生成与 Ground Truth 打标 (v2)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .baseband import (
    generate_baseband,
    default_bandwidth,
    actual_bandwidth,
    measure_obw99,
    MODULATIONS,
)
from .jamming import generate_interferer_waveform, JAMMING_TYPES
from .channel import (
    make_receiver,
    steering_vector,
    apply_freq_offset,
    expand_to_array,
    add_awgn,
    apply_lna_saturation,
)

DEFAULT_NUM_ANTENNAS = 4
DEFAULT_NUM_SAMPLES = 1024
DEFAULT_CENTER_FREQ_HZ = 2.4e9
DEFAULT_SAMPLING_RATE_HZ = 20e6

DOA_RANGE = (-60.0, 60.0)
FREQ_OFFSET_RANGE = (-0.5, 0.5)
POWER_RATIO_RANGE_DB = (0.0, 15.0)

FREQ_MIN_SEP = 0.15
DOA_MIN_SEP = 15.0

REAL_INTERFERERS = [
    "WiFi_OFDM",      # OFDM，像 802.11
    "LTE_QPSK",       # QPSK，像 4G/5G 控制信道
    "Bluetooth_GFSK", # GFSK，像蓝牙
    "Radar_LFM",      # LFM，像雷达脉冲
    "DSSS_BPSK",      # BPSK，像 GPS 或某些遥测
]
# 纯波形类干扰（走 jamming.py 的波形生成，而非 baseband 调制）
JAMMING_WAVEFORMS = {"single_tone", "swept", "pulse", "broadband", "nfm"}

INTERFERER_BANDWIDTH = {
    "single_tone": 0.01,
    "swept": 0.5,
    "pulse": 0.5,
    "broadband": 0.9,
    "nfm": 0.5,
}
PULSE_TYPE = "pulse"


def _valid_offset_range(bandwidth_normalized):
    """干扰频带不折叠（alias-free）的合法中心频偏范围。

    信号完整频带 [f_off - bw/2, f_off + bw/2] 必须落在奈奎斯特带 [-0.5, 0.5]
    内，否则频谱折叠会破坏干扰的带宽/中心频率/DOA 参数：
        |f_off| <= 0.5 - bw/2
    全带宽弹幕（bw=1.0，如 pulse/broadband）只能以 0 为中心。
    """
    half = float(bandwidth_normalized) / 2.0
    bound = max(0.0, 0.5 - half)
    return (-bound, bound)


def _make_receiver(**kw):
    return make_receiver(
        center_frequency_hz=kw.get("center_frequency_hz", DEFAULT_CENTER_FREQ_HZ),
        sampling_rate_hz=kw.get("sampling_rate_hz", DEFAULT_SAMPLING_RATE_HZ),
        antenna_spacing_wavelength=kw.get("antenna_spacing_wavelength", 0.5),
    )


def _sample_power_ratio(rng, min_inr_db=0.0, snr_db=10.0):
    """采样干扰功率比（dB，相对目标），并施加可检测性约束。

    干扰相对噪声的功率 INR = power_ratio_db + snr_db。若 min_inr_db > 0，
    则 power_ratio_db 至少为 max(0, min_inr_db - snr_db)，保证干扰在噪声之上
    （INR 低于下限的干扰物理上不可检测，样本无意义）。
    当 SNR 过低导致下限超出功率比上限（无解）时，保留原采样值——
    元数据 inr_db 会如实标注，评估端可据此分层。
    """
    power_ratio_db = float(rng.uniform(*POWER_RATIO_RANGE_DB))
    min_inr_db = float(min_inr_db)
    if min_inr_db > 0.0:
        lo = max(0.0, min_inr_db - float(snr_db))
        if lo <= POWER_RATIO_RANGE_DB[1] and power_ratio_db < lo:
            power_ratio_db = float(rng.uniform(lo, POWER_RATIO_RANGE_DB[1]))
    return power_ratio_db


def _sample_interferer(rng, num_samples, force_type=None, **kw):
    """随机采样一个干扰源参数。

    返回 dict，包含：
        waveform_type: None（baseband 调制）或 jamming 波形名
        modulation / display_name / freq_offset_normalized / doa_degree
        power_ratio_db / bandwidth_normalized / is_pulse / is_modulated / wave_kw
    """
    if force_type is None or force_type == "random":
        typ = str(rng.choice(REAL_INTERFERERS))
    else:
        typ = force_type

    # —— 纯波形类干扰（single_tone / swept / pulse / broadband / nfm）——
    if typ in JAMMING_WAVEFORMS:
        wave_kw = {}
        if typ == "swept":
            wave_kw["f_start"] = float(kw.get("f_start", -0.4))
            wave_kw["f_stop"] = float(kw.get("f_stop", 0.4))
            if any(k in kw for k in ("duty_cycle", "pri", "pulse_width")):
                for k in ("duty_cycle", "pri", "pulse_width"):
                    if kw.get(k) is not None:
                        wave_kw[k] = float(kw[k])
            else:
                # 默认脉冲化 LFM（真实雷达/ECM 脉冲串）：随机占空比 + 随机 PRI
                wave_kw["duty_cycle"] = float(rng.uniform(0.05, 0.4))
                wave_kw["pri"] = float(rng.uniform(0.1, 0.5))
        elif typ == "nfm":
            wave_kw["noise_bandwidth"] = float(kw.get("noise_bandwidth", rng.uniform(0.05, 0.25)))
            wave_kw["freq_deviation"] = float(kw.get("freq_deviation", rng.uniform(0.05, 0.3)))
            # 防折叠（实测标定）：NFM 的 OBW99 可达 4.2*(f_dev+noise_bw)（Carson 系数
            # 2.0 与 3.2/3.8 均低估，噪声积分后频谱显著展宽）。约束 f_dev+noise_bw <= 0.20，
            # 使 OBW99 <= 4.2*0.20 = 0.84，保证完整频带不折叠。
            if wave_kw["freq_deviation"] + wave_kw["noise_bandwidth"] > 0.20:
                scale = 0.20 / (wave_kw["freq_deviation"] + wave_kw["noise_bandwidth"])
                wave_kw["freq_deviation"] *= scale
                wave_kw["noise_bandwidth"] *= scale
        elif typ == "pulse":
            # 周期脉冲串（真实雷达脉冲）：随机 PRI + 随机占空比
            wave_kw["pri"] = float(kw.get("pri", rng.uniform(0.1, 0.5)))
            wave_kw["duty_cycle"] = float(kw.get("duty_cycle", rng.uniform(0.05, 0.3)))
        elif typ == "single_tone":
            wave_kw["frequency"] = float(kw.get("frequency", 0.0))
        is_pulse = (typ == "pulse") or (typ == "swept" and float(wave_kw.get("duty_cycle", 1.0)) < 1.0)

        # 波形类干扰的带宽语义（按波形参数计算，而非查表）：
        #   single_tone → 0（单音无带宽）；swept → 扫频范围；
        #   nfm → Carson 近似 2(f_dev + noise_bw)；pulse/broadband → 全带宽 1.0
        if typ == "single_tone":
            bandwidth = 0.0
        elif typ == "swept":
            bandwidth = abs(float(wave_kw.get("f_stop", 0.4)) - float(wave_kw.get("f_start", -0.4)))
        elif typ == "nfm":
            # 带宽语义 = 实测标定 OBW99 ≈ 4.2*(f_dev + noise_bw)（clip 0.95）
            bandwidth = min(
                0.95,
                4.2 * (float(wave_kw.get("freq_deviation", 0.2))
                       + float(wave_kw.get("noise_bandwidth", 0.1))),
            )
        else:  # pulse / broadband
            bandwidth = 1.0

        # 防折叠带宽：波形类的语义带宽已按实测标定，直接用于频偏合法范围
        bandwidth_fold = bandwidth

        # 中心频偏限制在「完整频带不折叠」的合法范围内（|f_off| <= 0.5 - bw/2）
        offset_lo, offset_hi = _valid_offset_range(bandwidth_fold)
        freq_offset = float(rng.uniform(offset_lo, offset_hi)) if offset_hi > offset_lo else 0.0

        power_ratio_db = _sample_power_ratio(
            rng, min_inr_db=kw.get("min_inr_db", 0.0), snr_db=kw.get("snr_db", 10.0)
        )

        return {
            "waveform_type": typ,
            "modulation": typ,
            "display_name": typ,
            "freq_offset_normalized": freq_offset,
            "doa_degree": float(rng.uniform(*DOA_RANGE)),
            "power_ratio_db": power_ratio_db,
            "bandwidth_normalized": bandwidth,
            "bandwidth_fold": bandwidth_fold,
            "is_pulse": is_pulse,
            "is_modulated": typ != "single_tone",
            "wave_kw": wave_kw,
        }

    # —— 真实调制类干扰（OFDM/QPSK/GFSK/LFM/BPSK）——
    modulation_map = {
        "WiFi_OFDM": "OFDM",
        "LTE_QPSK": "QPSK",
        "Bluetooth_GFSK": "GFSK",
        "Radar_LFM": "LFM",
        "DSSS_BPSK": "BPSK",
    }
    mod = modulation_map.get(typ, "QPSK")
    # 带宽作为待估计的干扰参数：在配置范围内随机采样（可被显式参数覆盖）
    bw_lo, bw_hi = kw.get("bandwidth_range", (0.05, 0.6))
    bandwidth_req = float(rng.uniform(float(bw_lo), float(bw_hi)))
    # 实际带宽（sps 整数化后），用于元数据理论字段
    bandwidth = actual_bandwidth(mod, num_samples, bandwidth_normalized=bandwidth_req)

    # 防折叠带宽：OFDM 的子载波 sinc 旁瓣使实测 OBW99 可达理论值的 ~1.7 倍
    # （小载波数时旁瓣能量占比大），频偏合法范围必须按保守上界计算
    if mod == "OFDM":
        bandwidth_fold = min(0.98, bandwidth + 0.10)
    else:
        bandwidth_fold = bandwidth

    # 中心频偏限制在「完整频带不折叠」的合法范围内
    offset_lo, offset_hi = _valid_offset_range(bandwidth_fold)
    freq_offset = float(rng.uniform(offset_lo, offset_hi)) if offset_hi > offset_lo else 0.0

    power_ratio_db = _sample_power_ratio(
        rng, min_inr_db=kw.get("min_inr_db", 0.0), snr_db=kw.get("snr_db", 10.0)
    )

    return {
        "waveform_type": None,
        "modulation": mod,
        "display_name": typ,          # 给 LLM 看的人类可读名称
        "freq_offset_normalized": freq_offset,
        "doa_degree": float(rng.uniform(*DOA_RANGE)),
        "power_ratio_db": power_ratio_db,
        "bandwidth_normalized": bandwidth,   # 理论带宽（元数据）
        "bandwidth_fold": bandwidth_fold,   # 防折叠带宽（频偏约束用）
        "is_pulse": False,
        "is_modulated": True,
        "wave_kw": {},
    }


def _sample_away_from(rng, forbidden, bounds, min_sep):
    """在 [lo, hi] 内均匀采样一个与所有 forbidden 值距离 >= min_sep 的值。

    无可行区间时返回 None。
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    intervals = [(lo, hi)]
    for v in forbidden:
        v = float(v)
        new = []
        for (l, h) in intervals:
            if v - min_sep > l:
                new.append((l, min(v - min_sep, h)))
            if v + min_sep < h:
                new.append((max(v + min_sep, l), h))
        intervals = new
    if not intervals:
        return None
    widths = [h - l for (l, h) in intervals]
    total = sum(widths)
    if total <= 1e-12:
        return None
    u = rng.uniform(0.0, total)
    acc = 0.0
    for (l, h), w in zip(intervals, widths):
        acc += w
        if u <= acc:
            return l + (u - (acc - w))
    return intervals[-1][1]  # 数值兜底


def _correct_separation(
    interferers,
    freq_min=FREQ_MIN_SEP,
    doa_min=DOA_MIN_SEP,
    max_attempts=20,
    rng=None,
):
    """迭代式参数分离修正：让所有干扰源两两满足频率/DOA 最小间隔。

    单遍修正可能因 freq_offset / doa 被裁剪到边界 ([-0.5,0.5] / [-60,60])
    而失败。这里改为 while 循环 + 最大尝试次数：
    每次对不满足约束的干扰源，在避开所有其他干扰源的合法区间内重新采样
    （先频率、后 DOA），直到全部满足或达到 max_attempts；最后再做一次
    确定性微调兜底，保证不会留下不可分离的干扰源。
    """
    if len(interferers) < 2:
        return
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(int(max_attempts)):
        ok = True
        for i in range(len(interferers)):
            for j in range(i + 1, len(interferers)):
                a, b = interferers[i], interferers[j]
                df = abs(a["freq_offset_normalized"] - b["freq_offset_normalized"])
                ddoa = abs(a["doa_degree"] - b["doa_degree"])
                if df >= freq_min or ddoa >= doa_min:
                    continue
                ok = False
                # 优先重采样 b 的频偏（避开所有其他干扰源的频率，
                # 且限制在自身「完整频带不折叠」的合法范围内）
                others_f = [
                    interferers[k]["freq_offset_normalized"]
                    for k in range(len(interferers))
                    if k != j
                ]
                b_bounds = _valid_offset_range(
                    b.get("bandwidth_fold", b.get("bandwidth_normalized", 0.0)))
                new_f = _sample_away_from(rng, others_f, b_bounds, freq_min)
                if new_f is not None:
                    b["freq_offset_normalized"] = new_f
                    continue
                # 频偏区间被压缩到无解 → 改重采样 DOA
                others_d = [
                    interferers[k]["doa_degree"]
                    for k in range(len(interferers))
                    if k != j
                ]
                new_d = _sample_away_from(rng, others_d, DOA_RANGE, doa_min)
                if new_d is not None:
                    b["doa_degree"] = new_d
        if ok:
            return

    # 兜底：确定性微调（尽力而为，避免无限循环）
    for i in range(len(interferers)):
        for j in range(i + 1, len(interferers)):
            a, b = interferers[i], interferers[j]
            df = abs(a["freq_offset_normalized"] - b["freq_offset_normalized"])
            ddoa = abs(a["doa_degree"] - b["doa_degree"])
            if df < freq_min and ddoa < doa_min:
                sign = 1.0 if b["freq_offset_normalized"] >= 0 else -1.0
                lo_b, hi_b = _valid_offset_range(
                    b.get("bandwidth_fold", b.get("bandwidth_normalized", 0.0)))
                b["freq_offset_normalized"] = float(np.clip(
                    b["freq_offset_normalized"] + sign * (freq_min - df + 1e-3),
                    lo_b, hi_b,
                ))
                b["doa_degree"] = float(np.clip(
                    b["doa_degree"] + np.sign(b["doa_degree"] or 1.0) * (doa_min - ddoa + 0.5),
                    *DOA_RANGE,
                ))


def _auto_label(source, target_bw, power_threshold_db=10.0, ratio_threshold=0.5):
    """干扰类别标注（co_channel / adjacent / blocking / pulse / none）。

    - pulse：脉冲类干扰（is_pulse=True）优先；
    - co_channel：ratio < 0.5（频偏落在目标带宽内）；
    - blocking：ratio >= ratio_threshold 且 power >= power_threshold_db
      （带外强信号 → 接收机 LNA 饱和，真实阻塞效应）；
    - adjacent：0.5 <= ratio < 2.0 且功率低于阻塞阈值；
    - none：ratio >= 2.0 且功率低于阻塞阈值（无害远端弱信号）。

    ratio = |freq_offset_normalized| / target_bw。
    """
    if source.get("is_pulse"):
        return "pulse"

    ratio = abs(float(source["freq_offset_normalized"])) / max(float(target_bw), 1e-9)
    power_db = float(source.get("power_ratio_db", 0.0))

    if ratio < 0.5:
        return "co_channel"
    if power_db >= float(power_threshold_db) and ratio >= float(ratio_threshold):
        return "blocking"
    if ratio < 2.0:
        return "adjacent"
    return "none"


def _resolve_modulation(modulation, rng):
    if modulation is None or modulation == "random":
        return str(rng.choice(sorted(MODULATIONS)))
    if modulation.upper() not in MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return modulation.upper()


def _normalized_to_hz(value, sampling_rate_hz):
    return float(value) * float(sampling_rate_hz)


def generate_signal_sample(
    num_sources=2,
    num_antennas=DEFAULT_NUM_ANTENNAS,
    num_samples=DEFAULT_NUM_SAMPLES,
    snr_db=10.0,
    modulation="random",
    interferer_type="random",
    center_frequency_hz=DEFAULT_CENTER_FREQ_HZ,
    sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
    seed=None,
    **kwargs,
):
    rng = np.random.default_rng(seed)
    receiver = _make_receiver(
        center_frequency_hz=center_frequency_hz,
        sampling_rate_hz=sampling_rate_hz,
    )
    num_antennas = int(num_antennas)
    num_samples = int(num_samples)
    fs = float(sampling_rate_hz)

    # —— 可选物理效应参数（全部有默认值，向后兼容）——
    lna_saturation_db = float(kwargs.get("lna_saturation_db", 6.0))
    lna_p = float(kwargs.get("lna_p", 2.0))
    array_gain_error_std = float(kwargs.get("array_gain_error_std", 0.0))
    array_phase_error_std_deg = float(kwargs.get("array_phase_error_std_deg", 0.0))
    # 带宽配置：None → 调制类型默认带宽；干扰带宽为待估计参数，默认随机采样
    target_bandwidth_normalized = kwargs.get("target_bandwidth_normalized", None)
    bw_lo, bw_hi = kwargs.get("interferer_bandwidth_range", (0.05, 0.6))
    # 防折叠：带宽必须 < 1.0（半带宽 < 0.5，否则目标/干扰频带必超出奈奎斯特带）
    interferer_bandwidth_range = (
        max(0.001, float(bw_lo)),
        min(0.98, float(bw_hi)),
    )
    # blocking 判定阈值（带外强信号）
    blocking_power_threshold_db = float(kwargs.get("blocking_power_threshold_db", 10.0))
    blocking_ratio_threshold = float(kwargs.get("blocking_ratio_threshold", 0.5))
    # 干扰可检测性约束：INR = power_ratio_db + snr_db >= min_inr_db（默认 3 dB）
    min_inr_db = float(kwargs.get("min_inr_db", 3.0))

    # 阵列失配：物理上一个阵列只有一组失配，所有源共享同一组（每个样本抽一次）
    if array_gain_error_std > 0.0 or array_phase_error_std_deg > 0.0:
        array_gain_error = rng.normal(0.0, array_gain_error_std, num_antennas)
        array_phase_error_rad = np.deg2rad(
            rng.normal(0.0, array_phase_error_std_deg, num_antennas)
        )
    else:
        array_gain_error = None
        array_phase_error_rad = None

    if num_sources == "random":
        num_sources = int(rng.choice([1, 2, 3]))
    else:
        num_sources = int(num_sources)

    target_mod = _resolve_modulation(modulation, rng)
    # 理论占用带宽（生成参数推导，用于防折叠校验与带宽配置）
    target_bw = actual_bandwidth(
        target_mod, num_samples, bandwidth_normalized=target_bandwidth_normalized
    )
    if target_bw >= 1.0 - 1e-9:
        raise ValueError(
            f"target bandwidth ({target_bw:.3f}) must be < 1.0 (Nyquist): "
            "the full band would alias"
        )

    target_wave = generate_baseband(
        target_mod, num_samples, rng=rng, sampling_rate=sampling_rate_hz,
        bandwidth_normalized=target_bandwidth_normalized,
    )
    # 可测量带宽（99% OBW）—— benchmark 带宽标签的统一定义
    target_bw_obw99 = measure_obw99(target_wave)
    target_wave = apply_freq_offset(target_wave, 0.0)
    target_array = expand_to_array(
        target_wave, 0.0, num_antennas,
        receiver["antenna_spacing_m"], receiver["wavelength_m"],
        gain_error=array_gain_error, phase_error_rad=array_phase_error_rad,
    )
    target_power = float(np.mean(np.abs(target_array) ** 2))

    interferers = [
        _sample_interferer(
            rng, num_samples, force_type=interferer_type,
            bandwidth_range=interferer_bandwidth_range,
            min_inr_db=min_inr_db, snr_db=snr_db,
        )
        for _ in range(max(0, num_sources - 1))
    ]
    # P1-2a：目标纳入分离约束集 —— 干扰必须与目标（f=0, doa=0）也可区分，
    # 避免"干扰与目标完全重合 → 不可辨识样本"。目标放在最前，修正只会作用于干扰。
    if interferers:
        target_ref = {
            "freq_offset_normalized": 0.0,
            "doa_degree": 0.0,
            "bandwidth_normalized": target_bw,
        }
        _correct_separation([target_ref] + interferers, rng=rng)
    else:
        _correct_separation(interferers, rng=rng)

    sources_meta = [{
        "source_id": 0,
        "role": "target",
        "modulation": target_mod,
        "power_ratio_db": 0.0,
        "freq_offset_normalized": 0.0,
        "freq_offset_hz": 0.0,
        "doa_degree": 0.0,
        "bandwidth_normalized": target_bw_obw99,      # 99% OBW（评估定义）
        "bandwidth_theoretical": target_bw,           # 理论占用带宽（参考）
        "bandwidth_hz": _normalized_to_hz(target_bw_obw99, fs),
    }]

    if interferers:
        mixer = target_array.copy()
        for i, itf in enumerate(interferers):
            if itf.get("waveform_type") in JAMMING_WAVEFORMS:
                wave = generate_interferer_waveform(
                    itf["waveform_type"], num_samples, rng=rng, **itf.get("wave_kw", {})
                )
                bw_theoretical = float(itf["bandwidth_normalized"])  # 波形类：语义带宽
            else:
                wave = generate_baseband(
                    itf["modulation"], num_samples, rng=rng,
                    bandwidth_normalized=itf.get("bandwidth_normalized"),
                )
                bw_theoretical = actual_bandwidth(
                    itf["modulation"], num_samples,
                    bandwidth_normalized=itf.get("bandwidth_normalized"),
                )
            wave = apply_freq_offset(wave, itf["freq_offset_normalized"])
            bw_obw99 = measure_obw99(wave)   # 可测量带宽（评估定义，频偏不影响带宽）
            arr = expand_to_array(
                wave, itf["doa_degree"], num_antennas,
                receiver["antenna_spacing_m"], receiver["wavelength_m"],
                gain_error=array_gain_error, phase_error_rad=array_phase_error_rad,
            )
            arr_power = float(np.mean(np.abs(arr) ** 2))
            if arr_power > 0:
                gain = np.sqrt((target_power * 10 ** (itf["power_ratio_db"] / 10.0)) / arr_power)
                mixer = mixer + gain * arr
            category = _auto_label(
                itf, target_bw_obw99,
                power_threshold_db=blocking_power_threshold_db,
                ratio_threshold=blocking_ratio_threshold,
            )
            sources_meta.append({
                "source_id": i + 1,
                "role": "interferer",
                "modulation": itf["modulation"],
                "display_name": itf["display_name"],
                "waveform_type": itf.get("waveform_type"),
                "power_ratio_db": float(itf["power_ratio_db"]),
                "inr_db": float(itf["power_ratio_db"]) + float(snr_db),
                "freq_offset_normalized": float(itf["freq_offset_normalized"]),
                "freq_offset_hz": _normalized_to_hz(itf["freq_offset_normalized"], fs),
                "doa_degree": float(itf["doa_degree"]),
                "bandwidth_normalized": bw_obw99,          # 99% OBW（评估定义）
                "bandwidth_theoretical": bw_theoretical,   # 理论占用带宽（参考）
                "bandwidth_fold": float(itf.get("bandwidth_fold", bw_theoretical)),
                "bandwidth_hz": _normalized_to_hz(bw_obw99, fs),
                "v2_category": category,
                "is_pulse": bool(itf["is_pulse"]),
                "is_modulated": bool(itf.get("is_modulated", True)),
                "waveform_params": dict(itf.get("wave_kw") or {}),
            })
    else:
        mixer = target_array.copy()

    # —— 接收机 LNA 饱和（真实 blocking 效应）——
    # 有 blocking 干扰（强功率 + 超出目标带宽）时，混合信号峰值超过线性动态范围，
    # Rapp 软限幅会把强干扰压缩，同时抑制/扭曲目标信号。
    blocking_present = any(
        m.get("v2_category") == "blocking" for m in sources_meta[1:]
    )
    lna_applied = False
    a_sat = 0.0
    if blocking_present:
        target_rms = float(np.sqrt(target_power))
        a_sat = target_rms * 10.0 ** (lna_saturation_db / 20.0)
        if a_sat > 0.0:
            mixer = apply_lna_saturation(mixer, a_sat, p=lna_p)
            lna_applied = True

    # SNR 定义为「目标功率 / 噪声功率」（干扰不参与 SNR 计算）
    iq = add_awgn(mixer, snr_db, rng=rng, ref_power=target_power)

    metadata = {
        "sample_id": "",
        "num_sources": len(sources_meta),
        "num_samples": num_samples,
        "snr_db": float(snr_db),
        "bandwidth_definition": "obw99",   # 带宽标签统一定义：99% 占用带宽（实测）
        "receiver": {
            "center_frequency_hz": receiver["center_frequency_hz"],
            "sampling_rate_hz": receiver["sampling_rate_hz"],
            "wavelength_m": receiver["wavelength_m"],
            "antenna_spacing_m": receiver["antenna_spacing_m"],
        },
        "lna_saturation": {
            "applied": lna_applied,
            "model": "rapp",
            "a_sat": float(a_sat) if lna_applied else None,
            "p": float(lna_p),
            "saturation_db": float(lna_saturation_db),
        },
        "array_mismatch": {
            "gain_error_std": float(array_gain_error_std),
            "phase_error_std_deg": float(array_phase_error_std_deg),
        },
        "sources": sources_meta,
    }
    return iq.astype(np.complex128), metadata


def split_metadata(metadata):
    """将完整元数据拆分为「Agent 观察」与「评估真值」两份。

    observation  (Agent 可见的先验)：接收机参数、目标调制类型、样本长度。
                不含任何干扰信息、SNR、LNA 触发状态等真值。
    ground_truth (评估端专用)：全部真值（源数、各源类别/CFO/DOA/带宽/INR、
                LNA 饱和状态等）。

    注意：metadata 深拷贝进 ground_truth，二者互不引用。
    """
    observation = {
        "sample_id": metadata.get("sample_id", ""),
        "num_samples": metadata.get("num_samples", 1024),
        "receiver": {
            "center_frequency_hz": metadata["receiver"]["center_frequency_hz"],
            "sampling_rate_hz": metadata["receiver"]["sampling_rate_hz"],
            "wavelength_m": metadata["receiver"]["wavelength_m"],
            "antenna_spacing_m": metadata["receiver"]["antenna_spacing_m"],
        },
        "target_modulation": metadata["sources"][0]["modulation"],
        # 设备规格（标准差），不含样本级失配向量
        "array_mismatch_config": dict(metadata.get("array_mismatch", {})),
    }
    ground_truth = json.loads(json.dumps(metadata))
    return observation, ground_truth


def _build_sample(base_seed, index, snr, src_count, num_antennas, num_samples,
                  modulation, interferer_type, **kwargs):
    """生成单个样本（模块级函数，供串行/多进程共用）。"""
    iq, meta = generate_signal_sample(
        num_sources=src_count,
        num_antennas=num_antennas,
        num_samples=num_samples,
        snr_db=snr,
        modulation=modulation,
        interferer_type=interferer_type,
        seed=base_seed + index,
        **kwargs,
    )
    meta["sample_id"] = f"sample_{index:05d}"
    return index, iq, meta


def _build_sample_from_args(args):
    """进程池工作入口（模块级、可 pickle）。"""
    (base_seed, index, snr, src_count, num_antennas, num_samples,
     modulation, interferer_type, kwargs) = args
    return _build_sample(base_seed, index, snr, src_count, num_antennas, num_samples,
                         modulation, interferer_type, **kwargs)


def generate_dataset(
    output_dir="dataset_v2",
    count=100,
    num_sources="random",
    snr_range=None,
    fixed_snr=None,
    seed=None,
    modulation="random",
    interferer_type="random",
    num_antennas=DEFAULT_NUM_ANTENNAS,
    num_samples=DEFAULT_NUM_SAMPLES,
    num_workers=1,
    progress=False,
    **kwargs,
):
    """批量生成数据集。

    输出三个 JSON：
        metadata.json      完整记录（含真值，仅供内部使用/调试，不得提供给 Agent）
        observations.json  Agent 可见先验（接收机参数 + 目标调制，无任何真值）
        ground_truth.json  评估端专用真值（源数、类别、各源参数、LNA/INR 等）

    num_workers > 1 时用进程池并行生成（每样本独立种子，结果与串行逐位一致）。
    progress=True 时显示 tqdm 进度条（需安装 tqdm，缺失时静默）。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    observations = []
    ground_truths = []
    base_seed = 0 if seed is None else int(seed)
    count = int(count)

    if progress:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
    else:
        tqdm = None

    # 每个样本的 SNR 与源数由该样本种子决定 → 打包进工作参数
    def _sample_args(i):
        rng = np.random.default_rng(base_seed + i)
        if fixed_snr is not None:
            snr = float(fixed_snr)
        elif snr_range is not None:
            snr = float(rng.uniform(float(snr_range[0]), float(snr_range[1])))
        else:
            snr = 10.0
        src_count = "random" if num_sources == "random" else int(num_sources)
        return (base_seed, i, snr, src_count, int(num_antennas), int(num_samples),
                modulation, interferer_type, kwargs)

    def _store(i, iq, meta):
        file_name = f"sample_{i:05d}.npy"
        np.save(out / file_name, iq)
        obs, gt = split_metadata(meta)
        records.append({"file": file_name, "metadata": meta})
        observations.append({"file": file_name, "observation": obs})
        ground_truths.append({"file": file_name, "ground_truth": gt})

    if int(num_workers) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            it = executor.map(
                _build_sample_from_args, (_sample_args(i) for i in range(count))
            )
            if tqdm:
                it = tqdm(it, total=count, desc="Generating", unit="sample")
            for i, iq, meta in it:
                _store(i, iq, meta)
    else:
        rng_iter = range(count)
        if tqdm:
            rng_iter = tqdm(rng_iter, total=count, desc="Generating", unit="sample")
        for i in rng_iter:
            _, iq, meta = _build_sample_from_args(_sample_args(i))
            _store(i, iq, meta)

    (out / "metadata.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "observations.json").write_text(
        json.dumps(observations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "ground_truth.json").write_text(
        json.dumps(ground_truths, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return records
