"""Signal-analysis tools with modulation-aware anomaly detection for LLM Agents."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

# ==================== 调制预期特征表（Modulation Profiles）====================
MODULATION_EXPECTED = {
    "OOK": {
        "papr_db_range": (8.0, 12.0),
        "spectral_flatness_range": (0.2, 0.5),
        "occupied_bandwidth_range": (0.1, 0.4),
        "drift_std_range": (0.0, 0.05),
        "description": "On-Off Keying: high PAPR by nature (switching), narrow-to-medium bandwidth"
    },
    "QPSK": {
        "papr_db_range": (3.0, 5.0),
        "spectral_flatness_range": (0.3, 0.6),
        "occupied_bandwidth_range": (0.15, 0.4),
        "drift_std_range": (0.0, 0.04),
        "description": "Quadrature Phase Shift Keying: constant envelope, smooth PAPR"
    },
    "BPSK": {
        "papr_db_range": (3.0, 5.0),
        "spectral_flatness_range": (0.3, 0.6),
        "occupied_bandwidth_range": (0.15, 0.4),
        "drift_std_range": (0.0, 0.04),
        "description": "Binary Phase Shift Keying: constant envelope, smooth PAPR"
    },
    "16QAM": {
        "papr_db_range": (4.0, 7.0),
        "spectral_flatness_range": (0.4, 0.7),
        "occupied_bandwidth_range": (0.2, 0.5),
        "drift_std_range": (0.0, 0.04),
        "description": "16-QAM: moderate PAPR"
    },
    "64QAM": {
        "papr_db_range": (5.0, 8.0),
        "spectral_flatness_range": (0.5, 0.8),
        "occupied_bandwidth_range": (0.25, 0.5),
        "drift_std_range": (0.0, 0.04),
        "description": "64-QAM: moderate-high PAPR"
    },
    "GFSK": {
        "papr_db_range": (2.0, 4.0),
        "spectral_flatness_range": (0.4, 0.7),
        "occupied_bandwidth_range": (0.2, 0.6),
        "drift_std_range": (0.0, 0.05),
        "description": "Gaussian Frequency Shift Keying: very smooth, low PAPR"
    },
    "OFDM": {
        "papr_db_range": (8.0, 12.0),
        "spectral_flatness_range": (0.6, 0.9),  # NATURALLY FLAT!
        "occupied_bandwidth_range": (0.5, 0.9),  # NATURALLY WIDE!
        "drift_std_range": (0.0, 0.05),
        "description": "OFDM: naturally flat spectrum (DO NOT mistake for broadband jamming)"
    },
    "FHSS": {
        "papr_db_range": (4.0, 7.0),
        "spectral_flatness_range": (0.3, 0.6),
        "occupied_bandwidth_range": (0.3, 0.7),
        "drift_std_range": (0.05, 0.15),  # NATURALLY HAS DRIFT!
        "description": "Frequency Hopping: naturally has frequency variations (DO NOT mistake for swept jamming)"
    },
    "LFM": {
        "papr_db_range": (2.0, 5.0),
        "spectral_flatness_range": (0.2, 0.5),
        "occupied_bandwidth_range": (0.3, 0.7),
        "drift_std_range": (0.06, 0.2),  # NATURALLY HAS LARGE DRIFT!
        "description": "Linear Frequency Modulation: naturally has large frequency sweep (DO NOT mistake for swept jamming)"
    },
}


def _load_iq(sample_path: str | Path) -> np.ndarray:
    data = np.asarray(np.load(sample_path))
    if data.ndim != 1:
        data = data.reshape(-1)
    if data.size == 0:
        raise ValueError("IQ sample is empty")
    return data.astype(np.complex128, copy=False)


def _get_modulation_profile(mod_type: str) -> dict[str, Any] | None:
    """Get the expected profile for a modulation type."""
    # 处理大小写和可能的变体
    normalized = str(mod_type).upper().strip()
    return MODULATION_EXPECTED.get(normalized, None)


def _calculate_anomaly_score(value: float, expected_range: tuple[float, float]) -> float:
    """
    Calculate how far the value deviates from the expected range.
    Returns:
        > 0: Above expected range (positive anomaly)
        < 0: Below expected range (negative anomaly)
        = 0: Within expected range
    """
    low, high = expected_range
    if low <= value <= high:
        return 0.0
    elif value > high:
        return value - high
    else:  # value < low
        return value - low  # negative value indicates below range


def _format_anomaly(score: float, feature_name: str, expected_range: tuple[float, float]) -> str:
    """Format anomaly score into human-readable text."""
    if score == 0:
        return f"{feature_name}: NORMAL (within expected range {expected_range[0]:.1f}-{expected_range[1]:.1f})"
    elif score > 0:
        return f"{feature_name}: ⚠️ ABNORMALLY HIGH (deviation +{score:.2f}, expected {expected_range[0]:.1f}-{expected_range[1]:.1f})"
    else:
        return f"{feature_name}: ⚠️ ABNORMALLY LOW (deviation {score:.2f}, expected {expected_range[0]:.1f}-{expected_range[1]:.1f})"


def analyze_spectrum(sample_path: str, mod_type: str = "unknown") -> dict[str, Any]:
    """
    Analyze FFT-based spectral features with modulation-aware anomaly detection.
    """
    iq = _load_iq(sample_path)
    n = iq.size
    eps = np.finfo(float).tiny

    # ----- 1. 基础 FFT 计算（不变）-----
    spectrum = np.fft.fftshift(np.fft.fft(iq - np.mean(iq)))
    power = np.abs(spectrum) ** 2 / max(n, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    peak_idx = int(np.argmax(power))

    # ----- 2. 99% 带宽 -----
    total = float(np.sum(power))
    if total > 0:
        cdf = np.cumsum(power) / total
        lo = float(freq[np.searchsorted(cdf, 0.005)])
        hi = float(freq[min(np.searchsorted(cdf, 0.995), n - 1)])
        bandwidth = max(0.0, hi - lo)
    else:
        bandwidth = 0.0

    # ----- 3. 谱平坦度 -----
    geom_mean = np.exp(np.mean(np.log(power + eps)))
    arith_mean = np.mean(power) + eps
    spectral_flatness = float(geom_mean / arith_mean)

    # ----- 4. 频偏漂移 -----
    num_segments = 8
    seg_size = n // num_segments
    segment_peaks = []
    if seg_size > 16:
        for i in range(num_segments):
            seg_iq = iq[i * seg_size : (i + 1) * seg_size]
            seg_spec = np.abs(np.fft.fft(seg_iq)) ** 2
            seg_freq = np.fft.fftfreq(seg_size, d=1.0)
            segment_peaks.append(float(seg_freq[np.argmax(seg_spec)]))
        freq_drift_std = float(np.std(segment_peaks))
    else:
        freq_drift_std = 0.0

    # ----- 5. 峰均比 -----
    peak_to_avg_db = float(10.0 * np.log10(max(power[peak_idx], eps) / max(np.mean(power), eps)))

    # ----- 6. 调制感知异常检测（新增！）-----
    profile = _get_modulation_profile(mod_type)

    anomaly_report = []
    spectral_insight = ""

    if profile:
        # 计算各项异常分数
        papr_anomaly = _calculate_anomaly_score(peak_to_avg_db, profile["papr_db_range"])
        flatness_anomaly = _calculate_anomaly_score(spectral_flatness, profile["spectral_flatness_range"])
        bandwidth_anomaly = _calculate_anomaly_score(bandwidth, profile["occupied_bandwidth_range"])
        drift_anomaly = _calculate_anomaly_score(freq_drift_std, profile["drift_std_range"])

        # 生成人类可读的报告
        anomaly_report.append(_format_anomaly(papr_anomaly, "Peak-to-average ratio", profile["papr_db_range"]))
        anomaly_report.append(_format_anomaly(flatness_anomaly, "Spectral flatness", profile["spectral_flatness_range"]))
        anomaly_report.append(_format_anomaly(bandwidth_anomaly, "Occupied bandwidth", profile["occupied_bandwidth_range"]))
        anomaly_report.append(_format_anomaly(drift_anomaly, "Frequency drift", profile["drift_std_range"]))

        # 生成综合频谱洞察
        if flatness_anomaly > 0.2 and bandwidth_anomaly > 0.1:
            spectral_insight = f"⚠️ SPECTRUM ANOMALY: Significantly flatter and wider than expected for {mod_type} (possible broadband jamming)."
        elif papr_anomaly > 3.0:
            spectral_insight = f"⚠️ SPECTRUM ANOMALY: Peak power abnormally high for {mod_type} (possible single-tone jamming)."
        elif drift_anomaly > 0.05:
            spectral_insight = f"⚠️ SPECTRUM ANOMALY: Frequency drift exceeds normal range for {mod_type} (possible swept jamming)."
        elif flatness_anomaly < -0.3 and papr_anomaly > 2.0:
            spectral_insight = f"⚠️ SPECTRUM ANOMALY: Unusually sharp peak for {mod_type} (possible single-tone jamming)."
        else:
            spectral_insight = f"✅ Spectrum features are within normal range for {mod_type} (no spectral jamming detected)."

    else:
        # 未知调制类型，只给原始数值
        anomaly_report = [
            f"Peak-to-average ratio: {peak_to_avg_db:.2f} dB (unknown modulation reference)",
            f"Spectral flatness: {spectral_flatness:.3f} (unknown modulation reference)",
            f"Occupied bandwidth: {bandwidth:.3f} (unknown modulation reference)",
            f"Frequency drift std: {freq_drift_std:.4f} (unknown modulation reference)",
        ]
        spectral_insight = f"⚠️ Unknown modulation '{mod_type}', providing raw measurements only."

    return {
        "num_samples": n,
        "peak_frequency_normalized": float(freq[peak_idx]),
        "peak_magnitude_db": float(20.0 * np.log10(max(np.abs(spectrum[peak_idx]) / n, eps))),
        "peak_to_average_power_ratio_db": peak_to_avg_db,
        "occupied_bandwidth_normalized": bandwidth,
        "spectral_flatness": spectral_flatness,
        "time_frequency_drift_std": freq_drift_std,
        # 新增字段：调制感知的异常报告
        "modulation_type": mod_type if mod_type != "unknown" else "unknown",
        "anomaly_report_spectral": "\n".join(anomaly_report),
        "spectral_insight": spectral_insight,
        # 保留原始数值以便模型参考（如果它想自己算）
        "raw_peak_to_average_db": peak_to_avg_db,
        "raw_spectral_flatness": spectral_flatness,
        "raw_bandwidth": bandwidth,
        "raw_drift_std": freq_drift_std,
    }


def detect_time_domain_features(sample_path: str, mod_type: str = "unknown") -> dict[str, Any]:
    """
    Extract PAPR and pulse-activity features with modulation-aware anomaly detection.
    """
    iq = _load_iq(sample_path)
    power = np.abs(iq) ** 2
    mean_power = float(np.mean(power))
    eps = np.finfo(float).tiny

    # ----- 1. 基础计算（不变）-----
    papr_db = float(10.0 * np.log10(max(float(np.max(power)), eps) / max(mean_power, eps)))

    median = float(np.median(power))
    mad = float(np.median(np.abs(power - median)))
    threshold = median + 3.0 * max(mad, eps)
    active = power > threshold
    duty_cycle = float(np.mean(active))

    # ----- 2. 调制感知异常检测（新增！）-----
    profile = _get_modulation_profile(mod_type)

    anomaly_report = []
    time_insight = ""

    if profile:
        # 计算 PAPR 异常
        papr_anomaly = _calculate_anomaly_score(papr_db, profile["papr_db_range"])

        # 生成报告
        anomaly_report.append(_format_anomaly(papr_anomaly, "PAPR", profile["papr_db_range"]))

        # 占空比分析（不同调制有不同的预期占空比）
        # 对于 OOK，占空比天然可能较低（取决于数据）
        if mod_type == "OOK":
            if duty_cycle > 0.6:
                time_insight = f"⚠️ TIME ANOMALY: OOK duty cycle unusually high ({duty_cycle:.2f})"
            else:
                time_insight = f"✅ OOK duty cycle normal ({duty_cycle:.2f})"
        elif papr_anomaly > 3.0 and duty_cycle < 0.35:
            time_insight = f"⚠️ TIME ANOMALY: PAPR abnormally high ({papr_db:.1f}dB) with low duty cycle ({duty_cycle:.2f}) - possible pulse jamming."
        elif papr_anomaly > 2.0:
            time_insight = f"⚠️ TIME ANOMALY: PAPR elevated for {mod_type} ({papr_db:.1f}dB) - check for potential interference."
        else:
            time_insight = f"✅ Time-domain features are within normal range for {mod_type} (no pulse jamming detected)."
    else:
        anomaly_report = [f"PAPR: {papr_db:.2f} dB (unknown modulation reference)"]
        time_insight = f"⚠️ Unknown modulation '{mod_type}', providing raw measurements only."

    return {
        "num_samples": int(iq.size),
        "papr_db": papr_db,
        "pulse_duty_cycle": duty_cycle,
        "active_sample_count": int(np.sum(active)),
        # 新增字段：调制感知的异常报告
        "modulation_type": mod_type if mod_type != "unknown" else "unknown",
        "anomaly_report_time": "\n".join(anomaly_report),
        "time_insight": time_insight,
        # 保留原始数值
        "raw_papr_db": papr_db,
        "raw_duty_cycle": duty_cycle,
    }


# 工具注册（保持不变，但为了兼容性，在调用时传入 mod_type）
TOOL_FUNCTIONS = {
    "analyze_spectrum": analyze_spectrum,
    "detect_time_domain_features": detect_time_domain_features,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_spectrum",
            "description": "Analyze FFT spectral features with modulation-aware anomaly detection. Returns both raw values and anomaly scores relative to the expected modulation profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "mod_type": {"type": "string", "description": "Modulation type (e.g., BPSK, QPSK, OFDM, LFM). Used for anomaly detection."}
                },
                "required": ["sample_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_time_domain_features",
            "description": "Extract PAPR and pulse duty cycle with modulation-aware anomaly detection. Returns both raw values and anomaly scores relative to the expected modulation profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "mod_type": {"type": "string", "description": "Modulation type (e.g., BPSK, QPSK, OFDM, LFM). Used for anomaly detection."}
                },
                "required": ["sample_path"]
            }
        }
    },
]
