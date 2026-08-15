"""Signal-analysis tools with enhanced DSP domain knowledge for LLM Agents."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np


def _load_iq(sample_path: str | Path) -> np.ndarray:
    data = np.asarray(np.load(sample_path))
    if data.ndim != 1:
        data = data.reshape(-1)
    if data.size == 0:
        raise ValueError("IQ sample is empty")
    return data.astype(np.complex128, copy=False)


def analyze_spectrum(sample_path: str) -> dict[str, Any]:
    """Calculate FFT-based spectral features, spectral flatness, and time-frequency drift."""
    iq = _load_iq(sample_path)
    n = iq.size
    
    # 1. 全局 FFT 频谱计算
    spectrum = np.fft.fftshift(np.fft.fft(iq - np.mean(iq)))
    power = np.abs(spectrum) ** 2 / max(n, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    peak_idx = int(np.argmax(power))
    eps = np.finfo(float).tiny
    
    # 2. 计算 99% 带宽
    total = float(np.sum(power))
    if total > 0:
        cdf = np.cumsum(power) / total
        lo = float(freq[np.searchsorted(cdf, 0.005)])
        hi = float(freq[min(np.searchsorted(cdf, 0.995), n - 1)])
        bandwidth = max(0.0, hi - lo)
    else:
        bandwidth = 0.0
        
    # 3. 计算谱平坦度 (Spectral Flatness) - 用于精准识别 broadband
    geom_mean = np.exp(np.mean(np.log(power + eps)))
    arith_mean = np.mean(power) + eps
    spectral_flatness = float(geom_mean / arith_mean)

    # 4. 短时 FFT 频偏漂移计算 - 用于精准识别 swept (扫频)
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

    # 5. 专家自动规则推理 (Expert Insights Generation)
    peak_to_avg_db = float(10.0 * np.log10(max(power[peak_idx], eps) / max(np.mean(power), eps)))
    
    insights = []
    if peak_to_avg_db > 15.0 and bandwidth < 0.1:
        insights.append("Strong Narrowband Peak detected (High chance of single_tone).")
    if freq_drift_std > 0.08:
        insights.append("Peak frequency shifts significantly over time (High chance of swept jamming).")
    if spectral_flatness > 0.6 and bandwidth > 0.5:
        insights.append("Spectrum is extremely flat across wide band (High chance of broadband jamming).")
    if not insights:
        insights.append("Spectrum appears standard or dominated by ambient noise.")

    return {
        "num_samples": n,
        "peak_frequency_normalized": float(freq[peak_idx]),
        "peak_magnitude_db": float(20.0 * np.log10(max(np.abs(spectrum[peak_idx]) / n, eps))),
        "peak_to_average_power_ratio_db": peak_to_avg_db,
        "occupied_bandwidth_normalized": float(bandwidth),
        "spectral_flatness": spectral_flatness,
        "time_frequency_drift_std": freq_drift_std,
        "expert_insight": " | ".join(insights)
    }


def detect_time_domain_features(sample_path: str) -> dict[str, Any]:
    """Extract PAPR and pulse-activity features with time-domain expert insights."""
    iq = _load_iq(sample_path)
    power = np.abs(iq) ** 2
    mean_power = float(np.mean(power))
    eps = np.finfo(float).tiny
    papr_db = float(10.0 * np.log10(max(float(np.max(power)), eps) / max(mean_power, eps)))
    
    median = float(np.median(power))
    mad = float(np.median(np.abs(power - median)))
    threshold = median + 3.0 * max(mad, eps)
    active = power > threshold
    duty_cycle = float(np.mean(active))

    # 专家自动规则推理
    if papr_db > 8.0 and duty_cycle < 0.35:
        insight = "High PAPR with low duty cycle observed (Strong sign of pulse jamming)."
    else:
        insight = "Time domain envelope is relatively continuous and smooth."

    return {
        "num_samples": int(iq.size),
        "papr_db": papr_db,
        "pulse_duty_cycle": duty_cycle,
        "active_sample_count": int(np.sum(active)),
        "expert_insight": insight
    }


TOOL_FUNCTIONS = {
    "analyze_spectrum": analyze_spectrum,
    "detect_time_domain_features": detect_time_domain_features,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "analyze_spectrum", "description": "Analyze FFT spectral features, spectral flatness, and peak frequency drift across time.", "parameters": {"type": "object", "properties": {"sample_path": {"type": "string"}}, "required": ["sample_path"]}}},
    {"type": "function", "function": {"name": "detect_time_domain_features", "description": "Measure PAPR and pulse duty cycle of an IQ .npy sample.", "parameters": {"type": "object", "properties": {"sample_path": {"type": "string"}}, "required": ["sample_path"]}}},
]