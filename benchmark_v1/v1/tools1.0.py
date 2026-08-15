"""Signal-analysis tools exposed to an evaluating LLM agent.

The functions in this module deliberately return JSON-serialisable dictionaries so
that they can be used both locally and as OpenAI/Qwen function-calling tools.
"""

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
    """Calculate FFT-based spectral features for a complex IQ sample.

    Frequencies are normalized to the sample rate (cycles/sample), in the range
    ``[-0.5, 0.5)``. Bandwidth is the occupied 99%-power bandwidth.
    """
    iq = _load_iq(sample_path)
    n = iq.size
    spectrum = np.fft.fftshift(np.fft.fft(iq - np.mean(iq)))
    power = np.abs(spectrum) ** 2 / max(n, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    peak_idx = int(np.argmax(power))
    eps = np.finfo(float).tiny
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(spectrum) / n, eps))
    total = float(np.sum(power))
    if total > 0:
        cdf = np.cumsum(power) / total
        lo = float(freq[np.searchsorted(cdf, 0.005)])
        hi = float(freq[min(np.searchsorted(cdf, 0.995), n - 1)])
        bandwidth = max(0.0, hi - lo)
    else:
        bandwidth = 0.0
    noise_floor = float(np.median(power))
    signal_power = float(power[peak_idx])
    snr_db = 10.0 * np.log10(max(signal_power, eps) / max(noise_floor, eps))
    return {
        "num_samples": n,
        "peak_frequency_normalized": float(freq[peak_idx]),
        "peak_magnitude_db": float(magnitude_db[peak_idx]),
        "occupied_bandwidth_normalized": float(bandwidth),
        "average_magnitude_db": float(20.0 * np.log10(max(float(np.mean(np.abs(iq))), eps))),
        "estimated_snr_db": float(snr_db),
    }


def detect_time_domain_features(sample_path: str) -> dict[str, Any]:
    """Extract PAPR and pulse-activity features from an IQ sample.

    Duty cycle is the fraction of samples above a robust threshold (median plus
    three median-absolute-deviations of instantaneous power).
    """
    iq = _load_iq(sample_path)
    power = np.abs(iq) ** 2
    mean_power = float(np.mean(power))
    eps = np.finfo(float).tiny
    papr_db = float(10.0 * np.log10(max(float(np.max(power)), eps) / max(mean_power, eps)))
    median = float(np.median(power))
    mad = float(np.median(np.abs(power - median)))
    threshold = median + 3.0 * max(mad, eps)
    active = power > threshold
    return {
        "num_samples": int(iq.size),
        "papr_db": papr_db,
        "pulse_duty_cycle": float(np.mean(active)),
        "active_sample_count": int(np.sum(active)),
        "power_mean": mean_power,
        "power_peak": float(np.max(power)),
        "power_threshold": threshold,
    }


TOOL_FUNCTIONS = {
    "analyze_spectrum": analyze_spectrum,
    "detect_time_domain_features": detect_time_domain_features,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "analyze_spectrum", "description": "Analyze FFT spectral features of an IQ .npy sample.", "parameters": {"type": "object", "properties": {"sample_path": {"type": "string"}}, "required": ["sample_path"]}}},
    {"type": "function", "function": {"name": "detect_time_domain_features", "description": "Measure PAPR and pulse duty cycle of an IQ .npy sample.", "parameters": {"type": "object", "properties": {"sample_path": {"type": "string"}}, "required": ["sample_path"]}}},
]
