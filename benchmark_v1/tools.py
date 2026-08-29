"""Signal-analysis tools with modulation-aware anomaly detection for LLM Agents."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

MODULATION_EXPECTED = {
    "OOK": {
        "papr_db_range": (8.0, 12.0),
        "spectral_flatness_range": (0.2, 0.5),
        "occupied_bandwidth_range": (0.1, 0.4),
        "drift_std_range": (0.0, 0.05),
    },
    "QPSK": {
        "papr_db_range": (3.0, 5.0),
        "spectral_flatness_range": (0.3, 0.6),
        "occupied_bandwidth_range": (0.15, 0.4),
        "drift_std_range": (0.0, 0.04),
    },
    "BPSK": {
        "papr_db_range": (3.0, 5.0),
        "spectral_flatness_range": (0.3, 0.6),
        "occupied_bandwidth_range": (0.15, 0.4),
        "drift_std_range": (0.0, 0.04),
    },
    "16QAM": {
        "papr_db_range": (4.0, 7.0),
        "spectral_flatness_range": (0.4, 0.7),
        "occupied_bandwidth_range": (0.2, 0.5),
        "drift_std_range": (0.0, 0.04),
    },
    "64QAM": {
        "papr_db_range": (5.0, 8.0),
        "spectral_flatness_range": (0.5, 0.8),
        "occupied_bandwidth_range": (0.25, 0.5),
        "drift_std_range": (0.0, 0.04),
    },
    "GFSK": {
        "papr_db_range": (2.0, 4.0),
        "spectral_flatness_range": (0.4, 0.7),
        "occupied_bandwidth_range": (0.2, 0.6),
        "drift_std_range": (0.0, 0.05),
    },
    "OFDM": {
        "papr_db_range": (8.0, 12.0),
        "spectral_flatness_range": (0.6, 0.9),
        "occupied_bandwidth_range": (0.5, 0.9),
        "drift_std_range": (0.0, 0.05),
    },
    "FHSS": {
        "papr_db_range": (4.0, 7.0),
        "spectral_flatness_range": (0.3, 0.6),
        "occupied_bandwidth_range": (0.3, 0.7),
        "drift_std_range": (0.05, 0.15),
    },
    "LFM": {
        "papr_db_range": (2.0, 5.0),
        "spectral_flatness_range": (0.2, 0.5),
        "occupied_bandwidth_range": (0.3, 0.7),
        "drift_std_range": (0.06, 0.2),
    },
}

CLEAN_THRESHOLD = 0.8


def _load_iq(sample_path: str | Path) -> np.ndarray:
    data = np.asarray(np.load(sample_path))
    if data.ndim != 1:
        data = data.reshape(-1)
    if data.size == 0:
        raise ValueError("IQ sample is empty")
    return data.astype(np.complex128, copy=False)


def _get_modulation_profile(mod_type: str) -> dict[str, Any] | None:
    normalized = str(mod_type).upper().strip()
    return MODULATION_EXPECTED.get(normalized, None)


def _calculate_anomaly_score(value: float, expected_range: tuple[float, float]) -> float:
    low, high = expected_range
    if low <= value <= high:
        return 0.0
    elif value > high:
        return value - high
    else:
        return value - low


def _format_anomaly(score: float, feature_name: str, expected_range: tuple[float, float]) -> str:
    if score == 0:
        return f"{feature_name}: NORMAL (range {expected_range[0]:.1f}-{expected_range[1]:.1f})"
    elif score > 0:
        return f"{feature_name}: HIGH (deviation +{score:.2f}, expected {expected_range[0]:.1f}-{expected_range[1]:.1f})"
    else:
        return f"{feature_name}: LOW (deviation {score:.2f}, expected {expected_range[0]:.1f}-{expected_range[1]:.1f})"


def analyze_spectrum(sample_path: str, mod_type: str = "unknown") -> dict[str, Any]:
    iq = _load_iq(sample_path)
    n = iq.size
    eps = np.finfo(float).tiny

    spectrum = np.fft.fftshift(np.fft.fft(iq - np.mean(iq)))
    power = np.abs(spectrum) ** 2 / max(n, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    peak_idx = int(np.argmax(power))

    total = float(np.sum(power))
    if total > 0:
        cdf = np.cumsum(power) / total
        lo = float(freq[np.searchsorted(cdf, 0.005)])
        hi = float(freq[min(np.searchsorted(cdf, 0.995), n - 1)])
        bandwidth = max(0.0, hi - lo)
    else:
        bandwidth = 0.0

    geom_mean = np.exp(np.mean(np.log(power + eps)))
    arith_mean = np.mean(power) + eps
    spectral_flatness = float(geom_mean / arith_mean)

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

    peak_to_avg_db = float(10.0 * np.log10(max(power[peak_idx], eps) / max(np.mean(power), eps)))

    profile = _get_modulation_profile(mod_type)
    anomaly_report = []
    spectral_insight = ""
    is_signal_clean = False
    total_anomaly = -1.0

    if profile:
        papr_anomaly = _calculate_anomaly_score(peak_to_avg_db, profile["papr_db_range"])
        flatness_anomaly = _calculate_anomaly_score(spectral_flatness, profile["spectral_flatness_range"])
        bandwidth_anomaly = _calculate_anomaly_score(bandwidth, profile["occupied_bandwidth_range"])
        drift_anomaly = _calculate_anomaly_score(freq_drift_std, profile["drift_std_range"])

        anomaly_report.append(_format_anomaly(papr_anomaly, "Peak-to-average ratio", profile["papr_db_range"]))
        anomaly_report.append(_format_anomaly(flatness_anomaly, "Spectral flatness", profile["spectral_flatness_range"]))
        anomaly_report.append(_format_anomaly(bandwidth_anomaly, "Occupied bandwidth", profile["occupied_bandwidth_range"]))
        anomaly_report.append(_format_anomaly(drift_anomaly, "Frequency drift", profile["drift_std_range"]))

        papr_width = max(profile["papr_db_range"][1] - profile["papr_db_range"][0], 1.0)
        flatness_width = max(profile["spectral_flatness_range"][1] - profile["spectral_flatness_range"][0], 0.1)
        bandwidth_width = max(profile["occupied_bandwidth_range"][1] - profile["occupied_bandwidth_range"][0], 0.1)
        drift_width = max(profile["drift_std_range"][1] - profile["drift_std_range"][0], 0.01)

        normalized_anomalies = [
            papr_anomaly / papr_width,
            flatness_anomaly / flatness_width,
            bandwidth_anomaly / bandwidth_width,
            drift_anomaly / drift_width,
        ]
        total_anomaly = float(np.sqrt(sum(a ** 2 for a in normalized_anomalies)))
        is_signal_clean = bool(total_anomaly < CLEAN_THRESHOLD)

        # 构建描述性洞察（不再给出分类建议）
        if is_signal_clean:
            spectral_insight = f"Frequency-domain features closely match the {mod_type} profile (anomaly distance {total_anomaly:.3f}). No strong spectral evidence of jamming."
        elif drift_anomaly > 0.08 and bandwidth_anomaly > 0.1:
            spectral_insight = f"Frequency drift ({drift_anomaly:.2f}) and bandwidth ({bandwidth_anomaly:.2f}) are significantly above {mod_type} expectations. This could indicate swept or broadband interference."
        elif flatness_anomaly > 0.2 and bandwidth_anomaly > 0.15:
            spectral_insight = f"Spectral flatness ({flatness_anomaly:.2f}) and bandwidth ({bandwidth_anomaly:.2f}) far exceed {mod_type} normal range. Broadband jamming is likely."
        elif papr_anomaly > 2.5:
            spectral_insight = f"Peak power is abnormally high (deviation {papr_anomaly:.2f}). This may be a single-tone jammer or a pulse—check time domain to distinguish."
        else:
            spectral_insight = f"Mild anomalies detected (total distance {total_anomaly:.3f}). Time-domain analysis is recommended for confirmation."
    else:
        anomaly_report = [
            f"PAPR: {peak_to_avg_db:.2f} dB (unknown reference)",
            f"Spectral flatness: {spectral_flatness:.3f} (unknown reference)",
            f"Bandwidth: {bandwidth:.3f} (unknown reference)",
            f"Drift std: {freq_drift_std:.4f} (unknown reference)",
        ]
        spectral_insight = f"Unknown modulation '{mod_type}', providing raw measurements only."

    return {
        "num_samples": int(n),
        "peak_frequency_normalized": float(freq[peak_idx]),
        "peak_magnitude_db": float(20.0 * np.log10(max(np.abs(spectrum[peak_idx]) / n, eps))),
        "peak_to_average_power_ratio_db": float(peak_to_avg_db),
        "occupied_bandwidth_normalized": float(bandwidth),
        "spectral_flatness": float(spectral_flatness),
        "time_frequency_drift_std": float(freq_drift_std),
        "modulation_type": str(mod_type) if mod_type != "unknown" else "unknown",
        "anomaly_report_spectral": "\n".join(anomaly_report),
        "spectral_insight": str(spectral_insight),
        "is_signal_clean": bool(is_signal_clean),
        "total_anomaly_distance": float(total_anomaly),
        "clean_threshold": float(CLEAN_THRESHOLD),
        "raw_peak_to_average_db": float(peak_to_avg_db),
        "raw_spectral_flatness": float(spectral_flatness),
        "raw_bandwidth": float(bandwidth),
        "raw_drift_std": float(freq_drift_std),
    }


def detect_time_domain_features(sample_path: str, mod_type: str = "unknown") -> dict[str, Any]:
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

    profile = _get_modulation_profile(mod_type)
    anomaly_report = []
    time_insight = ""
    is_time_clean = False

    if profile:
        papr_anomaly = _calculate_anomaly_score(papr_db, profile["papr_db_range"])
        anomaly_report.append(_format_anomaly(papr_anomaly, "PAPR", profile["papr_db_range"]))

        if mod_type == "OOK":
            if duty_cycle > 0.6:
                time_insight = f"OOK duty cycle unusually high ({duty_cycle:.2f})—possible interference."
                is_time_clean = False
            else:
                time_insight = f"OOK duty cycle normal ({duty_cycle:.2f})."
                is_time_clean = True
        else:
            if papr_anomaly > 2.5 and duty_cycle < 0.35:
                time_insight = f"PAPR is abnormally high (deviation {papr_anomaly:.2f}) with low duty cycle ({duty_cycle:.2f}). This strongly suggests pulse jamming."
                is_time_clean = False
            elif papr_anomaly > 1.5:
                time_insight = f"PAPR is elevated for {mod_type} (deviation {papr_anomaly:.2f})—check for interference."
                is_time_clean = False
            else:
                time_insight = f"Time-domain features are within normal range for {mod_type}."
                is_time_clean = True
    else:
        anomaly_report = [f"PAPR: {papr_db:.2f} dB (unknown reference)"]
        time_insight = f"Unknown modulation '{mod_type}', providing raw measurements only."

    return {
        "num_samples": int(iq.size),
        "papr_db": float(papr_db),
        "pulse_duty_cycle": float(duty_cycle),
        "active_sample_count": int(np.sum(active)),
        "modulation_type": str(mod_type) if mod_type != "unknown" else "unknown",
        "anomaly_report_time": "\n".join(anomaly_report),
        "time_insight": str(time_insight),
        "is_time_clean": bool(is_time_clean),
        "raw_papr_db": float(papr_db),
        "raw_duty_cycle": float(duty_cycle),
    }


TOOL_FUNCTIONS = {
    "analyze_spectrum": analyze_spectrum,
    "detect_time_domain_features": detect_time_domain_features,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_spectrum",
            "description": "Analyze FFT spectral features with modulation-aware anomaly detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "mod_type": {"type": "string", "description": "Modulation type for anomaly detection."}
                },
                "required": ["sample_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_time_domain_features",
            "description": "Extract PAPR and pulse duty cycle with modulation-aware anomaly detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "mod_type": {"type": "string", "description": "Modulation type for anomaly detection."}
                },
                "required": ["sample_path"]
            }
        }
    },
]
