"""Deterministic numerical primitives for Task 6 evidence estimators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PEAK_TIE_RTOL = 1.0e-8


class EstimationError(ValueError):
    """Base class for local typed estimator failures."""


class MalformedEstimationEvidence(EstimationError):
    """Stored numerical evidence cannot safely support an estimate."""


class EstimationBindingError(EstimationError):
    """Evidence and candidate provenance are inconsistent."""


class AmbiguousEstimationEvidence(EstimationError):
    """A request needs candidate evidence to disambiguate source blocks."""


class InvalidModulationFeatureParameters(EstimationError):
    """A modulation-evidence selection or control is outside the MVP contract."""


class InsufficientFeatureSamples(EstimationError):
    """A selected IQ unit is too short for the requested feature contract."""


class CandidateEvidenceMismatch(EstimationBindingError):
    """A Task 4 candidate is inconsistent with its declared upstream evidence."""


class UnusablePowerEvidence(MalformedEstimationEvidence):
    """Stored spectral evidence cannot support relative-power integration."""


class NonFiniteEstimationResult(EstimationError):
    """Finite inputs produced a non-finite derived Task 7 result."""


def validate_spectrum(
    frequency_hz: Any,
    power_linear: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional frequency and linear-power arrays."""

    raw_frequency = np.asarray(frequency_hz)
    raw_power = np.asarray(power_linear)
    if np.iscomplexobj(raw_frequency) or np.iscomplexobj(raw_power):
        raise MalformedEstimationEvidence(
            "Spectrum frequency and linear-power evidence must be real-valued."
        )
    frequency = np.asarray(raw_frequency, dtype=np.float64)
    power = np.asarray(raw_power, dtype=np.float64)
    if (
        frequency.ndim != 1
        or power.ndim != 1
        or frequency.size != power.size
        or frequency.size < 2
    ):
        raise MalformedEstimationEvidence(
            "Spectrum evidence needs matching one-dimensional bins."
        )
    if (
        not np.isfinite(frequency).all()
        or not np.isfinite(power).all()
        or np.any(power < 0.0)
        or np.any(frequency[1:] <= frequency[:-1])
    ):
        raise MalformedEstimationEvidence(
            "Spectrum evidence must have finite increasing frequencies and non-negative power."
        )
    return frequency, power


def validate_time_frequency_block(
    frequency_hz: Any,
    power_linear: Any,
    frame_start_samples: Any,
    frame_center_samples: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate a TF block before any consumer indexes its two-dimensional shape."""

    raw_frequency = np.asarray(frequency_hz)
    raw_power = np.asarray(power_linear)
    raw_frame_starts = np.asarray(frame_start_samples)
    raw_frame_centers = np.asarray(frame_center_samples)
    if any(
        np.iscomplexobj(value)
        for value in (
            raw_frequency,
            raw_power,
            raw_frame_starts,
            raw_frame_centers,
        )
    ):
        raise MalformedEstimationEvidence(
            "Time-frequency axes and linear-power evidence must be real-valued."
        )
    frequency = np.asarray(raw_frequency, dtype=np.float64)
    power = np.asarray(raw_power, dtype=np.float64)
    frame_starts = np.asarray(raw_frame_starts, dtype=np.float64)
    frame_centers = np.asarray(raw_frame_centers, dtype=np.float64)

    if frequency.ndim != 1 or frequency.size < 2:
        raise MalformedEstimationEvidence(
            "Time-frequency evidence needs a one-dimensional frequency axis."
        )
    if (
        not np.isfinite(frequency).all()
        or np.any(frequency[1:] <= frequency[:-1])
    ):
        raise MalformedEstimationEvidence(
            "Time-frequency frequencies must be finite and strictly increasing."
        )
    if power.ndim != 2:
        raise MalformedEstimationEvidence(
            "Time-frequency linear power must be two-dimensional."
        )
    if power.shape[0] != frequency.size or power.shape[1] < 1:
        raise MalformedEstimationEvidence(
            "Time-frequency power shape must match both evidence axes."
        )
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise MalformedEstimationEvidence(
            "Time-frequency power must be finite and non-negative."
        )
    if (
        frame_starts.ndim != 1
        or frame_centers.ndim != 1
        or frame_starts.size != power.shape[1]
        or frame_centers.size != power.shape[1]
    ):
        raise MalformedEstimationEvidence(
            "Time-frequency frame coordinates must match the power frame count."
        )
    if (
        not np.isfinite(frame_starts).all()
        or not np.isfinite(frame_centers).all()
        or (
            frame_starts.size > 1
            and np.any(frame_starts[1:] <= frame_starts[:-1])
        )
        or (
            frame_centers.size > 1
            and np.any(frame_centers[1:] <= frame_centers[:-1])
        )
    ):
        raise MalformedEstimationEvidence(
            "Time-frequency frame coordinates must be finite and strictly increasing."
        )
    return frequency, power, frame_starts, frame_centers


def bin_edges(frequency_hz: Any, sample_rate_hz: float) -> np.ndarray:
    """Build midpoint bin support and clip it to true physical Nyquist."""

    frequency, _ = validate_spectrum(
        frequency_hz,
        np.ones(np.asarray(frequency_hz).size, dtype=np.float64),
    )
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise MalformedEstimationEvidence(
            "Sample rate must be finite and positive."
        )
    nyquist_hz = sample_rate_hz / 2.0
    if np.any(frequency < -nyquist_hz) or np.any(frequency > nyquist_hz):
        raise MalformedEstimationEvidence(
            "Frequency evidence exceeds true physical Nyquist support."
        )

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            midpoints = frequency[:-1] / 2.0 + frequency[1:] / 2.0
            result = np.empty(frequency.size + 1, dtype=np.float64)
            result[1:-1] = midpoints
            result[0] = frequency[0] - (midpoints[0] - frequency[0])
            result[-1] = frequency[-1] + (frequency[-1] - midpoints[-1])
            result = np.clip(result, -nyquist_hz, nyquist_hz)
    except FloatingPointError as exc:
        raise MalformedEstimationEvidence(
            "Frequency-bin support arithmetic overflowed."
        ) from exc
    if not np.isfinite(result).all() or np.any(result[1:] <= result[:-1]):
        raise MalformedEstimationEvidence(
            "Frequency-bin support is non-finite or unresolved."
        )
    return result


def tied_peak_indices(power_linear: Any) -> np.ndarray:
    """Return all positive maxima tied under the Task 6 tolerance."""

    power = np.asarray(power_linear, dtype=np.float64)
    maximum = float(np.max(power))
    if maximum <= 0.0:
        return np.empty(0, dtype=np.int64)
    tolerance = max(
        PEAK_TIE_RTOL * abs(maximum),
        np.finfo(np.float64).tiny,
    )
    return np.flatnonzero(np.abs(power - maximum) <= tolerance)


def linear_power_crossing(
    frequency_0: float,
    power_0: float,
    frequency_1: float,
    power_1: float,
    threshold: float,
) -> float | None:
    """Interpolate a crossing between adjacent bin centres in linear power."""

    delta_0 = power_0 - threshold
    delta_1 = power_1 - threshold
    if power_0 == power_1 or (
        (delta_0 > 0.0 and delta_1 > 0.0)
        or (delta_0 < 0.0 and delta_1 < 0.0)
    ):
        return None
    fraction = (threshold - power_0) / (power_1 - power_0)
    value = frequency_0 + fraction * (frequency_1 - frequency_0)
    return float(value) if math.isfinite(value) else None


def complex_modulation_evidence(
    samples: Any,
    *,
    sample_rate_hz: float,
    max_lag_samples: int,
) -> tuple[dict[str, dict[str, object]], dict[str, int], bool]:
    """Compute scalar-only descriptive evidence for one complex IQ unit.

    The return value contains the six public feature maps, bounded count
    evidence, and a zero-energy flag.  It deliberately performs no waveform or
    modulation classification.
    """

    z = np.asarray(samples)
    if z.ndim != 1 or not np.issubdtype(z.dtype, np.complexfloating):
        raise InvalidModulationFeatureParameters(
            "Feature input must be one-dimensional complex IQ."
        )
    if (
        not math.isfinite(sample_rate_hz)
        or sample_rate_hz <= 0.0
        or not np.isfinite(z.real).all()
        or not np.isfinite(z.imag).all()
    ):
        raise NonFiniteEstimationResult(
            "Feature input and sample rate must be finite."
        )

    amplitude = _stable_complex_amplitude(z)
    envelope, amplitude_scaled, amplitude_floor_scaled, zero_energy = (
        envelope_modulation_evidence(amplitude)
    )
    if zero_energy:
        unavailable = _unavailable_feature_map("ZERO_ENERGY")
        return (
            {
                "envelope": envelope,
                "phase_increment": dict(unavailable),
                "instantaneous_frequency": dict(unavailable),
                "cumulants": dict(unavailable),
                "spectral_shape": dict(unavailable),
                "lag_correlation_evidence": dict(unavailable),
            },
            {
                "feature_samples": int(z.size),
                "eligible_phase_pairs": 0,
                "spectral_bins": 0,
                "tested_lags": 0,
            },
            True,
        )

    scale = float(np.max(amplitude))
    normalized_z = _normalize_complex_by_positive_scale(z, scale)
    phase_increment, instantaneous_frequency, eligible_pairs = _phase_evidence(
        normalized_z,
        amplitude_scaled,
        amplitude_floor_scaled,
        sample_rate_hz,
    )
    cumulants = _cumulant_evidence(normalized_z)
    spectral_shape, spectral_bins = _spectral_shape_evidence(
        normalized_z,
        sample_rate_hz,
    )
    lag_evidence, tested_lags = _lag_correlation_evidence(
        normalized_z,
        sample_rate_hz,
        max_lag_samples,
    )
    return (
        {
            "envelope": envelope,
            "phase_increment": phase_increment,
            "instantaneous_frequency": instantaneous_frequency,
            "cumulants": cumulants,
            "spectral_shape": spectral_shape,
            "lag_correlation_evidence": lag_evidence,
        },
        {
            "feature_samples": int(z.size),
            "eligible_phase_pairs": eligible_pairs,
            "spectral_bins": spectral_bins,
            "tested_lags": tested_lags,
        },
        False,
    )


def envelope_modulation_evidence(
    amplitude: Any,
) -> tuple[dict[str, object], np.ndarray, float, bool]:
    """Compute envelope/PAPR evidence with scale-safe linear arithmetic."""

    values = np.asarray(amplitude, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise NonFiniteEstimationResult("Envelope evidence must be finite and one-dimensional.")
    if np.any(values < 0.0):
        raise InvalidModulationFeatureParameters("Envelope amplitude cannot be negative.")
    scale = float(np.max(values))
    if scale == 0.0:
        return (
            {
                "status": "not_applicable",
                "reason": "ZERO_ENERGY",
                "mean_amplitude": None,
                "median_amplitude": None,
                "robust_spread_amplitude": None,
                "envelope_cv_robust": None,
                "constant_envelope_score": None,
                "zero_envelope_fraction": 1.0,
                "papr_db": None,
            },
            np.zeros_like(values),
            np.finfo(np.float64).tiny,
            True,
        )

    scaled = values / scale
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            mean_scaled = float(np.mean(scaled))
            median_scaled = float(np.median(scaled))
            robust_spread_scaled = 1.4826 * float(
                np.median(np.abs(scaled - median_scaled))
            )
            mean_power_scaled = float(np.mean(scaled * scaled))
            rms_scaled = math.sqrt(mean_power_scaled)
            floor_scaled = max(
                1.0e-12 * rms_scaled,
                np.finfo(np.float64).tiny / scale,
            )
            mean_amplitude = scale * mean_scaled
            median_amplitude = scale * median_scaled
            robust_spread = scale * robust_spread_scaled
            envelope_cv = (
                robust_spread_scaled / median_scaled
                if median_scaled > floor_scaled
                else None
            )
            constant_score = (
                1.0 / (1.0 + envelope_cv)
                if envelope_cv is not None
                else None
            )
            zero_fraction = float(np.mean(scaled <= floor_scaled))
            papr_db = 10.0 * math.log10(
                float(np.max(scaled * scaled)) / mean_power_scaled
            )
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
            raise NonFiniteEstimationResult(
                "Envelope feature arithmetic is non-finite."
            ) from exc
    _require_finite_result(
        scaled,
        mean_amplitude,
        median_amplitude,
        robust_spread,
        zero_fraction,
        papr_db,
    )
    if envelope_cv is not None:
        _require_finite_result(envelope_cv, constant_score)
    return (
        {
            "status": "success" if envelope_cv is not None else "partial",
            "reason": None if envelope_cv is not None else "ZERO_MEDIAN_ENVELOPE",
            "mean_amplitude": float(mean_amplitude),
            "median_amplitude": float(median_amplitude),
            "robust_spread_amplitude": float(robust_spread),
            "envelope_cv_robust": None if envelope_cv is None else float(envelope_cv),
            "constant_envelope_score": (
                None if constant_score is None else float(constant_score)
            ),
            "zero_envelope_fraction": zero_fraction,
            "papr_db": float(papr_db),
        },
        scaled,
        floor_scaled,
        False,
    )


def noncoherent_mean_amplitude(channels: Any) -> np.ndarray:
    """Return ``sqrt(mean_c(abs(x_c[n])**2))`` without coherent averaging."""

    values = np.asarray(channels)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or not np.issubdtype(values.dtype, np.complexfloating)
        or not np.isfinite(values.real).all()
        or not np.isfinite(values.imag).all()
    ):
        raise InvalidModulationFeatureParameters(
            "Mean-power evidence requires finite two-dimensional complex IQ."
        )
    component_scale = float(
        max(float(np.max(np.abs(values.real))), float(np.max(np.abs(values.imag))))
    )
    if component_scale == 0.0:
        return np.zeros(values.shape[1], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            real = np.asarray(values.real, dtype=np.float64) / component_scale
            imag = np.asarray(values.imag, dtype=np.float64) / component_scale
            amplitude = component_scale * np.sqrt(np.mean(real * real + imag * imag, axis=0))
    except FloatingPointError as exc:
        raise NonFiniteEstimationResult(
            "Non-coherent mean-power arithmetic is non-finite."
        ) from exc
    _require_finite_result(amplitude)
    return amplitude


def _stable_complex_amplitude(values: np.ndarray) -> np.ndarray:
    component_scale = float(
        max(float(np.max(np.abs(values.real))), float(np.max(np.abs(values.imag))))
    )
    if component_scale == 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            amplitude = component_scale * np.sqrt(
                (np.asarray(values.real, dtype=np.float64) / component_scale) ** 2
                + (np.asarray(values.imag, dtype=np.float64) / component_scale) ** 2
            )
    except FloatingPointError as exc:
        raise NonFiniteEstimationResult(
            "Complex-envelope arithmetic is non-finite."
        ) from exc
    _require_finite_result(amplitude)
    return amplitude


def _normalize_complex_by_positive_scale(
    values: np.ndarray,
    scale: float,
) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0.0:
        raise NonFiniteEstimationResult(
            "Complex normalization requires a finite positive scale."
        )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            real = np.asarray(values.real, dtype=np.float64) / scale
            imag = np.asarray(values.imag, dtype=np.float64) / scale
            normalized = np.asarray(real + 1j * imag, dtype=np.complex128)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise NonFiniteEstimationResult(
            "Complex normalization arithmetic is non-finite."
        ) from exc
    _require_finite_result(normalized)
    return normalized


def _phase_evidence(
    normalized_z: np.ndarray,
    amplitude_scaled: np.ndarray,
    amplitude_floor_scaled: float,
    sample_rate_hz: float,
) -> tuple[dict[str, object], dict[str, object], int]:
    eligible = (amplitude_scaled[1:] > amplitude_floor_scaled) & (
        amplitude_scaled[:-1] > amplitude_floor_scaled
    )
    eligible_count = int(np.count_nonzero(eligible))
    if eligible_count < 32:
        unavailable = _unavailable_feature_map("INSUFFICIENT_ELIGIBLE_PHASE_PAIRS")
        return dict(unavailable), dict(unavailable), eligible_count
    dphi = np.angle(normalized_z[1:] * np.conj(normalized_z[:-1]))[eligible]
    phasor_mean = np.mean(np.exp(1j * dphi))
    resultant = float(np.clip(abs(phasor_mean), 0.0, 1.0))
    circular_mean = float(np.angle(phasor_mean))
    circular_variance = 1.0 - resultant
    circular_std = math.sqrt(
        max(0.0, -2.0 * math.log(max(resultant, np.finfo(np.float64).tiny)))
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            instantaneous = np.asarray(sample_rate_hz, dtype=np.float64) * dphi / (
                2.0 * math.pi
            )
            median_hz = float(np.median(instantaneous))
            spread_hz = 1.4826 * float(
                np.median(np.abs(instantaneous - median_hz))
            )
            percentile_05, percentile_95 = np.percentile(
                instantaneous,
                [5.0, 95.0],
            )
    except (FloatingPointError, OverflowError) as exc:
        raise NonFiniteEstimationResult(
            "Instantaneous-frequency arithmetic is non-finite."
        ) from exc
    _require_finite_result(
        dphi,
        resultant,
        circular_mean,
        circular_variance,
        circular_std,
        instantaneous,
        median_hz,
        spread_hz,
        percentile_05,
        percentile_95,
    )
    return (
        {
            "status": "success",
            "reason": None,
            "circular_mean_rad": circular_mean,
            "mean_resultant_length": resultant,
            "circular_variance": circular_variance,
            "circular_std_rad": circular_std,
        },
        {
            "status": "success",
            "reason": None,
            "median_hz": median_hz,
            "robust_spread_hz": spread_hz,
            "percentile_05_hz": float(percentile_05),
            "percentile_95_hz": float(percentile_95),
        },
        eligible_count,
    )


def _cumulant_evidence(normalized_z: np.ndarray) -> dict[str, object]:
    if normalized_z.size < 64:
        return _unavailable_feature_map("INSUFFICIENT_CUMULANT_SAMPLES")
    centred = normalized_z - np.mean(normalized_z)
    scale = float(np.max(_stable_complex_amplitude(centred)))
    if scale == 0.0:
        return _unavailable_feature_map("ZERO_CENTRAL_SECOND_POWER")
    u = _normalize_complex_by_positive_scale(centred, scale)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            m20 = np.mean(u**2)
            m21 = float(np.mean(np.abs(u) ** 2))
            if m21 <= 0.0:
                return _unavailable_feature_map("ZERO_CENTRAL_SECOND_POWER")
            m40 = np.mean(u**4)
            m42 = float(np.mean(np.abs(u) ** 4))
            c20 = m20 / m21
            c40 = (m40 - 3.0 * m20**2) / (m21**2)
            c42 = (m42 - abs(m20) ** 2 - 2.0 * m21**2) / (m21**2)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            raise NonFiniteEstimationResult(
                "Cumulant arithmetic is non-finite."
            ) from exc
    _require_finite_result(c20, c40, c42)
    return {
        "status": "success",
        "reason": None,
        "c20_real": float(c20.real),
        "c20_imag": float(c20.imag),
        "c20_magnitude": float(abs(c20)),
        "c21": 1.0,
        "c40_real": float(c40.real),
        "c40_imag": float(c40.imag),
        "c40_magnitude": float(abs(c40)),
        "c42": float(c42),
    }


def _circular_bin_distance(first: int, second: int, bin_count: int) -> int:
    linear_distance = abs(first - second)
    return min(linear_distance, bin_count - linear_distance)


def _spectral_shape_evidence(
    normalized_z: np.ndarray,
    sample_rate_hz: float,
) -> tuple[dict[str, object], int]:
    if normalized_z.size < 64:
        return _unavailable_feature_map("INSUFFICIENT_SPECTRAL_SAMPLES"), 0
    maximum_nfft = min(int(normalized_z.size), 1024)
    nfft = 1 << (maximum_nfft.bit_length() - 1)
    window = np.hanning(nfft)
    transformed = np.fft.fftshift(np.fft.fft(normalized_z[:nfft] * window, n=nfft))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            q = np.asarray(np.abs(transformed) ** 2, dtype=np.float64)
            maximum = float(np.max(q))
            q_floor = max(maximum * 1.0e-12, np.finfo(np.float64).tiny)
            adjusted = q + q_floor
            flatness = float(np.exp(np.mean(np.log(adjusted))) / np.mean(adjusted))
            baseline = max(
                float(np.quantile(q, 0.20, method="linear")),
                q_floor,
            )
            threshold = baseline * (10.0 ** 0.6)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            raise NonFiniteEstimationResult(
                "Spectral-shape arithmetic is non-finite."
            ) from exc
    _require_finite_result(q, flatness, baseline, threshold)
    flatness = float(np.clip(flatness, 0.0, 1.0))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            frequency = np.fft.fftshift(
                np.fft.fftfreq(nfft) * np.float64(sample_rate_hz)
            )
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise NonFiniteEstimationResult(
            "Spectral frequency-axis arithmetic is non-finite."
        ) from exc
    _require_finite_result(frequency)
    local_maxima = [
        index
        for index in range(nfft)
        if q[index] > threshold
        and q[index] > q[(index - 1) % nfft]
        and q[index] > q[(index + 1) % nfft]
    ]
    ordered = sorted(local_maxima, key=lambda index: (-q[index], frequency[index]))
    selected: list[int] = []
    for index in ordered:
        if all(
            _circular_bin_distance(index, prior, nfft) >= 2
            for prior in selected
        ):
            selected.append(index)
    if selected:
        strongest = selected[0]
        strongest_frequency: float | None = float(frequency[strongest])
        strongest_level: float | None = float(10.0 * math.log10(q[strongest] / baseline))
        _require_finite_result(strongest_frequency, strongest_level)
    else:
        strongest_frequency = None
        strongest_level = None
    return (
        {
            "status": "success",
            "reason": None,
            "spectral_flatness": flatness,
            "peak_count": len(selected),
            "strongest_peak_frequency_hz": strongest_frequency,
            "strongest_peak_level_db": strongest_level,
        },
        nfft,
    )


def _lag_correlation_evidence(
    normalized_z: np.ndarray,
    sample_rate_hz: float,
    max_lag_samples: int,
) -> tuple[dict[str, object], int]:
    maximum = min(max_lag_samples, int(normalized_z.size) - 32)
    if maximum < 4:
        return _unavailable_feature_map("INSUFFICIENT_LAG_OVERLAP"), 0
    best_lag: int | None = None
    best_correlation = -1.0
    tested = 0
    for lag in range(4, maximum + 1):
        left = normalized_z[lag:]
        right = normalized_z[:-lag]
        left_power = float(np.sum(np.abs(left) ** 2))
        right_power = float(np.sum(np.abs(right) ** 2))
        denominator = math.sqrt(left_power * right_power)
        if denominator <= 0.0:
            continue
        correlation = float(abs(np.sum(left * np.conj(right))) / denominator)
        _require_finite_result(correlation)
        correlation = float(np.clip(correlation, 0.0, 1.0))
        tested += 1
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    if best_lag is None:
        return _unavailable_feature_map("NO_VALID_LAG_CORRELATION"), tested
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            peak_lag_seconds = float(
                np.float64(best_lag) / np.float64(sample_rate_hz)
            )
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise NonFiniteEstimationResult(
            "Lag-time arithmetic is non-finite."
        ) from exc
    _require_finite_result(peak_lag_seconds)
    return (
        {
            "status": "success",
            "reason": None,
            "tested_lag_count": tested,
            "peak_lag_samples": best_lag,
            "peak_lag_seconds": peak_lag_seconds,
            "peak_normalized_correlation": best_correlation,
        },
        tested,
    )


def _unavailable_feature_map(reason: str) -> dict[str, object]:
    return {"status": "not_applicable", "reason": reason}


def _require_finite_result(*values: object) -> None:
    for value in values:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise NonFiniteEstimationResult(
                "Feature arithmetic produced a non-finite result."
            )


# Accepted short aliases retained for internal compatibility with the first Task 6 pass.
spectrum = validate_spectrum
edges = bin_edges
peak_ties = tied_peak_indices
crossing = linear_power_crossing
