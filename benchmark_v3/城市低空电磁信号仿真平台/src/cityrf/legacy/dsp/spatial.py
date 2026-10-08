"""Internal narrowband spatial covariance, MDL, and ULA MUSIC primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


class SpatialError(ValueError):
    """Typed Task 5 failure with a stable safe code."""

    code = "INVALID_SPATIAL_PARAMETERS"

    def __init__(self, message: str = "Spatial parameters are invalid.") -> None:
        super().__init__(message)


class InvalidSpatialParameters(SpatialError):
    code = "INVALID_SPATIAL_PARAMETERS"


class SpatialEvidenceBindingError(SpatialError):
    """Candidate or spatial evidence is bound to an incompatible source/artifact/block."""

    code = "SPATIAL_EVIDENCE_BINDING_ERROR"

    def __init__(self, message: str = "Spatial evidence binding is incompatible.") -> None:
        super().__init__(message)


class MalformedSpatialEvidence(SpatialError):
    """A spatial or candidate payload does not satisfy its accepted internal shape/type contract."""

    code = "MALFORMED_SPATIAL_EVIDENCE"

    def __init__(self, message: str = "Spatial evidence payload is malformed.") -> None:
        super().__init__(message)


@dataclass(frozen=True, repr=False)
class SpatialEvidencePayload:
    """Private numerical state required by MDL and MUSIC."""

    raw_covariance: np.ndarray = field(repr=False)
    loaded_covariance: np.ndarray = field(repr=False)
    raw_eigenvalues: np.ndarray = field(repr=False)
    raw_eigenvectors: np.ndarray = field(repr=False)
    loaded_eigenvalues: np.ndarray = field(repr=False)
    loaded_eigenvectors: np.ndarray = field(repr=False)
    selected_channel_indices: tuple[int, ...]
    segment_ranges: tuple[tuple[int, int], ...]
    snapshot_count: int
    diagonal_loading: float
    rank_rtol: float
    raw_rank_threshold: float
    raw_effective_rank: int
    loaded_eigenvalue_tolerance: float
    loaded_condition_number: float | None
    synchronization: bool | None
    synchronization_conflict: bool
    calibration_state: str
    calibration_usable: bool
    geometry: Any = field(repr=False)
    manifest_reference_frequency_hz: float | None
    geometry_reference_frequency_hz: float | None
    physical_channel_count: int
    filtered: bool
    remove_mean: bool

    def __repr__(self) -> str:
        return (
            "SpatialEvidencePayload("
            f"channel_count={len(self.selected_channel_indices)}, snapshot_count={self.snapshot_count}, "
            f"calibration_state={self.calibration_state!r})"
        )


def covariance_from_segments(
    segments: list[np.ndarray], *, diagonal_loading_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | None]]:
    """Build weighted covariance and both descending eigensystems."""

    if not segments:
        raise InvalidSpatialParameters("At least one spatial segment is required.")
    channels = segments[0].shape[0]
    snapshot_count = sum(segment.shape[1] for segment in segments)
    if snapshot_count < channels:
        raise InvalidSpatialParameters("Insufficient simultaneous snapshots.")
    raw = sum((segment @ segment.conj().T for segment in segments), start=np.zeros((channels, channels), dtype=np.complex128)) / snapshot_count
    trace = float(np.trace(raw).real)
    if not math.isfinite(trace) or trace <= 0.0 or not np.isfinite(raw).all():
        raise InvalidSpatialParameters("Spatial covariance has no finite positive energy.")
    scale = trace / channels
    # Keep a margin below float64 overflow so adding the loading remains a
    # meaningful Hermitian eigensolve, without evaluating an overflowing divide.
    if diagonal_loading_fraction > 0.0 and math.log(diagonal_loading_fraction) + math.log(scale) > math.log(np.finfo(np.float64).max / 16.0):
        raise SpatialError("Spatial diagonal loading is not finite.")
    loading = diagonal_loading_fraction * scale
    loaded = raw + loading * np.eye(channels, dtype=np.complex128)
    try:
        raw_values, raw_vectors = np.linalg.eigh(raw)
        loaded_values, loaded_vectors = np.linalg.eigh(loaded)
    except np.linalg.LinAlgError as error:
        raise SpatialError("Spatial eigensolve failed.") from error
    raw_values, raw_vectors = raw_values[::-1], raw_vectors[:, ::-1]
    loaded_values, loaded_vectors = loaded_values[::-1], loaded_vectors[:, ::-1]
    if not (np.isfinite(raw_values).all() and np.isfinite(loaded_values).all()):
        raise SpatialError("Spatial eigensolve produced non-finite values.")
    raw_values = _normalize_eigenvalues(raw_values)
    loaded_values = _normalize_eigenvalues(loaded_values)
    tolerance = np.finfo(np.float64).eps * max(channels, snapshot_count)
    raw_threshold = tolerance * max(float(raw_values[0]), float(np.finfo(np.float64).tiny))
    raw_rank = int(np.count_nonzero(raw_values > raw_threshold))
    loaded_tolerance = np.finfo(np.float64).eps * channels * max(float(loaded_values[0]), float(np.finfo(np.float64).tiny))
    condition = (
        float(loaded_values[0] / loaded_values[-1])
        if loaded_values[-1] > loaded_tolerance and math.isfinite(float(loaded_values[-1]))
        else None
    )
    return raw, loaded, raw_values, raw_vectors, loaded_values, loaded_vectors, {
        "snapshot_count": snapshot_count,
        "diagonal_loading": loading,
        "rank_rtol": tolerance,
        "raw_rank_threshold": raw_threshold,
        "raw_effective_rank": raw_rank,
        "loaded_eigenvalue_tolerance": loaded_tolerance,
        "loaded_condition_number": condition,
    }


def _normalize_eigenvalues(values: np.ndarray) -> np.ndarray:
    """Clamp only floating-point-sized negative eigensolver residue to zero."""

    tolerance = 32.0 * np.finfo(np.float64).eps * values.size * max(1.0, float(np.max(np.abs(values))))
    if np.any(values < -tolerance):
        raise SpatialError("Spatial eigensolve produced materially negative eigenvalues.")
    return np.where(values < 0.0, 0.0, values)


def mdl_scores(raw_eigenvalues: np.ndarray, *, snapshot_count: int, max_sources: int | None) -> tuple[int, int, list[dict[str, float | int]]]:
    """Return standard raw-eigenvalue MDL source-count scores."""

    channels = raw_eigenvalues.size
    k_max = channels - 1 if max_sources is None else max_sources
    if k_max < 0 or k_max > channels - 1:
        raise InvalidSpatialParameters("Requested maximum source count is invalid.")
    if snapshot_count <= 1 or snapshot_count < channels or not np.isfinite(raw_eigenvalues).all() or np.any(raw_eigenvalues <= 0.0):
        raise SpatialError("Raw covariance eigenspectrum cannot support MDL.")
    records: list[dict[str, float | int]] = []
    for k in range(k_max + 1):
        tail = raw_eigenvalues[k:]
        geometric = float(np.exp(np.mean(np.log(tail))))
        arithmetic = float(np.mean(tail))
        score = float(-snapshot_count * (channels - k) * math.log(geometric / arithmetic) + 0.5 * k * (2 * channels - k) * math.log(snapshot_count))
        records.append({"k": k, "score": score})
    selected = min(records, key=lambda item: (float(item["score"]), int(item["k"])))
    return int(selected["k"]), k_max, records


def music_peaks(
    loaded_eigenvectors: np.ndarray,
    *,
    source_count: int,
    spacing_m: float,
    reference_frequency_hz: float,
    azimuth_min_deg: float,
    azimuth_max_deg: float,
    grid_step_deg: float,
    peak_separation_deg: float,
) -> tuple[list[dict[str, float]], dict[str, float | int], bool]:
    """Scan signed-broadside ULA MUSIC and extract deterministic separated peaks."""

    count = loaded_eigenvectors.shape[0]
    if source_count < 1 or source_count >= count:
        raise InvalidSpatialParameters("Source count is incompatible with selected channels.")
    q = math.floor((azimuth_max_deg - azimuth_min_deg) / grid_step_deg + 1e-12)
    if q + 1 > 100_001:
        raise InvalidSpatialParameters("Requested MUSIC grid exceeds 100001 points.")
    grid = azimuth_min_deg + grid_step_deg * np.arange(q + 1, dtype=np.float64)
    if abs(grid[-1] - azimuth_max_deg) <= 1e-12:
        grid[-1] = azimuth_max_deg
    noise = loaded_eigenvectors[:, source_count:]
    projector = noise @ noise.conj().T
    wavelength = 299_792_458.0 / reference_frequency_hz
    channels = np.arange(count, dtype=np.float64)
    powers = np.empty(grid.size, dtype=np.float64)
    floor = np.finfo(np.float64).eps * max(1.0, float(np.trace(projector).real))
    for index, angle in enumerate(grid):
        steering = np.exp(1j * 2.0 * np.pi * spacing_m / wavelength * math.sin(math.radians(float(angle))) * channels)
        denominator = max(float(np.real(steering.conj() @ projector @ steering)), floor)
        powers[index] = 1.0 / denominator
    if not np.isfinite(powers).all():
        raise SpatialError("MUSIC spectrum is non-finite.")
    candidates: list[int] = []
    for index, value in enumerate(powers):
        left = powers[index - 1] if index else -np.inf
        right = powers[index + 1] if index + 1 < powers.size else -np.inf
        if value >= left and value >= right and (value > left or value > right):
            # Lowest index represents a flat plateau.
            if index and value == left:
                continue
            candidates.append(index)
    median = max(float(np.median(powers)), np.finfo(np.float64).tiny)
    largest = max(float(np.max(powers)), np.finfo(np.float64).tiny)
    candidates.sort(key=lambda i: (-float(powers[i]), -10.0 * math.log10(powers[i] / median), float(grid[i])))
    accepted: list[int] = []
    for index in candidates:
        if all(abs(float(grid[index] - grid[other])) >= peak_separation_deg for other in accepted):
            accepted.append(index)
            if len(accepted) == source_count:
                break
    peaks = [
        {
            "azimuth_deg": float(grid[index]),
            "relative_to_largest_db": float(10.0 * math.log10(powers[index] / largest)),
            "peak_to_median_db": float(10.0 * math.log10(powers[index] / median)),
        }
        for index in accepted
    ]
    return peaks, {"first_deg": float(grid[0]), "last_deg": float(grid[-1]), "point_count": int(grid.size), "step_deg": grid_step_deg}, len(accepted) < source_count
