"""Internal activity and resolvable-emission detection algorithms.

The payloads in this module are deliberately artifact-only.  In particular,
threshold masks and full ridge frame lists are never suitable for a ToolResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
from scipy import ndimage

from cityrf.legacy.dsp.representations import SpectrumPayload, TimeFrequencyPayload
from cityrf.legacy.dsp.selection import SelectedSegment
from cityrf.legacy.schemas import FrequencyRange


_MAX_ABS_THRESHOLD_DB = 300.0


class DetectionError(ValueError):
    """Base class for typed, repairable detection failures."""


class InvalidDetectionParameters(DetectionError):
    """A method control or evidence combination is invalid."""


class InsufficientDetectionSamples(DetectionError):
    """The selected evidence cannot support a requested detector."""


class InapplicableDetectionEvidence(DetectionError):
    """An internal evidence payload is malformed for this detector."""


@dataclass(frozen=True)
class ActivityInterval:
    """A global, half-open detected activity range."""

    start: int
    stop: int
    peak_power: float
    noise_power: float
    boundary_truncated: bool


@dataclass(frozen=True, repr=False)
class ActivityDetectionResult:
    intervals: tuple[ActivityInterval, ...]
    selected_sample_count: int
    noise_power: float
    resolved_parameters: dict[str, int | float]
    boundary_truncated: bool

    def __repr__(self) -> str:
        return f"ActivityDetectionResult(interval_count={len(self.intervals)}, selected_sample_count={self.selected_sample_count})"


@dataclass(frozen=True, repr=False)
class EmissionCandidate:
    """One artifact-scoped resolvable region; no physical-source claim is made."""

    candidate_id: str
    source_ref: str
    source_artifact_id: str
    source_artifact_type: str
    method: str
    domain: str
    source_block_index: int
    sample_range: tuple[int, int]
    channel_indices: tuple[int, ...]
    frequency_range: FrequencyRange
    peak_frequency_hz: float
    peak_power_linear: float
    relative_power_db: float
    confidence: float
    lower_truncated: bool
    upper_truncated: bool
    mask_indices: tuple[tuple[int, int], ...] = field(default_factory=tuple, repr=False)
    track_frame_indices: tuple[int, ...] = field(default_factory=tuple, repr=False)
    track_frequency_bin_indices: tuple[int, ...] = field(default_factory=tuple, repr=False)
    slope_hz_per_sample: float | None = None
    max_observed_jump_bins: int | None = None

    def __repr__(self) -> str:
        return (
            "EmissionCandidate("
            f"candidate_id={self.candidate_id!r}, method={self.method!r}, domain={self.domain!r}, "
            f"sample_range={self.sample_range!r}, frequency_range={self.frequency_range!r})"
        )


@dataclass(frozen=True, repr=False)
class CandidateSetPayload:
    """Internal candidate-set artifact bound to its original evidence and source."""

    source_ref: str
    source_artifact_id: str
    source_artifact_type: str
    method: str
    domain: str
    candidates: tuple[EmissionCandidate, ...]

    def __repr__(self) -> str:
        return (
            "CandidateSetPayload("
            f"method={self.method!r}, domain={self.domain!r}, candidate_count={len(self.candidates)})"
        )


def detect_activity(
    segments: list[SelectedSegment],
    *,
    smoothing_window_samples: int,
    noise_quantile: float,
    high_threshold_db: float,
    low_threshold_db: float,
    min_active_samples: int,
    merge_gap_samples: int,
    padding_samples: int,
) -> ActivityDetectionResult:
    """Detect energy intervals independently within every selected source range.

    Smoothing is a centred moving average with edge replication.  Noise is the
    requested lower quantile of each independently smoothed range; a reported
    capture value is their selected-sample weighted mean.
    """

    _validate_activity_parameters(
        smoothing_window_samples=smoothing_window_samples,
        noise_quantile=noise_quantile,
        high_threshold_db=high_threshold_db,
        low_threshold_db=low_threshold_db,
        min_active_samples=min_active_samples,
        merge_gap_samples=merge_gap_samples,
        padding_samples=padding_samples,
    )
    if not segments:
        raise InsufficientDetectionSamples("No selected samples are available for activity detection.")
    if any(segment.iq.shape[1] < smoothing_window_samples for segment in segments):
        raise InsufficientDetectionSamples("A selected range is shorter than the resolved smoothing window.")

    all_intervals: list[ActivityInterval] = []
    weighted_noise = 0.0
    selected_sample_count = 0
    any_boundary_truncated = False
    for segment in segments:
        # Promote before magnitude/power arithmetic.  Otherwise complex64
        # produces float32 power and a positive float64 noise floor can be
        # downcast to zero during threshold comparisons.
        iq = np.asarray(segment.iq, dtype=np.complex128)
        power = np.mean(np.abs(iq) ** 2, axis=0, dtype=np.float64)
        smoothed = ndimage.uniform_filter1d(power, size=smoothing_window_samples, mode="nearest")
        noise_power = float(np.quantile(smoothed, noise_quantile))
        noise_power = max(noise_power, float(np.finfo(np.float64).tiny))
        high = noise_power * 10.0 ** (high_threshold_db / 10.0)
        low = noise_power * 10.0 ** (low_threshold_db / 10.0)
        local_intervals = _hysteresis_intervals(smoothed, high=high, low=low)
        local_intervals = _drop_short(local_intervals, min_active_samples)
        local_intervals = _merge_gaps(local_intervals, merge_gap_samples)
        local_intervals = _pad_intervals(
            local_intervals,
            padding_samples=padding_samples,
            segment_length=power.size,
        )
        for start, stop, padding_truncated in local_intervals:
            interval_truncated = padding_truncated or start == 0 or stop == power.size
            all_intervals.append(
                ActivityInterval(
                    start=segment.sample_range.start + start,
                    stop=segment.sample_range.start + stop,
                    peak_power=float(np.max(smoothed[start:stop])),
                    noise_power=noise_power,
                    boundary_truncated=interval_truncated,
                )
            )
            any_boundary_truncated = any_boundary_truncated or interval_truncated
        selected_sample_count += power.size
        weighted_noise += noise_power * power.size
    all_intervals.sort(key=lambda item: (item.start, item.stop))
    if any(right.start < left.stop for left, right in zip(all_intervals, all_intervals[1:])):
        raise InvalidDetectionParameters("Overlapping selected ranges are not supported for activity detection.")
    return ActivityDetectionResult(
        intervals=tuple(all_intervals),
        selected_sample_count=selected_sample_count,
        noise_power=weighted_noise / selected_sample_count,
        resolved_parameters={
            "smoothing_window_samples": smoothing_window_samples,
            "noise_quantile": noise_quantile,
            "high_threshold_db": high_threshold_db,
            "low_threshold_db": low_threshold_db,
            "min_active_samples": min_active_samples,
            "merge_gap_samples": merge_gap_samples,
            "padding_samples": padding_samples,
        },
        boundary_truncated=any_boundary_truncated,
    )


def spectral_region_candidates(
    payload: SpectrumPayload,
    *,
    source_ref: str,
    source_artifact_id: str,
    threshold_db: float,
    noise_quantile: float,
    min_region_bins: int,
    merge_gap_bins: int,
) -> tuple[EmissionCandidate, ...]:
    _validate_candidate_parameters(threshold_db, noise_quantile, min_region_bins, merge_gap_bins)
    candidates: list[EmissionCandidate] = []
    for source_block_index, block in enumerate(payload.blocks):
        candidates.extend(
            _spectral_block_candidates(
                frequency_hz=block.frequency_hz,
                power_linear=block.power_linear,
                sample_range=block.sample_range,
                channel_indices=block.channel_indices,
                source_ref=source_ref,
                source_artifact_id=source_artifact_id,
                threshold_db=threshold_db,
                noise_quantile=noise_quantile,
                min_region_bins=min_region_bins,
                merge_gap_bins=merge_gap_bins,
                sample_rate_hz=payload.sample_rate_hz,
                source_block_index=source_block_index,
            )
        )
    return _assign_stable_ids(candidates)


def tf_connected_region_candidates(
    payload: TimeFrequencyPayload,
    *,
    source_ref: str,
    source_artifact_id: str,
    threshold_db: float,
    noise_quantile: float,
    min_time_frames: int,
    min_frequency_bins: int,
    merge_gap_bins: int,
) -> tuple[EmissionCandidate, ...]:
    _validate_candidate_parameters(threshold_db, noise_quantile, min_frequency_bins, merge_gap_bins)
    if min_time_frames <= 0:
        raise InvalidDetectionParameters("min_time_frames must be a positive frame count.")
    candidates: list[EmissionCandidate] = []
    for source_block_index, block in enumerate(payload.blocks):
        power = block.power_linear
        noise = _robust_noise(power, noise_quantile)
        occupied = power >= noise * 10.0 ** (threshold_db / 10.0)
        if merge_gap_bins:
            occupied = ndimage.binary_closing(
                occupied,
                structure=np.ones((2 * merge_gap_bins + 1, 1), dtype=bool),
            )
        labels, label_count = ndimage.label(occupied, structure=np.ones((3, 3), dtype=np.uint8))
        for label_number in range(1, label_count + 1):
            coordinates = np.argwhere(labels == label_number)
            if coordinates.size == 0:
                continue
            frequency_indices = coordinates[:, 0]
            frame_indices = coordinates[:, 1]
            unique_frequency = np.unique(frequency_indices)
            unique_frames = np.unique(frame_indices)
            if unique_frequency.size < min_frequency_bins or unique_frames.size < min_time_frames:
                continue
            candidates.append(
                _tf_candidate(
                    frequency_hz=block.frequency_hz,
                    power_linear=power,
                    coordinates=coordinates,
                    frame_indices=unique_frames,
                    sample_range=block.sample_range,
                    channel_indices=block.channel_indices,
                    frame_start_samples=block.frame_start_samples,
                    window_length=payload.window_length,
                    source_ref=source_ref,
                    source_artifact_id=source_artifact_id,
                    method="tf_connected_regions",
                    noise_power=noise,
                    sample_rate_hz=payload.sample_rate_hz,
                    source_block_index=source_block_index,
                )
            )
    return _assign_stable_ids(candidates)


def tf_track_candidates(
    payload: TimeFrequencyPayload,
    *,
    source_ref: str,
    source_artifact_id: str,
    threshold_db: float,
    noise_quantile: float,
    min_track_frames: int,
    max_track_jump_bins: int,
    max_track_gap_frames: int,
) -> tuple[EmissionCandidate, ...]:
    if min_track_frames <= 0 or max_track_jump_bins < 0 or max_track_gap_frames < 0:
        raise InvalidDetectionParameters("Track frame, jump, and gap controls must be non-negative valid counts.")
    _validate_candidate_parameters(threshold_db, noise_quantile, 1, 0)
    candidates: list[EmissionCandidate] = []
    for source_block_index, block in enumerate(payload.blocks):
        noise = _robust_noise(block.power_linear, noise_quantile)
        occupied = block.power_linear >= noise * 10.0 ** (threshold_db / 10.0)
        for frame_indices, frequency_indices in _ridge_tracks(
            occupied, block.power_linear, max_track_jump_bins, max_track_gap_frames
        ):
            if len(frame_indices) < min_track_frames:
                continue
            coordinates = np.column_stack((frequency_indices, frame_indices))
            candidate = _tf_candidate(
                frequency_hz=block.frequency_hz,
                power_linear=block.power_linear,
                coordinates=coordinates,
                frame_indices=np.asarray(frame_indices, dtype=int),
                sample_range=block.sample_range,
                channel_indices=block.channel_indices,
                frame_start_samples=block.frame_start_samples,
                window_length=payload.window_length,
                source_ref=source_ref,
                source_artifact_id=source_artifact_id,
                method="tf_tracks",
                noise_power=noise,
                sample_rate_hz=payload.sample_rate_hz,
                source_block_index=source_block_index,
            )
            starts = block.frame_start_samples[np.asarray(frame_indices, dtype=int)]
            slope = float(np.polyfit(starts, block.frequency_hz[np.asarray(frequency_indices, dtype=int)], 1)[0])
            jumps = np.abs(np.diff(np.asarray(frequency_indices, dtype=int)))
            candidates.append(
                replace(
                    candidate,
                    track_frame_indices=tuple(int(index) for index in frame_indices),
                    track_frequency_bin_indices=tuple(int(index) for index in frequency_indices),
                    slope_hz_per_sample=slope,
                    max_observed_jump_bins=int(np.max(jumps)) if jumps.size else 0,
                )
            )
    return _assign_stable_ids(candidates)


def _validate_activity_parameters(**values: int | float) -> None:
    if values["smoothing_window_samples"] <= 0 or values["min_active_samples"] <= 0:
        raise InvalidDetectionParameters("Activity window and minimum duration must be positive sample counts.")
    if values["merge_gap_samples"] < 0 or values["padding_samples"] < 0:
        raise InvalidDetectionParameters("Activity gap and padding must be non-negative sample counts.")
    if not 0.0 < values["noise_quantile"] <= 0.5:
        raise InvalidDetectionParameters("noise_quantile must be in (0, 0.5].")
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise InvalidDetectionParameters("Detection controls must be finite.")
    if (
        abs(float(values["high_threshold_db"])) > _MAX_ABS_THRESHOLD_DB
        or abs(float(values["low_threshold_db"])) > _MAX_ABS_THRESHOLD_DB
    ):
        raise InvalidDetectionParameters(
            "Activity thresholds must be within the supported numerical dB range."
        )
    if values["high_threshold_db"] <= values["low_threshold_db"]:
        raise InvalidDetectionParameters("The activity high threshold must exceed the low hysteresis threshold.")


def _validate_candidate_parameters(
    threshold_db: float, noise_quantile: float, min_region_bins: int, merge_gap_bins: int
) -> None:
    if (
        not math.isfinite(threshold_db)
        or threshold_db <= 0.0
        or threshold_db > _MAX_ABS_THRESHOLD_DB
    ):
        raise InvalidDetectionParameters("The candidate threshold must be finite positive dB above noise.")
    if not 0.0 < noise_quantile <= 0.5:
        raise InvalidDetectionParameters("noise_quantile must be in (0, 0.5].")
    if min_region_bins <= 0 or merge_gap_bins < 0:
        raise InvalidDetectionParameters("Candidate region extents and gap controls are invalid.")


def _hysteresis_intervals(power: np.ndarray, *, high: float, low: float) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(power):
        if start is None and value >= high:
            start = index
        elif start is not None and value < low:
            intervals.append((start, index))
            start = None
    if start is not None:
        intervals.append((start, power.size))
    return intervals


def _drop_short(intervals: list[tuple[int, int]], minimum: int) -> list[tuple[int, int]]:
    return [(start, stop) for start, stop in intervals if stop - start >= minimum]


def _merge_gaps(intervals: list[tuple[int, int]], maximum_gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        previous_start, previous_stop = merged[-1]
        if start - previous_stop <= maximum_gap:
            merged[-1] = (previous_start, stop)
        else:
            merged.append((start, stop))
    return merged


def _pad_intervals(
    intervals: list[tuple[int, int]], *, padding_samples: int, segment_length: int
) -> list[tuple[int, int, bool]]:
    padded: list[tuple[int, int, bool]] = []
    for start, stop in intervals:
        padded_start = max(0, start - padding_samples)
        padded_stop = min(segment_length, stop + padding_samples)
        truncated = padded_start != start - padding_samples or padded_stop != stop + padding_samples
        if padded and padded_start <= padded[-1][1]:
            previous_start, previous_stop, previous_truncated = padded[-1]
            padded[-1] = (
                previous_start,
                max(previous_stop, padded_stop),
                previous_truncated or truncated,
            )
        else:
            padded.append((padded_start, padded_stop, truncated))
    return padded


def _robust_noise(power: np.ndarray, quantile: float) -> float:
    finite = np.asarray(power, dtype=np.float64)
    if finite.size == 0 or not np.isfinite(finite).all() or np.any(finite < 0.0):
        raise InapplicableDetectionEvidence("Detection evidence must be finite non-negative linear power.")
    # A noiseless synthetic capture can have a zero lower quantile.  Anchor its
    # floor to numerical dynamic range so floating-point STFT residue is not
    # promoted into an occupied region.
    return max(
        float(np.quantile(finite, quantile)),
        float(np.max(finite)) * 1.0e-12,
        float(np.finfo(np.float64).tiny),
    )


def _spectral_block_candidates(
    *,
    frequency_hz: np.ndarray,
    power_linear: np.ndarray,
    sample_range: tuple[int, int],
    channel_indices: tuple[int, ...],
    source_ref: str,
    source_artifact_id: str,
    threshold_db: float,
    noise_quantile: float,
    min_region_bins: int,
    merge_gap_bins: int,
    sample_rate_hz: float,
    source_block_index: int,
) -> list[EmissionCandidate]:
    if frequency_hz.size < 2:
        raise InsufficientDetectionSamples("Spectrum evidence needs at least two bins for edge-aware regions.")
    noise = _robust_noise(power_linear, noise_quantile)
    occupied = power_linear >= noise * 10.0 ** (threshold_db / 10.0)
    runs = _merge_index_runs(np.flatnonzero(occupied), merge_gap_bins)
    candidates: list[EmissionCandidate] = []
    for start, stop in runs:
        if stop - start < min_region_bins:
            continue
        peak_index = start + int(np.argmax(power_linear[start:stop]))
        frequency_range, lower_truncated, upper_truncated = _frequency_bounds(
            frequency_hz, start, stop - 1, sample_rate_hz=sample_rate_hz
        )
        peak_power = float(power_linear[peak_index])
        relative = _relative_db(peak_power, noise)
        candidates.append(
            EmissionCandidate(
                candidate_id="",
                source_ref=source_ref,
                source_artifact_id=source_artifact_id,
                source_artifact_type="spectrum",
                method="spectral_regions",
                domain="spectral",
                source_block_index=source_block_index,
                sample_range=sample_range,
                channel_indices=channel_indices,
                frequency_range=frequency_range,
                peak_frequency_hz=float(frequency_hz[peak_index]),
                peak_power_linear=peak_power,
                relative_power_db=relative,
                confidence=_confidence(relative, threshold_db),
                lower_truncated=lower_truncated,
                upper_truncated=upper_truncated,
            )
        )
    return candidates


def _tf_candidate(
    *,
    frequency_hz: np.ndarray,
    power_linear: np.ndarray,
    coordinates: np.ndarray,
    frame_indices: np.ndarray,
    sample_range: tuple[int, int],
    channel_indices: tuple[int, ...],
    frame_start_samples: np.ndarray,
    window_length: int,
    source_ref: str,
    source_artifact_id: str,
    method: str,
    noise_power: float,
    sample_rate_hz: float,
    source_block_index: int,
) -> EmissionCandidate:
    frequency_indices = coordinates[:, 0]
    first_frame = int(np.min(frame_indices))
    last_frame = int(np.max(frame_indices))
    start = int(frame_start_samples[first_frame])
    stop = min(sample_range[1], int(frame_start_samples[last_frame]) + window_length)
    low_index, high_index = int(np.min(frequency_indices)), int(np.max(frequency_indices))
    frequency_range, lower_truncated, upper_truncated = _frequency_bounds(
        frequency_hz, low_index, high_index, sample_rate_hz=sample_rate_hz
    )
    coordinate_power = power_linear[coordinates[:, 0], coordinates[:, 1]]
    peak_coordinate = coordinates[int(np.argmax(coordinate_power))]
    peak_power = float(np.max(coordinate_power))
    relative = _relative_db(peak_power, noise_power)
    return EmissionCandidate(
        candidate_id="",
        source_ref=source_ref,
        source_artifact_id=source_artifact_id,
        source_artifact_type="time_frequency",
        method=method,
        domain="time_frequency",
        source_block_index=source_block_index,
        sample_range=(start, stop),
        channel_indices=channel_indices,
        frequency_range=frequency_range,
        peak_frequency_hz=float(frequency_hz[int(peak_coordinate[0])]),
        peak_power_linear=peak_power,
        relative_power_db=relative,
        confidence=_confidence(relative, 0.0),
        lower_truncated=lower_truncated,
        upper_truncated=upper_truncated,
        mask_indices=tuple((int(freq), int(frame)) for freq, frame in coordinates),
    )


def _merge_index_runs(indices: np.ndarray, maximum_gap: int) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index_value in indices[1:]:
        index = int(index_value)
        if index - previous - 1 > maximum_gap:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return runs


def _frequency_bounds(
    frequency_hz: np.ndarray,
    low_index: int,
    high_index: int,
    *,
    sample_rate_hz: float,
) -> tuple[FrequencyRange, bool, bool]:
    spacing = float(np.median(np.diff(frequency_hz)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise InapplicableDetectionEvidence("Detection frequency bins must be strictly increasing and uniformly resolvable.")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise InapplicableDetectionEvidence("Detection evidence requires a finite positive sample rate.")
    nyquist_low = -sample_rate_hz / 2.0
    nyquist_high = sample_rate_hz / 2.0
    lower_raw = float(frequency_hz[low_index] - spacing / 2.0)
    upper_raw = float(frequency_hz[high_index] + spacing / 2.0)
    lower = max(nyquist_low, lower_raw)
    upper = min(nyquist_high, upper_raw)
    return (
        FrequencyRange(start_hz=lower, stop_hz=upper, reference="baseband"),
        low_index == 0 or lower != lower_raw,
        high_index == frequency_hz.size - 1 or upper != upper_raw,
    )


def _relative_db(value: float, noise: float) -> float:
    return float(10.0 * np.log10(max(value, np.finfo(np.float64).tiny) / max(noise, np.finfo(np.float64).tiny)))


def _confidence(relative_power_db: float, threshold_db: float) -> float:
    return float(min(1.0, max(0.0, (relative_power_db - threshold_db + 6.0) / 24.0)))


def _assign_stable_ids(candidates: list[EmissionCandidate]) -> tuple[EmissionCandidate, ...]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.sample_range[0],
            item.frequency_range.start_hz,
            -item.relative_power_db,
            item.sample_range[1],
            item.frequency_range.stop_hz,
            item.peak_frequency_hz,
            item.channel_indices,
            item.method,
            item.domain,
            -item.peak_power_linear,
            item.lower_truncated,
            item.upper_truncated,
            # This enumeration-derived value is only a final fallback after
            # every physical range/channel and semantic discriminator.
            item.source_block_index,
        ),
    )
    return tuple(replace(candidate, candidate_id=f"emission_{index}") for index, candidate in enumerate(ordered))


def _ridge_tracks(
    occupied: np.ndarray, power: np.ndarray, maximum_jump: int, maximum_gap: int
) -> list[tuple[list[int], list[int]]]:
    """Greedily link thresholded ridges; sorting makes public identities deterministic."""

    tracks: list[tuple[list[int], list[int]]] = []
    active: list[int] = []
    for frame in range(occupied.shape[1]):
        available = set(int(index) for index in np.flatnonzero(occupied[:, frame]))
        for track_index in sorted(active, key=lambda index: tracks[index][1][-1]):
            frames, frequencies = tracks[track_index]
            if frame - frames[-1] > maximum_gap + 1:
                active.remove(track_index)
                continue
            choices = [index for index in available if abs(index - frequencies[-1]) <= maximum_jump]
            if not choices:
                continue
            chosen = min(choices, key=lambda index: (abs(index - frequencies[-1]), -float(power[index, frame]), index))
            frames.append(frame)
            frequencies.append(chosen)
            available.remove(chosen)
        for frequency in sorted(available, key=lambda index: (-float(power[index, frame]), index)):
            tracks.append(([frame], [frequency]))
            active.append(len(tracks) - 1)
    return tracks
