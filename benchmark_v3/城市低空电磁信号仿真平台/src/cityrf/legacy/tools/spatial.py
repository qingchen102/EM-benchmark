"""Dependency-injected Task 5 spatial evidence and narrowband MUSIC executors."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cityrf.legacy.dsp.detection import CandidateSetPayload, EmissionCandidate
from cityrf.legacy.dsp.spatial import (
    InvalidSpatialParameters,
    MalformedSpatialEvidence,
    SpatialError,
    SpatialEvidenceBindingError,
    SpatialEvidencePayload,
    covariance_from_segments,
    music_peaks,
)
from cityrf.legacy.runtime import IQRepository, RunArtifactStore
from cityrf.legacy.schemas import ArrayGeometry, FrequencyRange, SampleRange, SignalSelection, ToolError, ToolResult, ToolStatus


_MAX_PUBLIC = 16


class ComputeSpatialEvidenceArgs(SignalSelection):
    """Complex multi-channel selection for fixed narrowband covariance evidence."""

    channel_mode: Literal["per_channel"] = "per_channel"
    candidate_set_id: str | None = None
    candidate_id: str | None = None
    remove_mean: bool = True
    diagonal_loading_fraction: float = Field(default=1e-3, ge=0.0)

    @model_validator(mode="after")
    def validate_spatial_selection(self) -> "ComputeSpatialEvidenceArgs":
        candidate_pair = (self.candidate_set_id is not None, self.candidate_id is not None)
        if candidate_pair[0] != candidate_pair[1]:
            raise ValueError("candidate_set_id and candidate_id must be supplied together")
        if self.frequency_range is not None and candidate_pair[0]:
            raise ValueError("CONFLICTING_SELECTION")
        if not math.isfinite(self.diagonal_loading_fraction):
            raise ValueError("diagonal_loading_fraction must be finite")
        return self


class EstimateDoaArgs(BaseModel):
    """MUSIC scan controls.  Only the spatial artifact is public input."""

    model_config = ConfigDict(extra="forbid")

    spatial_evidence_id: str
    source_count: int = Field(ge=1)
    azimuth_min_deg: float = -90.0
    azimuth_max_deg: float = 90.0
    grid_step_deg: float = 1.0
    peak_separation_deg: float = 5.0

    @model_validator(mode="after")
    def validate_scan(self) -> "EstimateDoaArgs":
        values = (self.azimuth_min_deg, self.azimuth_max_deg, self.grid_step_deg, self.peak_separation_deg)
        if not all(math.isfinite(value) for value in values) or not (-90.0 <= self.azimuth_min_deg < self.azimuth_max_deg <= 90.0) or self.grid_step_deg <= 0.0 or self.peak_separation_deg <= 0.0:
            raise ValueError("MUSIC scan bounds and resolutions are invalid")
        return self


def compute_spatial_evidence(args: ComputeSpatialEvidenceArgs, *, repository: IQRepository, artifact_store: RunArtifactStore) -> ToolResult:
    """Compute fixed ``narrowband_covariance_v1`` without exposing arrays."""

    frame = repository.get(args.source_ref)
    ranges, frequency_range, candidate_note = _resolve_ranges_and_frequency(args, frame.iq.shape[1], artifact_store)
    if not ranges:
        return _not_applicable("compute_spatial_evidence", "EMPTY_CANDIDATE_INTERSECTION", "Candidate and requested sample ranges do not intersect.", "Select an intersecting sample range.")
    _ensure_nonoverlap(ranges)
    if len(ranges) > 1 and args.range_mode != "aggregate":
        raise _coded("INVALID_RANGE_MODE", "Multiple spatial ranges require aggregate mode.")
    selected_indices = tuple(args.channel_indices) if args.channel_indices else tuple(range(frame.iq.shape[0]))
    if any(index < 0 or index >= frame.iq.shape[0] for index in selected_indices):
        raise _coded("INVALID_SPATIAL_PARAMETERS", "A selected channel is outside the capture.")
    if len(selected_indices) < 2:
        return _not_applicable("compute_spatial_evidence", "INSUFFICIENT_CHANNELS", "Select at least two physical channels.", "Select two or more channels.")
    calibration_state, factors, calibration_usable, calibration_warning = _resolve_calibration(frame.manifest, frame.iq.shape[0])
    segments: list[np.ndarray] = []
    warnings: list[str] = []
    if args.remove_mean:
        warnings.append("Per-channel temporal mean was removed independently in every segment; a genuine DC component can be suppressed.")
    if calibration_warning:
        warnings.append(calibration_warning)
    filtered = frequency_range is not None
    if filtered:
        warnings.append("FFT-bin band limiting was applied per segment; it reduces snapshot independence.")
    for start, stop in ranges:
        full = np.asarray(frame.iq[:, start:stop], dtype=np.complex128)
        if factors is not None:
            full = factors[:, None] * full
        matrix = full[np.asarray(selected_indices), :]
        if args.remove_mean:
            matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
        if frequency_range is not None:
            try:
                matrix = _band_limit(matrix, frame.manifest.sample_rate_hz, frequency_range, frame.manifest.center_frequency_hz)
            except InvalidSpatialParameters as error:
                if error.code == "EMPTY_FREQUENCY_SELECTION":
                    return _not_applicable("compute_spatial_evidence", error.code, "Frequency selection contains no FFT bins.", "Use a frequency range containing resolved bins.")
                raise
        if not np.isfinite(matrix).all():
            return _not_applicable("compute_spatial_evidence", "NO_SPATIAL_ENERGY", "Selected spatial samples are not finite.", "Select finite IQ samples.")
        segments.append(matrix)
    snapshots = sum(segment.shape[1] for segment in segments)
    if snapshots < len(selected_indices):
        return _not_applicable("compute_spatial_evidence", "INSUFFICIENT_SNAPSHOTS", "Selected simultaneous snapshots are fewer than channels.", "Select at least as many samples as channels.", {"snapshot_count": snapshots, "channel_count": len(selected_indices)})
    try:
        raw, loaded, raw_values, raw_vectors, loaded_values, loaded_vectors, stats = covariance_from_segments(segments, diagonal_loading_fraction=args.diagonal_loading_fraction)
    except SpatialError as error:
        code = "NO_SPATIAL_ENERGY" if "energy" in str(error).lower() else "NUMERICAL_FAILURE"
        status = ToolStatus.NOT_APPLICABLE if code == "NO_SPATIAL_ENERGY" else ToolStatus.ERROR
        result = ToolResult(tool_name="compute_spatial_evidence", status=status, error=ToolError(code=code, message="Spatial evidence could not be formed.", repair_hint="Select a nonzero finite spatial interval."))
        result.warnings = warnings[:_MAX_PUBLIC]
        return result
    if stats["raw_effective_rank"] < len(selected_indices):
        warnings.append("Raw spatial covariance is rank deficient or weakly conditioned.")
    if stats["loaded_condition_number"] is None:
        warnings.append("Loaded covariance condition number is undefined at numerical tolerance.")
    synchronization, sync_conflict = _resolve_synchronization(frame.manifest)
    if synchronization is not True or sync_conflict:
        warnings.append("Covariance is algebraically available, but synchronization is not verified for spatial interpretation.")
    if calibration_state == "ideal_verified":
        warnings.append("Calibration is ideal-verified; no measured per-channel correction was applied.")
    payload = SpatialEvidencePayload(
        raw, loaded, raw_values, raw_vectors, loaded_values, loaded_vectors, selected_indices, tuple(ranges), snapshots,
        float(stats["diagonal_loading"]), float(stats["rank_rtol"]), float(stats["raw_rank_threshold"]), int(stats["raw_effective_rank"]),
        float(stats["loaded_eigenvalue_tolerance"]), stats["loaded_condition_number"], synchronization, sync_conflict,
        calibration_state, calibration_usable, frame.manifest.array_geometry, frame.manifest.reference_frequency_hz,
        frame.manifest.array_geometry.reference_frequency_hz if frame.manifest.array_geometry else None,
        frame.iq.shape[0], filtered, args.remove_mean,
    )
    public_values = [float(value) for value in raw_values[:_MAX_PUBLIC]]
    metadata = {
        "channel_count": len(selected_indices), "snapshot_count": snapshots, "eigenvalue_total_count": len(raw_values),
        "eigenvalue_disclosed_count": len(public_values), "eigenvalue_truncated": len(raw_values) > len(public_values),
    }
    selection = args.model_copy(update={"sample_ranges": [SampleRange(start=start, stop=stop) for start, stop in ranges]})
    artifact_id = artifact_store.put("spatial_evidence", args.source_ref, payload, metadata, selection=selection, parameters={"ranges": ranges, "channels": selected_indices, "frequency": frequency_range.model_dump(mode="json") if frequency_range else None, "remove_mean": args.remove_mean, "loading": args.diagonal_loading_fraction}, meaning="Narrowband spatial covariance evidence; numerical arrays remain internal.")
    disclosed_warnings = warnings[:_MAX_PUBLIC]
    return ToolResult(tool_name="compute_spatial_evidence", status=ToolStatus.SUCCESS, data={"spatial_evidence_id": artifact_id, **metadata, "eigenvalues": public_values, "effective_rank": stats["raw_effective_rank"], "loaded_condition_number": stats["loaded_condition_number"], "diagonal_loading": stats["diagonal_loading"], "warning_total_count": len(warnings), "warning_disclosed_count": len(disclosed_warnings), "warnings_truncated": len(warnings) > len(disclosed_warnings)}, semantics={"eigenvalues": "Descending covariance eigenvalues before loading; this spectrum supports MDL.", "snapshot_count": "Number of simultaneous selected complex channel vectors.", "effective_rank": "Rank computed from the covariance before diagonal loading."}, quality={"finite_outputs": True, "rank_rtol": stats["rank_rtol"], "rank_threshold": stats["raw_rank_threshold"], "loaded_eigenvalue_tolerance": stats["loaded_eigenvalue_tolerance"]}, warnings=disclosed_warnings, artifacts=[artifact_store.get(artifact_id).summary], provenance={"method": "narrowband_covariance_v1", "frequency_selection": candidate_note or _frequency_note(frequency_range), "calibration_state": calibration_state, "synchronization_verified": synchronization is True and not sync_conflict, "segment_count": len(ranges)})


def estimate_doa(args: EstimateDoaArgs, *, artifact_store: RunArtifactStore) -> ToolResult:
    """Estimate signed ULA MUSIC directions from one spatial-evidence artifact."""

    record = artifact_store.get(args.spatial_evidence_id, expected_type="spatial_evidence")
    payload = record.payload
    if not isinstance(payload, SpatialEvidencePayload) or not _is_structurally_valid_spatial_payload(payload):
        raise MalformedSpatialEvidence()
    if payload.synchronization is not True or payload.synchronization_conflict:
        return _not_applicable("estimate_doa", "UNVERIFIED_SYNCHRONIZATION", "MUSIC requires explicitly verified synchronization.", "Provide verified synchronized array metadata.")
    if args.source_count >= len(payload.selected_channel_indices):
        return _not_applicable("estimate_doa", "INVALID_SOURCE_COUNT", "Source count must be smaller than selected channel count.", "Reduce source_count.")
    code, spacing, reference = _verify_ula(payload)
    if code:
        return _not_applicable("estimate_doa", code, "Array metadata cannot verify a usable ULA manifold.", "Provide consistent ULA geometry and reference frequency.")
    if not payload.calibration_usable:
        return _not_applicable("estimate_doa", "UNVERIFIED_CALIBRATION", "MUSIC requires verified calibration state.", "Provide ideal_verified or complete applied inverse calibration.")
    try:
        peaks, grid, partial = music_peaks(payload.loaded_eigenvectors, source_count=args.source_count, spacing_m=spacing, reference_frequency_hz=reference, azimuth_min_deg=args.azimuth_min_deg, azimuth_max_deg=args.azimuth_max_deg, grid_step_deg=args.grid_step_deg, peak_separation_deg=args.peak_separation_deg)
    except SpatialError:
        return ToolResult(tool_name="estimate_doa", status=ToolStatus.ERROR, error=ToolError(code="NUMERICAL_FAILURE", message="MUSIC numerical processing failed.", repair_hint="Use a better-conditioned spatial evidence selection."))
    warnings: list[str] = []
    if spacing / (299_792_458.0 / reference) > 0.5:
        warnings.append("Element spacing exceeds half wavelength; grating-lobe ambiguity is possible.")
    if abs(spacing / (299_792_458.0 / reference) - 0.5) <= 1e-12 and grid["first_deg"] <= -90 and grid["last_deg"] >= 90:
        warnings.append("Half-wavelength ULA endfire directions are ambiguous.")
    if partial:
        warnings.append("Fewer separated MUSIC peaks were found than requested source_count.")
    disclosed = peaks[:_MAX_PUBLIC]
    disclosed_warnings = warnings[:_MAX_PUBLIC]
    return ToolResult(tool_name="estimate_doa", status=ToolStatus.PARTIAL if partial else ToolStatus.SUCCESS, data={"method": "music_v1", "peaks": disclosed, "peak_total_count": len(peaks), "peak_disclosed_count": len(disclosed), "peaks_truncated": len(peaks) > len(disclosed), "grid": grid, "warning_total_count": len(warnings), "warning_disclosed_count": len(disclosed_warnings), "warnings_truncated": len(warnings) > len(disclosed_warnings)}, semantics={"peaks": "Signed broadside ULA azimuth peaks in degrees; zero is broadside."}, quality={"finite_outputs": True, "angular_grid_resolution_deg": args.grid_step_deg}, warnings=disclosed_warnings, provenance={"reference_frequency_hz": reference, "selected_channel_count": len(payload.selected_channel_indices)})


def _resolve_ranges_and_frequency(args: ComputeSpatialEvidenceArgs, sample_count: int, artifact_store: RunArtifactStore):
    ranges = [(item.start, item.stop) for item in args.sample_ranges] or [(0, sample_count)]
    if any(start < 0 or stop > sample_count for start, stop in ranges):
        raise _coded("INVALID_SPATIAL_PARAMETERS", "A selected range is outside the capture.")
    if args.candidate_set_id is None:
        return ranges, args.frequency_range, None
    record = artifact_store.get(args.candidate_set_id, expected_type="candidate_set", expected_source_ref=args.source_ref)
    if not isinstance(record.payload, CandidateSetPayload):
        raise MalformedSpatialEvidence("Candidate evidence payload is malformed.")
    payload = record.payload
    if not _is_structurally_valid_candidate_set(payload):
        raise MalformedSpatialEvidence("Candidate evidence payload is malformed.")
    if payload.source_ref != args.source_ref or payload.source_artifact_type not in {"spectrum", "time_frequency"}:
        raise SpatialEvidenceBindingError()
    upstream = artifact_store.get(
        payload.source_artifact_id,
        expected_type=payload.source_artifact_type,
        expected_source_ref=args.source_ref,
    )
    matches = [candidate for candidate in record.payload.candidates if candidate.candidate_id == args.candidate_id and candidate.source_ref == args.source_ref]
    if len(matches) != 1:
        raise SpatialEvidenceBindingError()
    candidate = matches[0]
    if (
        candidate.source_artifact_id != payload.source_artifact_id
        or candidate.source_artifact_type != payload.source_artifact_type
        or type(candidate.source_block_index) is not int
        or candidate.source_block_index < 0
        or not _is_strict_sample_range(candidate.sample_range)
        or not _is_strict_channel_indices(candidate.channel_indices)
        or candidate.frequency_range.reference != "baseband"
    ):
        raise SpatialEvidenceBindingError()
    blocks = getattr(upstream.payload, "blocks", None)
    if not isinstance(blocks, tuple) or candidate.source_block_index >= len(blocks):
        raise SpatialEvidenceBindingError()
    block = blocks[candidate.source_block_index]
    block_range = getattr(block, "sample_range", None)
    block_channels = getattr(block, "channel_indices", None)
    if (
        not _is_strict_sample_range(block_range)
        or not _is_strict_channel_indices(block_channels)
        or candidate.sample_range[0] < block_range[0]
        or candidate.sample_range[1] > block_range[1]
        or candidate.channel_indices != block_channels
    ):
        raise SpatialEvidenceBindingError()
    if not _candidate_membership_matches_block(candidate, payload.source_artifact_type, block):
        raise SpatialEvidenceBindingError()
    _ensure_nonoverlap(ranges)
    effective = [(max(start, candidate.sample_range[0]), min(stop, candidate.sample_range[1])) for start, stop in ranges]
    effective = [(start, stop) for start, stop in effective if stop > start]
    if not effective:
        return [], candidate.frequency_range, {"source": "candidate", "start_hz": candidate.frequency_range.start_hz, "stop_hz": candidate.frequency_range.stop_hz}
    return effective, candidate.frequency_range, {"source": "candidate", "start_hz": candidate.frequency_range.start_hz, "stop_hz": candidate.frequency_range.stop_hz}


def _is_strict_sample_range(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and type(value[0]) is int
        and type(value[1]) is int
        and 0 <= value[0] < value[1]
    )


def _is_structurally_valid_spatial_payload(payload: SpatialEvidencePayload) -> bool:
    channels = payload.selected_channel_indices
    physical_count = payload.physical_channel_count
    if not _is_strict_channel_indices(channels) or type(physical_count) is not int or physical_count <= 0:
        return False
    if max(channels) >= physical_count:
        return False
    count = len(channels)
    ranges = payload.segment_ranges
    snapshot_count = payload.snapshot_count
    if type(ranges) is not tuple or not ranges or not all(_is_strict_sample_range(item) for item in ranges):
        return False
    ordered = sorted(ranges)
    if any(next_start < stop for (_, stop), (next_start, _) in zip(ordered, ordered[1:])):
        return False
    if type(snapshot_count) is not int or snapshot_count != sum(stop - start for start, stop in ranges) or snapshot_count < count:
        return False
    matrix_shape = (count, count)
    if not (
        _is_complex_array(payload.raw_covariance, matrix_shape)
        and _is_complex_array(payload.loaded_covariance, matrix_shape)
        and _is_real_array(payload.raw_eigenvalues, (count,))
        and _is_complex_array(payload.raw_eigenvectors, matrix_shape)
        and _is_real_array(payload.loaded_eigenvalues, (count,))
        and _is_complex_array(payload.loaded_eigenvectors, matrix_shape)
    ):
        return False
    if not np.allclose(payload.raw_covariance, payload.raw_covariance.conj().T) or not np.allclose(payload.loaded_covariance, payload.loaded_covariance.conj().T):
        return False
    if not np.allclose(payload.raw_eigenvectors.conj().T @ payload.raw_eigenvectors, np.eye(count)) or not np.allclose(payload.loaded_eigenvectors.conj().T @ payload.loaded_eigenvectors, np.eye(count)):
        return False
    if np.any(np.diff(payload.raw_eigenvalues) > 0.0) or np.any(np.diff(payload.loaded_eigenvalues) > 0.0) or np.any(payload.raw_eigenvalues < 0.0) or np.any(payload.loaded_eigenvalues < 0.0):
        return False
    nonnegative_scalars = (
        payload.diagonal_loading,
        payload.rank_rtol,
        payload.raw_rank_threshold,
        payload.loaded_eigenvalue_tolerance,
    )
    if not all(_is_finite_float(value) and value >= 0.0 for value in nonnegative_scalars):
        return False
    if type(payload.raw_effective_rank) is not int or not 0 <= payload.raw_effective_rank <= count:
        return False
    condition = payload.loaded_condition_number
    if condition is not None and (not _is_finite_float(condition) or condition <= 0.0):
        return False
    if payload.synchronization is not None and type(payload.synchronization) is not bool:
        return False
    if type(payload.synchronization_conflict) is not bool or not _is_nonempty_string(payload.calibration_state) or type(payload.calibration_usable) is not bool:
        return False
    if type(payload.filtered) is not bool or type(payload.remove_mean) is not bool:
        return False
    if not _is_optional_positive_float(payload.manifest_reference_frequency_hz) or not _is_optional_positive_float(payload.geometry_reference_frequency_hz):
        return False
    geometry = payload.geometry
    if geometry is not None:
        if not _is_structurally_valid_geometry(geometry):
            return False
        if geometry.reference_frequency_hz != payload.geometry_reference_frequency_hz:
            return False
        if geometry.synchronized is not None and geometry.synchronized != payload.synchronization:
            return False
    return True


def _is_complex_array(value: object, shape: tuple[int, ...]) -> bool:
    return type(value) is np.ndarray and value.shape == shape and np.issubdtype(value.dtype, np.complexfloating) and bool(np.isfinite(value).all())


def _is_real_array(value: object, shape: tuple[int, ...]) -> bool:
    return type(value) is np.ndarray and value.shape == shape and np.issubdtype(value.dtype, np.floating) and bool(np.isfinite(value).all())


def _is_optional_positive_float(value: object) -> bool:
    return value is None or (_is_finite_float(value) and value > 0.0)


def _is_structurally_valid_geometry(value: object) -> bool:
    if not isinstance(value, ArrayGeometry):
        return False
    if type(value.geometry_type) is not str or value.geometry_type not in {"ula", "custom"}:
        return False
    if value.element_count is not None and (type(value.element_count) is not int or value.element_count <= 0):
        return False
    positions = value.element_positions_m
    if type(positions) is not list or not all(_is_finite_float(position) for position in positions):
        return False
    if value.element_count is not None and positions and value.element_count != len(positions):
        return False
    if value.geometry_type == "ula" and any(right <= left for left, right in zip(positions, positions[1:])):
        return False
    if not _is_optional_positive_float(value.element_spacing_m) or not _is_optional_positive_float(value.reference_frequency_hz):
        return False
    if value.synchronized is not None and type(value.synchronized) is not bool:
        return False
    return type(value.calibration) is dict


def _is_structurally_valid_candidate_set(payload: CandidateSetPayload) -> bool:
    """Validate forged dataclass internals before using them as keys or evidence."""

    if not (
        _is_nonempty_string(payload.source_ref)
        and _is_nonempty_string(payload.source_artifact_id)
        and _is_nonempty_string(payload.source_artifact_type)
        and _is_nonempty_string(payload.method)
        and _is_nonempty_string(payload.domain)
        and type(payload.candidates) is tuple
        and all(isinstance(candidate, EmissionCandidate) for candidate in payload.candidates)
    ):
        return False
    return all(_is_structurally_valid_candidate(candidate) for candidate in payload.candidates)


def _is_structurally_valid_candidate(candidate: EmissionCandidate) -> bool:
    strings = (
        candidate.candidate_id,
        candidate.source_ref,
        candidate.source_artifact_id,
        candidate.source_artifact_type,
        candidate.method,
        candidate.domain,
    )
    if not all(_is_nonempty_string(value) for value in strings):
        return False
    if type(candidate.source_block_index) is not int:
        return False
    if not _is_strict_sample_range(candidate.sample_range) or not _is_strict_channel_indices(candidate.channel_indices):
        return False
    if not _is_strict_frequency_range(candidate.frequency_range):
        return False
    if not all(
        _is_finite_float(value)
        for value in (
            candidate.peak_frequency_hz,
            candidate.peak_power_linear,
            candidate.relative_power_db,
            candidate.confidence,
        )
    ):
        return False
    if type(candidate.lower_truncated) is not bool or type(candidate.upper_truncated) is not bool:
        return False
    if not _is_membership_pair_tuple(candidate.mask_indices):
        return False
    if not _is_nonnegative_index_tuple(candidate.track_frame_indices) or not _is_nonnegative_index_tuple(candidate.track_frequency_bin_indices):
        return False
    if len(candidate.track_frame_indices) != len(candidate.track_frequency_bin_indices):
        return False
    if candidate.slope_hz_per_sample is not None and not _is_finite_float(candidate.slope_hz_per_sample):
        return False
    maximum_jump = candidate.max_observed_jump_bins
    return maximum_jump is None or (type(maximum_jump) is int and maximum_jump >= 0)


def _is_membership_pair_tuple(value: object) -> bool:
    return (
        type(value) is tuple
        and all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is int
            and type(pair[1]) is int
            and pair[0] >= 0
            and pair[1] >= 0
            for pair in value
        )
    )


def _is_nonnegative_index_tuple(value: object) -> bool:
    return type(value) is tuple and all(type(index) is int and index >= 0 for index in value)


def _candidate_membership_matches_block(candidate: EmissionCandidate, artifact_type: str, block: object) -> bool:
    mask_pairs = candidate.mask_indices
    track_pairs = tuple(zip(candidate.track_frequency_bin_indices, candidate.track_frame_indices))
    if artifact_type == "spectrum":
        return not mask_pairs and not track_pairs and candidate.slope_hz_per_sample is None and candidate.max_observed_jump_bins is None
    power = getattr(block, "power_linear", None)
    frequency_hz = getattr(block, "frequency_hz", None)
    if type(power) is not np.ndarray or power.ndim != 2 or type(frequency_hz) is not np.ndarray or frequency_hz.ndim != 1 or frequency_hz.shape[0] != power.shape[0]:
        return False
    frequency_count, frame_count = power.shape
    all_pairs = mask_pairs + track_pairs
    if any(frequency >= frequency_count or frame >= frame_count for frequency, frame in all_pairs):
        return False
    if any(not candidate.frequency_range.start_hz <= float(frequency_hz[frequency]) <= candidate.frequency_range.stop_hz for frequency, _ in all_pairs):
        return False
    if len(set(mask_pairs)) != len(mask_pairs) or len(set(track_pairs)) != len(track_pairs):
        return False
    if track_pairs and not set(track_pairs).issubset(set(mask_pairs)):
        return False
    if candidate.method == "tf_connected_regions":
        return bool(mask_pairs) and not track_pairs and candidate.slope_hz_per_sample is None and candidate.max_observed_jump_bins is None
    if candidate.method != "tf_tracks" or not mask_pairs or not track_pairs or candidate.slope_hz_per_sample is None or candidate.max_observed_jump_bins is None:
        return False
    if any(current <= previous for previous, current in zip(candidate.track_frame_indices, candidate.track_frame_indices[1:])):
        return False
    jumps = [abs(current - previous) for previous, current in zip(candidate.track_frequency_bin_indices, candidate.track_frequency_bin_indices[1:])]
    return candidate.max_observed_jump_bins == (max(jumps) if jumps else 0)


def _is_nonempty_string(value: object) -> bool:
    return type(value) is str and bool(value)


def _is_finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _is_strict_frequency_range(value: object) -> bool:
    if not isinstance(value, FrequencyRange):
        return False
    start_hz = value.start_hz
    stop_hz = value.stop_hz
    reference = value.reference
    return (
        _is_finite_float(start_hz)
        and _is_finite_float(stop_hz)
        and start_hz < stop_hz
        and type(reference) is str
        and reference in {"baseband", "capture_center", "absolute_rf"}
    )


def _is_strict_channel_indices(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(type(index) is int and index >= 0 for index in value)
        and len(set(value)) == len(value)
    )


def _ensure_nonoverlap(ranges: list[tuple[int, int]]) -> None:
    ordered = sorted(ranges)
    if any(next_start < stop for (_, stop), (next_start, _) in zip(ordered, ordered[1:])):
        raise _coded("OVERLAPPING_SAMPLE_RANGES", "Spatial sample ranges must not overlap.")


def _band_limit(matrix: np.ndarray, sample_rate_hz: float, frequency_range: FrequencyRange, center_frequency_hz: float | None) -> np.ndarray:
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0 or (center_frequency_hz is not None and not math.isfinite(center_frequency_hz)):
        raise _coded("INVALID_SPATIAL_PARAMETERS", "Capture frequency metadata is not finite.")
    if frequency_range.reference == "absolute_rf":
        if center_frequency_hz is None:
            raise _coded("INVALID_SPATIAL_PARAMETERS", "Absolute RF selection needs capture center frequency.")
        lower, upper = frequency_range.start_hz - center_frequency_hz, frequency_range.stop_hz - center_frequency_hz
    else:
        lower, upper = frequency_range.start_hz, frequency_range.stop_hz
    if lower < -sample_rate_hz / 2.0 or upper > sample_rate_hz / 2.0:
        raise _coded("INVALID_SPATIAL_PARAMETERS", "Frequency selection exceeds baseband Nyquist support.")
    bins = np.fft.fftfreq(matrix.shape[1], d=1.0 / sample_rate_hz)
    mask = (bins >= lower) & (bins <= upper)
    if not np.any(mask):
        raise _coded("EMPTY_FREQUENCY_SELECTION", "Frequency selection contains no FFT bins.")
    return np.fft.ifft(np.fft.fft(matrix, axis=1) * mask[None, :], axis=1)


def _resolve_synchronization(manifest):
    geometry_value = manifest.array_geometry.synchronized if manifest.array_geometry else None
    top_value = manifest.synchronized
    conflict = geometry_value is not None and top_value is not None and geometry_value != top_value
    return (geometry_value if geometry_value is not None else top_value), conflict


def _resolve_calibration(manifest, count: int):
    geometry = manifest.array_geometry.calibration if manifest.array_geometry else {}
    top = manifest.calibration or {}
    conflict = bool(geometry and top and geometry != top)
    value = geometry or top
    if conflict or not value:
        return "unverified", None, False, "Calibration is absent or conflicting; covariance is uncalibrated."
    if value == {"state": "ideal_verified"}:
        return "ideal_verified", None, True, ""
    if value.get("state") != "applied" or set(value) != {"state", "inverse_complex_corrections"} or not isinstance(value.get("inverse_complex_corrections"), list):
        return "unverified", None, False, "Calibration is malformed; covariance is uncalibrated."
    rows = value["inverse_complex_corrections"]
    if len(rows) != count:
        return "unverified", None, False, "Calibration corrections do not cover every physical channel."
    factors = np.empty(count, dtype=np.complex128)
    seen: set[int] = set()
    try:
        for row in rows:
            if set(row) != {"channel_index", "real", "imag"}:
                raise ValueError
            index = row["channel_index"]
            factor = complex(row["real"], row["imag"])
            if not isinstance(index, int) or index < 0 or index >= count or index in seen or not math.isfinite(factor.real) or not math.isfinite(factor.imag) or factor == 0:
                raise ValueError
            seen.add(index); factors[index] = factor
    except (TypeError, ValueError):
        return "unverified", None, False, "Calibration corrections are malformed."
    if seen != set(range(count)):
        return "unverified", None, False, "Calibration corrections do not cover every physical channel."
    return "applied", factors, True, ""


def _verify_ula(payload: SpatialEvidencePayload):
    geometry = payload.geometry
    if geometry is None:
        return "MISSING_ARRAY_GEOMETRY", 0.0, 0.0
    if geometry.geometry_type != "ula":
        return "UNSUPPORTED_ARRAY_GEOMETRY", 0.0, 0.0
    declared_count = geometry.element_count if geometry.element_count is not None else (len(geometry.element_positions_m) if geometry.element_positions_m else None)
    if declared_count != payload.physical_channel_count:
        return "ARRAY_CHANNEL_MISMATCH", 0.0, 0.0
    channels = payload.selected_channel_indices
    if tuple(sorted(channels)) != channels or any(right != left + 1 for left, right in zip(channels, channels[1:])):
        return "UNVERIFIED_UNIFORM_ULA", 0.0, 0.0
    positions = geometry.element_positions_m
    spacing = geometry.element_spacing_m
    if positions:
        deltas = np.diff(np.asarray(positions, dtype=float))
        if not np.isfinite(deltas).all() or np.any(deltas <= 0.0):
            return "UNVERIFIED_UNIFORM_ULA", 0.0, 0.0
        d = float(deltas[0])
        tolerance = max(1e-12, 1e-9 * max(abs(d), abs(deltas).max()))
        if np.any(np.abs(deltas - d) > tolerance) or (spacing is not None and abs(spacing - d) > tolerance):
            return "UNVERIFIED_UNIFORM_ULA", 0.0, 0.0
        spacing = d
    if spacing is None or not math.isfinite(spacing) or spacing <= 0.0:
        return "UNVERIFIED_UNIFORM_ULA", 0.0, 0.0
    a, b = payload.geometry_reference_frequency_hz, payload.manifest_reference_frequency_hz
    if (a is not None and (not math.isfinite(a) or a <= 0.0)) or (b is not None and (not math.isfinite(b) or b <= 0.0)):
        return "CONFLICTING_REFERENCE_FREQUENCY", 0.0, 0.0
    if a is not None and b is not None and abs(a - b) > max(1.0, 1e-9 * max(abs(a), abs(b))):
        return "CONFLICTING_REFERENCE_FREQUENCY", 0.0, 0.0
    reference = a if a is not None else b
    if reference is None:
        return "MISSING_REFERENCE_FREQUENCY", 0.0, 0.0
    return "", float(spacing), float(reference)


def _frequency_note(value: FrequencyRange | None) -> dict[str, object]:
    return {"source": "none"} if value is None else {"source": "explicit", "start_hz": value.start_hz, "stop_hz": value.stop_hz, "reference": value.reference}


def _coded(code: str, message: str) -> InvalidSpatialParameters:
    error = InvalidSpatialParameters(message); error.code = code; return error


def _not_applicable(tool_name: str, code: str, message: str, hint: str, details: dict[str, int] | None = None) -> ToolResult:
    return ToolResult(tool_name=tool_name, status=ToolStatus.NOT_APPLICABLE, error=ToolError(code=code, message=message, repair_hint=hint, details=details or {}))
