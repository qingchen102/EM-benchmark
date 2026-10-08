"""Controlled Task 6 frequency, bandwidth, and pulse estimators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal

from cityrf.legacy.dsp.detection import CandidateSetPayload, EmissionCandidate
from cityrf.legacy.dsp.estimation import (
    AmbiguousEstimationEvidence,
    CandidateEvidenceMismatch,
    EstimationBindingError,
    InsufficientFeatureSamples,
    InvalidModulationFeatureParameters,
    MalformedEstimationEvidence,
    NonFiniteEstimationResult,
    UnusablePowerEvidence,
    bin_edges,
    complex_modulation_evidence,
    envelope_modulation_evidence,
    linear_power_crossing,
    noncoherent_mean_amplitude,
    tied_peak_indices,
    validate_spectrum,
    validate_time_frequency_block,
)
from cityrf.legacy.dsp.representations import (
    SpectrumBlock,
    SpectrumPayload,
    TimeFrequencyBlock,
    TimeFrequencyPayload,
)
from cityrf.legacy.dsp.selection import select_iq
from cityrf.legacy.runtime import IQRepository, RunArtifactStore
from cityrf.legacy.schemas import (
    FrequencyRange,
    SampleRange,
    SignalSelection,
    ToolError,
    ToolResult,
    ToolStatus,
)


_MAX_PUBLIC_ESTIMATE_ITEMS = 8
_PHYSICAL_CONTROLS = {
    "noise_quantile",
    "occupied_fraction",
    "threshold_db_above_noise",
    "drop_db",
}


class EstimateFrequencyArgs(BaseModel):
    """Request one frequency estimator bound to stored evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_ref: str = Field(min_length=1)
    method: Literal["quadratic_fft_peak", "spectral_centroid", "tf_ridge_fit"]
    spectrum_id: str | None = Field(default=None, min_length=1)
    time_frequency_id: str | None = Field(default=None, min_length=1)
    candidate_set_id: str | None = Field(default=None, min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    noise_quantile: float = Field(default=0.20, gt=0.0, le=0.5)

    @model_validator(mode="after")
    def validate_frequency_evidence(self) -> Self:
        needs_spectrum = self.method in {
            "quadratic_fft_peak",
            "spectral_centroid",
        }
        if needs_spectrum != (self.spectrum_id is not None):
            raise ValueError("method requires exactly its matching primary evidence ID")
        if needs_spectrum == (self.time_frequency_id is not None):
            raise ValueError("method requires exactly its matching primary evidence ID")
        if (self.candidate_set_id is None) != (self.candidate_id is None):
            raise ValueError("candidate_set_id and candidate_id must be supplied together")
        return self


class EstimateBandwidthArgs(BaseModel):
    """Request one bandwidth definition bound to stored evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_ref: str = Field(min_length=1)
    method: Literal[
        "occupied_power",
        "threshold_edges",
        "peak_drop",
        "tf_region_width",
    ]
    spectrum_id: str | None = Field(default=None, min_length=1)
    time_frequency_id: str | None = Field(default=None, min_length=1)
    candidate_set_id: str | None = Field(default=None, min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    occupied_fraction: float = Field(default=0.99, gt=0.0, lt=1.0)
    threshold_db_above_noise: float = Field(default=6.0, gt=0.0)
    noise_quantile: float = Field(default=0.20, gt=0.0, le=0.5)
    drop_db: float = Field(default=3.0, gt=0.0)

    @model_validator(mode="after")
    def validate_bandwidth_evidence(self) -> Self:
        needs_spectrum = self.method in {
            "occupied_power",
            "threshold_edges",
            "peak_drop",
        }
        if needs_spectrum != (self.spectrum_id is not None):
            raise ValueError("method requires exactly its matching primary evidence ID")
        if needs_spectrum == (self.time_frequency_id is not None):
            raise ValueError("method requires exactly its matching primary evidence ID")
        has_candidate_set = self.candidate_set_id is not None
        has_candidate = self.candidate_id is not None
        if has_candidate_set != has_candidate:
            raise ValueError("candidate_set_id and candidate_id must be supplied together")
        if self.method == "tf_region_width" and not has_candidate:
            raise ValueError("tf_region_width requires a candidate pair")
        return self


class PulseIntervalInput(SampleRange):
    """Task 4-compatible activity range with carried boundary truncation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    boundary_truncated: bool = False

    @model_validator(mode="before")
    @classmethod
    def accept_task4_activity_entry(cls, value: object) -> object:
        if isinstance(value, PulseIntervalInput):
            return value
        if isinstance(value, SampleRange):
            return {**value.model_dump(), "boundary_truncated": False}
        if isinstance(value, Mapping):
            copied = dict(value)
            if set(copied) - {
                "start",
                "stop",
                "boundary_truncated",
                "peak_to_noise_db",
            }:
                raise ValueError("activity interval contains unsupported fields")
            if "peak_to_noise_db" in copied:
                evidence = copied.pop("peak_to_noise_db")
                if (
                    isinstance(evidence, bool)
                    or not isinstance(evidence, (int, float))
                    or not math.isfinite(float(evidence))
                ):
                    raise ValueError(
                        "activity peak-to-noise evidence must be finite"
                    )
            return copied
        return value


class EstimatePulseParametersArgs(BaseModel):
    """Request pulse statistics from direct global activity ranges."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_ref: str = Field(min_length=1)
    activity_ranges: list[PulseIntervalInput] = Field(min_length=1)
    exclude_capture_boundary_intervals: bool = True


class ExtractModulationFeaturesArgs(SignalSelection):
    """Select one IQ interval for scalar modulation-description evidence."""

    model_config = ConfigDict(extra="forbid")

    candidate_set_id: str | None = Field(default=None, min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    min_feature_samples: int = Field(default=128, ge=64, le=8192)
    max_lag_samples: int = Field(default=64, ge=4, le=512)

    @model_validator(mode="after")
    def validate_modulation_feature_scope(self) -> Self:
        has_candidate_set = self.candidate_set_id is not None
        has_candidate = self.candidate_id is not None
        if has_candidate_set != has_candidate:
            raise ValueError("candidate_set_id and candidate_id must be supplied together")
        if self.frequency_range is not None:
            raise ValueError("frequency_range is not accepted by modulation feature extraction")
        if self.max_lag_samples >= self.min_feature_samples:
            raise ValueError("max_lag_samples must be smaller than min_feature_samples")
        if len(self.channel_indices) > 16:
            raise ValueError("at most 16 explicit channel indices are accepted")
        if self.range_mode != "independent":
            raise ValueError("modulation feature extraction requires independent range mode")
        if has_candidate and self.sample_ranges:
            raise ValueError("candidate mode uses only the bound candidate sample range")
        if not has_candidate and len(self.sample_ranges) > 1:
            raise ValueError("direct mode accepts at most one sample range")
        return self


class MeasurePowerRelationsArgs(BaseModel):
    """Bind one protected prior and one spectral candidate to one spectrum block."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_ref: str = Field(min_length=1)
    spectrum_id: str = Field(min_length=1)
    candidate_set_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    noise_quantile: float = Field(default=0.20, gt=0.0, le=0.5)


@dataclass(frozen=True)
class _SpectrumView:
    sample_rate_hz: float
    center_frequency_hz: float | None
    record: Any
    payload: SpectrumPayload
    candidate: EmissionCandidate | None
    frequency_hz: np.ndarray
    power_linear: np.ndarray
    bin_edges_hz: np.ndarray
    scope_mask: np.ndarray
    scope_lower_hz: float
    scope_upper_hz: float
    observation_lower_hz: float
    observation_upper_hz: float
    scope: str


@dataclass(frozen=True)
class _TfBlockView:
    block: TimeFrequencyBlock
    frequency_hz: np.ndarray
    power_linear: np.ndarray
    frame_start_samples: np.ndarray
    frame_center_samples: np.ndarray
    bin_edges_hz: np.ndarray


@dataclass(frozen=True)
class _TfEvidence:
    sample_rate_hz: float
    center_frequency_hz: float | None
    record: Any
    payload: TimeFrequencyPayload
    blocks: tuple[_TfBlockView, ...]
    candidate: EmissionCandidate | None


def _load_frame(repository: IQRepository, source_ref: str) -> tuple[Any, float, float | None]:
    frame = repository.get(source_ref)
    raw_sample_rate_hz = frame.manifest.sample_rate_hz
    raw_center = frame.manifest.center_frequency_hz
    if not _task7_is_finite_real(raw_sample_rate_hz) or (
        raw_center is not None and not _task7_is_finite_real(raw_center)
    ):
        raise MalformedEstimationEvidence(
            "Repository sampling metadata is invalid."
        )
    sample_rate_hz = float(raw_sample_rate_hz)
    if sample_rate_hz <= 0.0:
        raise MalformedEstimationEvidence(
            "Repository sample rate is invalid."
        )
    center_frequency_hz = None if raw_center is None else float(raw_center)
    return frame, sample_rate_hz, center_frequency_hz


def _not_applicable(
    tool_name: str,
    code: str,
    message: str,
    *,
    data: dict[str, object],
    provenance: dict[str, object],
    quality: dict[str, object] | None = None,
    warnings: list[str] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        status=ToolStatus.NOT_APPLICABLE,
        data=data,
        semantics={},
        quality=quality or {},
        warnings=warnings or [],
        error=ToolError(code=code, message=message),
        provenance=provenance,
    )


def _is_integer_index(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _frequency_range(candidate: EmissionCandidate) -> tuple[float, float]:
    frequency_range = candidate.frequency_range
    if frequency_range.reference != "baseband":
        raise EstimationBindingError(
            "Candidate frequency reference is incompatible with stored evidence."
        )
    lower = float(frequency_range.start_hz)
    upper = float(frequency_range.stop_hz)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise EstimationBindingError(
            "Candidate frequency scope is invalid."
        )
    return lower, upper


def _support_intersects_scope(
    support_edges_hz: np.ndarray,
    frequency_index: int,
    lower_hz: float,
    upper_hz: float,
) -> bool:
    return max(float(support_edges_hz[frequency_index]), lower_hz) < min(
        float(support_edges_hz[frequency_index + 1]), upper_hz
    )


def _resolve_candidate(
    artifact_store: RunArtifactStore,
    args: EstimateFrequencyArgs | EstimateBandwidthArgs,
    record: Any,
    payload: SpectrumPayload | TimeFrequencyPayload,
    *,
    expected_domain: Literal["spectral", "time_frequency"],
    sample_rate_hz: float,
    tf_blocks: tuple[_TfBlockView, ...] | None = None,
) -> EmissionCandidate | None:
    if args.candidate_id is None:
        return None

    candidate_record = artifact_store.get(
        args.candidate_set_id or "",
        expected_type="candidate_set",
        expected_source_ref=args.source_ref,
    )
    candidate_payload = candidate_record.payload
    if (
        not isinstance(candidate_payload, CandidateSetPayload)
        or candidate_payload.source_ref != args.source_ref
        or candidate_payload.source_artifact_id != record.artifact_id
        or candidate_payload.source_artifact_type != record.artifact_type
        or candidate_payload.domain != expected_domain
    ):
        raise EstimationBindingError(
            "Candidate evidence binding is malformed."
        )

    matches = [
        item
        for item in candidate_payload.candidates
        if item.candidate_id == args.candidate_id
    ]
    if len(matches) != 1:
        raise EstimationBindingError(
            "Candidate identifier is invalid for this candidate artifact."
        )
    candidate = matches[0]
    valid_candidate_methods = (
        {"spectral_regions"}
        if expected_domain == "spectral"
        else {"tf_connected_regions", "tf_tracks"}
    )
    if (
        candidate.source_ref != args.source_ref
        or candidate.domain != expected_domain
        or candidate.method != candidate_payload.method
        or candidate.method not in valid_candidate_methods
        or candidate.source_artifact_id != record.artifact_id
        or candidate.source_artifact_type != record.artifact_type
        or not _is_integer_index(candidate.source_block_index)
        or not 0 <= candidate.source_block_index < len(payload.blocks)
    ):
        raise EstimationBindingError(
            "Candidate source block binding is invalid."
        )

    block = payload.blocks[candidate.source_block_index]
    if (
        block.sample_range != candidate.sample_range
        or block.channel_indices != candidate.channel_indices
    ):
        raise EstimationBindingError(
            "Candidate block provenance does not match evidence."
        )

    lower_hz, upper_hz = _frequency_range(candidate)
    if expected_domain == "spectral":
        if not isinstance(block, SpectrumBlock):
            raise MalformedEstimationEvidence(
                "Spectrum source block is malformed."
            )
        frequency_hz, power_linear = validate_spectrum(
            block.frequency_hz,
            block.power_linear,
        )
        support = bin_edges(frequency_hz, sample_rate_hz)
        if (
            candidate.mask_indices
            or candidate.track_frame_indices
            or candidate.track_frequency_bin_indices
        ):
            raise EstimationBindingError(
                "Spectral candidates cannot contain time-frequency membership."
            )
        if lower_hz < support[0] or upper_hz > support[-1]:
            raise EstimationBindingError(
                "Candidate frequency scope exceeds source-block observation support."
            )
        return candidate

    if tf_blocks is None:
        raise MalformedEstimationEvidence(
            "Validated time-frequency blocks are required for candidate binding."
        )
    block_view = tf_blocks[candidate.source_block_index]
    support = block_view.bin_edges_hz
    frequency_count, frame_count = block_view.power_linear.shape
    if lower_hz < support[0] or upper_hz > support[-1]:
        raise EstimationBindingError(
            "Candidate frequency scope exceeds source-block observation support."
        )

    mask_pairs: list[tuple[int, int]] = []
    for pair in candidate.mask_indices:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise EstimationBindingError(
                "Candidate mask membership is malformed."
            )
        frequency_index, frame_index = pair
        if (
            not _is_integer_index(frequency_index)
            or not _is_integer_index(frame_index)
            or not 0 <= frequency_index < frequency_count
            or not 0 <= frame_index < frame_count
        ):
            raise EstimationBindingError(
                "Candidate mask indices are out of bounds."
            )
        if not _support_intersects_scope(
            support,
            int(frequency_index),
            lower_hz,
            upper_hz,
        ):
            raise EstimationBindingError(
                "Candidate mask membership lies outside candidate frequency scope."
            )
        mask_pairs.append((int(frequency_index), int(frame_index)))

    track_frames = candidate.track_frame_indices
    track_bins = candidate.track_frequency_bin_indices
    if len(track_frames) != len(track_bins):
        raise EstimationBindingError(
            "Candidate track membership arrays must have equal length."
        )
    track_pairs: list[tuple[int, int]] = []
    for frame_index, frequency_index in zip(track_frames, track_bins):
        if (
            not _is_integer_index(frequency_index)
            or not _is_integer_index(frame_index)
            or not 0 <= frequency_index < frequency_count
            or not 0 <= frame_index < frame_count
        ):
            raise EstimationBindingError(
                "Candidate track indices are out of bounds."
            )
        if not _support_intersects_scope(
            support,
            int(frequency_index),
            lower_hz,
            upper_hz,
        ):
            raise EstimationBindingError(
                "Candidate track membership lies outside candidate frequency scope."
            )
        track_pairs.append((int(frequency_index), int(frame_index)))

    if mask_pairs and track_pairs and not set(track_pairs).issubset(set(mask_pairs)):
        raise EstimationBindingError(
            "Candidate track membership is inconsistent with its stored mask."
        )
    if candidate.method == "tf_tracks":
        if not track_pairs:
            raise EstimationBindingError(
                "TF-track candidates require stored track membership."
            )
        if any(
            current_frame <= previous_frame
            for (_, previous_frame), (_, current_frame) in zip(
                track_pairs,
                track_pairs[1:],
            )
        ):
            raise EstimationBindingError(
                "TF-track frame membership must be strictly increasing."
            )
    elif not mask_pairs:
        raise EstimationBindingError(
            "TF-region candidates require stored mask membership."
        )
    return candidate


def _provenance(
    record: Any,
    candidate: EmissionCandidate | None,
    args: EstimateFrequencyArgs | EstimateBandwidthArgs,
    resolved_parameters: dict[str, object],
    ignored_controls: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    result: dict[str, object] = {
        "source_evidence_id": record.artifact_id,
        "source_evidence_type": record.artifact_type,
        "resolved_parameters": resolved_parameters,
    }
    if candidate is not None:
        result.update(
            {
                "candidate_set_id": args.candidate_set_id,
                "candidate_id": candidate.candidate_id,
                "evidence_block_index": candidate.source_block_index,
            }
        )
    if ignored_controls:
        result["ignored_controls"] = list(ignored_controls)
    return result


def _evidence_data(
    candidate: EmissionCandidate | None,
    args: EstimateFrequencyArgs | EstimateBandwidthArgs,
) -> dict[str, object]:
    if candidate is None:
        return {}
    return {
        "evidence_block_index": candidate.source_block_index,
        "candidate_set_id": args.candidate_set_id,
        "candidate_id": candidate.candidate_id,
    }


def _ignored_controls(
    args: BaseModel,
    used_controls: set[str],
) -> list[str]:
    return sorted(
        name
        for name in _PHYSICAL_CONTROLS - used_controls
        if name in args.model_fields_set
    )


def _ignored_warnings(ignored_controls: list[str]) -> list[str]:
    return (
        ["Explicit method-irrelevant controls were ignored."]
        if ignored_controls
        else []
    )


def _absolute_rf(
    center_frequency_hz: float | None,
    baseband_frequency_hz: float,
    warnings: list[str],
) -> float | None:
    if center_frequency_hz is None:
        warnings.append(
            "Capture center frequency is unavailable; absolute RF is null."
        )
        return None
    absolute = center_frequency_hz + baseband_frequency_hz
    if not math.isfinite(absolute):
        raise MalformedEstimationEvidence(
            "Derived absolute RF is non-finite."
        )
    return absolute


def _frequency_semantics() -> dict[str, str]:
    """Return the shared concise contract for every successful frequency result."""

    return {
        "method": "Estimator method identifier reported in data.method.",
        "scope": "Evidence scope reported in data.scope, such as candidate or full block.",
        "baseband_frequency_hz": (
            "Estimated shifted-baseband frequency in Hz, referenced to the capture center."
        ),
        "frequency_offset_hz": (
            "Signed baseband Hz offset from the capture center; equal to baseband_frequency_hz."
        ),
        "frequency_uncertainty_hz": (
            "Method-specific frequency uncertainty in Hz; supporting evidence is in quality."
        ),
        "absolute_rf_hz": (
            "Capture center frequency plus baseband_frequency_hz; null when the capture center is unavailable."
        ),
    }


def _normalize_power(power_linear: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale finite non-negative power without changing ratio-based estimates."""

    scale = float(np.max(power_linear))
    if not math.isfinite(scale) or scale < 0.0:
        raise MalformedEstimationEvidence(
            "Estimator power scale is invalid."
        )
    if scale == 0.0:
        return np.zeros_like(power_linear, dtype=np.float64), 1.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = np.asarray(power_linear / scale, dtype=np.float64)
    except FloatingPointError as exc:
        raise MalformedEstimationEvidence(
            "Estimator power normalization overflowed."
        ) from exc
    if not np.isfinite(normalized).all():
        raise MalformedEstimationEvidence(
            "Estimator power normalization is non-finite."
        )
    return normalized, scale


def _load_spectrum_view(
    args: EstimateFrequencyArgs | EstimateBandwidthArgs,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> _SpectrumView:
    _, sample_rate_hz, center_frequency_hz = _load_frame(
        repository,
        args.source_ref,
    )
    record = artifact_store.get(
        args.spectrum_id or "",
        expected_type="spectrum",
        expected_source_ref=args.source_ref,
    )
    payload = record.payload
    if not isinstance(payload, SpectrumPayload):
        raise MalformedEstimationEvidence(
            "Spectrum artifact payload is malformed."
        )
    payload_sample_rate = float(payload.sample_rate_hz)
    if (
        not math.isfinite(payload_sample_rate)
        or payload_sample_rate <= 0.0
        or payload_sample_rate != sample_rate_hz
    ):
        raise MalformedEstimationEvidence(
            "Spectrum payload sample-rate binding is invalid."
        )

    candidate = _resolve_candidate(
        artifact_store,
        args,
        record,
        payload,
        expected_domain="spectral",
        sample_rate_hz=sample_rate_hz,
    )
    if candidate is None:
        frequency_hz, power_linear = validate_spectrum(
            payload.frequency_hz,
            payload.power_linear,
        )
        scope = "entire_artifact_aggregate"
    else:
        block = payload.blocks[candidate.source_block_index]
        frequency_hz, power_linear = validate_spectrum(
            block.frequency_hz,
            block.power_linear,
        )
        scope = "candidate"

    support_edges = bin_edges(frequency_hz, sample_rate_hz)
    observation_lower_hz = float(support_edges[0])
    observation_upper_hz = float(support_edges[-1])
    if candidate is None:
        scope_lower_hz = observation_lower_hz
        scope_upper_hz = observation_upper_hz
    else:
        scope_lower_hz, scope_upper_hz = _frequency_range(candidate)

    scope_mask = (support_edges[:-1] < scope_upper_hz) & (
        support_edges[1:] > scope_lower_hz
    )
    if not np.any(scope_mask):
        raise EstimationBindingError(
            "Candidate scope has no resolvable spectrum bins."
        )
    indices = np.flatnonzero(scope_mask)
    if indices.size > 1 and not np.all(np.diff(indices) == 1):
        raise MalformedEstimationEvidence(
            "Resolved spectrum scope must be contiguous."
        )
    return _SpectrumView(
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_frequency_hz,
        record=record,
        payload=payload,
        candidate=candidate,
        frequency_hz=frequency_hz,
        power_linear=power_linear,
        bin_edges_hz=support_edges,
        scope_mask=scope_mask,
        scope_lower_hz=scope_lower_hz,
        scope_upper_hz=scope_upper_hz,
        observation_lower_hz=observation_lower_hz,
        observation_upper_hz=observation_upper_hz,
        scope=scope,
    )


def _load_tf_evidence(
    args: EstimateFrequencyArgs | EstimateBandwidthArgs,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> _TfEvidence:
    _, sample_rate_hz, center_frequency_hz = _load_frame(
        repository,
        args.source_ref,
    )
    record = artifact_store.get(
        args.time_frequency_id or "",
        expected_type="time_frequency",
        expected_source_ref=args.source_ref,
    )
    payload = record.payload
    if not isinstance(payload, TimeFrequencyPayload):
        raise MalformedEstimationEvidence(
            "Time-frequency artifact payload is malformed."
        )
    payload_sample_rate = float(payload.sample_rate_hz)
    if (
        not math.isfinite(payload_sample_rate)
        or payload_sample_rate <= 0.0
        or payload_sample_rate != sample_rate_hz
    ):
        raise MalformedEstimationEvidence(
            "Time-frequency payload sample-rate binding is invalid."
        )
    if not payload.blocks:
        raise MalformedEstimationEvidence(
            "Time-frequency payload contains no source blocks."
        )

    validated_blocks: list[_TfBlockView] = []
    for block in payload.blocks:
        if not isinstance(block, TimeFrequencyBlock):
            raise MalformedEstimationEvidence(
                "Time-frequency source block is malformed."
            )
        frequency_hz, power_linear, starts, centers = validate_time_frequency_block(
            block.frequency_hz,
            block.power_linear,
            block.frame_start_samples,
            block.frame_center_samples,
        )
        validated_blocks.append(
            _TfBlockView(
                block=block,
                frequency_hz=frequency_hz,
                power_linear=power_linear,
                frame_start_samples=starts,
                frame_center_samples=centers,
                bin_edges_hz=bin_edges(frequency_hz, sample_rate_hz),
            )
        )
    blocks = tuple(validated_blocks)
    candidate = _resolve_candidate(
        artifact_store,
        args,
        record,
        payload,
        expected_domain="time_frequency",
        sample_rate_hz=sample_rate_hz,
        tf_blocks=blocks,
    )
    return _TfEvidence(
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_frequency_hz,
        record=record,
        payload=payload,
        blocks=blocks,
        candidate=candidate,
    )


def estimate_frequency(
    args: EstimateFrequencyArgs,
    *,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> ToolResult:
    """Estimate frequency from validated spectrum or TF evidence."""

    if args.method == "tf_ridge_fit":
        return _estimate_tf_ridge_frequency(
            args,
            repository=repository,
            artifact_store=artifact_store,
        )

    view = _load_spectrum_view(args, repository, artifact_store)
    indices = np.flatnonzero(view.scope_mask)
    used_controls = {"noise_quantile"} if args.method == "spectral_centroid" else set()
    ignored = _ignored_controls(args, used_controls)
    warnings = _ignored_warnings(ignored)
    resolved = (
        {"noise_quantile": args.noise_quantile}
        if args.method == "spectral_centroid"
        else {}
    )
    provenance = _provenance(
        view.record,
        view.candidate,
        args,
        resolved,
        ignored,
    )

    if args.method == "spectral_centroid":
        selected_power = view.power_linear[view.scope_mask]
        normalized_power, power_scale = _normalize_power(selected_power)
        baseline_normalized = float(
            np.quantile(
                normalized_power,
                args.noise_quantile,
                method="linear",
            )
        )
        weights = np.maximum(normalized_power - baseline_normalized, 0.0)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return _not_applicable(
                "estimate_frequency",
                "NO_EXCESS_SPECTRAL_POWER",
                "No excess spectral power is available for a centroid.",
                data={"method": args.method, "scope": view.scope},
                provenance=provenance,
            )
        frequencies = view.frequency_hz[view.scope_mask]
        normalized_weights = weights / total_weight
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                baseband = float(np.sum(frequencies * normalized_weights))
                frequency_scale = max(
                    1.0,
                    float(np.max(np.abs(frequencies))),
                    abs(baseband),
                )
                normalized_offsets = (
                    frequencies - baseband
                ) / frequency_scale
                normalized_variance = float(
                    np.sum(normalized_weights * normalized_offsets**2)
                )
                spread = float(
                    frequency_scale * math.sqrt(max(0.0, normalized_variance))
                )
                effective_bins = float(
                    1.0 / np.sum(normalized_weights**2)
                )
                baseline = float(baseline_normalized * power_scale)
        except (FloatingPointError, OverflowError) as exc:
            raise MalformedEstimationEvidence(
                "Spectral-centroid arithmetic overflowed."
            ) from exc
        if not all(
            math.isfinite(value)
            for value in (baseband, spread, effective_bins, baseline)
        ):
            raise MalformedEstimationEvidence(
                "Spectral-centroid arithmetic produced a non-finite value."
            )
        resolution = float(np.median(np.diff(view.frequency_hz)))
        uncertainty = max(
            resolution / 2.0,
            spread / math.sqrt(effective_bins),
        )
        absolute = _absolute_rf(
            view.center_frequency_hz,
            baseband,
            warnings,
        )
        data = {
            "method": args.method,
            "baseband_frequency_hz": baseband,
            "frequency_offset_hz": baseband,
            "absolute_rf_hz": absolute,
            "frequency_uncertainty_hz": uncertainty,
            "scope": view.scope,
            **_evidence_data(view.candidate, args),
            "method_details": {
                "baseline_power_linear": baseline,
                "frequency_spread_hz": spread,
                "effective_bin_count": effective_bins,
            },
        }
        return ToolResult(
            tool_name="estimate_frequency",
            status=ToolStatus.SUCCESS,
            data=data,
            semantics=_frequency_semantics(),
            quality={
                "finite_inputs": True,
                "in_scope_bin_count": int(indices.size),
                "frequency_resolution_hz": resolution,
            },
            warnings=warnings,
            provenance=provenance,
        )

    ties = tied_peak_indices(view.power_linear[view.scope_mask])
    if ties.size != 1:
        code = "NO_POSITIVE_PEAK" if ties.size == 0 else "AMBIGUOUS_DOMINANT_PEAK"
        return _not_applicable(
            "estimate_frequency",
            code,
            "A unique dominant spectral peak is required.",
            data={"method": args.method, "scope": view.scope},
            provenance=provenance,
            warnings=warnings,
        )
    peak_index = int(indices[int(ties[0])])
    if peak_index == 0 or peak_index == view.frequency_hz.size - 1:
        return _not_applicable(
            "estimate_frequency",
            "PEAK_AT_OBSERVATION_EDGE",
            "The dominant peak lacks an observed neighbour.",
            data={"method": args.method, "scope": view.scope},
            provenance=provenance,
            warnings=warnings,
        )
    if not (
        view.scope_mask[peak_index - 1]
        and view.scope_mask[peak_index + 1]
    ):
        return _not_applicable(
            "estimate_frequency",
            "PEAK_AT_SCOPE_EDGE",
            "The dominant peak lacks a neighbour inside scope.",
            data={"method": args.method, "scope": view.scope},
            provenance=provenance,
            warnings=warnings,
        )

    local_power = view.power_linear[peak_index - 1 : peak_index + 2]
    db_power = 10.0 * np.log10(
        np.maximum(local_power, np.finfo(np.float64).tiny)
    )
    denominator = db_power[0] - 2.0 * db_power[1] + db_power[2]
    delta_bins = (
        0.5 * (db_power[0] - db_power[2]) / denominator
        if denominator < 0.0
        else float("nan")
    )
    if not math.isfinite(delta_bins) or abs(delta_bins) >= 0.5:
        return _not_applicable(
            "estimate_frequency",
            "NO_RESOLVED_QUADRATIC_PEAK",
            "The dominant peak cannot support quadratic interpolation.",
            data={"method": args.method, "scope": view.scope},
            provenance=provenance,
            warnings=warnings,
        )
    resolution = float(np.median(np.diff(view.frequency_hz)))
    baseband = float(
        view.frequency_hz[peak_index] + delta_bins * resolution
    )
    absolute = _absolute_rf(
        view.center_frequency_hz,
        baseband,
        warnings,
    )
    diagnostics = _empty_edge_diagnostics()
    data = {
        "method": args.method,
        "baseband_frequency_hz": baseband,
        "frequency_offset_hz": baseband,
        "absolute_rf_hz": absolute,
        "frequency_uncertainty_hz": resolution / 2.0,
        "scope": view.scope,
        **_evidence_data(view.candidate, args),
        "edge_diagnostics": diagnostics,
        "method_details": {
            "interpolation_offset_bins": float(delta_bins),
        },
    }
    return ToolResult(
        tool_name="estimate_frequency",
        status=ToolStatus.SUCCESS,
        data=data,
        semantics=_frequency_semantics(),
        quality={
            "finite_inputs": True,
            "frequency_resolution_hz": resolution,
            "in_scope_bin_count": int(indices.size),
        },
        warnings=warnings,
        provenance=provenance,
    )


def _empty_edge_diagnostics() -> dict[str, bool]:
    return {
        "lower_scope_truncated": False,
        "upper_scope_truncated": False,
        "lower_observation_truncated": False,
        "upper_observation_truncated": False,
    }


def _missing_edge_diagnostics(
    view: _SpectrumView,
    *,
    lower_missing: bool,
    upper_missing: bool,
) -> dict[str, bool]:
    diagnostics = _empty_edge_diagnostics()
    tolerance = np.finfo(np.float64).eps * max(
        1.0,
        abs(view.observation_lower_hz),
        abs(view.observation_upper_hz),
    )
    if lower_missing:
        if view.scope_lower_hz > view.observation_lower_hz + tolerance:
            diagnostics["lower_scope_truncated"] = True
        else:
            diagnostics["lower_observation_truncated"] = True
    if upper_missing:
        if view.scope_upper_hz < view.observation_upper_hz - tolerance:
            diagnostics["upper_scope_truncated"] = True
        else:
            diagnostics["upper_observation_truncated"] = True
    return diagnostics


def _db_power_ratio(db_value: float) -> float:
    exponent = db_value / 10.0
    if exponent > math.log10(np.finfo(np.float64).max):
        raise MalformedEstimationEvidence(
            "Requested dB control exceeds finite numerical range."
        )
    if exponent < math.log10(np.nextafter(0.0, 1.0)):
        return 0.0
    return float(10.0**exponent)


def _bandwidth_result(
    args: EstimateBandwidthArgs,
    view_record: Any,
    candidate: EmissionCandidate | None,
    *,
    bandwidth_hz: float,
    lower_edge_hz: float,
    upper_edge_hz: float,
    definition: str,
    scope: str,
    edge_diagnostics: dict[str, bool],
    method_details: dict[str, object],
    resolved_parameters: dict[str, object],
    quality: dict[str, object],
    observation_support_hz: tuple[float, float],
    status: ToolStatus = ToolStatus.SUCCESS,
    warnings: list[str] | None = None,
    extra_data: dict[str, object] | None = None,
) -> ToolResult:
    if (
        not math.isfinite(bandwidth_hz)
        or not math.isfinite(lower_edge_hz)
        or not math.isfinite(upper_edge_hz)
        or bandwidth_hz < 0.0
        or not math.isclose(
            upper_edge_hz - lower_edge_hz,
            bandwidth_hz,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise MalformedEstimationEvidence(
            "Bandwidth result edges are inconsistent or non-finite."
        )
    result_quality = dict(quality)
    resolution = result_quality.get("frequency_resolution_hz")
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, (int, float))
        or not math.isfinite(float(resolution))
        or float(resolution) <= 0.0
    ):
        raise MalformedEstimationEvidence(
            "Bandwidth confidence requires a finite positive frequency resolution."
        )
    observation_lower, observation_upper = observation_support_hz
    if (
        not math.isfinite(observation_lower)
        or not math.isfinite(observation_upper)
        or observation_upper <= observation_lower
    ):
        raise MalformedEstimationEvidence(
            "Bandwidth confidence requires valid observable frequency support."
        )
    edge_distance_bins = min(
        max(0.0, lower_edge_hz - observation_lower),
        max(0.0, observation_upper - upper_edge_hz),
    ) / float(resolution)
    low_confidence_reasons: list[str] = []
    if any(edge_diagnostics.values()):
        low_confidence_reasons.append("reported support is truncated")
    if status is ToolStatus.PARTIAL:
        low_confidence_reasons.append("result status is partial")
    if edge_distance_bins <= 2.0 + 1.0e-12:
        low_confidence_reasons.append(
            "a reported edge is within two frequency bins of observable support"
        )
    result_quality.update(
        {
            "minimum_observation_edge_distance_bins": edge_distance_bins,
            "low_confidence": bool(low_confidence_reasons),
        }
    )

    ignored = _ignored_controls(args, set(resolved_parameters))
    result_warnings = list(warnings or []) + _ignored_warnings(ignored)
    if low_confidence_reasons:
        result_warnings.append(
            "Low confidence bandwidth estimate: "
            + "; ".join(low_confidence_reasons)
            + "."
        )
    data: dict[str, object] = {
        "method": args.method,
        "bandwidth_hz": bandwidth_hz,
        "lower_edge_hz": lower_edge_hz,
        "upper_edge_hz": upper_edge_hz,
        "definition": definition,
        "scope": scope,
        **_evidence_data(candidate, args),
        "edge_diagnostics": edge_diagnostics,
        **(extra_data or {}),
        "method_details": method_details,
    }
    return ToolResult(
        tool_name="estimate_bandwidth",
        status=status,
        data=data,
        semantics={
            "bandwidth_hz": "Baseband-Hz width using the stated definition.",
            "lower_edge_hz": "Lower representative or measured baseband-Hz edge.",
            "upper_edge_hz": "Upper representative or measured baseband-Hz edge.",
        },
        quality=result_quality,
        warnings=result_warnings,
        provenance=_provenance(
            view_record,
            candidate,
            args,
            resolved_parameters,
            ignored,
        ),
    )


def estimate_bandwidth(
    args: EstimateBandwidthArgs,
    *,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> ToolResult:
    """Estimate one explicitly named bandwidth definition."""

    if args.method == "tf_region_width":
        return _estimate_tf_region_width(
            args,
            repository=repository,
            artifact_store=artifact_store,
        )

    view = _load_spectrum_view(args, repository, artifact_store)
    global_indices = np.flatnonzero(view.scope_mask)
    selected_power = view.power_linear[view.scope_mask]
    resolution = float(np.median(np.diff(view.frequency_hz)))

    if args.method == "occupied_power":
        widths = np.diff(view.bin_edges_hz)[view.scope_mask]
        overlaps = np.maximum(
            0.0,
            np.minimum(
                view.bin_edges_hz[1:][view.scope_mask],
                view.scope_upper_hz,
            )
            - np.maximum(
                view.bin_edges_hz[:-1][view.scope_mask],
                view.scope_lower_hz,
            ),
        )
        normalized_power, _ = _normalize_power(selected_power)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if view.payload.scaling == "power_density":
                    contributions = normalized_power * overlaps
                else:
                    contributions = normalized_power * overlaps / widths
                contribution_scale = float(np.max(contributions))
                if contribution_scale > 0.0:
                    contributions = contributions / contribution_scale
        except FloatingPointError as exc:
            raise MalformedEstimationEvidence(
                "Occupied-power arithmetic overflowed."
            ) from exc
        if not np.isfinite(contributions).all():
            raise MalformedEstimationEvidence(
                "Occupied-power arithmetic produced a non-finite value."
            )
        total_power = float(np.sum(contributions))
        resolved = {"occupied_fraction": args.occupied_fraction}
        provenance = _provenance(
            view.record,
            view.candidate,
            args,
            resolved,
        )
        if total_power <= 0.0:
            return _not_applicable(
                "estimate_bandwidth",
                "NO_OCCUPIED_POWER",
                "No positive in-scope power is available.",
                data={"method": args.method, "scope": view.scope},
                provenance=provenance,
            )

        target = args.occupied_fraction * total_power
        best: tuple[float, float, float, float] | None = None
        for lower_local in range(contributions.size):
            running = 0.0
            for upper_local in range(lower_local, contributions.size):
                running += float(contributions[upper_local])
                if running >= target:
                    lower_global = int(global_indices[lower_local])
                    upper_global = int(global_indices[upper_local])
                    lower = max(
                        float(view.bin_edges_hz[lower_global]),
                        view.scope_lower_hz,
                    )
                    upper = min(
                        float(view.bin_edges_hz[upper_global + 1]),
                        view.scope_upper_hz,
                    )
                    candidate_interval = (
                        upper - lower,
                        lower,
                        upper,
                        running,
                    )
                    if best is None or candidate_interval[:3] < best[:3]:
                        best = candidate_interval
                    break
        if best is None:
            raise MalformedEstimationEvidence(
                "Occupied-power interval search did not converge."
            )
        bandwidth, lower, upper, enclosed = best
        return _bandwidth_result(
            args,
            view.record,
            view.candidate,
            bandwidth_hz=float(bandwidth),
            lower_edge_hz=float(lower),
            upper_edge_hz=float(upper),
            definition="Narrowest contiguous interval enclosing requested occupied linear power.",
            scope=view.scope,
            edge_diagnostics=_empty_edge_diagnostics(),
            method_details={"enclosed_fraction": float(enclosed / total_power)},
            resolved_parameters=resolved,
            quality={
                "finite_inputs": True,
                "in_scope_bin_count": int(global_indices.size),
                "frequency_resolution_hz": resolution,
                "edge_interpolation": "none_bin_support_edges",
            },
            observation_support_hz=(
                view.observation_lower_hz,
                view.observation_upper_hz,
            ),
        )

    ties = tied_peak_indices(selected_power)
    peak_local = int(ties[0]) if ties.size else int(np.argmax(selected_power))
    peak_global = int(global_indices[peak_local])

    if args.method == "threshold_edges":
        normalized_power, power_scale = _normalize_power(selected_power)
        baseline_normalized = float(
            np.quantile(
                normalized_power,
                args.noise_quantile,
                method="linear",
            )
        )
        threshold_ratio = _db_power_ratio(args.threshold_db_above_noise)
        threshold_exceeds_normalized_range = (
            baseline_normalized > 0.0
            and threshold_ratio > 1.0 / baseline_normalized
        )
        threshold = (
            0.0
            if baseline_normalized == 0.0
            else (
                float("inf")
                if threshold_exceeds_normalized_range
                else baseline_normalized * threshold_ratio
            )
        )
        resolved = {
            "threshold_db_above_noise": args.threshold_db_above_noise,
            "noise_quantile": args.noise_quantile,
        }
        provenance = _provenance(
            view.record,
            view.candidate,
            args,
            resolved,
        )
        if threshold_exceeds_normalized_range or normalized_power[peak_local] <= threshold:
            return _not_applicable(
                "estimate_bandwidth",
                "NO_THRESHOLD_COMPONENT",
                "No positive peak exceeds the threshold.",
                data={"method": args.method, "scope": view.scope},
                provenance=provenance,
            )

        lower_local = upper_local = peak_local
        while lower_local > 0 and normalized_power[lower_local - 1] >= threshold:
            lower_local -= 1
        while (
            upper_local < normalized_power.size - 1
            and normalized_power[upper_local + 1] >= threshold
        ):
            upper_local += 1

        lower_crossing = None
        if lower_local > 0:
            lower_crossing = linear_power_crossing(
                float(view.frequency_hz[global_indices[lower_local - 1]]),
                float(normalized_power[lower_local - 1]),
                float(view.frequency_hz[global_indices[lower_local]]),
                float(normalized_power[lower_local]),
                threshold,
            )
        upper_crossing = None
        if upper_local < selected_power.size - 1:
            upper_crossing = linear_power_crossing(
                float(view.frequency_hz[global_indices[upper_local]]),
                float(normalized_power[upper_local]),
                float(view.frequency_hz[global_indices[upper_local + 1]]),
                float(normalized_power[upper_local + 1]),
                threshold,
            )

        lower_missing = lower_crossing is None
        upper_missing = upper_crossing is None
        diagnostics = _missing_edge_diagnostics(
            view,
            lower_missing=lower_missing,
            upper_missing=upper_missing,
        )
        lower = (
            max(
                float(view.bin_edges_hz[global_indices[lower_local]]),
                view.scope_lower_hz,
            )
            if lower_crossing is None
            else max(lower_crossing, view.scope_lower_hz)
        )
        upper = (
            min(
                float(view.bin_edges_hz[global_indices[upper_local] + 1]),
                view.scope_upper_hz,
            )
            if upper_crossing is None
            else min(upper_crossing, view.scope_upper_hz)
        )

        warnings: list[str] = []
        occupied = normalized_power >= threshold
        components = np.cumsum(
            occupied & np.concatenate(([True], ~occupied[:-1]))
        )
        tied_components = {
            int(components[int(index)])
            for index in ties
            if occupied[int(index)]
        }
        if len(tied_components) > 1:
            warnings.append(
                "Equal-height disconnected maximum components were resolved to the lowest-frequency component."
            )
        status = (
            ToolStatus.PARTIAL
            if lower_missing or upper_missing
            else ToolStatus.SUCCESS
        )
        return _bandwidth_result(
            args,
            view.record,
            view.candidate,
            bandwidth_hz=float(upper - lower),
            lower_edge_hz=float(lower),
            upper_edge_hz=float(upper),
            definition="Contiguous above-noise threshold component around the deterministic dominant bin.",
            scope=view.scope,
            edge_diagnostics=diagnostics,
            method_details={
                "baseline_power_linear": float(
                    baseline_normalized * power_scale
                ),
                "threshold_power_linear": float(threshold * power_scale),
            },
            resolved_parameters=resolved,
            quality={
                "finite_inputs": True,
                "in_scope_bin_count": int(global_indices.size),
                "frequency_resolution_hz": resolution,
                "edge_interpolation": "linear_power",
            },
            observation_support_hz=(
                view.observation_lower_hz,
                view.observation_upper_hz,
            ),
            status=status,
            warnings=warnings,
        )

    resolved = {"drop_db": args.drop_db}
    provenance = _provenance(
        view.record,
        view.candidate,
        args,
        resolved,
    )
    if ties.size != 1:
        code = "NO_POSITIVE_PEAK" if ties.size == 0 else "AMBIGUOUS_DOMINANT_PEAK"
        return _not_applicable(
            "estimate_bandwidth",
            code,
            "A unique dominant peak is required.",
            data={"method": args.method, "scope": view.scope},
            provenance=provenance,
        )
    if peak_global == 0 or peak_global == view.frequency_hz.size - 1:
        return _not_applicable(
            "estimate_bandwidth",
            "PEAK_AT_OBSERVATION_EDGE",
            "The dominant peak lacks an observed neighbour.",
            data={"method": args.method, "scope": view.scope},
            provenance=provenance,
        )
    if peak_local == 0 or peak_local == selected_power.size - 1:
        return _not_applicable(
            "estimate_bandwidth",
            "PEAK_AT_SCOPE_EDGE",
            "The dominant peak lacks a neighbour inside scope.",
            data={"method": args.method, "scope": view.scope},
            provenance=provenance,
        )

    normalized_power, _ = _normalize_power(selected_power)
    threshold = float(
        normalized_power[peak_local] * _db_power_ratio(-args.drop_db)
    )
    lower_crossing = None
    for local_index in range(peak_local - 1, -1, -1):
        if normalized_power[local_index] <= threshold:
            candidate_crossing = linear_power_crossing(
                float(view.frequency_hz[global_indices[local_index]]),
                float(normalized_power[local_index]),
                float(view.frequency_hz[global_indices[local_index + 1]]),
                float(normalized_power[local_index + 1]),
                threshold,
            )
            if (
                candidate_crossing is not None
                and candidate_crossing >= view.scope_lower_hz
            ):
                lower_crossing = candidate_crossing
            break

    upper_crossing = None
    for local_index in range(peak_local, normalized_power.size - 1):
        if normalized_power[local_index + 1] <= threshold:
            candidate_crossing = linear_power_crossing(
                float(view.frequency_hz[global_indices[local_index]]),
                float(normalized_power[local_index]),
                float(view.frequency_hz[global_indices[local_index + 1]]),
                float(normalized_power[local_index + 1]),
                threshold,
            )
            if (
                candidate_crossing is not None
                and candidate_crossing <= view.scope_upper_hz
            ):
                upper_crossing = candidate_crossing
            break

    lower_missing = lower_crossing is None
    upper_missing = upper_crossing is None
    diagnostics = _missing_edge_diagnostics(
        view,
        lower_missing=lower_missing,
        upper_missing=upper_missing,
    )
    if lower_missing or upper_missing:
        data = {
            "method": args.method,
            "scope": view.scope,
            **_evidence_data(view.candidate, args),
            "edge_diagnostics": diagnostics,
        }
        return _not_applicable(
            "estimate_bandwidth",
            "NO_TWO_SIDED_PEAK_DROP_CROSSINGS",
            "Two in-scope threshold crossings are required for peak-drop bandwidth.",
            data=data,
            provenance=provenance,
            quality={
                "finite_inputs": True,
                "in_scope_bin_count": int(global_indices.size),
                "frequency_resolution_hz": resolution,
            },
        )

    definition = (
        "3 dB bandwidth from nearest two-sided peak-drop crossings."
        if args.drop_db == 3.0
        else "Nearest two-sided peak-drop bandwidth."
    )
    return _bandwidth_result(
        args,
        view.record,
        view.candidate,
        bandwidth_hz=float(upper_crossing - lower_crossing),
        lower_edge_hz=float(lower_crossing),
        upper_edge_hz=float(upper_crossing),
        definition=definition,
        scope=view.scope,
        edge_diagnostics=_empty_edge_diagnostics(),
        method_details={
            "drop_db": args.drop_db,
            "peak_frequency_hz": float(view.frequency_hz[peak_global]),
            "peak_power_db_relative": 0.0,
        },
        resolved_parameters=resolved,
        quality={
            "finite_inputs": True,
            "in_scope_bin_count": int(global_indices.size),
            "frequency_resolution_hz": resolution,
            "edge_interpolation": "linear_power",
        },
        observation_support_hz=(
            view.observation_lower_hz,
            view.observation_upper_hz,
        ),
    )


def _candidate_region_pairs(
    candidate: EmissionCandidate,
) -> list[tuple[int, int]]:
    """Return full region membership, falling back to a track-only candidate."""

    if candidate.mask_indices:
        return [
            (int(frequency), int(frame))
            for frequency, frame in candidate.mask_indices
        ]
    return [
        (int(frequency), int(frame))
        for frame, frequency in zip(
            candidate.track_frame_indices,
            candidate.track_frequency_bin_indices,
        )
    ]


def _candidate_ridge_pairs(
    block: _TfBlockView,
    candidate: EmissionCandidate,
) -> list[tuple[int, int]]:
    """Resolve exact tracks or one strongest mask bin per occupied region frame."""

    if candidate.method == "tf_tracks":
        return [
            (int(frequency_index), int(frame_index))
            for frame_index, frequency_index in zip(
                candidate.track_frame_indices,
                candidate.track_frequency_bin_indices,
            )
        ]

    by_frame: dict[int, list[int]] = {}
    for frequency_index, frame_index in candidate.mask_indices:
        by_frame.setdefault(frame_index, []).append(frequency_index)

    ridge: list[tuple[int, int]] = []
    for frame_index, frequency_indices in sorted(by_frame.items()):
        unique_indices = sorted(set(frequency_indices))
        powers = block.power_linear[unique_indices, frame_index]
        ties = tied_peak_indices(powers)
        selected_local = int(ties[0]) if ties.size else 0
        ridge.append((unique_indices[selected_local], frame_index))
    return ridge


def _estimate_tf_ridge_frequency(
    args: EstimateFrequencyArgs,
    *,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> ToolResult:
    evidence = _load_tf_evidence(args, repository, artifact_store)
    candidate = evidence.candidate
    if candidate is None:
        if len(evidence.blocks) != 1:
            raise AmbiguousEstimationEvidence(
                "Multiple time-frequency blocks require a candidate."
            )
        block_index = 0
        block = evidence.blocks[0]
        pairs: list[tuple[int, int]] = []
        for frame_index in range(block.power_linear.shape[1]):
            tied = tied_peak_indices(block.power_linear[:, frame_index])
            if tied.size:
                pairs.append((int(tied[0]), frame_index))
        scope = "single_block"
    else:
        block_index = candidate.source_block_index
        block = evidence.blocks[block_index]
        pairs = _candidate_ridge_pairs(block, candidate)
        scope = "candidate"

    provenance = _provenance(
        evidence.record,
        candidate,
        args,
        {},
    )
    pairs.sort(key=lambda pair: pair[1])
    if len(pairs) < 2:
        return _not_applicable(
            "estimate_frequency",
            "INSUFFICIENT_RIDGE_EVIDENCE",
            "At least two ridge points are required.",
            data={"method": args.method, "scope": scope},
            provenance=provenance,
        )

    time_seconds = np.asarray(
        [
            block.frame_center_samples[frame_index] / evidence.sample_rate_hz
            for _, frame_index in pairs
        ],
        dtype=np.float64,
    )
    frequency_hz = np.asarray(
        [block.frequency_hz[frequency_index] for frequency_index, _ in pairs],
        dtype=np.float64,
    )
    if np.unique(time_seconds).size < 2:
        return _not_applicable(
            "estimate_frequency",
            "INSUFFICIENT_RIDGE_EVIDENCE",
            "Distinct ridge times are required.",
            data={"method": args.method, "scope": scope},
            provenance=provenance,
        )

    design = np.column_stack((time_seconds, np.ones(time_seconds.size)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, frequency_hz, rcond=None)
    if rank < 2 or not np.isfinite(coefficients).all():
        return _not_applicable(
            "estimate_frequency",
            "INSUFFICIENT_RIDGE_EVIDENCE",
            "Ridge evidence is rank deficient.",
            data={"method": args.method, "scope": scope},
            provenance=provenance,
        )
    slope, intercept = (float(coefficients[0]), float(coefficients[1]))
    fitted = slope * time_seconds + intercept
    rms_residual = float(
        math.sqrt(np.mean((frequency_hz - fitted) ** 2))
    )
    if not math.isfinite(rms_residual):
        return _not_applicable(
            "estimate_frequency",
            "INSUFFICIENT_RIDGE_EVIDENCE",
            "Ridge residual is non-finite.",
            data={"method": args.method, "scope": scope},
            provenance=provenance,
        )

    midpoint_seconds = float((time_seconds[0] + time_seconds[-1]) / 2.0)
    baseband = float(slope * midpoint_seconds + intercept)
    warnings: list[str] = []
    absolute = _absolute_rf(
        evidence.center_frequency_hz,
        baseband,
        warnings,
    )
    summary = [
        {
            "frame_center_sample": int(block.frame_center_samples[frame_index]),
            "frequency_hz": float(block.frequency_hz[frequency_index]),
        }
        for frequency_index, frame_index in pairs[:_MAX_PUBLIC_ESTIMATE_ITEMS]
    ]
    if len(pairs) > len(summary):
        warnings.append("Ridge trajectory disclosure was truncated.")
    resolution = float(np.median(np.diff(block.frequency_hz)))
    data = {
        "method": args.method,
        "baseband_frequency_hz": baseband,
        "frequency_offset_hz": baseband,
        "absolute_rf_hz": absolute,
        "frequency_uncertainty_hz": resolution / 2.0,
        "scope": scope,
        **_evidence_data(candidate, args),
        "reference_time_sample": int(round(midpoint_seconds * evidence.sample_rate_hz)),
        "trajectory_summary": summary,
        "trajectory_total_count": len(pairs),
        "trajectory_disclosure_truncated": len(pairs) > len(summary),
        "method_details": {
            "chirp_rate_hz_per_s": slope,
            "rms_residual_hz": rms_residual,
            "fitted_point_count": len(pairs),
        },
    }
    return ToolResult(
        tool_name="estimate_frequency",
        status=ToolStatus.SUCCESS,
        data=data,
        semantics=_frequency_semantics(),
        quality={
            "finite_inputs": True,
            "fit_point_count": len(pairs),
            "rms_residual_hz": rms_residual,
            "frequency_resolution_hz": resolution,
        },
        warnings=warnings,
        provenance=provenance,
    )


def _tf_width_edge_diagnostics(
    block: _TfBlockView,
    candidate: EmissionCandidate,
    frame_edges: list[tuple[float, float]],
) -> dict[str, bool]:
    lower_scope, upper_scope = _frequency_range(candidate)
    observation_lower = float(block.bin_edges_hz[0])
    observation_upper = float(block.bin_edges_hz[-1])
    tolerance = np.finfo(np.float64).eps * max(
        1.0,
        abs(observation_lower),
        abs(observation_upper),
    )
    return {
        "lower_scope_truncated": any(
            abs(lower - lower_scope) <= tolerance
            and lower_scope > observation_lower + tolerance
            for lower, _ in frame_edges
        ),
        "upper_scope_truncated": any(
            abs(upper - upper_scope) <= tolerance
            and upper_scope < observation_upper - tolerance
            for _, upper in frame_edges
        ),
        "lower_observation_truncated": any(
            abs(lower - observation_lower) <= tolerance
            for lower, _ in frame_edges
        ),
        "upper_observation_truncated": any(
            abs(upper - observation_upper) <= tolerance
            for _, upper in frame_edges
        ),
    }


def _estimate_tf_region_width(
    args: EstimateBandwidthArgs,
    *,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> ToolResult:
    evidence = _load_tf_evidence(args, repository, artifact_store)
    candidate = evidence.candidate
    if candidate is None:
        raise EstimationBindingError(
            "TF-region width requires candidate evidence."
        )
    block = evidence.blocks[candidate.source_block_index]
    lower_scope, upper_scope = _frequency_range(candidate)
    membership = _candidate_region_pairs(candidate)
    by_frame: dict[int, list[int]] = {}
    for frequency_index, frame_index in membership:
        by_frame.setdefault(frame_index, []).append(frequency_index)
    provenance = _provenance(
        evidence.record,
        candidate,
        args,
        {},
    )
    if not by_frame:
        return _not_applicable(
            "estimate_bandwidth",
            "EMPTY_TF_REGION",
            "No occupied candidate frames are available.",
            data={"method": args.method, "scope": "candidate"},
            provenance=provenance,
        )

    records: list[dict[str, float | int]] = []
    hole_run_lengths: list[int] = []
    frames_with_internal_holes = 0
    for frame_index, frequency_indices in sorted(by_frame.items()):
        unique_indices = sorted(set(frequency_indices))
        lower = max(
            float(block.bin_edges_hz[unique_indices[0]]),
            lower_scope,
            -evidence.sample_rate_hz / 2.0,
        )
        upper = min(
            float(block.bin_edges_hz[unique_indices[-1] + 1]),
            upper_scope,
            evidence.sample_rate_hz / 2.0,
        )
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise EstimationBindingError(
                "Candidate membership has no positive clipped frequency support."
            )
        records.append(
            {
                "frame_index": frame_index,
                "frame_center_sample": int(
                    block.frame_center_samples[frame_index]
                ),
                "lower_edge_hz": lower,
                "upper_edge_hz": upper,
                "span_hz": upper - lower,
            }
        )
        frame_hole_runs = [
            right - left - 1
            for left, right in zip(unique_indices, unique_indices[1:])
            if right - left > 1
        ]
        if frame_hole_runs:
            frames_with_internal_holes += 1
            hole_run_lengths.extend(frame_hole_runs)

    ordered_by_span = sorted(
        records,
        key=lambda item: (float(item["span_hz"]), int(item["frame_index"])),
    )
    middle = len(ordered_by_span) // 2
    if len(ordered_by_span) % 2:
        central = [ordered_by_span[middle]]
    else:
        central = [ordered_by_span[middle - 1], ordered_by_span[middle]]
    representative_lower = float(
        np.mean([float(item["lower_edge_hz"]) for item in central])
    )
    representative_upper = float(
        np.mean([float(item["upper_edge_hz"]) for item in central])
    )
    median_span = representative_upper - representative_lower
    spans = np.asarray(
        [float(item["span_hz"]) for item in records],
        dtype=np.float64,
    )
    if not math.isclose(
        median_span,
        float(np.median(spans)),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise MalformedEstimationEvidence(
            "Representative TF edges do not preserve median span."
        )

    time_ordered = sorted(records, key=lambda item: int(item["frame_index"]))
    summary = [
        {
            "frame_center_sample": int(item["frame_center_sample"]),
            "span_hz": float(item["span_hz"]),
        }
        for item in time_ordered[:_MAX_PUBLIC_ESTIMATE_ITEMS]
    ]
    warnings: list[str] = []
    if len(records) > len(summary):
        warnings.append("Per-frame span disclosure was truncated.")
    if len(records) == 1 and len(membership) == 1:
        warnings.append(
            "A one-bin track span is one clipped frequency-bin support width."
        )
    diagnostics = _tf_width_edge_diagnostics(
        block,
        candidate,
        [
            (float(item["lower_edge_hz"]), float(item["upper_edge_hz"]))
            for item in records
        ],
    )
    return _bandwidth_result(
        args,
        evidence.record,
        candidate,
        bandwidth_hz=median_span,
        lower_edge_hz=representative_lower,
        upper_edge_hz=representative_upper,
        definition=(
            "Median per-frame occupied frequency span with representative edges "
            "from the median-span frame or averaged central frames."
        ),
        scope="candidate",
        edge_diagnostics=diagnostics,
        method_details={
            "minimum_span_hz": float(np.min(spans)),
            "maximum_span_hz": float(np.max(spans)),
            "valid_frame_count": len(records),
            "candidate_frame_count": len(by_frame),
            "representative_edge_rule": "median_span_frame_or_average_central_frames",
        },
        resolved_parameters={},
        quality={
            "finite_inputs": True,
            "frames_with_internal_holes": frames_with_internal_holes,
            "internal_hole_run_count": len(hole_run_lengths),
            "maximum_internal_hole_bins": max(hole_run_lengths, default=0),
            "width_dispersion_hz": float(np.max(spans) - np.min(spans)),
            "frequency_resolution_hz": float(
                np.median(np.diff(block.frequency_hz))
            ),
        },
        observation_support_hz=(
            float(block.bin_edges_hz[0]),
            float(block.bin_edges_hz[-1]),
        ),
        warnings=warnings,
        extra_data={
            "per_frame_span_summary": summary,
            "per_frame_span_total_count": len(records),
            "per_frame_span_disclosure_truncated": len(records) > len(summary),
        },
    )


def estimate_pulse_parameters(
    args: EstimatePulseParametersArgs,
    *,
    repository: IQRepository,
) -> ToolResult:
    """Estimate pulse statistics while preserving Task 4 boundary evidence."""

    frame, sample_rate_hz, _ = _load_frame(repository, args.source_ref)
    capture_num_samples = int(frame.iq.shape[1])
    intervals = args.activity_ranges
    if any(interval.stop > capture_num_samples for interval in intervals):
        raise EstimationBindingError(
            "Pulse intervals must remain within capture bounds."
        )
    if any(
        right.start <= left.start or right.start < left.stop
        for left, right in zip(intervals, intervals[1:])
    ):
        raise EstimationBindingError(
            "Pulse intervals must be strictly sorted and non-overlapping."
        )

    effective_truncation = [
        interval.boundary_truncated
        or interval.start == 0
        or interval.stop == capture_num_samples
        for interval in intervals
    ]
    included = [
        interval
        for interval, truncated in zip(intervals, effective_truncation)
        if not (
            args.exclude_capture_boundary_intervals
            and truncated
        )
    ]
    widths = np.asarray(
        [interval.length for interval in included],
        dtype=np.float64,
    )
    warnings: list[str] = []
    if widths.size == 0:
        width: dict[str, object] = {
            "status": "not_estimated",
            "reason": "no_nontruncated_pulses",
        }
    else:
        median_width = float(np.median(widths))
        minimum_width = int(np.min(widths))
        maximum_width = int(np.max(widths))
        width = {
            "median_samples": median_width,
            "minimum_samples": minimum_width,
            "maximum_samples": maximum_width,
            "median_seconds": median_width / sample_rate_hz,
            "minimum_seconds": minimum_width / sample_rate_hz,
            "maximum_seconds": maximum_width / sample_rate_hz,
        }

    if len(included) < 2:
        pri: dict[str, object] = {
            "status": "not_estimated",
            "reason": "fewer_than_two_nontruncated_pulses",
        }
        prf_hz = None
        jitter: dict[str, object] = {"status": "not_estimated"}
    else:
        pri_samples = np.diff(
            np.asarray([interval.start for interval in included], dtype=np.float64)
        )
        median_pri = float(np.median(pri_samples))
        jitter_samples = float(np.std(pri_samples))
        pri = {
            "median_samples": median_pri,
            "median_seconds": median_pri / sample_rate_hz,
        }
        prf_hz = sample_rate_hz / median_pri
        jitter = {
            "std_samples": jitter_samples,
            "std_seconds": jitter_samples / sample_rate_hz,
            "percent_of_median": 100.0 * jitter_samples / median_pri,
        }
        if len(included) == 2:
            warnings.append("PRI is based on two included pulses.")

    if any(effective_truncation) and args.exclude_capture_boundary_intervals:
        warnings.append(
            "Boundary-truncated intervals were excluded from width and PRI statistics."
        )
    summaries = [
        {
            "start_sample": interval.start,
            "stop_sample": interval.stop,
            "width_samples": interval.length,
            "boundary_truncated": bool(truncated),
        }
        for interval, truncated in list(
            zip(intervals, effective_truncation)
        )[:_MAX_PUBLIC_ESTIMATE_ITEMS]
    ]
    if len(intervals) > len(summaries):
        warnings.append("Pulse interval disclosure was truncated.")

    return ToolResult(
        tool_name="estimate_pulse_parameters",
        status=ToolStatus.SUCCESS,
        data={
            "pulse_count": len(intervals),
            "width": width,
            "pri": pri,
            "prf_hz": prf_hz,
            "duty_cycle": sum(interval.length for interval in intervals)
            / capture_num_samples,
            "jitter": jitter,
            "boundary_counts": {
                "included": len(included),
                "excluded": len(intervals) - len(included),
            },
            "capture_context": {
                "capture_num_samples": capture_num_samples,
                "sample_rate_hz": sample_rate_hz,
            },
            "interval_summaries": summaries,
            "interval_total_count": len(intervals),
            "interval_disclosure_truncated": len(intervals) > len(summaries),
        },
        semantics={
            "duty_cycle": "All supplied active samples divided by full capture samples.",
            "pri": "Successive included start-to-start intervals in samples and seconds.",
        },
        quality={
            "interval_count": len(intervals),
            "usable_pri_count": max(0, len(included) - 1),
            "exclude_capture_boundary_intervals": args.exclude_capture_boundary_intervals,
        },
        warnings=warnings,
        provenance={"source_ref": args.source_ref},
    )


@dataclass(frozen=True)
class _Task7CandidateBinding:
    """Private, fully revalidated binding from one candidate to one source block."""

    candidate_record: Any
    candidate_payload: CandidateSetPayload
    candidate: EmissionCandidate
    source_record: Any
    source_payload: SpectrumPayload | TimeFrequencyPayload
    source_block: SpectrumBlock | TimeFrequencyBlock
    support_edges_hz: np.ndarray
    sample_rate_hz: float


def extract_modulation_features(
    args: ExtractModulationFeaturesArgs,
    *,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> ToolResult:
    """Return bounded descriptive modulation evidence without classifying IQ."""

    frame, sample_rate_hz, _ = _load_frame(repository, args.source_ref)
    if args.candidate_id is None:
        segments = select_iq(frame, args)
        if len(segments) != 1:
            raise InvalidModulationFeatureParameters(
                "Feature extraction requires exactly one resolved sample range."
            )
        segment = segments[0]
        sample_start = segment.sample_range.start
        sample_stop = segment.sample_range.stop
        channel_indices = tuple(segment.channel_indices)
        selected_iq = np.asarray(segment.iq)
        preprocessing: dict[str, object] = {
            "mode": "direct_iq",
            "band_filter": "none",
            "frequency_correction": "none",
        }
        analysis_mode = "direct"
    else:
        binding = _resolve_task7_candidate(
            source_ref=args.source_ref,
            candidate_set_id=args.candidate_set_id or "",
            candidate_id=args.candidate_id,
            frame=frame,
            artifact_store=artifact_store,
        )
        candidate = binding.candidate
        candidate_channels = tuple(candidate.channel_indices)
        if args.channel_indices:
            requested_channels = tuple(args.channel_indices)
            if any(index not in candidate_channels for index in requested_channels):
                raise InvalidModulationFeatureParameters(
                    "Requested channels must be a subset of candidate channels."
                )
            channel_indices = requested_channels
        else:
            channel_indices = candidate_channels
        if len(channel_indices) == 0 or len(channel_indices) > 16:
            raise InvalidModulationFeatureParameters(
                "Resolved feature selection must contain between one and 16 channels."
            )
        sample_start, sample_stop = candidate.sample_range
        selected_iq = np.asarray(
            frame.iq[np.asarray(channel_indices), sample_start:sample_stop]
        )
        candidate_lower_hz, candidate_upper_hz = _task7_frequency_bounds(candidate)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate_center_hz = float(
                    np.float64(candidate_lower_hz) / 2.0
                    + np.float64(candidate_upper_hz) / 2.0
                )
                candidate_bandwidth_hz = float(
                    np.float64(candidate_upper_hz) - np.float64(candidate_lower_hz)
                )
        except (FloatingPointError, OverflowError) as exc:
            raise NonFiniteEstimationResult(
                "Candidate band arithmetic is non-finite."
            ) from exc
        _require_task7_finite(candidate_center_hz, candidate_bandwidth_hz)
        selected_iq = _candidate_filtered_iq(
            selected_iq,
            sample_start=sample_start,
            sample_stop=sample_stop,
            sample_rate_hz=sample_rate_hz,
            center_hz=candidate_center_hz,
            bandwidth_hz=candidate_bandwidth_hz,
            min_feature_samples=args.min_feature_samples,
        )
        preprocessing = {
            "mode": "candidate_iq",
            "candidate_set_id": args.candidate_set_id,
            "candidate_id": candidate.candidate_id,
            "source_artifact_id": binding.source_record.artifact_id,
            "source_artifact_type": binding.source_record.artifact_type,
            "source_block_index": candidate.source_block_index,
            "sample_start": sample_start,
            "sample_stop": sample_stop,
            "candidate_lower_hz": candidate_lower_hz,
            "candidate_upper_hz": candidate_upper_hz,
            "candidate_center_hz": candidate_center_hz,
            "candidate_bandwidth_hz": candidate_bandwidth_hz,
            "downconversion": "candidate_center_to_zero_hz",
            "filter_name": "candidate_bandpass_v1",
            "filter_taps": 33,
            "kaiser_beta": 8.6,
            "zero_phase": True,
        }
        analysis_mode = "candidate"

    if len(channel_indices) == 0 or len(channel_indices) > 16:
        raise InvalidModulationFeatureParameters(
            "Resolved feature selection must contain between one and 16 channels."
        )
    sample_count = sample_stop - sample_start
    if sample_count < args.min_feature_samples:
        raise InsufficientFeatureSamples(
            "The selected range is shorter than min_feature_samples."
        )

    channel_results: list[dict[str, object]] = []
    unit_counts: list[dict[str, object]] = []
    any_zero_energy = False
    any_unavailable = False
    if args.channel_mode == "mean_power":
        amplitude = noncoherent_mean_amplitude(selected_iq)
        envelope, _, _, zero_energy = envelope_modulation_evidence(amplitude)
        reason = "ZERO_ENERGY" if zero_energy else "NONCOHERENT_POWER_HAS_NO_COMPLEX_PHASE"
        unavailable = {"status": "not_applicable", "reason": reason}
        channel_results.append(
            {
                "channel_indices": list(channel_indices),
                "envelope": envelope,
                "phase_increment": dict(unavailable),
                "instantaneous_frequency": dict(unavailable),
                "cumulants": dict(unavailable),
                "spectral_shape": dict(unavailable),
                "lag_correlation_evidence": dict(unavailable),
            }
        )
        unit_counts.append(
            {
                "result_index": 0,
                "feature_samples": sample_count,
                "eligible_phase_pairs": 0,
                "spectral_bins": 0,
                "tested_lags": 0,
            }
        )
        any_zero_energy = zero_energy
        any_unavailable = True
    else:
        unit_indices = (0,) if args.channel_mode == "single" else tuple(range(len(channel_indices)))
        for result_index, local_index in enumerate(unit_indices):
            evidence, counts, zero_energy = complex_modulation_evidence(
                selected_iq[local_index],
                sample_rate_hz=sample_rate_hz,
                max_lag_samples=args.max_lag_samples,
            )
            channel_results.append(
                {
                    "channel_index": channel_indices[local_index],
                    **evidence,
                }
            )
            unit_counts.append({"result_index": result_index, **counts})
            any_zero_energy = any_zero_energy or zero_energy
            any_unavailable = any_unavailable or any(
                feature.get("status") != "success" for feature in evidence.values()
            )

    all_zero_energy = all(
        item["envelope"].get("reason") == "ZERO_ENERGY" for item in channel_results
    )
    if all_zero_energy:
        status = ToolStatus.NOT_APPLICABLE
        error = ToolError(
            code="ZERO_ENERGY",
            message="Selected IQ has no positive energy for feature extraction.",
        )
    elif any_unavailable or any_zero_energy:
        status = ToolStatus.PARTIAL
        error = None
    else:
        status = ToolStatus.SUCCESS
        error = None
    warnings: list[str] = []
    if args.channel_mode == "mean_power":
        warnings.append(
            "Non-coherent mean power has no complex phase, cumulant, spectral, or lag evidence."
        )
    elif any_unavailable:
        warnings.append("At least one otherwise applicable feature is unavailable.")

    return ToolResult(
        tool_name="extract_modulation_features",
        status=status,
        data={
            "feature_set": "modulation_evidence_v1",
            "analysis_mode": analysis_mode,
            "sample_range": {"start": sample_start, "stop": sample_stop},
            "channel_mode": args.channel_mode,
            "channel_result_count": len(channel_results),
            "channel_results": channel_results,
        },
        semantics={
            "feature_set": "Scalar descriptive evidence only; it is not a modulation or waveform classifier.",
            "sample_range": "One global half-open capture sample interval used by every returned feature.",
            "envelope": "Linear IQ-amplitude summaries and unitless robust shape evidence.",
            "phase_increment": "Eligible adjacent-sample circular phase-increment statistics in radians.",
            "instantaneous_frequency": "Adjacent-pair frequency evidence in Hz; it is not a carrier estimate.",
            "cumulants": "Scale-normalized central complex cumulant evidence without a class label.",
            "spectral_shape": "Private Hann-periodogram flatness and bounded peak summaries in baseband Hz.",
            "lag_correlation_evidence": "Generic normalized complex lag correlation without protocol interpretation.",
        },
        quality={
            "finite_outputs": True,
            "unit_counts": unit_counts,
        },
        warnings=warnings,
        error=error,
        provenance={
            "preprocessing": preprocessing,
            "sample_rate_hz": sample_rate_hz,
        },
    )


def measure_power_relations(
    args: MeasurePowerRelationsArgs,
    *,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> ToolResult:
    """Integrate protected/candidate observed power in one bound spectrum block."""

    frame, sample_rate_hz, _ = _load_frame(repository, args.source_ref)
    protected = frame.manifest.protected_channel
    if protected is None:
        return ToolResult(
            tool_name="measure_power_relations",
            status=ToolStatus.NOT_APPLICABLE,
            data={"method": "integrated_relative_power_v1"},
            semantics={},
            quality={},
            error=ToolError(
                code="PROTECTED_CHANNEL_UNAVAILABLE",
                message="Protected-channel configuration is unavailable.",
            ),
            provenance={},
        )

    binding = _resolve_task7_candidate(
        source_ref=args.source_ref,
        candidate_set_id=args.candidate_set_id,
        candidate_id=args.candidate_id,
        frame=frame,
        artifact_store=artifact_store,
        required_spectrum_id=args.spectrum_id,
        power_consumer=True,
    )
    if not isinstance(binding.source_payload, SpectrumPayload) or not isinstance(
        binding.source_block, SpectrumBlock
    ):
        raise CandidateEvidenceMismatch(
            "Power relations require a bound spectral candidate."
        )
    block = binding.source_block
    try:
        frequency_hz, power_linear = validate_spectrum(
            block.frequency_hz,
            block.power_linear,
        )
        edges_hz = bin_edges(frequency_hz, sample_rate_hz)
    except MalformedEstimationEvidence as exc:
        raise UnusablePowerEvidence(
            "Spectrum block cannot support power integration."
        ) from exc
    if (
        not isinstance(binding.source_payload.scaling, str)
        or binding.source_payload.scaling not in {
            "complex",
            "magnitude",
            "power",
            "power_density",
        }
    ):
        raise UnusablePowerEvidence("Spectrum scaling is unsupported for power integration.")

    if not _task7_is_finite_real(
        protected.center_offset_hz
    ) or not _task7_is_finite_real(protected.nominal_bandwidth_hz):
        raise UnusablePowerEvidence("Protected-channel configuration is numerically invalid.")
    protected_center_hz = float(protected.center_offset_hz)
    protected_width_hz = float(protected.nominal_bandwidth_hz)
    if protected_width_hz <= 0.0:
        raise UnusablePowerEvidence("Protected-channel configuration is numerically invalid.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            protected_half_width_hz = np.float64(protected_width_hz) / 2.0
            protected_requested = (
                float(np.float64(protected_center_hz) - protected_half_width_hz),
                float(np.float64(protected_center_hz) + protected_half_width_hz),
            )
    except (FloatingPointError, OverflowError) as exc:
        raise NonFiniteEstimationResult(
            "Protected-band endpoint arithmetic is non-finite."
        ) from exc
    _require_task7_finite(protected_requested)
    candidate_requested = _task7_frequency_bounds(binding.candidate)
    observable_support = (float(edges_hz[0]), float(edges_hz[-1]))
    protected_effective = _band_intersection(protected_requested, observable_support)
    candidate_effective = _band_intersection(candidate_requested, observable_support)
    if protected_effective is None or candidate_effective is None:
        return _power_not_applicable(
            args,
            binding,
            code="BAND_UNOBSERVED",
            message="A requested analysis band is outside observable support.",
            protected_requested=protected_requested,
            candidate_requested=candidate_requested,
            observable_support=observable_support,
        )

    bin_widths = np.diff(edges_hz)
    if not np.isfinite(bin_widths).all() or np.any(bin_widths <= 0.0):
        raise UnusablePowerEvidence("Spectrum bin support contains zero-width bins.")
    scale = float(np.max(power_linear))
    if not math.isfinite(scale) or scale < 0.0:
        raise UnusablePowerEvidence("Spectrum power scale is invalid.")
    normalized_power = power_linear / scale if scale > 0.0 else np.zeros_like(power_linear)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            density_normalized = (
                normalized_power
                if binding.source_payload.scaling == "power_density"
                else normalized_power / bin_widths
            )
            protected_observed_normalized = _integrate_density_band(
                density_normalized, edges_hz, protected_effective
            )
            candidate_observed_normalized = _integrate_density_band(
                density_normalized, edges_hz, candidate_effective
            )
            protected_observed = scale * protected_observed_normalized
            candidate_observed = scale * candidate_observed_normalized
    except (FloatingPointError, OverflowError) as exc:
        raise NonFiniteEstimationResult(
            "Observed-band integration is non-finite."
        ) from exc
    _require_task7_finite(protected_observed, candidate_observed, density_normalized)

    protected_coverage = _coverage_fraction(protected_requested, protected_effective)
    candidate_coverage = _coverage_fraction(candidate_requested, candidate_effective)
    overlap = _band_intersection(protected_effective, candidate_effective)
    overlap_width = 0.0 if overlap is None else overlap[1] - overlap[0]
    protected_effective_width = protected_effective[1] - protected_effective[0]
    candidate_effective_width = candidate_effective[1] - candidate_effective[0]

    noise_mask = np.asarray(
        [
            _band_disjoint((float(edges_hz[index]), float(edges_hz[index + 1])), protected_effective)
            and _band_disjoint(
                (float(edges_hz[index]), float(edges_hz[index + 1])),
                candidate_effective,
            )
            for index in range(power_linear.size)
        ],
        dtype=bool,
    )
    noise_values = density_normalized[noise_mask]
    warnings: list[str] = []
    noise_density: float | None
    protected_noise: float | None
    candidate_noise: float | None
    protected_excess: float | None
    candidate_excess: float | None
    if noise_values.size == 0:
        noise_density = None
        protected_noise = None
        candidate_noise = None
        protected_excess = None
        candidate_excess = None
        noise_status = "unavailable"
        noise_reason: str | None = "NO_DISJOINT_NOISE_BINS"
    else:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                noise_normalized = float(
                    np.quantile(noise_values, args.noise_quantile, method="linear")
                )
                noise_density = scale * noise_normalized
                protected_noise = noise_density * protected_effective_width
                candidate_noise = noise_density * candidate_effective_width
                protected_excess = max(protected_observed - protected_noise, 0.0)
                candidate_excess = max(candidate_observed - candidate_noise, 0.0)
        except (FloatingPointError, OverflowError) as exc:
            raise NonFiniteEstimationResult(
                "Noise-baseline arithmetic is non-finite."
            ) from exc
        _require_task7_finite(
            noise_density,
            protected_noise,
            candidate_noise,
            protected_excess,
            candidate_excess,
        )
        noise_status = "success"
        noise_reason = None

    tolerance_hz = 1.0e-12 * sample_rate_hz
    fully_observed = (
        (protected_requested[1] - protected_requested[0])
        - protected_effective_width
        <= tolerance_hz
        and (candidate_requested[1] - candidate_requested[0])
        - candidate_effective_width
        <= tolerance_hz
    )
    ratio_status: str
    ratio_reason: str | None
    j_over_s_db: float | None = None
    sir_db: float | None = None
    if noise_values.size == 0:
        warnings.append("No fully disjoint noise bins are available for correction.")
    if overlap_width > 0.0:
        ratio_status = "unresolvable"
        ratio_reason = "BAND_OVERLAP_UNSEPARATED"
        warnings.append("Overlapping observed bands cannot be separated from mixed IQ.")
    elif not fully_observed:
        ratio_status = "unavailable"
        ratio_reason = "BAND_SUPPORT_CLIPPED"
        warnings.append("Requested band support is clipped; ratios are unavailable.")
    elif noise_values.size == 0:
        ratio_status = "unavailable"
        ratio_reason = "NO_DISJOINT_NOISE_BINS"
    elif protected_excess is None or candidate_excess is None or protected_excess <= 0.0 or candidate_excess <= 0.0:
        ratio_status = "unavailable"
        ratio_reason = "NONPOSITIVE_EXCESS_POWER"
        warnings.append("Noise-corrected excess power is not positive in both bands.")
    else:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                j_over_s_db = float(
                    10.0
                    * (
                        math.log10(candidate_excess)
                        - math.log10(protected_excess)
                    )
                )
                sir_db = -j_over_s_db
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise NonFiniteEstimationResult(
                "Power-ratio logarithmic arithmetic is non-finite."
            ) from exc
        _require_task7_finite(j_over_s_db, sir_db)
        ratio_status = "success"
        ratio_reason = None

    result_status = ToolStatus.SUCCESS if ratio_status == "success" else ToolStatus.PARTIAL
    return ToolResult(
        tool_name="measure_power_relations",
        status=result_status,
        data={
            "method": "integrated_relative_power_v1",
            "protected_band": _power_band_summary(
                protected_requested,
                protected_effective,
                protected_coverage,
                protected_observed,
                protected_noise,
                protected_excess,
            ),
            "candidate_band": _power_band_summary(
                candidate_requested,
                candidate_effective,
                candidate_coverage,
                candidate_observed,
                candidate_noise,
                candidate_excess,
            ),
            "noise_evidence": {
                "status": noise_status,
                "reason": noise_reason,
                "density_baseline": noise_density,
                "noise_bin_count": int(noise_values.size),
                "noise_quantile": args.noise_quantile,
            },
            "overlap": {
                "overlap_width_hz": overlap_width,
                "protected_overlap_fraction": overlap_width / protected_effective_width,
                "candidate_overlap_fraction": overlap_width / candidate_effective_width,
            },
            "ratios": {
                "status": ratio_status,
                "reason": ratio_reason,
                "j_over_s_db": j_over_s_db,
                "sir_db": sir_db,
            },
        },
        semantics={
            "observed_power": "Integrated relative capture-power units in one bound spectrum block.",
            "noise_power": "Lower-quantile density baseline integrated over the effective band width.",
            "excess_power": "Observed band capture power minus integrated noise, floored at zero.",
            "coverage_fraction": "Effective observed band width divided by requested band width.",
            "j_over_s_db": "Candidate-to-protected noise-corrected observed-band capture-power ratio in dB, not separated transmitter power.",
            "sir_db": "Protected-to-candidate noise-corrected observed-band capture-power ratio in dB and the negative of J over S.",
            "overlap": "Effective-band overlap with separate protected and candidate width denominators.",
        },
        quality={
            "finite_inputs": True,
            "frequency_bin_count": int(frequency_hz.size),
            "noise_bin_count": int(noise_values.size),
            "spectrum_scaling": binding.source_payload.scaling,
            "fully_observed": fully_observed,
            "bands_disjoint": overlap_width == 0.0,
        },
        warnings=warnings,
        provenance={
            "spectrum_id": binding.source_record.artifact_id,
            "candidate_set_id": args.candidate_set_id,
            "candidate_id": binding.candidate.candidate_id,
            "evidence_block_index": binding.candidate.source_block_index,
            "source_evidence_type": "spectrum",
        },
    )


def _task7_is_finite_real(value: object) -> bool:
    """Return whether an internal scalar is a finite, non-Boolean real."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _validated_task7_sample_range(
    value: object,
    *,
    capture_sample_count: int,
    description: str,
) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise CandidateEvidenceMismatch(f"{description} sample range is malformed.")
    start, stop = value
    if not _is_integer_index(start) or not _is_integer_index(stop):
        raise CandidateEvidenceMismatch(f"{description} sample range is malformed.")
    resolved = (int(start), int(stop))
    if resolved[0] < 0 or resolved[1] <= resolved[0] or resolved[1] > capture_sample_count:
        raise CandidateEvidenceMismatch(
            f"{description} sample range lies outside the capture."
        )
    return resolved


def _validated_task7_channels(
    value: object,
    *,
    capture_channel_count: int,
    description: str,
) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise CandidateEvidenceMismatch(f"{description} channels are malformed.")
    if any(not _is_integer_index(index) for index in value):
        raise CandidateEvidenceMismatch(f"{description} channels are malformed.")
    resolved = tuple(int(index) for index in value)
    if len(set(resolved)) != len(resolved) or any(
        index < 0 or index >= capture_channel_count for index in resolved
    ):
        raise CandidateEvidenceMismatch(
            f"{description} channels lie outside the capture or contain duplicates."
        )
    return resolved


def _validate_task7_candidate_set_header(
    payload: CandidateSetPayload,
    *,
    source_ref: str,
) -> tuple[str, str]:
    string_fields = (
        payload.source_ref,
        payload.source_artifact_id,
        payload.source_artifact_type,
        payload.method,
        payload.domain,
    )
    if any(not isinstance(value, str) or not value for value in string_fields):
        raise CandidateEvidenceMismatch("Candidate-set provenance is malformed.")
    if payload.source_ref != source_ref:
        raise CandidateEvidenceMismatch("Candidate-set source binding is malformed.")
    if payload.source_artifact_type not in {"spectrum", "time_frequency"}:
        raise CandidateEvidenceMismatch("Candidate-set upstream type is unsupported.")
    expected_domain = (
        "spectral" if payload.source_artifact_type == "spectrum" else "time_frequency"
    )
    valid_methods = (
        {"spectral_regions"}
        if payload.source_artifact_type == "spectrum"
        else {"tf_connected_regions", "tf_tracks"}
    )
    if payload.domain != expected_domain or payload.method not in valid_methods:
        raise CandidateEvidenceMismatch(
            "Candidate-set method or evidence domain is inconsistent."
        )
    if type(payload.candidates) is not tuple:
        raise CandidateEvidenceMismatch("Candidate collection must be a tuple.")
    if any(
        not isinstance(candidate, EmissionCandidate)
        or not isinstance(candidate.candidate_id, str)
        for candidate in payload.candidates
    ):
        raise CandidateEvidenceMismatch("Candidate collection contains malformed entries.")
    candidate_ids = tuple(candidate.candidate_id for candidate in payload.candidates)
    expected_candidate_ids = tuple(
        f"emission_{index}" for index in range(len(payload.candidates))
    )
    if candidate_ids != expected_candidate_ids:
        raise CandidateEvidenceMismatch(
            "Candidate collection does not follow the Task 4 ID contract."
        )
    return expected_domain, payload.source_artifact_type


def _validate_task7_candidate_structure(
    candidate: EmissionCandidate,
    *,
    capture_sample_count: int,
    capture_channel_count: int,
) -> tuple[tuple[int, int], tuple[int, ...], tuple[float, float]]:
    required_strings = (
        candidate.candidate_id,
        candidate.source_ref,
        candidate.source_artifact_id,
        candidate.source_artifact_type,
        candidate.method,
        candidate.domain,
    )
    if any(not isinstance(value, str) or not value for value in required_strings):
        raise CandidateEvidenceMismatch("Candidate provenance contains malformed strings.")
    if not _is_integer_index(candidate.source_block_index):
        raise CandidateEvidenceMismatch("Candidate source block index is malformed.")
    sample_range = _validated_task7_sample_range(
        candidate.sample_range,
        capture_sample_count=capture_sample_count,
        description="Candidate",
    )
    channels = _validated_task7_channels(
        candidate.channel_indices,
        capture_channel_count=capture_channel_count,
        description="Candidate",
    )
    bounds = _task7_frequency_bounds(candidate)
    if any(
        not _task7_is_finite_real(value)
        for value in (
            candidate.peak_frequency_hz,
            candidate.peak_power_linear,
            candidate.relative_power_db,
            candidate.confidence,
        )
    ):
        raise CandidateEvidenceMismatch("Candidate scalar evidence is malformed.")
    if candidate.peak_power_linear < 0.0 or not 0.0 <= candidate.confidence <= 1.0:
        raise CandidateEvidenceMismatch("Candidate scalar evidence is outside its domain.")
    if type(candidate.lower_truncated) is not bool or type(candidate.upper_truncated) is not bool:
        raise CandidateEvidenceMismatch("Candidate truncation flags are malformed.")
    if (
        type(candidate.mask_indices) is not tuple
        or type(candidate.track_frame_indices) is not tuple
        or type(candidate.track_frequency_bin_indices) is not tuple
    ):
        raise CandidateEvidenceMismatch("Candidate membership containers must be tuples.")
    if candidate.slope_hz_per_sample is not None and not _task7_is_finite_real(
        candidate.slope_hz_per_sample
    ):
        raise CandidateEvidenceMismatch("Candidate slope evidence is malformed.")
    if candidate.max_observed_jump_bins is not None and (
        not _is_integer_index(candidate.max_observed_jump_bins)
        or int(candidate.max_observed_jump_bins) < 0
    ):
        raise CandidateEvidenceMismatch("Candidate jump evidence is malformed.")
    return sample_range, channels, bounds


def _task7_upstream_error(
    *,
    power_consumer: bool,
    message: str,
) -> MalformedEstimationEvidence:
    error_type = UnusablePowerEvidence if power_consumer else MalformedEstimationEvidence
    return error_type(message)


def _resolve_task7_candidate(
    *,
    source_ref: str,
    candidate_set_id: str,
    candidate_id: str,
    frame: Any,
    artifact_store: RunArtifactStore,
    required_spectrum_id: str | None = None,
    power_consumer: bool = False,
) -> _Task7CandidateBinding:
    """Revalidate the full Task 4 candidate/upstream provenance chain."""

    candidate_record = artifact_store.get(
        candidate_set_id,
        expected_type="candidate_set",
        expected_source_ref=source_ref,
    )
    payload = candidate_record.payload
    if not isinstance(payload, CandidateSetPayload):
        raise CandidateEvidenceMismatch("Candidate-set payload binding is malformed.")
    expected_domain, source_artifact_type = _validate_task7_candidate_set_header(
        payload,
        source_ref=source_ref,
    )
    if required_spectrum_id is not None and (
        source_artifact_type != "spectrum"
        or payload.source_artifact_id != required_spectrum_id
    ):
        raise CandidateEvidenceMismatch("Candidate set is not bound to the supplied spectrum.")
    source_record = artifact_store.get(
        payload.source_artifact_id,
        expected_type=source_artifact_type,
        expected_source_ref=source_ref,
    )
    source_payload = source_record.payload
    expected_payload_type = (
        SpectrumPayload if source_artifact_type == "spectrum" else TimeFrequencyPayload
    )
    if not isinstance(source_payload, expected_payload_type):
        raise _task7_upstream_error(
            power_consumer=power_consumer,
            message="Candidate upstream payload is malformed.",
        )
    if not _task7_is_finite_real(source_payload.sample_rate_hz):
        raise _task7_upstream_error(
            power_consumer=power_consumer,
            message="Candidate upstream sample-rate binding is invalid.",
        )
    if type(source_payload.blocks) is not tuple or not source_payload.blocks:
        raise _task7_upstream_error(
            power_consumer=power_consumer,
            message="Candidate upstream block collection is malformed.",
        )
    payload_sample_rate = float(source_payload.sample_rate_hz)
    sample_rate_hz = float(frame.manifest.sample_rate_hz)
    if (
        payload_sample_rate <= 0.0
        or payload_sample_rate != sample_rate_hz
    ):
        raise _task7_upstream_error(
            power_consumer=power_consumer,
            message="Candidate upstream sample-rate binding is invalid.",
        )

    matches = [candidate for candidate in payload.candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise CandidateEvidenceMismatch("Candidate identifier is invalid for this artifact.")
    candidate = matches[0]
    capture_channel_count, capture_sample_count = frame.iq.shape
    candidate_range, candidate_channels, (
        candidate_lower_hz,
        candidate_upper_hz,
    ) = _validate_task7_candidate_structure(
        candidate,
        capture_sample_count=capture_sample_count,
        capture_channel_count=capture_channel_count,
    )
    if (
        candidate.source_ref != source_ref
        or candidate.source_artifact_id != source_record.artifact_id
        or candidate.source_artifact_type != source_record.artifact_type
        or candidate.domain != expected_domain
        or candidate.method != payload.method
        or not 0 <= int(candidate.source_block_index) < len(source_payload.blocks)
    ):
        raise CandidateEvidenceMismatch("Candidate source provenance is inconsistent.")
    source_block = source_payload.blocks[int(candidate.source_block_index)]
    expected_block_type = (
        SpectrumBlock if source_artifact_type == "spectrum" else TimeFrequencyBlock
    )
    if not isinstance(source_block, expected_block_type):
        raise _task7_upstream_error(
            power_consumer=power_consumer,
            message="Candidate source block is malformed.",
        )
    block_range = _validated_task7_sample_range(
        source_block.sample_range,
        capture_sample_count=capture_sample_count,
        description="Source block",
    )
    block_channels = _validated_task7_channels(
        source_block.channel_indices,
        capture_channel_count=capture_channel_count,
        description="Source block",
    )
    sample_start, sample_stop = candidate_range
    block_start, block_stop = block_range
    if (
        sample_start < block_start
        or sample_stop > block_stop
        or candidate_channels != block_channels
    ):
        raise CandidateEvidenceMismatch("Candidate range or channel provenance is inconsistent.")

    if isinstance(source_block, TimeFrequencyBlock):
        if not _is_integer_index(source_payload.window_length) or int(
            source_payload.window_length
        ) <= 0:
            raise CandidateEvidenceMismatch("TF window length is malformed.")
        raw_frame_starts = source_block.frame_start_samples
        if (
            not isinstance(raw_frame_starts, np.ndarray)
            or raw_frame_starts.ndim != 1
            or np.issubdtype(raw_frame_starts.dtype, np.bool_)
            or not np.issubdtype(raw_frame_starts.dtype, np.integer)
        ):
            raise CandidateEvidenceMismatch(
                "TF frame-start samples must be a one-dimensional integer array."
            )
        if raw_frame_starts.size > 1 and np.any(
            raw_frame_starts[1:] <= raw_frame_starts[:-1]
        ):
            raise CandidateEvidenceMismatch(
                "TF frame-start mapping must be strictly increasing."
            )
        if np.any(raw_frame_starts < block_start) or np.any(
            raw_frame_starts >= block_stop
        ):
            raise CandidateEvidenceMismatch(
                "TF frame-start mapping lies outside its source block."
            )

    try:
        if isinstance(source_block, SpectrumBlock):
            frequency_hz, _ = validate_spectrum(
                source_block.frequency_hz,
                source_block.power_linear,
            )
            support_edges = bin_edges(frequency_hz, sample_rate_hz)
        else:
            frequency_hz, _, _, _ = validate_time_frequency_block(
                source_block.frequency_hz,
                source_block.power_linear,
                source_block.frame_start_samples,
                source_block.frame_center_samples,
            )
            support_edges = bin_edges(frequency_hz, sample_rate_hz)
    except (
        MalformedEstimationEvidence,
        TypeError,
        ValueError,
        OverflowError,
        FloatingPointError,
    ) as exc:
        raise _task7_upstream_error(
            power_consumer=power_consumer,
            message="Candidate source numerical evidence is malformed.",
        ) from exc
    if candidate_lower_hz < support_edges[0] or candidate_upper_hz > support_edges[-1]:
        raise CandidateEvidenceMismatch("Candidate frequency bounds exceed observable support.")

    if isinstance(source_block, SpectrumBlock):
        if candidate_range != block_range:
            raise CandidateEvidenceMismatch("Spectral candidate range must equal its source block.")
        if (
            candidate.method != "spectral_regions"
            or candidate.mask_indices
            or candidate.track_frame_indices
            or candidate.track_frequency_bin_indices
        ):
            raise CandidateEvidenceMismatch("Spectral candidate membership is malformed.")
    else:
        pairs = _validated_task7_tf_membership(
            candidate,
            source_block,
            source_payload,
            support_edges,
            candidate_lower_hz,
            candidate_upper_hz,
        )
        frame_indices = [frame_index for _, frame_index in pairs]
        frame_starts = source_block.frame_start_samples
        expected_start = int(frame_starts[min(frame_indices)])
        expected_stop = min(
            block_stop,
            int(frame_starts[max(frame_indices)]) + int(source_payload.window_length),
        )
        if candidate_range != (expected_start, expected_stop):
            raise CandidateEvidenceMismatch(
                "TF candidate range does not match stored membership frame mapping."
            )

    return _Task7CandidateBinding(
        candidate_record=candidate_record,
        candidate_payload=payload,
        candidate=candidate,
        source_record=source_record,
        source_payload=source_payload,
        source_block=source_block,
        support_edges_hz=support_edges,
        sample_rate_hz=sample_rate_hz,
    )


def _validated_task7_tf_membership(
    candidate: EmissionCandidate,
    block: TimeFrequencyBlock,
    payload: TimeFrequencyPayload,
    support_edges_hz: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> list[tuple[int, int]]:
    power_shape = np.asarray(block.power_linear).shape
    if len(power_shape) != 2:
        raise MalformedEstimationEvidence("TF power evidence is malformed.")
    frequency_count, frame_count = power_shape

    def validate_pairs(values: object) -> list[tuple[int, int]]:
        if type(values) is not tuple:
            raise CandidateEvidenceMismatch(
                "TF candidate membership container must be a tuple."
            )
        resolved: list[tuple[int, int]] = []
        for pair in values:
            if type(pair) is not tuple or len(pair) != 2:
                raise CandidateEvidenceMismatch("TF candidate membership is malformed.")
            frequency_index, frame_index = pair
            if (
                not _is_integer_index(frequency_index)
                or not _is_integer_index(frame_index)
                or not 0 <= int(frequency_index) < frequency_count
                or not 0 <= int(frame_index) < frame_count
            ):
                raise CandidateEvidenceMismatch("TF candidate membership is out of bounds.")
            frequency_index = int(frequency_index)
            frame_index = int(frame_index)
            if not _support_intersects_scope(
                support_edges_hz,
                frequency_index,
                lower_hz,
                upper_hz,
            ):
                raise CandidateEvidenceMismatch("TF candidate membership exceeds its frequency bounds.")
            resolved.append((frequency_index, frame_index))
        if len(set(resolved)) != len(resolved):
            raise CandidateEvidenceMismatch(
                "TF candidate membership contains duplicate coordinates."
            )
        return resolved

    mask_pairs = validate_pairs(candidate.mask_indices)
    if (
        type(candidate.track_frame_indices) is not tuple
        or type(candidate.track_frequency_bin_indices) is not tuple
    ):
        raise CandidateEvidenceMismatch(
            "TF candidate track membership containers must be tuples."
        )
    if len(candidate.track_frame_indices) != len(candidate.track_frequency_bin_indices):
        raise CandidateEvidenceMismatch("TF candidate track arrays have unequal length.")
    track_pairs = validate_pairs(
        tuple(zip(candidate.track_frequency_bin_indices, candidate.track_frame_indices))
    )
    if mask_pairs and track_pairs and not set(track_pairs).issubset(set(mask_pairs)):
        raise CandidateEvidenceMismatch("TF candidate track is inconsistent with its mask.")
    if candidate.method == "tf_connected_regions":
        if not mask_pairs:
            raise CandidateEvidenceMismatch("TF region candidate has no stored mask membership.")
        return mask_pairs
    if candidate.method == "tf_tracks":
        if not track_pairs or any(
            current[1] <= previous[1]
            for previous, current in zip(track_pairs, track_pairs[1:])
        ):
            raise CandidateEvidenceMismatch("TF track membership must have increasing frames.")
        return track_pairs
    raise CandidateEvidenceMismatch("TF candidate method is unsupported.")


def _candidate_filtered_iq(
    selected_iq: np.ndarray,
    *,
    sample_start: int,
    sample_stop: int,
    sample_rate_hz: float,
    center_hz: float,
    bandwidth_hz: float,
    min_feature_samples: int,
) -> np.ndarray:
    sample_count = sample_stop - sample_start
    if sample_count < 100 or sample_count < min_feature_samples:
        raise InsufficientFeatureSamples(
            "Candidate filtering requires at least 100 samples and min_feature_samples."
        )
    cutoff_hz = bandwidth_hz / 2.0
    if (
        not math.isfinite(cutoff_hz)
        or not 0.0 < cutoff_hz < sample_rate_hz / 2.0
    ):
        raise InvalidModulationFeatureParameters(
            "Candidate bandwidth cannot support the fixed band filter."
        )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized_center = float(
                np.float64(center_hz) / np.float64(sample_rate_hz)
            )
            global_samples = np.arange(sample_start, sample_stop, dtype=np.float64)
            phase = -2.0 * math.pi * normalized_center * global_samples
            oscillator = np.exp(1j * phase)
            mixed = (
                np.asarray(selected_iq, dtype=np.complex128)
                * oscillator[np.newaxis, :]
            )
            taps = signal.firwin(
                33,
                cutoff=cutoff_hz,
                fs=sample_rate_hz,
                window=("kaiser", 8.6),
            )
            if (
                not math.isfinite(normalized_center)
                or not np.isfinite(phase).all()
                or not np.isfinite(oscillator.real).all()
                or not np.isfinite(oscillator.imag).all()
                or not np.isfinite(mixed.real).all()
                or not np.isfinite(mixed.imag).all()
                or not np.isfinite(taps).all()
            ):
                raise NonFiniteEstimationResult(
                    "Candidate preprocessing produced non-finite intermediates."
                )
            filtered = np.stack(
                [signal.filtfilt(taps, [1.0], channel) for channel in mixed],
                axis=0,
            )
    except NonFiniteEstimationResult:
        raise
    except (FloatingPointError, OverflowError) as exc:
        raise NonFiniteEstimationResult(
            "Candidate preprocessing arithmetic is non-finite."
        ) from exc
    except RuntimeWarning as exc:
        raise NonFiniteEstimationResult(
            "Candidate preprocessing emitted a numerical warning."
        ) from exc
    except ValueError as exc:
        raise InvalidModulationFeatureParameters(
            "Candidate filter could not process the selected range."
        ) from exc
    if not np.isfinite(filtered.real).all() or not np.isfinite(filtered.imag).all():
        raise NonFiniteEstimationResult("Candidate preprocessing produced non-finite IQ.")
    return filtered


def _task7_frequency_bounds(candidate: EmissionCandidate) -> tuple[float, float]:
    frequency_range = candidate.frequency_range
    if not isinstance(frequency_range, FrequencyRange):
        raise CandidateEvidenceMismatch("Candidate frequency range is malformed.")
    if frequency_range.reference != "baseband":
        raise CandidateEvidenceMismatch("Candidate frequency reference must be baseband.")
    if not _task7_is_finite_real(frequency_range.start_hz) or not _task7_is_finite_real(
        frequency_range.stop_hz
    ):
        raise CandidateEvidenceMismatch("Candidate frequency bounds are invalid.")
    lower_hz = float(frequency_range.start_hz)
    upper_hz = float(frequency_range.stop_hz)
    if not math.isfinite(lower_hz) or not math.isfinite(upper_hz) or upper_hz <= lower_hz:
        raise CandidateEvidenceMismatch("Candidate frequency bounds are invalid.")
    return lower_hz, upper_hz


def _band_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float] | None:
    lower = max(first[0], second[0])
    upper = min(first[1], second[1])
    return (lower, upper) if upper > lower else None


def _band_disjoint(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[1] <= second[0] or first[0] >= second[1]


def _coverage_fraction(
    requested: tuple[float, float],
    effective: tuple[float, float],
) -> float:
    coverage = (effective[1] - effective[0]) / (requested[1] - requested[0])
    _require_task7_finite(coverage)
    return float(min(1.0, max(0.0, coverage)))


def _integrate_density_band(
    density: np.ndarray,
    edges_hz: np.ndarray,
    band: tuple[float, float],
) -> float:
    overlap = np.maximum(
        0.0,
        np.minimum(edges_hz[1:], band[1]) - np.maximum(edges_hz[:-1], band[0]),
    )
    value = float(np.sum(density * overlap))
    _require_task7_finite(overlap, value)
    return value


def _power_band_summary(
    requested: tuple[float, float],
    effective: tuple[float, float],
    coverage_fraction: float,
    observed_power: float,
    noise_power: float | None,
    excess_power: float | None,
) -> dict[str, object]:
    return {
        "requested_lower_hz": requested[0],
        "requested_upper_hz": requested[1],
        "requested_width_hz": requested[1] - requested[0],
        "effective_lower_hz": effective[0],
        "effective_upper_hz": effective[1],
        "effective_width_hz": effective[1] - effective[0],
        "coverage_fraction": coverage_fraction,
        "observed_power": observed_power,
        "noise_power": noise_power,
        "excess_power": excess_power,
    }


def _power_not_applicable(
    args: MeasurePowerRelationsArgs,
    binding: _Task7CandidateBinding,
    *,
    code: str,
    message: str,
    protected_requested: tuple[float, float],
    candidate_requested: tuple[float, float],
    observable_support: tuple[float, float],
) -> ToolResult:
    return ToolResult(
        tool_name="measure_power_relations",
        status=ToolStatus.NOT_APPLICABLE,
        data={
            "method": "integrated_relative_power_v1",
            "requested_protected_band": {
                "lower_hz": protected_requested[0],
                "upper_hz": protected_requested[1],
            },
            "requested_candidate_band": {
                "lower_hz": candidate_requested[0],
                "upper_hz": candidate_requested[1],
            },
            "observable_support": {
                "lower_hz": observable_support[0],
                "upper_hz": observable_support[1],
            },
        },
        semantics={
            "observable_support": "True-Nyquist-clipped baseband Hz support of the bound spectrum block."
        },
        quality={"finite_inputs": True},
        error=ToolError(code=code, message=message),
        provenance={
            "spectrum_id": binding.source_record.artifact_id,
            "candidate_set_id": args.candidate_set_id,
            "candidate_id": binding.candidate.candidate_id,
            "evidence_block_index": binding.candidate.source_block_index,
        },
    )


def _require_task7_finite(*values: object) -> None:
    if any(not np.isfinite(np.asarray(value)).all() for value in values):
        raise NonFiniteEstimationResult("Task 7 arithmetic produced a non-finite result.")
