"""Typed, dependency-injected detection executors for later tool registration."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from cityrf.legacy.dsp.detection import (
    CandidateSetPayload,
    DetectionError,
    InapplicableDetectionEvidence,
    InvalidDetectionParameters,
    detect_activity,
    spectral_region_candidates,
    tf_connected_region_candidates,
    tf_track_candidates,
)
from cityrf.legacy.dsp.representations import SpectrumPayload, TimeFrequencyPayload
from cityrf.legacy.dsp.selection import select_iq
from cityrf.legacy.runtime import IQRepository, RunArtifactStore
from cityrf.legacy.schemas import SampleRange, ToolResult, ToolStatus


_MAX_PUBLIC_CANDIDATES = 8


class DetectActivityIntervalsArgs(BaseModel):
    """Explicit non-coherent activity selection and adaptive-energy controls."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)
    sample_ranges: list[SampleRange] = Field(default_factory=list)
    range_mode: Literal["independent", "aggregate"] = "independent"
    channel_indices: list[int] = Field(default_factory=list)
    smoothing_window_samples: int = Field(default=9, ge=1)
    noise_quantile: float = Field(default=0.25)
    high_threshold_db: float = Field(default=10.0)
    low_threshold_db: float = Field(default=6.0)
    min_active_samples: int = Field(default=8, ge=1)
    merge_gap_samples: int = Field(default=0, ge=0)
    padding_samples: int = Field(default=0, ge=0)


class DetectEmissionsArgs(BaseModel):
    """Detection controls for exactly one typed spectrum or TF artifact."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)
    method: Literal["spectral_regions", "tf_connected_regions", "tf_tracks"]
    spectrum_id: str | None = None
    time_frequency_id: str | None = None
    threshold_db: float = 12.0
    noise_quantile: float = 0.25
    min_region_bins: int = Field(default=1, ge=1)
    merge_gap_bins: int = Field(default=0, ge=0)
    min_time_frames: int = Field(default=2, ge=1)
    min_frequency_bins: int = Field(default=1, ge=1)
    min_track_frames: int = Field(default=4, ge=1)
    max_track_jump_bins: int = Field(default=2, ge=0)
    max_track_gap_frames: int = Field(default=0, ge=0)


class CandidateCountArgs(BaseModel):
    """Resolvable-region count controls; deliberately not a physical-source estimator."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["candidate_count"]
    source_ref: str = Field(min_length=1)
    candidate_set_id: str = Field(min_length=1)
    min_candidate_confidence: float = 0.0
    max_sources: int = Field(default=8, ge=1)


class MdlCountArgs(BaseModel):
    """Spatial-covariance MDL controls sharing the public source-count entrypoint."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["mdl"]
    source_ref: str = Field(min_length=1)
    spatial_evidence_id: str = Field(min_length=1)
    max_sources: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Upper bound on total resolvable spatial components, including the "
            "protected target; it is not the maximum number of interferers."
        ),
    )


SourceCountBranch = Annotated[CandidateCountArgs | MdlCountArgs, Field(discriminator="method")]


class EstimateSourceCountArgs(RootModel[SourceCountBranch]):
    """Top-level discriminated public source-count argument schema."""


def detect_activity_intervals(args: DetectActivityIntervalsArgs, *, repository: IQRepository) -> ToolResult:
    """Run fixed ``adaptive_energy_v1`` and disclose only bounded range summaries."""

    _validate_activity_args(args)
    frame = repository.get(args.source_ref)
    # Activity always means non-coherent mean channel power, not complex averaging.
    from cityrf.legacy.schemas import SignalSelection

    selection = SignalSelection(
        source_ref=args.source_ref,
        sample_ranges=args.sample_ranges,
        range_mode=args.range_mode,
        channel_indices=args.channel_indices,
        channel_mode="mean_power",
    )
    result = detect_activity(
        select_iq(frame, selection),
        smoothing_window_samples=args.smoothing_window_samples,
        noise_quantile=args.noise_quantile,
        high_threshold_db=args.high_threshold_db,
        low_threshold_db=args.low_threshold_db,
        min_active_samples=args.min_active_samples,
        merge_gap_samples=args.merge_gap_samples,
        padding_samples=args.padding_samples,
    )
    intervals = [
        {
            "start": interval.start,
            "stop": interval.stop,
            "peak_to_noise_db": _db_ratio(interval.peak_power, interval.noise_power),
            "boundary_truncated": interval.boundary_truncated,
        }
        for interval in result.intervals[:_MAX_PUBLIC_CANDIDATES]
    ]
    active_samples = sum(interval.stop - interval.start for interval in result.intervals)
    return ToolResult(
        tool_name="detect_activity_intervals",
        status=ToolStatus.SUCCESS,
        data={
            "method": "adaptive_energy_v1",
            "activity_ranges": intervals,
            "interval_count": len(result.intervals),
            "selected_sample_count": result.selected_sample_count,
            "duty_cycle": active_samples / result.selected_sample_count,
            "noise_power": result.noise_power,
            "boundary_truncated": result.boundary_truncated,
        },
        semantics={
            "activity_ranges": "Half-open global capture-coordinate [start, stop) sample ranges directly usable by pulse estimation.",
            "duty_cycle": "Detected active samples divided by the sum of selected sample counts, not their enclosing span.",
            "noise_power": "Selected-sample weighted robust lower-quantile smoothed linear power estimate.",
        },
        quality={"finite_outputs": True, "smoothing_alignment": "centred moving average with edge replication"},
        warnings=(
            ["Activity interval disclosure was truncated; rerun on narrower selected sample ranges to inspect omitted intervals."]
            if len(result.intervals) > _MAX_PUBLIC_CANDIDATES
            else []
        ),
        provenance={
            "resolved_parameters": result.resolved_parameters,
            "channel_aggregation": "noncoherent_mean_power",
            "range_processing": "each selected range independently; no smoothing, hysteresis, merging, or padding crosses a gap",
        },
    )


def detect_emissions(args: DetectEmissionsArgs, *, artifact_store: RunArtifactStore) -> ToolResult:
    """Detect resolvable spectrum or TF regions and retain full candidates internally."""

    _validate_emission_args(args)
    if args.method == "spectral_regions":
        record = artifact_store.get(args.spectrum_id or "", expected_type="spectrum", expected_source_ref=args.source_ref)
        if not isinstance(record.payload, SpectrumPayload):
            raise InapplicableDetectionEvidence("Spectrum artifact payload is malformed.")
        candidates = spectral_region_candidates(
            record.payload,
            source_ref=args.source_ref,
            source_artifact_id=record.artifact_id,
            threshold_db=args.threshold_db,
            noise_quantile=args.noise_quantile,
            min_region_bins=args.min_region_bins,
            merge_gap_bins=args.merge_gap_bins,
        )
        domain = "spectral"
    else:
        record = artifact_store.get(args.time_frequency_id or "", expected_type="time_frequency", expected_source_ref=args.source_ref)
        if not isinstance(record.payload, TimeFrequencyPayload):
            raise InapplicableDetectionEvidence("Time-frequency artifact payload is malformed.")
        if args.method == "tf_connected_regions":
            candidates = tf_connected_region_candidates(
                record.payload,
                source_ref=args.source_ref,
                source_artifact_id=record.artifact_id,
                threshold_db=args.threshold_db,
                noise_quantile=args.noise_quantile,
                min_time_frames=args.min_time_frames,
                min_frequency_bins=args.min_frequency_bins,
                merge_gap_bins=args.merge_gap_bins,
            )
        else:
            candidates = tf_track_candidates(
                record.payload,
                source_ref=args.source_ref,
                source_artifact_id=record.artifact_id,
                threshold_db=args.threshold_db,
                noise_quantile=args.noise_quantile,
                min_track_frames=args.min_track_frames,
                max_track_jump_bins=args.max_track_jump_bins,
                max_track_gap_frames=args.max_track_gap_frames,
            )
        domain = "time_frequency"
    payload = CandidateSetPayload(
        source_ref=args.source_ref,
        source_artifact_id=record.artifact_id,
        source_artifact_type=record.artifact_type,
        method=args.method,
        domain=domain,
        candidates=candidates,
    )
    candidate_set_id = artifact_store.put(
        "candidate_set",
        args.source_ref,
        payload,
        {"candidate_count": len(candidates), "method": args.method, "evidence_domain": domain},
        selection=record.selection,
        parameters=args.model_dump(mode="json"),
        meaning="Resolvable emission regions; full numerical masks and tracks remain internal.",
    )
    disclosed = [_candidate_summary(candidate) for candidate in candidates[:_MAX_PUBLIC_CANDIDATES]]
    warnings = ["Candidate count is a resolvable-region count, not physical transmitter or source truth."]
    if len(candidates) > len(disclosed):
        warnings.append("Candidate disclosure was truncated; the candidate artifact retains all candidates.")
    return ToolResult(
        tool_name="detect_emissions",
        status=ToolStatus.SUCCESS,
        data={
            "candidate_set_id": candidate_set_id,
            "candidate_count": len(candidates),
            "disclosed_candidate_count": len(disclosed),
            "candidates": disclosed,
        },
        semantics={
            "candidates": "Bounded resolvable-region summaries; candidate IDs are scoped to candidate_set_id.",
            "frequency_range": "Bin-edge-aware baseband Hz bounds clipped to observable Nyquist support.",
            "relative_power_db": "Peak linear power in dB above the resolved robust lower-quantile noise estimate.",
        },
        quality={"finite_outputs": True, "evidence_domain": domain},
        warnings=warnings,
        artifacts=[artifact_store.get(candidate_set_id).summary],
        provenance={
            "source_evidence_type": record.artifact_type,
            "source_evidence_id": record.artifact_id,
            "resolved_parameters": args.model_dump(mode="json", exclude_none=True),
            "tf_connectivity": "8-connected frequency-time neighbourhood" if args.method == "tf_connected_regions" else "not_applicable",
        },
    )


def estimate_source_count(args: EstimateSourceCountArgs, *, artifact_store: RunArtifactStore) -> ToolResult:
    """Dispatch candidate-region or spatial-MDL source count without a second API."""

    branch = args.root
    if isinstance(branch, MdlCountArgs):
        return _estimate_mdl_source_count(branch, artifact_store=artifact_store)
    return _estimate_candidate_source_count(branch, artifact_store=artifact_store)


def _estimate_candidate_source_count(args: CandidateCountArgs, *, artifact_store: RunArtifactStore) -> ToolResult:
    """Return the unchanged Task 4 capped resolvable-region count."""

    if not math.isfinite(args.min_candidate_confidence) or not 0.0 <= args.min_candidate_confidence <= 1.0:
        raise InvalidDetectionParameters("min_candidate_confidence must be a finite value in [0, 1].")
    record = artifact_store.get(args.candidate_set_id, expected_type="candidate_set", expected_source_ref=args.source_ref)
    payload = record.payload
    if not isinstance(payload, CandidateSetPayload):
        raise DetectionError("Candidate artifact payload is malformed.")
    if payload.source_ref != args.source_ref or payload.source_artifact_type not in {"spectrum", "time_frequency"}:
        raise DetectionError("Candidate artifact binding is malformed.")
    # Resolve the full evidence chain on every consumption.  A candidate-set
    # record can outlive neither a valid same-run source artifact nor its
    # declared type/source binding.
    artifact_store.get(
        payload.source_artifact_id,
        expected_type=payload.source_artifact_type,
        expected_source_ref=args.source_ref,
    )
    qualified = [candidate for candidate in payload.candidates if candidate.confidence >= args.min_candidate_confidence]
    estimate = min(len(qualified), args.max_sources)
    estimator_capped = len(qualified) > args.max_sources
    public_disclosed = min(len(payload.candidates), _MAX_PUBLIC_CANDIDATES)
    warnings = ["This is a resolvable-region count, not physical transmitter or source truth."]
    if len(payload.candidates) > public_disclosed:
        warnings.append("Candidate disclosure truncation is separate from estimator capping.")
    if estimator_capped:
        warnings.append("The resolvable-region estimate was capped by max_sources.")
    return ToolResult(
        tool_name="estimate_source_count",
        status=ToolStatus.SUCCESS,
        data={
            "method": "candidate_count",
            "evidence_domain": payload.domain,
            "total_stored_candidates": len(payload.candidates),
            "publicly_disclosed_candidates": public_disclosed,
            "confidence_qualified_candidates": len(qualified),
            "resolvable_region_count": estimate,
            "estimator_capped": estimator_capped,
            "disclosure_truncated": len(payload.candidates) > public_disclosed,
        },
        semantics={
            "resolvable_region_count": "Confidence-qualified candidate regions capped by max_sources; it is not physical transmitter/source truth.",
            "evidence_domain": "spectral or time_frequency evidence used for this resolvable-region count.",
        },
        quality={"finite_outputs": True},
        warnings=warnings,
        provenance={"candidate_set_id": args.candidate_set_id, "source_evidence_type": payload.source_artifact_type},
    )


def _estimate_mdl_source_count(args: MdlCountArgs, *, artifact_store: RunArtifactStore) -> ToolResult:
    """Return raw-eigenvalue MDL resolvable spatial covariance components."""

    from cityrf.legacy.dsp.spatial import SpatialEvidencePayload, SpatialError, mdl_scores

    record = artifact_store.get(args.spatial_evidence_id, expected_type="spatial_evidence", expected_source_ref=args.source_ref)
    payload = record.payload
    if not isinstance(payload, SpatialEvidencePayload):
        raise DetectionError("Spatial evidence payload is malformed.")
    if payload.synchronization is not True or payload.synchronization_conflict:
        from cityrf.legacy.schemas import ToolError

        return ToolResult(tool_name="estimate_source_count", status=ToolStatus.NOT_APPLICABLE, error=ToolError(code="UNVERIFIED_SYNCHRONIZATION", message="MDL requires explicitly verified synchronization.", repair_hint="Provide verified synchronized array metadata."))
    channels = len(payload.selected_channel_indices)
    if args.max_sources is not None and args.max_sources > channels - 1:
        from cityrf.legacy.schemas import ToolError

        return ToolResult(tool_name="estimate_source_count", status=ToolStatus.NOT_APPLICABLE, error=ToolError(code="INVALID_MAX_SOURCES", message="max_sources exceeds the spatial channel limit.", repair_hint="Use a value no larger than channel_count minus one."))
    try:
        selected, k_max, scores = mdl_scores(payload.raw_eigenvalues, snapshot_count=payload.snapshot_count, max_sources=args.max_sources)
    except SpatialError:
        from cityrf.legacy.schemas import ToolError

        return ToolResult(tool_name="estimate_source_count", status=ToolStatus.NOT_APPLICABLE, error=ToolError(code="INVALID_MDL_EIGENSPECTRUM", message="Raw covariance eigenspectrum cannot support MDL.", repair_hint="Select finite positive-rank spatial evidence."))
    disclosed = sorted(scores, key=lambda item: (float(item["score"]), int(item["k"])))[:16]
    warnings = ["MDL estimates resolvable spatial covariance components, not physical transmitter or interference truth."]
    if len(scores) > len(disclosed):
        warnings.append("MDL score disclosure was truncated to the lowest scores.")
    return ToolResult(
        tool_name="estimate_source_count",
        status=ToolStatus.SUCCESS,
        data={"method": "mdl", "evidence_domain": "spatial_covariance", "resolvable_spatial_component_count": selected, "observation_count": payload.snapshot_count, "k_max": k_max, "evaluated_score_count": len(scores), "disclosed_score_count": len(disclosed), "scores_truncated": len(scores) > len(disclosed), "scores": disclosed, "warning_total_count": len(warnings), "warning_disclosed_count": len(warnings), "warnings_truncated": False},
        semantics={"resolvable_spatial_component_count": "MDL-selected resolvable spatial covariance components, not physical transmitter or interference truth."},
        quality={"finite_outputs": True},
        warnings=warnings,
        provenance={"spatial_evidence_id": args.spatial_evidence_id, "eigenvalue_source": "raw_covariance"},
    )


def _validate_activity_args(args: DetectActivityIntervalsArgs) -> None:
    if len(set(args.channel_indices)) != len(args.channel_indices) or any(index < 0 for index in args.channel_indices):
        raise InvalidDetectionParameters("channel_indices must be distinct non-negative indices.")
    if any(right.start < left.stop for left, right in zip(sorted(args.sample_ranges, key=lambda value: value.start), sorted(args.sample_ranges, key=lambda value: value.start)[1:])):
        raise InvalidDetectionParameters("Activity sample ranges must not overlap.")


def _validate_emission_args(args: DetectEmissionsArgs) -> None:
    spectral = args.method == "spectral_regions"
    if spectral != (args.spectrum_id is not None) or (not spectral) != (args.time_frequency_id is not None):
        raise InvalidDetectionParameters("Detection method requires exactly its matching evidence artifact ID.")
    if not math.isfinite(args.threshold_db) or not math.isfinite(args.noise_quantile):
        raise InvalidDetectionParameters("Detection thresholds must be finite.")


def _candidate_summary(candidate) -> dict[str, object]:
    result: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "sample_range": {"start": candidate.sample_range[0], "stop": candidate.sample_range[1]},
        "frequency_range": candidate.frequency_range.model_dump(mode="json"),
        "peak_frequency_hz": candidate.peak_frequency_hz,
        "relative_power_db": candidate.relative_power_db,
        "confidence": candidate.confidence,
        "lower_truncated": candidate.lower_truncated,
        "upper_truncated": candidate.upper_truncated,
    }
    if candidate.slope_hz_per_sample is not None:
        result.update(
            {
                "track_frame_count": len(candidate.track_frame_indices),
                "slope_hz_per_sample": candidate.slope_hz_per_sample,
                "max_observed_jump_bins": candidate.max_observed_jump_bins,
            }
        )
    return result


def _db_ratio(value: float, reference: float) -> float:
    return float(10.0 * math.log10(max(value, float.fromhex("0x1.0p-1022")) / max(reference, float.fromhex("0x1.0p-1022"))))
