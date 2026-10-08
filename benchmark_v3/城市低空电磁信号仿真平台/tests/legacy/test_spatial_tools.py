"""TDD contracts for Task 5 narrowband spatial evidence and MUSIC."""

from __future__ import annotations

from dataclasses import replace
import math
from unittest.mock import patch
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from cityrf.legacy.dsp.detection import CandidateSetPayload, EmissionCandidate
from cityrf.legacy.dsp.representations import (
    SpectrumBlock,
    SpectrumPayload,
    TimeFrequencyBlock,
    TimeFrequencyPayload,
)
from cityrf.legacy.dsp.spatial import (
    InvalidSpatialParameters,
    MalformedSpatialEvidence,
    SpatialError,
    SpatialEvidenceBindingError,
    covariance_from_segments,
)
from cityrf.legacy.runtime import (
    ArtifactExpired,
    ArtifactNotFound,
    ArtifactSourceMismatch,
    ArtifactTypeMismatch,
    CrossRunArtifact,
    RunArtifactStore,
)
from cityrf.legacy.schemas import ArrayGeometry, FrequencyRange, SampleRange, ToolStatus
from cityrf.legacy.tools.detection import EstimateSourceCountArgs, estimate_source_count
from cityrf.legacy.tools.spatial import (
    ComputeSpatialEvidenceArgs,
    EstimateDoaArgs,
    compute_spatial_evidence,
    estimate_doa,
)


_C = 299_792_458.0
_FC = 2_400_000_000.0
_SPACING = _C / _FC / 2.0


def _store() -> RunArtifactStore:
    return RunArtifactStore(run_id="spatial_test_run")


def _geometry(*, sync: bool | None = True, calibration: dict | None = None) -> ArrayGeometry:
    return ArrayGeometry(
        geometry_type="ula",
        element_count=4,
        element_spacing_m=_SPACING,
        reference_frequency_hz=_FC,
        synchronized=sync,
        calibration={"state": "ideal_verified"} if calibration is None else calibration,
    )


def _ula_iq(angles_deg: tuple[float, ...] = (20.0,), *, samples: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(501)
    time = np.arange(samples, dtype=np.float64)
    iq = np.zeros((4, samples), dtype=np.complex128)
    for index, angle in enumerate(angles_deg):
        steering = np.exp(1j * np.pi * np.sin(np.deg2rad(angle)) * np.arange(4))
        waveform = np.exp(2j * np.pi * (0.071 + 0.017 * index) * time + 1j * (0.13 * index))
        iq += steering[:, None] * waveform
    iq += 0.03 * (rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape))
    return iq.astype(np.complex64)


def _spatial_args(source_ref: str, **overrides: object) -> ComputeSpatialEvidenceArgs:
    values: dict[str, object] = {"source_ref": source_ref, "range_mode": "aggregate"}
    values.update(overrides)
    return ComputeSpatialEvidenceArgs(**values)


def _candidate_spectrum() -> SpectrumPayload:
    axis = np.array([-128.0, 128.0])
    values = np.ones(2)
    block = SpectrumBlock(axis, values, values.astype(np.complex128), (0, 1024), (0, 1, 2, 3))
    return SpectrumPayload(axis, values, values.astype(np.complex128), (block,), ((0, 1024),), "fft", "power", 1024.0, 512.0)


def _candidate_time_frequency() -> TimeFrequencyPayload:
    frequency_hz = np.array([40.0, 80.0, 120.0, 160.0])
    power_linear = np.ones((4, 3), dtype=np.float64)
    values = power_linear.astype(np.complex128)
    block = TimeFrequencyBlock(
        frequency_hz,
        values,
        power_linear,
        np.array([0, 100, 200]),
        np.array([50, 150, 250]),
        (0, 1024),
        (0, 1, 2, 3),
    )
    return TimeFrequencyPayload((block,), "complex_stft", 1024.0, 40.0, 100, 100, 4)


def _candidate_payload(source_ref: str, upstream_id: str, **overrides: object) -> CandidateSetPayload:
    candidate = EmissionCandidate(
        candidate_id="emission_0",
        source_ref=source_ref,
        source_artifact_id=upstream_id,
        source_artifact_type="spectrum",
        method="spectral_regions",
        domain="spectral",
        source_block_index=0,
        sample_range=(100, 400),
        channel_indices=(0, 1, 2, 3),
        frequency_range=FrequencyRange(start_hz=40.0, stop_hz=120.0, reference="baseband"),
        peak_frequency_hz=80.0,
        peak_power_linear=1.0,
        relative_power_db=20.0,
        confidence=1.0,
        lower_truncated=False,
        upper_truncated=False,
    )
    return CandidateSetPayload(
        source_ref,
        upstream_id,
        "spectrum",
        "spectral_regions",
        "spectral",
        (replace(candidate, **overrides),),
    )


def _compute(register_iq, iq: np.ndarray, **kwargs: object):
    repository, source_ref = register_iq("spatial", iq, array_geometry=_geometry(), synchronized=True, reference_frequency_hz=_FC)
    store = _store()
    result = compute_spatial_evidence(_spatial_args(source_ref, **kwargs), repository=repository, artifact_store=store)
    return repository, source_ref, store, result


def test_covariance_has_weighted_hermitian_raw_and_loaded_eigensystems(register_iq):
    _, source_ref, store, result = _compute(register_iq, _ula_iq(), diagonal_loading_fraction=0.01)

    assert result.status is ToolStatus.SUCCESS
    assert result.data["snapshot_count"] == 1024
    assert result.data["channel_count"] == 4
    assert result.data["eigenvalue_total_count"] == 4
    assert result.data["eigenvalue_disclosed_count"] == 4
    artifact = store.get(result.data["spatial_evidence_id"], expected_type="spatial_evidence", expected_source_ref=source_ref)
    payload = artifact.payload
    assert np.allclose(payload.raw_covariance, payload.raw_covariance.conj().T)
    assert np.allclose(payload.loaded_covariance, payload.loaded_covariance.conj().T)
    assert np.all(np.diff(payload.raw_eigenvalues) <= 0.0)
    assert np.all(np.diff(payload.loaded_eigenvalues) <= 0.0)
    assert np.all(payload.raw_eigenvalues >= -1e-10)
    assert result.data["diagonal_loading"] == pytest.approx(0.01 * np.trace(payload.raw_covariance).real / 4)
    dumped = result.model_dump_json()
    assert "covariance_matrix" not in dumped.lower() and "eigenvectors" not in dumped.lower()


def test_ranges_are_sample_weighted_and_overlap_is_rejected(register_iq):
    iq = _ula_iq(samples=1024)
    _, source_ref, store, result = _compute(
        register_iq,
        iq,
        sample_ranges=[SampleRange(start=0, stop=128), SampleRange(start=512, stop=1024)],
        remove_mean=False,
    )
    payload = store.get(result.data["spatial_evidence_id"]).payload
    x0, x1 = iq[:, :128], iq[:, 512:]
    expected = (x0 @ x0.conj().T + x1 @ x1.conj().T) / 640
    assert result.data["snapshot_count"] == 640
    assert np.allclose(payload.raw_covariance, expected, atol=1e-5)
    with pytest.raises(InvalidSpatialParameters) as raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, sample_ranges=[SampleRange(start=0, stop=128), SampleRange(start=127, stop=256)]),
            repository=_, artifact_store=store,
        )
    assert not isinstance(raised.value, (SpatialEvidenceBindingError, MalformedSpatialEvidence))
    assert getattr(raised.value, "code", None) == "OVERLAPPING_SAMPLE_RANGES"


def test_candidate_selection_uses_stored_frequency_and_intersection(register_iq):
    repository, source_ref = register_iq("candidate_spatial", _ula_iq(), array_geometry=_geometry(), synchronized=True, reference_frequency_hz=_FC)
    store = _store()
    candidate = EmissionCandidate(
        candidate_id="emission_0", source_ref=source_ref, source_artifact_id="spectrum_upstream", source_artifact_type="spectrum",
        method="spectral_regions", domain="spectral", source_block_index=0, sample_range=(100, 400), channel_indices=(0, 1, 2, 3),
        frequency_range=FrequencyRange(start_hz=40.0, stop_hz=120.0, reference="baseband"), peak_frequency_hz=80.0,
        peak_power_linear=1.0, relative_power_db=20.0, confidence=1.0, lower_truncated=False, upper_truncated=False,
    )
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    candidate = candidate.__class__(**{**candidate.__dict__, "source_artifact_id": upstream_id})
    candidate_set_id = store.put("candidate_set", source_ref, CandidateSetPayload(source_ref, upstream_id, "spectrum", "spectral_regions", "spectral", (candidate,)), {"candidate_count": 1})
    result = compute_spatial_evidence(
        _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0", sample_ranges=[SampleRange(start=200, stop=500)]),
        repository=repository, artifact_store=store,
    )
    assert result.status is ToolStatus.SUCCESS
    assert result.data["snapshot_count"] == 200
    assert result.provenance["frequency_selection"]["source"] == "candidate"
    assert result.provenance["frequency_selection"]["start_hz"] == 40.0
    empty = compute_spatial_evidence(
        _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0", sample_ranges=[SampleRange(start=500, stop=600)]),
        repository=repository, artifact_store=store,
    )
    assert empty.status is ToolStatus.NOT_APPLICABLE and empty.error.code == "EMPTY_CANDIDATE_INTERSECTION"
    with pytest.raises(Exception, match="CONFLICTING_SELECTION"):
        ComputeSpatialEvidenceArgs(source_ref=source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0", frequency_range=FrequencyRange(start_hz=1, stop_hz=2, reference="baseband"))


def test_insufficient_channels_and_dc_mean_removal_are_safe(register_iq):
    repository, source_ref = register_iq("one_channel", _ula_iq()[:1], array_geometry=None)
    result = compute_spatial_evidence(_spatial_args(source_ref), repository=repository, artifact_store=_store())
    assert result.status is ToolStatus.NOT_APPLICABLE and result.error.code == "INSUFFICIENT_CHANNELS"

    dc = np.exp(1j * np.arange(4)[:, None] * 0.2) * np.ones((1, 256), dtype=np.complex64)
    _, _, _, retained = _compute(register_iq, dc, remove_mean=False)
    _, _, _, removed = _compute(register_iq, dc, remove_mean=True)
    assert retained.status is ToolStatus.SUCCESS
    assert removed.status is ToolStatus.NOT_APPLICABLE and removed.error.code == "NO_SPATIAL_ENERGY"
    assert any("mean" in warning.lower() and "dc" in warning.lower() for warning in removed.warnings)


def test_covariance_warns_unsynchronized_while_mdl_and_music_reject(register_iq):
    repository, source_ref = register_iq("unsync", _ula_iq(), array_geometry=_geometry(sync=False), synchronized=False, reference_frequency_hz=_FC)
    store = _store()
    evidence = compute_spatial_evidence(_spatial_args(source_ref), repository=repository, artifact_store=store)
    assert evidence.status is ToolStatus.SUCCESS
    assert any("synchron" in warning.lower() for warning in evidence.warnings)
    mdl = estimate_source_count(EstimateSourceCountArgs.model_validate({"method": "mdl", "source_ref": source_ref, "spatial_evidence_id": evidence.data["spatial_evidence_id"]}), artifact_store=store)
    doa = estimate_doa(EstimateDoaArgs(spatial_evidence_id=evidence.data["spatial_evidence_id"], source_count=1), artifact_store=store)
    assert mdl.status is ToolStatus.NOT_APPLICABLE and mdl.error.code == "UNVERIFIED_SYNCHRONIZATION"
    assert doa.status is ToolStatus.NOT_APPLICABLE and doa.error.code == "UNVERIFIED_SYNCHRONIZATION"


def test_mdl_uses_raw_eigenvalues_and_unified_discriminator(register_iq):
    _, source_ref, store, evidence = _compute(register_iq, _ula_iq())
    args = EstimateSourceCountArgs.model_validate({"method": "mdl", "source_ref": source_ref, "spatial_evidence_id": evidence.data["spatial_evidence_id"]})
    result = estimate_source_count(args, artifact_store=store)
    assert result.status is ToolStatus.SUCCESS
    assert result.data["method"] == "mdl" and result.data["evidence_domain"] == "spatial_covariance"
    assert result.data["resolvable_spatial_component_count"] == 1
    assert result.data["observation_count"] == 1024
    assert result.data["k_max"] == 3
    with pytest.raises(ValidationError):
        EstimateSourceCountArgs.model_validate({"method": "mdl", "source_ref": source_ref, "spatial_evidence_id": "x", "candidate_set_id": "no"})
    invalid = estimate_source_count(EstimateSourceCountArgs.model_validate({"method": "mdl", "source_ref": source_ref, "spatial_evidence_id": evidence.data["spatial_evidence_id"], "max_sources": 4}), artifact_store=store)
    assert invalid.status is ToolStatus.NOT_APPLICABLE and invalid.error.code == "INVALID_MAX_SOURCES"


def test_music_uses_signed_v3_steering_and_separated_peaks(register_iq):
    _, _, store, one = _compute(register_iq, _ula_iq((20.0,)))
    signed = estimate_doa(EstimateDoaArgs(spatial_evidence_id=one.data["spatial_evidence_id"], source_count=1, grid_step_deg=0.25), artifact_store=store)
    assert signed.status is ToolStatus.SUCCESS
    assert signed.data["peaks"][0]["azimuth_deg"] == pytest.approx(20.0, abs=1.0)

    _, _, store, two = _compute(register_iq, _ula_iq((-20.0, 35.0)))
    result = estimate_doa(EstimateDoaArgs(spatial_evidence_id=two.data["spatial_evidence_id"], source_count=2, grid_step_deg=0.25), artifact_store=store)
    peaks = sorted(peak["azimuth_deg"] for peak in result.data["peaks"])
    assert result.status is ToolStatus.SUCCESS
    assert peaks[0] == pytest.approx(-20.0, abs=2.0) and peaks[1] == pytest.approx(35.0, abs=2.0)
    invalid = estimate_doa(EstimateDoaArgs(spatial_evidence_id=two.data["spatial_evidence_id"], source_count=4), artifact_store=store)
    assert invalid.status is ToolStatus.NOT_APPLICABLE and invalid.error.code == "INVALID_SOURCE_COUNT"


def test_music_requires_verified_geometry_reference_and_calibration(register_iq):
    iq = _ula_iq()
    repository, source_ref = register_iq("no_cal", iq, array_geometry=_geometry(calibration={}), synchronized=True, reference_frequency_hz=_FC)
    store = _store()
    evidence = compute_spatial_evidence(_spatial_args(source_ref), repository=repository, artifact_store=store)
    result = estimate_doa(EstimateDoaArgs(spatial_evidence_id=evidence.data["spatial_evidence_id"], source_count=1), artifact_store=store)
    assert result.status is ToolStatus.NOT_APPLICABLE and result.error.code == "UNVERIFIED_CALIBRATION"

    repository, source_ref = register_iq("no_ref", iq, array_geometry=ArrayGeometry(geometry_type="ula", element_count=4, element_spacing_m=_SPACING, synchronized=True, calibration={"state": "ideal_verified"}), synchronized=True, reference_frequency_hz=None, center_frequency_hz=_FC)
    store = _store()
    evidence = compute_spatial_evidence(_spatial_args(source_ref), repository=repository, artifact_store=store)
    result = estimate_doa(EstimateDoaArgs(spatial_evidence_id=evidence.data["spatial_evidence_id"], source_count=1), artifact_store=store)
    assert result.status is ToolStatus.NOT_APPLICABLE and result.error.code == "MISSING_REFERENCE_FREQUENCY"


def test_public_sequence_caps_and_grid_bounds(register_iq):
    _, _, store, evidence = _compute(register_iq, _ula_iq())
    result = estimate_doa(EstimateDoaArgs(spatial_evidence_id=evidence.data["spatial_evidence_id"], source_count=1, azimuth_min_deg=-3.0, azimuth_max_deg=3.1, grid_step_deg=2.0), artifact_store=store)
    assert result.data["grid"]["first_deg"] == -3.0
    assert result.data["grid"]["last_deg"] == 3.0
    assert result.data["grid"]["point_count"] == 4
    assert len(result.data["peaks"]) <= 16
    assert result.data["peak_total_count"] >= result.data["peak_disclosed_count"]


def test_candidate_selection_revalidates_the_complete_evidence_chain(register_iq):
    repository, source_ref = register_iq("candidate_chain", _ula_iq(), array_geometry=_geometry(), synchronized=True, reference_frequency_hz=_FC)
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    candidate = EmissionCandidate(
        candidate_id="emission_0", source_ref=source_ref, source_artifact_id=upstream_id, source_artifact_type="spectrum",
        method="spectral_regions", domain="spectral", source_block_index=0, sample_range=(100, 400), channel_indices=(0, 1, 2, 3),
        frequency_range=FrequencyRange(start_hz=40.0, stop_hz=120.0, reference="baseband"), peak_frequency_hz=80.0,
        peak_power_linear=1.0, relative_power_db=20.0, confidence=1.0, lower_truncated=False, upper_truncated=False,
    )
    forged = replace(candidate, source_artifact_id="spectrum_unknown_handle")
    candidate_set_id = store.put("candidate_set", source_ref, CandidateSetPayload(source_ref, upstream_id, "spectrum", "spectral_regions", "spectral", (forged,)), {"candidate_count": 1})
    with pytest.raises(SpatialEvidenceBindingError) as raised:
        compute_spatial_evidence(_spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"), repository=repository, artifact_store=store)
    assert isinstance(raised.value, SpatialError)
    assert "spectrum_unknown_handle" not in str(raised.value)


def test_wrong_spatial_and_candidate_payload_types_raise_malformed_evidence(register_iq):
    repository, source_ref = register_iq(
        "malformed_evidence",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    hostile_payload = {"payload_marker": "must_not_escape"}
    spatial_id = store.put("spatial_evidence", source_ref, hostile_payload, {"channel_count": 4})
    candidate_set_id = store.put("candidate_set", source_ref, hostile_payload, {"candidate_count": 1})

    with pytest.raises(MalformedSpatialEvidence) as spatial_raised:
        estimate_doa(EstimateDoaArgs(spatial_evidence_id=spatial_id, source_count=1), artifact_store=store)
    with pytest.raises(MalformedSpatialEvidence) as candidate_raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_secret_marker"),
            repository=repository,
            artifact_store=store,
        )

    for raised in (spatial_raised, candidate_raised):
        assert isinstance(raised.value, SpatialError)
        assert "must_not_escape" not in str(raised.value)
        assert "emission_secret_marker" not in str(raised.value)
        assert repr(hostile_payload) not in str(raised.value)


@pytest.mark.parametrize(
    "payload_overrides",
    [
        pytest.param({"raw_covariance": np.eye(3, dtype=np.complex128)}, id="raw-covariance-shape"),
        pytest.param({"loaded_covariance": np.full((4, 4), "array_marker")}, id="loaded-covariance-dtype"),
        pytest.param({"raw_eigenvalues": np.ones(4, dtype=np.complex128)}, id="raw-eigenvalue-dtype"),
        pytest.param({"raw_eigenvectors": np.eye(4, dtype=np.float64)}, id="raw-eigenvector-dtype"),
        pytest.param({"loaded_eigenvalues": np.array([4.0, 3.0, math.nan, 1.0])}, id="loaded-eigenvalue-nonfinite"),
        pytest.param({"loaded_eigenvectors": None}, id="null-loaded-eigenvectors"),
        pytest.param({"loaded_eigenvectors": np.ones((4, 3), dtype=np.complex128)}, id="loaded-eigenvector-shape"),
        pytest.param({"loaded_eigenvectors": np.full((4, 4), math.nan + 0j)}, id="loaded-eigenvector-nonfinite"),
        pytest.param({"selected_channel_indices": None}, id="null-selected-channels"),
        pytest.param({"selected_channel_indices": [0, 1, 2, 3]}, id="list-selected-channels"),
        pytest.param({"selected_channel_indices": (0, 1, 1, 3)}, id="duplicate-selected-channels"),
        pytest.param({"selected_channel_indices": (0, 1, 2, 4)}, id="selected-channel-out-of-bounds"),
        pytest.param({"segment_ranges": None}, id="null-segment-ranges"),
        pytest.param({"segment_ranges": [(0, 1024)]}, id="list-segment-ranges"),
        pytest.param({"segment_ranges": ((400, 100),)}, id="reversed-segment-range"),
        pytest.param({"segment_ranges": ((0, 600), (500, 1024))}, id="overlapping-segment-ranges"),
        pytest.param({"snapshot_count": False}, id="bool-snapshot-count"),
        pytest.param({"snapshot_count": 1023}, id="snapshot-count-mismatch"),
        pytest.param({"diagonal_loading": math.nan}, id="nonfinite-loading"),
        pytest.param({"rank_rtol": None}, id="null-rank-tolerance"),
        pytest.param({"raw_rank_threshold": -1.0}, id="negative-rank-threshold"),
        pytest.param({"raw_effective_rank": False}, id="bool-effective-rank"),
        pytest.param({"loaded_eigenvalue_tolerance": math.inf}, id="nonfinite-loaded-tolerance"),
        pytest.param({"loaded_condition_number": math.nan}, id="nonfinite-condition-number"),
        pytest.param({"synchronization": "synchronization_marker"}, id="non-bool-synchronization"),
        pytest.param({"synchronization_conflict": None}, id="null-synchronization-conflict"),
        pytest.param({"calibration_state": None}, id="null-calibration-state"),
        pytest.param({"calibration_usable": 1}, id="integer-calibration-usable"),
        pytest.param({"geometry": {"geometry_marker": "must_not_escape"}}, id="mapping-geometry"),
        pytest.param({"manifest_reference_frequency_hz": math.inf}, id="nonfinite-manifest-reference"),
        pytest.param({"geometry_reference_frequency_hz": math.nan}, id="nonfinite-geometry-reference"),
        pytest.param({"geometry_reference_frequency_hz": _FC * 2.0}, id="geometry-reference-mismatch"),
        pytest.param({"physical_channel_count": 3}, id="physical-channel-count-too-small"),
        pytest.param({"filtered": None}, id="null-filtered"),
        pytest.param({"remove_mean": 1}, id="integer-remove-mean"),
    ],
)
def test_forged_spatial_payload_internals_raise_safe_malformed(
    register_iq, payload_overrides: dict[str, object]
):
    _, _, store, evidence = _compute(register_iq, _ula_iq())
    record = store.get(evidence.data["spatial_evidence_id"])
    forged = replace(record.payload, **payload_overrides)
    forged_id = store.put(
        "spatial_evidence",
        record.source_ref,
        forged,
        {"channel_count": 4},
        selection=record.selection,
    )

    with pytest.raises(MalformedSpatialEvidence) as raised:
        estimate_doa(EstimateDoaArgs(spatial_evidence_id=forged_id, source_count=1), artifact_store=store)

    assert str(raised.value) == "Spatial evidence payload is malformed."
    for marker in ("array_marker", "synchronization_marker", "geometry_marker", "must_not_escape"):
        assert marker not in str(raised.value)
    assert repr(payload_overrides) not in str(raised.value)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        pytest.param("source_ref", None, id="null-source-ref"),
        pytest.param("source_artifact_id", None, id="null-artifact-id"),
        pytest.param("source_artifact_id", "", id="empty-artifact-id"),
        pytest.param("source_artifact_type", ["payload_type_marker"], id="unhashable-artifact-type"),
        pytest.param("source_artifact_type", None, id="null-artifact-type"),
        pytest.param("method", None, id="null-method"),
        pytest.param("domain", ["payload_domain_marker"], id="non-string-domain"),
        pytest.param("candidates", "list", id="non-tuple-candidates"),
    ],
)
def test_candidate_set_forged_container_and_key_types_raise_safe_malformed(
    register_iq, field: str, forged_value: object
):
    repository, source_ref = register_iq(
        f"malformed_set_{field}",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    payload = _candidate_payload(source_ref, upstream_id)
    if field == "candidates":
        forged_value = list(payload.candidates)
    payload = replace(payload, **{field: forged_value})
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    with pytest.raises(MalformedSpatialEvidence) as raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
            repository=repository,
            artifact_store=store,
        )

    assert str(raised.value) == "Candidate evidence payload is malformed."
    assert "payload_type_marker" not in str(raised.value)
    assert "payload_domain_marker" not in str(raised.value)


@pytest.mark.parametrize(
    "candidate_element",
    [
        pytest.param({"candidate_element_marker": "must_not_escape"}, id="mapping"),
        pytest.param(None, id="null"),
    ],
)
def test_candidate_set_non_candidate_elements_raise_safe_malformed(register_iq, candidate_element: object):
    repository, source_ref = register_iq(
        "malformed_candidate_element",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    payload = replace(_candidate_payload(source_ref, upstream_id), candidates=(candidate_element,))
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    with pytest.raises(MalformedSpatialEvidence) as raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
            repository=repository,
            artifact_store=store,
        )

    assert str(raised.value) == "Candidate evidence payload is malformed."
    assert "candidate_element_marker" not in str(raised.value)
    assert "must_not_escape" not in str(raised.value)
    assert repr(candidate_element) not in str(raised.value)


@pytest.mark.parametrize(
    "candidate_overrides",
    [
        pytest.param({"candidate_id": None}, id="null-candidate-id"),
        pytest.param({"source_ref": ["candidate_source_marker"]}, id="non-string-source-ref"),
        pytest.param({"source_artifact_id": ""}, id="empty-artifact-id"),
        pytest.param({"source_artifact_type": ["candidate_type_marker"]}, id="non-string-artifact-type"),
        pytest.param({"method": None}, id="null-method"),
        pytest.param({"domain": 1}, id="non-string-domain"),
        pytest.param({"source_block_index": False}, id="bool-block-index"),
        pytest.param({"source_block_index": 0.0}, id="float-block-index"),
        pytest.param({"source_block_index": "candidate_block_marker"}, id="string-block-index"),
        pytest.param({"sample_range": [100, 400]}, id="list-sample-range"),
        pytest.param({"sample_range": (100,)}, id="short-sample-range"),
        pytest.param({"sample_range": (100.0, 400)}, id="float-range-bound"),
        pytest.param({"sample_range": (100, True)}, id="bool-range-bound"),
        pytest.param({"sample_range": (-1, 400)}, id="negative-range-bound"),
        pytest.param({"sample_range": (400, 100)}, id="reversed-range"),
        pytest.param({"channel_indices": [0, 1, 2, 3]}, id="list-channel-indices"),
        pytest.param({"channel_indices": ()}, id="empty-channel-indices"),
        pytest.param({"channel_indices": (False, 1, 2, 3)}, id="bool-channel-index"),
        pytest.param({"channel_indices": (0.0, 1, 2, 3)}, id="float-channel-index"),
        pytest.param({"channel_indices": (-1, 1, 2, 3)}, id="negative-channel-index"),
        pytest.param({"channel_indices": (0, 1, 1, 3)}, id="duplicate-channel-index"),
        pytest.param({"frequency_range": None}, id="null-frequency-range"),
        pytest.param(
            {"frequency_range": FrequencyRange.model_construct(start_hz=math.nan, stop_hz=120.0, reference="baseband")},
            id="nan-frequency-start",
        ),
        pytest.param(
            {"frequency_range": FrequencyRange.model_construct(start_hz=40.0, stop_hz=math.inf, reference="baseband")},
            id="infinite-frequency-stop",
        ),
        pytest.param({"peak_frequency_hz": math.nan}, id="nan-peak-frequency"),
        pytest.param({"peak_power_linear": math.inf}, id="infinite-peak-power"),
        pytest.param({"relative_power_db": -math.inf}, id="infinite-relative-power"),
        pytest.param({"confidence": math.nan}, id="nan-confidence"),
    ],
)
def test_candidate_forged_structure_and_nonfinite_physics_raise_safe_malformed(
    register_iq, candidate_overrides: dict[str, object]
):
    repository, source_ref = register_iq(
        "malformed_candidate_fields",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    payload = _candidate_payload(source_ref, upstream_id)
    payload = replace(payload, candidates=(replace(payload.candidates[0], **candidate_overrides),))
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    with pytest.raises(MalformedSpatialEvidence) as raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
            repository=repository,
            artifact_store=store,
        )

    assert str(raised.value) == "Candidate evidence payload is malformed."
    for marker in (
        "candidate_source_marker",
        "candidate_type_marker",
        "candidate_block_marker",
    ):
        assert marker not in str(raised.value)


@pytest.mark.parametrize(
    "candidate_overrides",
    [
        pytest.param({"mask_indices": None}, id="null-mask"),
        pytest.param({"mask_indices": [(0, 0)]}, id="list-mask"),
        pytest.param({"mask_indices": ((0,),)}, id="short-mask-pair"),
        pytest.param({"mask_indices": ((False, 0),)}, id="bool-mask-index"),
        pytest.param({"mask_indices": ((-1, 0),)}, id="negative-mask-index"),
        pytest.param({"mask_indices": (("membership_marker", 0),)}, id="string-mask-index"),
        pytest.param({"track_frame_indices": None}, id="null-track-frames"),
        pytest.param({"track_frame_indices": [0]}, id="list-track-frames"),
        pytest.param({"track_frame_indices": (False,)}, id="bool-track-frame"),
        pytest.param({"track_frame_indices": (-1,)}, id="negative-track-frame"),
        pytest.param({"track_frequency_bin_indices": None}, id="null-track-bins"),
        pytest.param({"track_frequency_bin_indices": [0]}, id="list-track-bins"),
        pytest.param({"track_frequency_bin_indices": (False,)}, id="bool-track-bin"),
        pytest.param({"track_frequency_bin_indices": (-1,)}, id="negative-track-bin"),
        pytest.param(
            {"track_frame_indices": (0, 1), "track_frequency_bin_indices": (0,)},
            id="track-length-mismatch",
        ),
        pytest.param({"slope_hz_per_sample": math.nan}, id="nan-slope"),
        pytest.param({"slope_hz_per_sample": math.inf}, id="infinite-slope"),
        pytest.param({"slope_hz_per_sample": False}, id="bool-slope"),
        pytest.param({"slope_hz_per_sample": "slope_marker"}, id="string-slope"),
        pytest.param({"max_observed_jump_bins": False}, id="bool-maximum-jump"),
        pytest.param({"max_observed_jump_bins": -1}, id="negative-maximum-jump"),
        pytest.param({"max_observed_jump_bins": 1.0}, id="float-maximum-jump"),
    ],
)
def test_candidate_membership_structure_raises_safe_malformed(
    register_iq, candidate_overrides: dict[str, object]
):
    repository, source_ref = register_iq(
        "malformed_candidate_membership",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    payload = _candidate_payload(source_ref, upstream_id)
    candidate = replace(payload.candidates[0], **candidate_overrides)
    payload = replace(payload, candidates=(candidate,))
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    with pytest.raises(MalformedSpatialEvidence) as raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
            repository=repository,
            artifact_store=store,
        )

    assert str(raised.value) == "Candidate evidence payload is malformed."
    for marker in ("membership_marker", "slope_marker"):
        assert marker not in str(raised.value)
    assert repr(candidate_overrides) not in str(raised.value)


@pytest.mark.parametrize(
    ("artifact_type", "candidate_overrides"),
    [
        pytest.param("spectrum", {"mask_indices": ((0, 0),)}, id="spectral-membership"),
        pytest.param(
            "time_frequency",
            {"method": "tf_connected_regions", "domain": "time_frequency", "mask_indices": ((4, 0),)},
            id="mask-frequency-out-of-bounds",
        ),
        pytest.param(
            "time_frequency",
            {
                "method": "tf_tracks",
                "domain": "time_frequency",
                "mask_indices": ((0, 0),),
                "track_frame_indices": (1,),
                "track_frequency_bin_indices": (1,),
                "slope_hz_per_sample": 0.0,
                "max_observed_jump_bins": 0,
            },
            id="track-not-in-mask",
        ),
        pytest.param(
            "time_frequency",
            {
                "method": "tf_tracks",
                "domain": "time_frequency",
                "mask_indices": ((0, 0), (1, 1)),
                "track_frame_indices": (0, 1),
                "track_frequency_bin_indices": (0, 1),
                "slope_hz_per_sample": 0.4,
                "max_observed_jump_bins": 99,
            },
            id="maximum-jump-inconsistent",
        ),
    ],
)
def test_candidate_membership_disagreement_raises_safe_binding_error(
    register_iq, artifact_type: str, candidate_overrides: dict[str, object]
):
    repository, source_ref = register_iq(
        "candidate_membership_binding",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_payload = _candidate_spectrum() if artifact_type == "spectrum" else _candidate_time_frequency()
    upstream_id = store.put(artifact_type, source_ref, upstream_payload, {"kind": "test"})
    payload = _candidate_payload(source_ref, upstream_id)
    candidate = replace(
        payload.candidates[0],
        source_artifact_type=artifact_type,
        **candidate_overrides,
    )
    payload = replace(
        payload,
        source_artifact_type=artifact_type,
        method=candidate.method,
        domain=candidate.domain,
        candidates=(candidate,),
    )
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    with pytest.raises(SpatialEvidenceBindingError) as raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
            repository=repository,
            artifact_store=store,
        )

    assert str(raised.value) == "Spatial evidence binding is incompatible."
    assert repr(candidate_overrides) not in str(raised.value)


@pytest.mark.parametrize(
    "candidate_overrides",
    [
        pytest.param(
            {"method": "tf_connected_regions", "mask_indices": ((0, 0), (1, 1))},
            id="connected-region",
        ),
        pytest.param(
            {
                "method": "tf_tracks",
                "mask_indices": ((0, 0), (1, 1)),
                "track_frame_indices": (0, 1),
                "track_frequency_bin_indices": (0, 1),
                "slope_hz_per_sample": 0.4,
                "max_observed_jump_bins": 1,
            },
            id="track",
        ),
    ],
)
def test_candidate_valid_time_frequency_membership_remains_accepted(
    register_iq, candidate_overrides: dict[str, object]
):
    repository, source_ref = register_iq(
        "candidate_valid_membership",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_id = store.put("time_frequency", source_ref, _candidate_time_frequency(), {"kind": "test"})
    payload = _candidate_payload(source_ref, upstream_id)
    candidate = replace(
        payload.candidates[0],
        source_artifact_type="time_frequency",
        domain="time_frequency",
        **candidate_overrides,
    )
    payload = replace(
        payload,
        source_artifact_type="time_frequency",
        method=candidate.method,
        domain="time_frequency",
        candidates=(candidate,),
    )
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    result = compute_spatial_evidence(
        _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
        repository=repository,
        artifact_store=store,
    )

    assert result.status is ToolStatus.SUCCESS


@pytest.mark.parametrize(
    "candidate_overrides",
    [
        pytest.param({"source_artifact_id": "spectrum_forged"}, id="artifact-id"),
        pytest.param({"source_artifact_type": "time_frequency"}, id="artifact-type"),
        pytest.param({"source_block_index": 3}, id="block-index-too-large"),
        pytest.param({"source_block_index": -1}, id="negative-block-index"),
        pytest.param({"sample_range": (100, 1025)}, id="range-outside-block"),
        pytest.param({"channel_indices": (0, 1)}, id="different-channels"),
    ],
)
def test_candidate_provenance_disagreement_raises_binding_error(
    register_iq, candidate_overrides: dict[str, object]
):
    repository, source_ref = register_iq(
        "binding_evidence",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    payload = _candidate_payload(source_ref, upstream_id, **candidate_overrides)
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    with pytest.raises(SpatialEvidenceBindingError) as raised:
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
            repository=repository,
            artifact_store=store,
        )

    assert isinstance(raised.value, SpatialError)
    assert candidate_set_id not in str(raised.value)
    assert "spectrum_forged" not in str(raised.value)
    assert "must_not_escape" not in str(raised.value)


@pytest.mark.parametrize(
    ("payload_update", "requested_candidate_id"),
    [
        ({"source_ref": "iq_other"}, "emission_0"),
        ({"source_artifact_type": "unsupported"}, "emission_0"),
        ({}, "emission_missing"),
    ],
)
def test_candidate_set_binding_and_scoped_id_disagreement_raise_binding_error(
    register_iq, payload_update: dict[str, object], requested_candidate_id: str
):
    repository, source_ref = register_iq(
        "set_binding",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    payload = _candidate_payload(source_ref, upstream_id)
    payload = replace(payload, **payload_update)
    candidate_set_id = store.put("candidate_set", source_ref, payload, {"candidate_count": 1})

    with pytest.raises(SpatialEvidenceBindingError) as raised:
        compute_spatial_evidence(
            _spatial_args(
                source_ref,
                candidate_set_id=candidate_set_id,
                candidate_id=requested_candidate_id,
            ),
            repository=repository,
            artifact_store=store,
        )

    assert isinstance(raised.value, SpatialError)
    assert requested_candidate_id not in str(raised.value)


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("unknown", ArtifactNotFound),
        ("expired", ArtifactExpired),
        ("cross_run", CrossRunArtifact),
        ("wrong_type", ArtifactTypeMismatch),
        ("wrong_source", ArtifactSourceMismatch),
    ],
)
def test_candidate_artifact_lifecycle_type_and_source_errors_propagate(
    register_iq, case: str, expected_error: type[Exception]
):
    repository, source_ref = register_iq(
        f"lifecycle_{case}",
        _ula_iq(),
        array_geometry=_geometry(),
        synchronized=True,
        reference_frequency_hz=_FC,
    )
    store = _store()
    candidate_set_id = "candidate_set_unknown"
    if case == "expired":
        candidate_set_id = store.put("candidate_set", source_ref, object(), {"candidate_count": 0})
        store.clear()
    elif case == "cross_run":
        foreign_store = RunArtifactStore(run_id="foreign_spatial_run")
        candidate_set_id = foreign_store.put("candidate_set", source_ref, object(), {"candidate_count": 0})
    elif case == "wrong_type":
        candidate_set_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"bin_count": 2})
    elif case == "wrong_source":
        candidate_set_id = store.put("candidate_set", "iq_other", object(), {"candidate_count": 0})

    with pytest.raises(expected_error):
        compute_spatial_evidence(
            _spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"),
            repository=repository,
            artifact_store=store,
        )


def test_candidate_spatial_artifact_stores_effective_ranges(register_iq):
    repository, source_ref = register_iq("candidate_ranges", _ula_iq(), array_geometry=_geometry(), synchronized=True, reference_frequency_hz=_FC)
    store = _store()
    upstream_id = store.put("spectrum", source_ref, _candidate_spectrum(), {"kind": "test"})
    candidate = EmissionCandidate(
        candidate_id="emission_0", source_ref=source_ref, source_artifact_id=upstream_id, source_artifact_type="spectrum",
        method="spectral_regions", domain="spectral", source_block_index=0, sample_range=(100, 400), channel_indices=(0, 1, 2, 3),
        frequency_range=FrequencyRange(start_hz=40.0, stop_hz=120.0, reference="baseband"), peak_frequency_hz=80.0,
        peak_power_linear=1.0, relative_power_db=20.0, confidence=1.0, lower_truncated=False, upper_truncated=False,
    )
    candidate_set_id = store.put("candidate_set", source_ref, CandidateSetPayload(source_ref, upstream_id, "spectrum", "spectral_regions", "spectral", (candidate,)), {"candidate_count": 1})
    result = compute_spatial_evidence(_spatial_args(source_ref, candidate_set_id=candidate_set_id, candidate_id="emission_0"), repository=repository, artifact_store=store)
    record = store.get(result.data["spatial_evidence_id"])
    assert [(item.start, item.stop) for item in record.selection.sample_ranges] == [(100, 400)]


def test_reference_frequency_is_finite_and_defensively_rejected(register_iq):
    with pytest.raises(ValidationError):
        ArrayGeometry(element_count=4, element_spacing_m=0.1, reference_frequency_hz=math.inf)
    _, _, store, evidence = _compute(register_iq, _ula_iq())
    record = store.get(evidence.data["spatial_evidence_id"])
    forged = replace(record.payload, geometry_reference_frequency_hz=math.inf)
    bad_id = store.put("spatial_evidence", record.source_ref, forged, {"channel_count": 4}, selection=record.selection)
    with pytest.raises(MalformedSpatialEvidence) as raised:
        estimate_doa(EstimateDoaArgs(spatial_evidence_id=bad_id, source_count=1), artifact_store=store)
    assert str(raised.value) == "Spatial evidence payload is malformed."


def test_endfire_warning_requires_actual_scanned_endpoints(register_iq):
    _, _, store, evidence = _compute(register_iq, _ula_iq())
    result = estimate_doa(EstimateDoaArgs(spatial_evidence_id=evidence.data["spatial_evidence_id"], source_count=1, grid_step_deg=7.0), artifact_store=store)
    assert result.data["grid"]["last_deg"] == 85.0
    assert not any("endfire" in warning.lower() for warning in result.warnings)


def test_huge_loading_is_a_safe_numerical_failure_without_warning(register_iq):
    repository, source_ref = register_iq("huge_loading", _ula_iq(), array_geometry=_geometry(), synchronized=True, reference_frequency_hz=_FC)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_spatial_evidence(_spatial_args(source_ref, diagonal_loading_fraction=1e308), repository=repository, artifact_store=_store())
    assert not caught
    assert result.status is ToolStatus.ERROR and result.error.code == "NUMERICAL_FAILURE"


def test_eigenspectrum_normalizes_tiny_negative_and_rejects_material_negative():
    segment = np.eye(2, dtype=np.complex128)
    vectors = np.eye(2, dtype=np.complex128)
    with patch("cityrf.legacy.dsp.spatial.np.linalg.eigh", side_effect=[(np.array([-1e-18, 1.0]), vectors), (np.array([-1e-18, 1.0]), vectors)]):
        _, _, values, _, _, _, _ = covariance_from_segments([segment], diagonal_loading_fraction=0.0)
    assert values[-1] == 0.0
    with patch("cityrf.legacy.dsp.spatial.np.linalg.eigh", side_effect=[(np.array([-1e-3, 1.0]), vectors), (np.array([-1e-3, 1.0]), vectors)]):
        with pytest.raises(SpatialError):
            covariance_from_segments([segment], diagonal_loading_fraction=0.0)
