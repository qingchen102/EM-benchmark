"""Strict-TDD contracts for activity and emission-candidate detection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cityrf.legacy.dsp.detection import (
    InapplicableDetectionEvidence,
    InsufficientDetectionSamples,
    InvalidDetectionParameters,
    spectral_region_candidates,
)
from cityrf.legacy.dsp.representations import SpectrumBlock, SpectrumPayload
from cityrf.legacy.runtime import RunArtifactStore
from cityrf.legacy.runtime.artifact_store import (
    ArtifactNotFound,
    ArtifactSourceMismatch,
    ArtifactTypeMismatch,
    CrossRunArtifact,
)
from cityrf.legacy.schemas import FrequencyRange, SampleRange, ToolStatus
from cityrf.legacy.tools.detection import (
    DetectActivityIntervalsArgs,
    DetectEmissionsArgs,
    EstimateSourceCountArgs,
    detect_activity_intervals,
    detect_emissions,
    estimate_source_count,
)
from cityrf.legacy.tools.representations import (
    ComputeSpectrumArgs,
    ComputeTimeFrequencyArgs,
    compute_spectrum,
    compute_time_frequency,
)


def _store() -> RunArtifactStore:
    return RunArtifactStore(run_id="detection_contract_run")


def _pulse_fixture() -> np.ndarray:
    rng = np.random.default_rng(904)
    sample_count = 1_024
    samples = np.arange(sample_count)
    gate = np.zeros(sample_count, dtype=float)
    for start in (100, 300, 500, 700):
        gate[start : start + 40] = 1.0
    tone = gate * np.exp(2j * np.pi * 0.09 * samples)
    noise = 0.01 * (rng.standard_normal(sample_count) + 1j * rng.standard_normal(sample_count))
    # Opposite channel phase proves power, rather than complex-IQ, averaging.
    return np.vstack((tone + noise, -tone + noise)).astype(np.complex64)


def test_activity_recovers_half_open_pulses_from_noncoherent_channel_power(register_iq):
    repository, source_ref = register_iq("activity_pulses", _pulse_fixture())

    result = detect_activity_intervals(
        DetectActivityIntervalsArgs(
            source_ref=source_ref,
            smoothing_window_samples=5,
            noise_quantile=0.25,
            high_threshold_db=10.0,
            low_threshold_db=6.0,
            min_active_samples=20,
            merge_gap_samples=0,
            padding_samples=0,
        ),
        repository=repository,
    )

    ranges = result.data["activity_ranges"]
    assert result.status is ToolStatus.SUCCESS
    assert len(ranges) == 4
    for observed, expected_start in zip(ranges, (100, 300, 500, 700), strict=True):
        assert abs(observed["start"] - expected_start) <= 3
        assert abs(observed["stop"] - (expected_start + 40)) <= 3
        assert observed["stop"] > observed["start"]
    assert result.data["duty_cycle"] == pytest.approx(160 / 1_024, abs=0.03)
    assert result.provenance["channel_aggregation"] == "noncoherent_mean_power"
    assert result.semantics["activity_ranges"].startswith("Half-open")
    assert "envelope" not in result.model_dump_json().lower()


def test_activity_never_bridges_unselected_gaps_and_uses_selected_denominator(register_iq):
    repository, source_ref = register_iq("activity_ranges", _pulse_fixture())

    result = detect_activity_intervals(
        DetectActivityIntervalsArgs(
            source_ref=source_ref,
            sample_ranges=[SampleRange(start=80, stop=180), SampleRange(start=300, stop=380)],
            range_mode="aggregate",
            smoothing_window_samples=3,
            high_threshold_db=10.0,
            low_threshold_db=6.0,
            min_active_samples=10,
            merge_gap_samples=100,
            padding_samples=10,
        ),
        repository=repository,
    )

    ranges = result.data["activity_ranges"]
    assert all(80 <= interval["start"] < interval["stop"] <= 180 or 300 <= interval["start"] < interval["stop"] <= 380 for interval in ranges)
    assert result.data["selected_sample_count"] == 180
    assert result.data["duty_cycle"] == pytest.approx(sum(item["stop"] - item["start"] for item in ranges) / 180)
    assert result.data["boundary_truncated"] is True


def test_activity_no_signal_is_a_successful_empty_result(register_iq):
    rng = np.random.default_rng(8)
    noise = 0.01 * (rng.standard_normal(512) + 1j * rng.standard_normal(512))
    repository, source_ref = register_iq("activity_none", noise)

    result = detect_activity_intervals(
        DetectActivityIntervalsArgs(source_ref=source_ref, high_threshold_db=18.0, low_threshold_db=12.0),
        repository=repository,
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.data["activity_ranges"] == []
    assert result.data["interval_count"] == 0


def test_activity_all_zero_complex64_is_successful_empty_result(register_iq):
    repository, source_ref = register_iq(
        "activity_all_zero", np.zeros(256, dtype=np.complex64)
    )

    result = detect_activity_intervals(
        DetectActivityIntervalsArgs(source_ref=source_ref), repository=repository
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.data["activity_ranges"] == []
    assert result.data["interval_count"] == 0
    assert result.data["duty_cycle"] == 0.0
    assert np.isfinite(result.data["noise_power"])


def test_activity_boundary_truncation_is_scoped_to_each_interval(register_iq):
    rng = np.random.default_rng(117)
    samples = np.arange(256)
    gate = np.zeros(256)
    gate[0:24] = 1.0
    gate[120:144] = 1.0
    noise = 0.005 * (rng.standard_normal(256) + 1j * rng.standard_normal(256))
    iq = gate * np.exp(2j * np.pi * 0.08 * samples) + noise
    repository, source_ref = register_iq("activity_truncation", iq)

    result = detect_activity_intervals(
        DetectActivityIntervalsArgs(
            source_ref=source_ref,
            smoothing_window_samples=1,
            high_threshold_db=10.0,
            low_threshold_db=6.0,
            min_active_samples=12,
            padding_samples=8,
        ),
        repository=repository,
    )

    boundary, interior = result.data["activity_ranges"]
    assert boundary["start"] == 0
    assert boundary["boundary_truncated"] is True
    assert 100 < interior["start"] < interior["stop"] < 170
    assert interior["boundary_truncated"] is False


def test_activity_disclosure_warning_advises_narrower_rerun_without_claiming_artifact_access(
    register_iq,
):
    rng = np.random.default_rng(441)
    samples = np.arange(1_024)
    gate = np.zeros(1_024)
    for start in range(20, 920, 90):
        gate[start : start + 20] = 1.0
    noise = 0.003 * (rng.standard_normal(1_024) + 1j * rng.standard_normal(1_024))
    repository, source_ref = register_iq(
        "activity_disclosure", gate * np.exp(2j * np.pi * 0.07 * samples) + noise
    )

    result = detect_activity_intervals(
        DetectActivityIntervalsArgs(
            source_ref=source_ref,
            smoothing_window_samples=1,
            min_active_samples=10,
            high_threshold_db=10.0,
            low_threshold_db=6.0,
        ),
        repository=repository,
    )

    warning = " ".join(result.warnings).lower()
    assert result.data["interval_count"] == 10
    assert len(result.data["activity_ranges"]) == 8
    assert "rerun" in warning and "narrower" in warning
    assert "remain available" not in warning


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"high_threshold_db": 6.0, "low_threshold_db": 6.0}, InvalidDetectionParameters),
        ({"smoothing_window_samples": 1_025}, InsufficientDetectionSamples),
    ],
)
def test_activity_invalid_controls_raise_typed_errors(register_iq, kwargs, error):
    repository, source_ref = register_iq("activity_invalid", _pulse_fixture())

    with pytest.raises(error):
        detect_activity_intervals(DetectActivityIntervalsArgs(source_ref=source_ref, **kwargs), repository=repository)


def test_spectral_regions_store_two_stable_baseband_candidates(register_iq):
    sample_count = 2_048
    samples = np.arange(sample_count)
    iq = np.exp(2j * np.pi * 0.117 * samples) + 0.7 * np.exp(2j * np.pi * -0.211 * samples)
    repository, source_ref = register_iq("two_tones", iq)
    store = _store()
    spectrum = compute_spectrum(
        ComputeSpectrumArgs(source_ref=source_ref, profile="custom", nfft=2_048, detrend="none"),
        repository=repository,
        artifact_store=store,
    )

    result = detect_emissions(
        DetectEmissionsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum.data["spectrum_id"],
            method="spectral_regions",
            threshold_db=14.0,
            min_region_bins=1,
            merge_gap_bins=1,
        ),
        artifact_store=store,
    )

    candidates = result.data["candidates"]
    peaks = sorted(candidate["peak_frequency_hz"] / 1_024.0 for candidate in candidates)
    assert result.status is ToolStatus.SUCCESS
    assert len(candidates) >= 2
    assert any(abs(peak + 0.211) < 2 / sample_count for peak in peaks)
    assert any(abs(peak - 0.117) < 2 / sample_count for peak in peaks)
    assert all(candidate["frequency_range"]["reference"] == "baseband" for candidate in candidates)
    candidate_artifact = store.get(result.data["candidate_set_id"], expected_type="candidate_set")
    assert len(candidate_artifact.payload.candidates) >= 2
    assert [candidate.candidate_id for candidate in candidate_artifact.payload.candidates[:2]] == ["emission_0", "emission_1"]
    assert "physical transmitter" in " ".join(result.warnings).lower()
    assert "power_linear" not in result.model_dump_json()


def test_candidate_edges_use_true_nyquist_for_cropped_spectrum(register_iq):
    sample_rate_hz = 1_024.0
    samples = np.arange(1_024)
    nyquist_tone = np.exp(1j * np.pi * samples)
    repository, source_ref = register_iq(
        "cropped_nyquist", nyquist_tone, sample_rate_hz=sample_rate_hz
    )
    store = _store()
    spectrum = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=source_ref,
            profile="custom",
            nfft=1_024,
            detrend="none",
            frequency_range=FrequencyRange(
                start_hz=-sample_rate_hz / 2.0,
                stop_hz=-480.0,
                reference="baseband",
            ),
        ),
        repository=repository,
        artifact_store=store,
    )
    spectrum_payload = store.get(
        spectrum.data["spectrum_id"], expected_type="spectrum"
    ).payload

    detected = detect_emissions(
        DetectEmissionsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum.data["spectrum_id"],
            method="spectral_regions",
            threshold_db=12.0,
        ),
        artifact_store=store,
    )

    candidate = min(
        detected.data["candidates"], key=lambda item: abs(item["peak_frequency_hz"] + 512.0)
    )
    assert spectrum_payload.sample_rate_hz == sample_rate_hz
    assert candidate["frequency_range"]["start_hz"] == -sample_rate_hz / 2.0
    assert candidate["frequency_range"]["stop_hz"] <= sample_rate_hz / 2.0
    assert candidate["lower_truncated"] is True


def test_spectral_candidates_retain_stable_source_block_indices(register_iq):
    samples = np.arange(1_024)
    iq = np.concatenate(
        (
            np.exp(2j * np.pi * 0.10 * samples[:512]),
            np.exp(2j * np.pi * -0.20 * samples[512:]),
        )
    )
    repository, source_ref = register_iq("candidate_blocks", iq)
    store = _store()
    spectrum = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=source_ref,
            profile="custom",
            nfft=512,
            detrend="none",
            sample_ranges=[SampleRange(start=0, stop=512), SampleRange(start=512, stop=1_024)],
        ),
        repository=repository,
        artifact_store=store,
    )
    detected = detect_emissions(
        DetectEmissionsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum.data["spectrum_id"],
            method="spectral_regions",
            threshold_db=12.0,
        ),
        artifact_store=store,
    )

    payload = store.get(detected.data["candidate_set_id"], expected_type="candidate_set").payload
    assert {candidate.source_block_index for candidate in payload.candidates} == {0, 1}
    assert all(
        candidate.sample_range == ((0, 512) if candidate.source_block_index == 0 else (512, 1_024))
        for candidate in payload.candidates
    )


def test_stable_candidate_ids_do_not_depend_on_equal_block_scan_order():
    frequency_hz = np.array([-1.0, 0.0, 1.0])
    power_linear = np.array([1.0e-6, 1.0, 1.0e-6])

    def block(channel_index: int) -> SpectrumBlock:
        return SpectrumBlock(
            frequency_hz=frequency_hz,
            power_linear=power_linear,
            representation=power_linear,
            sample_range=(100, 200),
            channel_indices=(channel_index,),
        )

    def payload(blocks: tuple[SpectrumBlock, ...]) -> SpectrumPayload:
        return SpectrumPayload(
            frequency_hz=frequency_hz,
            power_linear=power_linear,
            representation=power_linear,
            blocks=blocks,
            segment_ranges=((100, 200),),
            method="fft",
            scaling="power",
            sample_rate_hz=8.0,
            frequency_resolution_hz=1.0,
        )

    forward = spectral_region_candidates(
        payload((block(0), block(1))),
        source_ref="iq_stable_order",
        source_artifact_id="spectrum_stable_order",
        threshold_db=12.0,
        noise_quantile=0.25,
        min_region_bins=1,
        merge_gap_bins=0,
    )
    reversed_blocks = spectral_region_candidates(
        payload((block(1), block(0))),
        source_ref="iq_stable_order",
        source_artifact_id="spectrum_stable_order",
        threshold_db=12.0,
        noise_quantile=0.25,
        min_region_bins=1,
        merge_gap_bins=0,
    )

    forward_ids = {candidate.channel_indices: candidate.candidate_id for candidate in forward}
    reversed_ids = {candidate.channel_indices: candidate.candidate_id for candidate in reversed_blocks}
    assert forward_ids == reversed_ids
    assert forward_ids[(0,)] == "emission_0"


def test_connected_tf_regions_report_original_sample_bounds(register_iq):
    sample_count = 1_024
    samples = np.arange(sample_count)
    gate = np.zeros(sample_count)
    gate[256:640] = 1.0
    iq = gate * np.exp(2j * np.pi * 0.16 * samples)
    repository, source_ref = register_iq("gated_tf", iq)
    store = _store()
    tf = compute_time_frequency(
        ComputeTimeFrequencyArgs(source_ref=source_ref, profile="custom", window_length=64, hop_length=16, nfft=64, detrend="none"),
        repository=repository,
        artifact_store=store,
    )

    result = detect_emissions(
        DetectEmissionsArgs(
            source_ref=source_ref,
            time_frequency_id=tf.data["time_frequency_id"],
            method="tf_connected_regions",
            threshold_db=12.0,
            min_time_frames=4,
            min_frequency_bins=1,
            merge_gap_bins=1,
        ),
        artifact_store=store,
    )

    candidate = max(result.data["candidates"], key=lambda value: value["relative_power_db"])
    # Per contract, bounds are the first complete included frame start through
    # the final included start plus one 64-sample analysis window.
    assert abs(candidate["sample_range"]["start"] - 256) <= 64
    assert abs(candidate["sample_range"]["stop"] - 640) <= 64
    assert candidate["sample_range"]["start"] < candidate["sample_range"]["stop"]
    assert candidate["frequency_range"]["start_hz"] < candidate["frequency_range"]["stop_hz"]


def test_tf_tracks_recover_a_positive_slope_chirp_with_bounded_jumps(register_iq, complex_linear_chirp):
    repository, source_ref = register_iq("track_chirp", complex_linear_chirp)
    store = _store()
    tf = compute_time_frequency(
        ComputeTimeFrequencyArgs(source_ref=source_ref, profile="custom", window_length=64, hop_length=16, nfft=128, detrend="none"),
        repository=repository,
        artifact_store=store,
    )

    result = detect_emissions(
        DetectEmissionsArgs(
            source_ref=source_ref,
            time_frequency_id=tf.data["time_frequency_id"],
            method="tf_tracks",
            threshold_db=8.0,
            min_track_frames=12,
            max_track_jump_bins=4,
            max_track_gap_frames=1,
        ),
        artifact_store=store,
    )

    candidate = max(result.data["candidates"], key=lambda value: value["track_frame_count"])
    assert candidate["track_frame_count"] >= 12
    assert candidate["slope_hz_per_sample"] > 0.0
    assert candidate["max_observed_jump_bins"] <= 4
    assert "lfm" not in result.model_dump_json().lower()


def test_candidate_count_uses_all_stored_candidates_and_caps_only_estimate(register_iq):
    sample_count = 2_048
    samples = np.arange(sample_count)
    iq = np.exp(2j * np.pi * 0.10 * samples) + 0.9 * np.exp(2j * np.pi * -0.20 * samples)
    repository, source_ref = register_iq("count_tones", iq)
    store = _store()
    spectrum = compute_spectrum(
        ComputeSpectrumArgs(source_ref=source_ref, profile="custom", nfft=2_048, detrend="none"),
        repository=repository,
        artifact_store=store,
    )
    detected = detect_emissions(
        DetectEmissionsArgs(source_ref=source_ref, spectrum_id=spectrum.data["spectrum_id"], method="spectral_regions", threshold_db=14.0),
        artifact_store=store,
    )

    result = estimate_source_count(
        EstimateSourceCountArgs.model_validate({"method": "candidate_count", "source_ref": source_ref, "candidate_set_id": detected.data["candidate_set_id"], "min_candidate_confidence": 0.0, "max_sources": 1}),
        artifact_store=store,
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.data["total_stored_candidates"] >= 2
    assert result.data["confidence_qualified_candidates"] >= 2
    assert result.data["resolvable_region_count"] == 1
    assert result.data["estimator_capped"] is True
    assert result.data["evidence_domain"] == "spectral"
    assert "not physical" in " ".join(result.warnings).lower()


def _forge_candidate_evidence_reference(
    store: RunArtifactStore,
    candidate_set_id: str,
    *,
    source_ref: str,
    source_artifact_id: str,
    source_artifact_type: str = "spectrum",
) -> str:
    original = store.get(candidate_set_id, expected_type="candidate_set").payload
    payload = replace(
        original,
        source_artifact_id=source_artifact_id,
        source_artifact_type=source_artifact_type,
    )
    return store.put(
        "candidate_set",
        source_ref,
        payload,
        {
            "candidate_count": len(payload.candidates),
            "method": payload.method,
            "evidence_domain": payload.domain,
        },
    )


def _one_detected_candidate_set(register_iq, *, store: RunArtifactStore, name: str):
    samples = np.arange(1_024)
    repository, source_ref = register_iq(name, np.exp(2j * np.pi * 0.125 * samples))
    spectrum = compute_spectrum(
        ComputeSpectrumArgs(source_ref=source_ref, profile="custom", nfft=1_024),
        repository=repository,
        artifact_store=store,
    )
    detected = detect_emissions(
        DetectEmissionsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum.data["spectrum_id"],
            method="spectral_regions",
        ),
        artifact_store=store,
    )
    return repository, source_ref, spectrum.data["spectrum_id"], detected.data["candidate_set_id"]


def test_candidate_count_resolves_nonexistent_upstream_evidence(register_iq):
    store = _store()
    _, source_ref, _, candidate_set_id = _one_detected_candidate_set(
        register_iq, store=store, name="missing_upstream"
    )
    forged_id = _forge_candidate_evidence_reference(
        store,
        candidate_set_id,
        source_ref=source_ref,
        source_artifact_id="spectrum_unknown_handle",
    )

    with pytest.raises(ArtifactNotFound):
        estimate_source_count(
            EstimateSourceCountArgs.model_validate({"method": "candidate_count", "source_ref": source_ref, "candidate_set_id": forged_id}),
            artifact_store=store,
        )


def test_candidate_count_resolves_upstream_evidence_type(register_iq):
    store = _store()
    repository, source_ref, _, candidate_set_id = _one_detected_candidate_set(
        register_iq, store=store, name="wrong_upstream_type"
    )
    time_frequency = compute_time_frequency(
        ComputeTimeFrequencyArgs(
            source_ref=source_ref,
            profile="custom",
            window_length=64,
            hop_length=32,
            nfft=64,
        ),
        repository=repository,
        artifact_store=store,
    )
    forged_id = _forge_candidate_evidence_reference(
        store,
        candidate_set_id,
        source_ref=source_ref,
        source_artifact_id=time_frequency.data["time_frequency_id"],
        source_artifact_type="spectrum",
    )

    with pytest.raises(ArtifactTypeMismatch):
        estimate_source_count(
            EstimateSourceCountArgs.model_validate({"method": "candidate_count", "source_ref": source_ref, "candidate_set_id": forged_id}),
            artifact_store=store,
        )


def test_candidate_count_resolves_upstream_evidence_source(register_iq):
    store = _store()
    _, source_ref, _, candidate_set_id = _one_detected_candidate_set(
        register_iq, store=store, name="upstream_source_one"
    )
    _, other_source_ref, other_spectrum_id, _ = _one_detected_candidate_set(
        register_iq, store=store, name="upstream_source_two"
    )
    assert other_source_ref != source_ref
    forged_id = _forge_candidate_evidence_reference(
        store,
        candidate_set_id,
        source_ref=source_ref,
        source_artifact_id=other_spectrum_id,
    )

    with pytest.raises(ArtifactSourceMismatch):
        estimate_source_count(
            EstimateSourceCountArgs.model_validate({"method": "candidate_count", "source_ref": source_ref, "candidate_set_id": forged_id}),
            artifact_store=store,
        )


def test_candidate_count_resolves_upstream_evidence_run(register_iq):
    first_store = RunArtifactStore(run_id="upstream_first_run")
    _, source_ref, spectrum_id, candidate_set_id = _one_detected_candidate_set(
        register_iq, store=first_store, name="upstream_cross_run"
    )
    second_store = RunArtifactStore(run_id="upstream_second_run")
    original = first_store.get(candidate_set_id, expected_type="candidate_set").payload
    forged_id = second_store.put(
        "candidate_set",
        source_ref,
        replace(original, source_artifact_id=spectrum_id),
        {
            "candidate_count": len(original.candidates),
            "method": original.method,
            "evidence_domain": original.domain,
        },
    )

    with pytest.raises(CrossRunArtifact):
        estimate_source_count(
            EstimateSourceCountArgs.model_validate({"method": "candidate_count", "source_ref": source_ref, "candidate_set_id": forged_id}),
            artifact_store=second_store,
        )


@pytest.mark.parametrize(
    ("artifact_type", "method", "payload"),
    [
        ("spectrum", "spectral_regions", {}),
        ("time_frequency", "tf_connected_regions", object()),
    ],
)
def test_detect_emissions_rejects_malformed_typed_artifact_payloads(
    artifact_type, method, payload
):
    store = _store()
    source_ref = "iq_malformed_payload"
    artifact_id = store.put(
        artifact_type,
        source_ref,
        payload,
        {"evidence_kind": artifact_type},
    )
    args = DetectEmissionsArgs(
        source_ref=source_ref,
        method=method,
        spectrum_id=artifact_id if artifact_type == "spectrum" else None,
        time_frequency_id=artifact_id if artifact_type == "time_frequency" else None,
    )

    with pytest.raises(InapplicableDetectionEvidence):
        detect_emissions(args, artifact_store=store)


def test_activity_rejects_extreme_finite_db_before_exponentiation(register_iq):
    repository, source_ref = register_iq(
        "activity_extreme_db", np.ones(128, dtype=np.complex64)
    )

    with pytest.raises(InvalidDetectionParameters):
        detect_activity_intervals(
            DetectActivityIntervalsArgs(
                source_ref=source_ref,
                high_threshold_db=1.0e308,
                low_threshold_db=0.0,
            ),
            repository=repository,
        )


def test_spectral_detection_rejects_extreme_finite_db_before_exponentiation(register_iq):
    samples = np.arange(256)
    repository, source_ref = register_iq(
        "spectral_extreme_db", np.exp(2j * np.pi * 0.125 * samples)
    )
    store = _store()
    spectrum = compute_spectrum(
        ComputeSpectrumArgs(source_ref=source_ref, profile="custom", nfft=256),
        repository=repository,
        artifact_store=store,
    )

    with pytest.raises(InvalidDetectionParameters):
        detect_emissions(
            DetectEmissionsArgs(
                source_ref=source_ref,
                spectrum_id=spectrum.data["spectrum_id"],
                method="spectral_regions",
                threshold_db=1.0e308,
            ),
            artifact_store=store,
        )


def test_tf_detection_rejects_extreme_finite_db_before_exponentiation(register_iq):
    samples = np.arange(256)
    repository, source_ref = register_iq(
        "tf_extreme_db", np.exp(2j * np.pi * 0.125 * samples)
    )
    store = _store()
    time_frequency = compute_time_frequency(
        ComputeTimeFrequencyArgs(
            source_ref=source_ref,
            profile="custom",
            window_length=64,
            hop_length=32,
            nfft=64,
        ),
        repository=repository,
        artifact_store=store,
    )

    with pytest.raises(InvalidDetectionParameters):
        detect_emissions(
            DetectEmissionsArgs(
                source_ref=source_ref,
                time_frequency_id=time_frequency.data["time_frequency_id"],
                method="tf_connected_regions",
                threshold_db=1.0e308,
            ),
            artifact_store=store,
        )


def test_candidate_consumers_reject_source_binding_mismatch(register_iq, complex_tone):
    repository, source_ref = register_iq("binding_one", complex_tone)
    _, wrong_source_ref = register_iq("binding_two", complex_tone)
    store = _store()
    spectrum = compute_spectrum(
        ComputeSpectrumArgs(source_ref=source_ref, profile="custom", nfft=1_024), repository=repository, artifact_store=store
    )

    with pytest.raises(Exception):
        detect_emissions(
            DetectEmissionsArgs(source_ref=wrong_source_ref, spectrum_id=spectrum.data["spectrum_id"], method="spectral_regions"),
            artifact_store=store,
        )
