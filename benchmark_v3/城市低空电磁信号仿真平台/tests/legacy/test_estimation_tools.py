"""Task 6 estimator contracts (written before the estimator modules)."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import numpy as np
import pytest

import cityrf.legacy.dsp.estimation as estimation_dsp
import cityrf.legacy.tools.estimation as estimation_tools
from cityrf.legacy.dsp.detection import CandidateSetPayload, EmissionCandidate
from cityrf.legacy.dsp.estimation import (
    CandidateEvidenceMismatch,
    EstimationBindingError,
    InsufficientFeatureSamples,
    InvalidModulationFeatureParameters,
    MalformedEstimationEvidence,
    NonFiniteEstimationResult,
    UnusablePowerEvidence,
    _lag_correlation_evidence,
    _spectral_shape_evidence,
)
from cityrf.legacy.dsp.representations import SpectrumBlock, SpectrumPayload, TimeFrequencyBlock, TimeFrequencyPayload
from cityrf.legacy.runtime import IQRepository, RunArtifactStore
from cityrf.legacy.schemas import FrequencyRange, IQSampleManifest, ProtectedChannel, SignalSelection, ToolStatus
from cityrf.legacy.tools.representations import ComputeSpectrumArgs, ComputeTimeFrequencyArgs, compute_spectrum, compute_time_frequency
from cityrf.legacy.tools.estimation import (
    EstimateBandwidthArgs,
    EstimateFrequencyArgs,
    EstimatePulseParametersArgs,
    ExtractModulationFeaturesArgs,
    MeasurePowerRelationsArgs,
    PulseIntervalInput,
    estimate_bandwidth,
    estimate_frequency,
    estimate_pulse_parameters,
    extract_modulation_features,
    measure_power_relations,
    _task7_is_finite_real,
)


def _spectrum(store, source_ref, frequency, power, *, scaling="power"):
    block = SpectrumBlock(np.asarray(frequency, float), np.asarray(power, float), np.asarray(power, float), (0, 1024), (0,))
    payload = SpectrumPayload(block.frequency_hz, block.power_linear, block.representation, (block,), ((0, 1024),), "fft", scaling, 1024.0, 1.0)
    return store.put("spectrum", source_ref, payload, {"method": "fft"}, selection=SignalSelection(source_ref=source_ref))


def _candidate(store, source_ref, artifact_id, artifact_type, domain, candidate):
    payload = CandidateSetPayload(source_ref, artifact_id, artifact_type, candidate.method, domain, (candidate,))
    return store.put("candidate_set", source_ref, payload, {"candidate_count": 1}, selection=SignalSelection(source_ref=source_ref))


def _candidate_for(source_ref, artifact_id, artifact_type, domain, *, block=0, method=None, mask=(), track_frames=(), track_bins=(), bounds=(-4.5, 4.5)):
    resolved_method = method or (
        "spectral_regions" if domain == "spectral" else "tf_connected_regions"
    )
    return EmissionCandidate("emission_0", source_ref, artifact_id, artifact_type, resolved_method, domain, block, (0, 1024), (0,), FrequencyRange(start_hz=bounds[0], stop_hz=bounds[1], reference="baseband"), 0.0, 1.0, 1.0, 1.0, False, False, tuple(mask), tuple(track_frames), tuple(track_bins))


def _time_frequency(store, source_ref, frequency, power, centers=None):
    frequency_array = np.asarray(frequency, dtype=float)
    power_array = np.asarray(power, dtype=float)
    frame_count = power_array.shape[1] if power_array.ndim == 2 else 1
    center_array = (
        np.arange(frame_count, dtype=float) * 100 + 50
        if centers is None
        else np.asarray(centers)
    )
    starts = np.arange(frame_count, dtype=float) * 100
    block = TimeFrequencyBlock(
        frequency_hz=frequency_array,
        values=np.asarray(power_array, dtype=np.complex128),
        power_linear=power_array,
        frame_start_samples=starts,
        frame_center_samples=center_array,
        sample_range=(0, 1024),
        channel_indices=(0,),
    )
    payload = TimeFrequencyPayload(
        blocks=(block,),
        output="power_spectrogram",
        sample_rate_hz=1024.0,
        frequency_resolution_hz=1.0,
        time_resolution_samples=100,
        window_length=100,
        nfft=max(2, frequency_array.size),
    )
    evidence_id = store.put(
        "time_frequency",
        source_ref,
        payload,
        {"frame_count": frame_count},
        selection=SignalSelection(source_ref=source_ref),
    )
    return evidence_id, block


def test_argument_models_bind_evidence_and_reject_nonfinite_controls():
    with pytest.raises(ValueError):
        EstimateFrequencyArgs(source_ref="s", method="quadratic_fft_peak")
    with pytest.raises(ValueError):
        EstimateBandwidthArgs(source_ref="s", method="tf_region_width", time_frequency_id="tf")
    with pytest.raises(ValueError):
        EstimateBandwidthArgs(source_ref="s", method="peak_drop", spectrum_id="sp", drop_db=float("nan"))
    assert PulseIntervalInput.model_validate({"start": 0, "stop": 2, "peak_to_noise_db": 9}).boundary_truncated is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_every_physical_control_rejects_nonfinite_values(value):
    with pytest.raises(ValueError): EstimateFrequencyArgs(source_ref="s", method="spectral_centroid", spectrum_id="sp", noise_quantile=value)
    with pytest.raises(ValueError): EstimateBandwidthArgs(source_ref="s", method="occupied_power", spectrum_id="sp", occupied_fraction=value)
    with pytest.raises(ValueError): EstimateBandwidthArgs(source_ref="s", method="threshold_edges", spectrum_id="sp", threshold_db_above_noise=value)
    with pytest.raises(ValueError): EstimateBandwidthArgs(source_ref="s", method="threshold_edges", spectrum_id="sp", noise_quantile=value)
    with pytest.raises(ValueError): EstimateBandwidthArgs(source_ref="s", method="peak_drop", spectrum_id="sp", drop_db=value)


def test_quadratic_frequency_centroid_and_ignored_control(register_iq):
    repository, source_ref = register_iq("est", np.ones(1024, np.complex64))
    store = RunArtifactStore("est")
    frequency = np.arange(-4, 5, dtype=float)
    spectrum_id = _spectrum(store, source_ref, frequency, [1, 1, 2, 8, 16, 8, 2, 1, 1])
    peak = estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref, method="quadratic_fft_peak", spectrum_id=spectrum_id, noise_quantile=0.49), repository=repository, artifact_store=store)
    assert peak.status is ToolStatus.SUCCESS
    assert peak.data["baseband_frequency_hz"] == pytest.approx(0.0)
    assert "noise_quantile" in peak.provenance["ignored_controls"]
    centroid = estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref, method="spectral_centroid", spectrum_id=spectrum_id, noise_quantile=.2), repository=repository, artifact_store=store)
    assert centroid.data["baseband_frequency_hz"] == pytest.approx(0.0)
    assert centroid.data["absolute_rf_hz"] == pytest.approx(2_400_000_000.0)


def test_centroid_locks_linear_quantile_and_threshold_linear_crossings(register_iq):
    repository, source_ref = register_iq("edges", np.ones(1024, np.complex64)); store = RunArtifactStore("edges")
    spectrum_id = _spectrum(store, source_ref, [-2, -1, 0, 1, 2], [1, 5, 10, 5, 1])
    centroid = estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref, method="spectral_centroid", spectrum_id=spectrum_id, noise_quantile=.25), repository=repository, artifact_store=store)
    assert centroid.data["method_details"]["baseline_power_linear"] == pytest.approx(1)
    threshold = estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref, method="threshold_edges", spectrum_id=spectrum_id, noise_quantile=.25, threshold_db_above_noise=3.010299956639812), repository=repository, artifact_store=store)
    assert threshold.data["lower_edge_hz"] == pytest.approx(-1.75)
    assert threshold.data["upper_edge_hz"] == pytest.approx(1.75)
    assert not any(threshold.data["edge_diagnostics"].values())


def test_nonbin_tone_and_halfbin_plateau_contract(register_iq):
    samples = np.arange(1024)
    repository, source_ref = register_iq("tone", np.exp(2j * np.pi * 129.25 * samples / 1024).astype(np.complex64))
    store = RunArtifactStore("tone")
    spectrum = compute_spectrum(ComputeSpectrumArgs(source_ref=source_ref, method="fft", profile="custom", nfft=1024, window="hann", detrend="none", scaling="power"), repository=repository, artifact_store=store)
    result = estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref, method="quadratic_fft_peak", spectrum_id=spectrum.data["spectrum_id"]), repository=repository, artifact_store=store)
    assert result.status is ToolStatus.SUCCESS
    assert result.data["baseband_frequency_hz"] == pytest.approx(129.25, abs=.25)
    assert abs(result.data["method_details"]["interpolation_offset_bins"]) < .5
    repository2, source2 = register_iq("half", np.exp(2j * np.pi * 129.5 * samples / 1024).astype(np.complex64))
    store2 = RunArtifactStore("half")
    spectrum2 = compute_spectrum(ComputeSpectrumArgs(source_ref=source2, method="fft", profile="custom", nfft=1024, window="hann", detrend="none", scaling="power"), repository=repository2, artifact_store=store2)
    plateau = estimate_frequency(EstimateFrequencyArgs(source_ref=source2, method="quadratic_fft_peak", spectrum_id=spectrum2.data["spectrum_id"]), repository=repository2, artifact_store=store2)
    assert plateau.status is ToolStatus.NOT_APPLICABLE and plateau.error.code == "AMBIGUOUS_DOMINANT_PEAK"


def test_all_four_bandwidth_definitions_and_true_three_db(register_iq):
    repository, source_ref = register_iq("bw", np.ones(1024, np.complex64))
    store = RunArtifactStore("bw")
    frequency = np.arange(-16, 17, dtype=float)
    power = 10.0 ** (-np.abs(frequency) / 10.0)
    spectrum_id = _spectrum(store, source_ref, frequency, power)
    three = estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref, method="peak_drop", spectrum_id=spectrum_id, drop_db=3), repository=repository, artifact_store=store)
    six = estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref, method="peak_drop", spectrum_id=spectrum_id, drop_db=6), repository=repository, artifact_store=store)
    assert three.status is ToolStatus.SUCCESS and three.data["bandwidth_hz"] == pytest.approx(6, abs=.05)
    assert "3 dB" in three.data["definition"]
    assert six.data["bandwidth_hz"] == pytest.approx(12, abs=.05)
    occupied = estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref, method="occupied_power", spectrum_id=spectrum_id), repository=repository, artifact_store=store)
    threshold = estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref, method="threshold_edges", spectrum_id=spectrum_id, threshold_db_above_noise=1), repository=repository, artifact_store=store)
    assert occupied.status is ToolStatus.SUCCESS
    assert threshold.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL}


def test_peak_drop_never_searches_beyond_candidate_scope(register_iq):
    repository, source_ref = register_iq("peak_scope", np.ones(1024, np.complex64))
    store = RunArtifactStore("peak_scope")
    frequency = np.arange(-4, 5, dtype=float)
    power = np.array([0.1, 0.2, 0.4, 0.8, 1.0, 0.8, 0.4, 0.2, 0.1])
    spectrum_id = _spectrum(store, source_ref, frequency, power)
    candidate = _candidate_for(
        source_ref,
        spectrum_id,
        "spectrum",
        "spectral",
        bounds=(-1.5, 3.5),
    )
    candidate_set_id = _candidate(
        store, source_ref, spectrum_id, "spectrum", "spectral", candidate
    )

    result = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="peak_drop",
            spectrum_id=spectrum_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
            drop_db=3.0,
        ),
        repository=repository,
        artifact_store=store,
    )

    assert result.status is ToolStatus.NOT_APPLICABLE
    assert result.error.code == "NO_TWO_SIDED_PEAK_DROP_CROSSINGS"
    assert result.data["edge_diagnostics"] == {
        "lower_scope_truncated": True,
        "upper_scope_truncated": False,
        "lower_observation_truncated": False,
        "upper_observation_truncated": False,
    }


def test_threshold_equal_disconnected_maxima_choose_lower_component_with_warning(register_iq):
    repository, source_ref = register_iq("threshold_tie", np.ones(1024, np.complex64))
    store = RunArtifactStore("threshold_tie")
    frequency = np.arange(-5, 6, dtype=float)
    power = np.array([1, 1, 5, 10, 5, 1, 5, 10, 5, 1, 1], dtype=float)
    spectrum_id = _spectrum(store, source_ref, frequency, power)

    result = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="threshold_edges",
            spectrum_id=spectrum_id,
            noise_quantile=0.2,
            threshold_db_above_noise=3.010299956639812,
        ),
        repository=repository,
        artifact_store=store,
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.data["lower_edge_hz"] == pytest.approx(-3.75)
    assert result.data["upper_edge_hz"] == pytest.approx(-0.25)
    assert any(
        "equal-height disconnected" in warning.lower()
        for warning in result.warnings
    )


def test_power_density_invariance_candidate_binding_and_nyquist(register_iq):
    repository, source_ref = register_iq("bind", np.ones(1024, np.complex64))
    store = RunArtifactStore("bind")
    frequency = np.array([-5., -3., -2., -1., 0., 1., 2., 4., 7.]); powers = np.array([1, 1, 2, 4, 16, 4, 2, 1, 1.], float)
    power_id = _spectrum(store, source_ref, frequency, powers)
    bin_widths = np.diff(np.r_[frequency[0] - (frequency[1] - frequency[0]) / 2, (frequency[:-1] + frequency[1:]) / 2, frequency[-1] + (frequency[-1] - frequency[-2]) / 2])
    density_id = _spectrum(store, source_ref, frequency, powers / bin_widths, scaling="power_density")
    a = estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref, method="occupied_power", spectrum_id=power_id), repository=repository, artifact_store=store)
    b = estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref, method="occupied_power", spectrum_id=density_id), repository=repository, artifact_store=store)
    assert a.data["bandwidth_hz"] == pytest.approx(b.data["bandwidth_hz"], abs=1e-12)
    candidate = _candidate_for(source_ref, power_id, "spectrum", "spectral", bounds=(-1.5, 1.5))
    candidate_id = _candidate(store, source_ref, power_id, "spectrum", "spectral", candidate)
    scoped = estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref, method="spectral_centroid", spectrum_id=power_id, candidate_set_id=candidate_id, candidate_id="emission_0"), repository=repository, artifact_store=store)
    assert scoped.data["evidence_block_index"] == 0
    assert -512 <= scoped.data["baseband_frequency_hz"] <= 512


def test_multiblock_source_block_binding_and_forged_index_fail(register_iq):
    repository, source_ref = register_iq("multi", np.ones(1024, np.complex64)); store=RunArtifactStore("multi")
    f=np.arange(-4,5,dtype=float); first=SpectrumBlock(f,np.array([1,1,1,1,16,1,1,1,1.]),np.ones(9),(0,1024),(0,)); second=SpectrumBlock(f,np.array([1,1,1,16,1,1,1,1,1.]),np.ones(9),(0,1024),(0,))
    payload=SpectrumPayload(f, first.power_linear, first.representation, (first,second), ((0,1024),), "fft","power",1024.,1.)
    evidence=store.put("spectrum",source_ref,payload,{"method":"fft"},selection=SignalSelection(source_ref=source_ref))
    c=_candidate_for(source_ref,evidence,"spectrum","spectral",block=1); cs=_candidate(store,source_ref,evidence,"spectrum","spectral",c)
    result=estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref,method="quadratic_fft_peak",spectrum_id=evidence,candidate_set_id=cs,candidate_id="emission_0"),repository=repository,artifact_store=store)
    assert result.data["evidence_block_index"] == 1 and result.data["baseband_frequency_hz"] == pytest.approx(-1)
    bad=_candidate_for(source_ref,evidence,"spectrum","spectral",block=9); badid=_candidate(store,source_ref,evidence,"spectrum","spectral",bad)
    with pytest.raises(EstimationBindingError): estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref,method="quadratic_fft_peak",spectrum_id=evidence,candidate_set_id=badid,candidate_id="emission_0"),repository=repository,artifact_store=store)
    wrong_source = replace(c, source_ref="foreign_source")
    wrongid = _candidate(store, source_ref, evidence, "spectrum", "spectral", wrong_source)
    with pytest.raises(EstimationBindingError): estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref,method="quadratic_fft_peak",spectrum_id=evidence,candidate_set_id=wrongid,candidate_id="emission_0"),repository=repository,artifact_store=store)
    wrong_channels = replace(c, channel_indices=(1,))
    wrong_channels_id = _candidate(store, source_ref, evidence, "spectrum", "spectral", wrong_channels)
    with pytest.raises(EstimationBindingError): estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref,method="quadratic_fft_peak",spectrum_id=evidence,candidate_set_id=wrong_channels_id,candidate_id="emission_0"),repository=repository,artifact_store=store)


def test_payload_sample_rate_mismatch_is_typed_before_result(register_iq):
    repository, source_ref = register_iq("fs", np.ones(1024, np.complex64)); store = RunArtifactStore("fs")
    f = np.arange(-4, 5, dtype=float); block = SpectrumBlock(f, np.ones(9), np.ones(9), (0, 1024), (0,))
    forged = SpectrumPayload(f, np.ones(9), np.ones(9), (block,), ((0, 1024),), "fft", "power", 512., 1.)
    evidence = store.put("spectrum", source_ref, forged, {"method": "fft"}, selection=SignalSelection(source_ref=source_ref))
    with pytest.raises(MalformedEstimationEvidence): estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref, method="spectral_centroid", spectrum_id=evidence), repository=repository, artifact_store=store)


def test_ridge_fit_tf_region_width_holes_and_public_budget(register_iq, complex_linear_chirp):
    repository, source_ref=register_iq("chirp",complex_linear_chirp); store=RunArtifactStore("chirp")
    tf=compute_time_frequency(ComputeTimeFrequencyArgs(source_ref=source_ref,profile="custom",window_length=128,hop_length=64,nfft=128,output="power_spectrogram",detrend="none"),repository=repository,artifact_store=store)
    ridge=estimate_frequency(EstimateFrequencyArgs(source_ref=source_ref,method="tf_ridge_fit",time_frequency_id=tf.data["time_frequency_id"]),repository=repository,artifact_store=store)
    assert ridge.status is ToolStatus.SUCCESS and ridge.data["method_details"]["chirp_rate_hz_per_s"] > 0 and len(ridge.data["trajectory_summary"]) <= 8
    block=TimeFrequencyBlock(np.arange(-4,5,dtype=float),np.ones((9,3),complex),np.ones((9,3)),np.array([0,100,200]),np.array([2,102,202]),(0,1024),(0,))
    payload=TimeFrequencyPayload((block,),"power_spectrogram",1024.,1.,100,4,9); evidence=store.put("time_frequency",source_ref,payload,{"frame_count":3},selection=SignalSelection(source_ref=source_ref))
    candidate=_candidate_for(source_ref,evidence,"time_frequency","time_frequency",mask=((1,0),(3,0),(2,1),(3,1),(4,1),(1,2))); cs=_candidate(store,source_ref,evidence,"time_frequency","time_frequency",candidate)
    width=estimate_bandwidth(EstimateBandwidthArgs(source_ref=source_ref,method="tf_region_width",time_frequency_id=evidence,candidate_set_id=cs,candidate_id="emission_0"),repository=repository,artifact_store=store)
    assert width.status is ToolStatus.SUCCESS and width.data["bandwidth_hz"] == pytest.approx(3) and width.data["method_details"]["minimum_span_hz"] == pytest.approx(1) and width.data["method_details"]["maximum_span_hz"] == pytest.approx(3) and width.quality["frames_with_internal_holes"] == 1


def test_tf_region_width_clips_membership_to_candidate_and_reports_representative_edges(register_iq):
    repository, source_ref = register_iq("tf_clip", np.ones(1024, np.complex64))
    store = RunArtifactStore("tf_clip")
    frequency = np.arange(-4, 5, dtype=float)
    evidence_id, _ = _time_frequency(store, source_ref, frequency, np.ones((9, 3)))
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        bounds=(-3.25, 3.25),
        mask=((1, 0), (3, 1), (4, 1), (5, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    result = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="tf_region_width",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    # Clipped spans are 0.75, 3.0, and 4.75 Hz; the middle actual frame is representative.
    assert result.data["bandwidth_hz"] == pytest.approx(3.0)
    assert result.data["lower_edge_hz"] == pytest.approx(-1.5)
    assert result.data["upper_edge_hz"] == pytest.approx(1.5)
    assert result.data["method_details"]["representative_edge_rule"] == "median_span_frame_or_average_central_frames"
    assert "representative" in result.data["definition"].lower()


def test_tf_region_width_even_median_averages_central_frame_edges(register_iq):
    repository, source_ref = register_iq("tf_even", np.ones(1024, np.complex64))
    store = RunArtifactStore("tf_even")
    frequency = np.arange(-4, 5, dtype=float)
    evidence_id, _ = _time_frequency(store, source_ref, frequency, np.ones((9, 2)))
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        mask=((0, 0), (3, 1), (4, 1), (5, 1)),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    result = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="tf_region_width",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert result.data["bandwidth_hz"] == pytest.approx(2.0)
    assert result.data["lower_edge_hz"] == pytest.approx(-3.0)
    assert result.data["upper_edge_hz"] == pytest.approx(-1.0)


def test_tf_track_width_uses_track_membership_without_rectangular_fill(register_iq):
    repository, source_ref = register_iq("tf_track_width", np.ones(1024, np.complex64))
    store = RunArtifactStore("tf_track_width")
    frequency = np.arange(-4, 5, dtype=float)
    evidence_id, _ = _time_frequency(store, source_ref, frequency, np.ones((9, 3)))
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        method="tf_tracks",
        mask=(),
        track_frames=(0, 1, 2),
        track_bins=(2, 3, 4),
        bounds=(-2.5, 0.5),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    result = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="tf_region_width",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.data["bandwidth_hz"] == pytest.approx(1.0)
    assert result.data["method_details"]["valid_frame_count"] == 3


@pytest.mark.parametrize(
    "candidate_change",
    [
        {"mask_indices": ((99, 0),)},
        {"track_frame_indices": (0, 1), "track_frequency_bin_indices": (2,)},
        {"track_frame_indices": (0,), "track_frequency_bin_indices": (99,)},
        {"frequency_range": FrequencyRange(start_hz=-2.0, stop_hz=2.0, reference="capture_center")},
    ],
)
def test_candidate_rejects_every_invalid_membership_even_when_method_does_not_use_it(register_iq, candidate_change):
    repository, source_ref = register_iq("tf_candidate_invalid", np.ones(1024, np.complex64))
    store = RunArtifactStore("tf_candidate_invalid")
    frequency = np.arange(-4, 5, dtype=float)
    evidence_id, _ = _time_frequency(store, source_ref, frequency, np.ones((9, 3)))
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        mask=((2, 0),),
        track_frames=(0,),
        track_bins=(2,),
        bounds=(-2.5, -1.5),
    )
    candidate = replace(candidate, **candidate_change)
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    with pytest.raises(EstimationBindingError):
        estimate_bandwidth(
            EstimateBandwidthArgs(
                source_ref=source_ref,
                method="tf_region_width",
                time_frequency_id=evidence_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_spectral_candidate_rejects_tf_membership_even_when_unused(register_iq):
    repository, source_ref = register_iq("spectral_membership", np.ones(1024, np.complex64))
    store = RunArtifactStore("spectral_membership")
    frequency = np.arange(-4, 5, dtype=float)
    spectrum_id = _spectrum(store, source_ref, frequency, np.ones(9))
    candidate = _candidate_for(
        source_ref,
        spectrum_id,
        "spectrum",
        "spectral",
        mask=((0, 0),),
    )
    candidate_set_id = _candidate(
        store, source_ref, spectrum_id, "spectrum", "spectral", candidate
    )

    with pytest.raises(EstimationBindingError):
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="spectral_centroid",
                spectrum_id=spectrum_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_tf_candidate_membership_outside_candidate_frequency_range_is_rejected(register_iq):
    repository, source_ref = register_iq("tf_outside_scope", np.ones(1024, np.complex64))
    store = RunArtifactStore("tf_outside_scope")
    frequency = np.arange(-4, 5, dtype=float)
    evidence_id, _ = _time_frequency(store, source_ref, frequency, np.ones((9, 2)))
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        mask=((0, 0),),
        bounds=(-1.5, 1.5),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    with pytest.raises(EstimationBindingError):
        estimate_bandwidth(
            EstimateBandwidthArgs(
                source_ref=source_ref,
                method="tf_region_width",
                time_frequency_id=evidence_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


@pytest.mark.parametrize(
    ("frequency", "power", "centers"),
    [
        (np.arange(4), np.ones(4), np.array([50])),
        (np.arange(4), np.ones((3, 2)), np.array([50, 150])),
        (np.array([0.0, 0.0, 1.0, 2.0]), np.ones((4, 2)), np.array([50, 150])),
        (np.arange(4), np.array([[1.0, np.nan]] * 4), np.array([50, 150])),
        (np.arange(4), np.ones((4, 2)), np.array([[50, 150]])),
        (np.arange(4), np.ones((4, 2)), np.array([50])),
    ],
)
def test_every_tf_consumer_rejects_malformed_block_before_shape_indexing(register_iq, frequency, power, centers):
    repository, source_ref = register_iq("tf_malformed", np.ones(1024, np.complex64))
    store = RunArtifactStore("tf_malformed")
    evidence_id, _ = _time_frequency(store, source_ref, frequency, power, centers=centers)

    with pytest.raises(MalformedEstimationEvidence):
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="tf_ridge_fit",
                time_frequency_id=evidence_id,
            ),
            repository=repository,
            artifact_store=store,
        )


def test_tf_ridge_selects_one_strongest_mask_bin_per_occupied_frame(register_iq):
    repository, source_ref = register_iq(
        "tf_mask_ridge",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_mask_ridge")
    frequency = np.arange(-2, 3, dtype=float)
    power = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [9.0, 0.0],
            [9.0, 8.0],
            [0.0, 8.0],
        ]
    )
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        frequency,
        power,
        centers=np.array([0.0, 256.0]),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        bounds=(-2.5, 2.5),
        mask=((0, 0), (2, 0), (3, 0), (1, 1), (3, 1), (4, 1)),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    ridge = estimate_frequency(
        EstimateFrequencyArgs(
            source_ref=source_ref,
            method="tf_ridge_fit",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )
    width = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="tf_region_width",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert ridge.data["method_details"]["chirp_rate_hz_per_s"] == pytest.approx(4.0)
    assert ridge.data["method_details"]["rms_residual_hz"] == pytest.approx(0.0)
    assert ridge.data["method_details"]["fitted_point_count"] == 2
    assert [
        point["frequency_hz"] for point in ridge.data["trajectory_summary"]
    ] == [0.0, 1.0]
    assert width.data["bandwidth_hz"] == pytest.approx(4.0)


def test_tf_tracks_ridge_uses_exact_track_over_stronger_superset_mask(register_iq):
    repository, source_ref = register_iq(
        "tf_exact_track",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_exact_track")
    frequency = np.arange(-2, 3, dtype=float)
    power = np.array(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [0.0, 0.0],
            [0.0, 9.0],
            [9.0, 0.0],
        ]
    )
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        frequency,
        power,
        centers=np.array([0.0, 256.0]),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        method="tf_tracks",
        bounds=(-2.5, 2.5),
        mask=((0, 0), (4, 0), (1, 1), (3, 1)),
        track_frames=(0, 1),
        track_bins=(0, 1),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    ridge = estimate_frequency(
        EstimateFrequencyArgs(
            source_ref=source_ref,
            method="tf_ridge_fit",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )
    width = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="tf_region_width",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert [
        point["frequency_hz"] for point in ridge.data["trajectory_summary"]
    ] == [-2.0, -1.0]
    assert ridge.data["method_details"]["chirp_rate_hz_per_s"] == pytest.approx(4.0)
    assert ridge.data["method_details"]["rms_residual_hz"] == pytest.approx(0.0)
    assert ridge.data["method_details"]["fitted_point_count"] == 2
    assert width.data["bandwidth_hz"] == pytest.approx(4.0)


def test_region_ridge_uses_strongest_mask_bins_when_track_is_also_present(register_iq):
    repository, source_ref = register_iq(
        "tf_region_with_track",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_region_with_track")
    frequency = np.arange(-2, 3, dtype=float)
    power = np.array(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [0.0, 0.0],
            [0.0, 9.0],
            [9.0, 0.0],
        ]
    )
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        frequency,
        power,
        centers=np.array([0.0, 256.0]),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        method="tf_connected_regions",
        bounds=(-2.5, 2.5),
        mask=((0, 0), (4, 0), (1, 1), (3, 1)),
        track_frames=(0, 1),
        track_bins=(0, 1),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    result = estimate_frequency(
        EstimateFrequencyArgs(
            source_ref=source_ref,
            method="tf_ridge_fit",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert [
        point["frequency_hz"] for point in result.data["trajectory_summary"]
    ] == [2.0, 1.0]
    assert result.data["method_details"]["chirp_rate_hz_per_s"] == pytest.approx(-4.0)


def test_tf_tracks_preserves_exact_track_on_power_ties(register_iq):
    repository, source_ref = register_iq(
        "tf_track_ties",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_track_ties")
    frequency = np.arange(-2, 3, dtype=float)
    power = np.array(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [5.0, 0.0],
            [0.0, 5.0],
            [0.0, 0.0],
        ]
    )
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        frequency,
        power,
        centers=np.array([0.0, 256.0]),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        method="tf_tracks",
        bounds=(-2.5, 2.5),
        mask=((0, 0), (2, 0), (1, 1), (3, 1)),
        track_frames=(0, 1),
        track_bins=(2, 3),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    result = estimate_frequency(
        EstimateFrequencyArgs(
            source_ref=source_ref,
            method="tf_ridge_fit",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert [
        point["frequency_hz"] for point in result.data["trajectory_summary"]
    ] == [0.0, 1.0]
    assert result.data["method_details"]["chirp_rate_hz_per_s"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("track_frames", "track_bins"),
    [
        ((), ()),
        ((0, 0), (0, 1)),
        ((1, 0), (1, 0)),
        ((0, 1), (0,)),
        ((0, 1), (0, 99)),
    ],
)
def test_tf_tracks_rejects_missing_tied_unordered_or_malformed_track(
    register_iq,
    track_frames,
    track_bins,
):
    repository, source_ref = register_iq(
        "tf_bad_track",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_bad_track")
    frequency = np.arange(-2, 3, dtype=float)
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        frequency,
        np.ones((5, 2)),
        centers=np.array([0.0, 256.0]),
    )
    mask = tuple(
        (frequency_index, frame_index)
        for frame_index in range(2)
        for frequency_index in range(5)
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        method="tf_tracks",
        bounds=(-2.5, 2.5),
        mask=mask,
        track_frames=track_frames,
        track_bins=track_bins,
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    with pytest.raises(EstimationBindingError):
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="tf_ridge_fit",
                time_frequency_id=evidence_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_candidate_set_method_must_match_candidate_membership_method(register_iq):
    repository, source_ref = register_iq(
        "tf_method_binding",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_method_binding")
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        np.arange(-2, 3, dtype=float),
        np.ones((5, 2)),
        centers=np.array([0.0, 256.0]),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        method="tf_tracks",
        bounds=(-2.5, 2.5),
        mask=((0, 0), (1, 1)),
        track_frames=(0, 1),
        track_bins=(0, 1),
    )
    payload = CandidateSetPayload(
        source_ref,
        evidence_id,
        "time_frequency",
        "tf_connected_regions",
        "time_frequency",
        (candidate,),
    )
    candidate_set_id = store.put(
        "candidate_set",
        source_ref,
        payload,
        {"candidate_count": 1},
        selection=SignalSelection(source_ref=source_ref),
    )

    with pytest.raises(EstimationBindingError):
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="tf_ridge_fit",
                time_frequency_id=evidence_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_tf_region_ridge_rejects_missing_mask_instead_of_using_track(register_iq):
    repository, source_ref = register_iq(
        "tf_region_missing_mask",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_region_missing_mask")
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        np.arange(-2, 3, dtype=float),
        np.ones((5, 2)),
        centers=np.array([0.0, 256.0]),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        method="tf_connected_regions",
        bounds=(-2.5, 2.5),
        mask=(),
        track_frames=(0, 1),
        track_bins=(0, 1),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    with pytest.raises(EstimationBindingError):
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="tf_ridge_fit",
                time_frequency_id=evidence_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_spectral_bandwidth_methods_warn_near_observation_support_and_report_interpolation(register_iq):
    repository, source_ref = register_iq(
        "bandwidth_confidence",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("bandwidth_confidence")
    spectrum_id = _spectrum(
        store,
        source_ref,
        np.arange(-4, 5, dtype=float),
        np.array([0.1, 0.6, 1.0, 0.6, 0.2, 0.1, 0.1, 0.1, 0.1]),
    )
    requests = (
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="occupied_power",
            spectrum_id=spectrum_id,
            occupied_fraction=0.5,
        ),
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="threshold_edges",
            spectrum_id=spectrum_id,
            noise_quantile=0.2,
            threshold_db_above_noise=3.010299956639812,
        ),
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="peak_drop",
            spectrum_id=spectrum_id,
            drop_db=3.0,
        ),
    )

    for request in requests:
        result = estimate_bandwidth(
            request,
            repository=repository,
            artifact_store=store,
        )
        assert result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL}
        assert "edge_interpolation" in result.quality
        assert any("low confidence" in warning.lower() for warning in result.warnings)


def test_tf_bandwidth_warns_for_truncated_support_and_reports_width_dispersion(register_iq):
    repository, source_ref = register_iq(
        "tf_width_confidence",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_width_confidence")
    frequency = np.arange(-4, 5, dtype=float)
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        frequency,
        np.ones((9, 2)),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        bounds=(-4.5, -1.5),
        mask=((0, 0), (1, 0), (0, 1), (1, 1), (2, 1)),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    result = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="tf_region_width",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.data["edge_diagnostics"]["lower_observation_truncated"] is True
    assert "width_dispersion_hz" in result.quality
    assert any("low confidence" in warning.lower() for warning in result.warnings)


def test_large_finite_spectrum_power_is_stable_for_all_estimators(register_iq):
    repository, source_ref = register_iq(
        "finite_extreme_power",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("finite_extreme_power")
    frequency = np.arange(-3, 4, dtype=float)
    power = np.array([1.0, 2.0, 6.0, 10.0, 6.0, 2.0, 1.0]) * 1.0e307
    spectrum_id = _spectrum(store, source_ref, frequency, power)

    centroid = estimate_frequency(
        EstimateFrequencyArgs(
            source_ref=source_ref,
            method="spectral_centroid",
            spectrum_id=spectrum_id,
            noise_quantile=0.2,
        ),
        repository=repository,
        artifact_store=store,
    )
    assert centroid.status is ToolStatus.SUCCESS
    assert centroid.data["baseband_frequency_hz"] == pytest.approx(0.0)
    assert np.isfinite(centroid.data["frequency_uncertainty_hz"])

    bandwidth_requests = (
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="occupied_power",
            spectrum_id=spectrum_id,
            occupied_fraction=0.5,
        ),
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="threshold_edges",
            spectrum_id=spectrum_id,
            threshold_db_above_noise=3.0,
            noise_quantile=0.2,
        ),
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="peak_drop",
            spectrum_id=spectrum_id,
            drop_db=3.0,
        ),
    )
    for request in bandwidth_requests:
        result = estimate_bandwidth(
            request,
            repository=repository,
            artifact_store=store,
        )
        assert result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL}
        assert np.isfinite(result.data["bandwidth_hz"])


def test_finite_extreme_frequency_axis_raises_typed_without_runtime_warning(register_iq):
    repository, source_ref = register_iq(
        "finite_extreme_axis",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("finite_extreme_axis")
    spectrum_id = _spectrum(
        store,
        source_ref,
        np.array([-1.0e308, 0.0, 1.0e308]),
        np.ones(3),
    )

    with pytest.raises(MalformedEstimationEvidence):
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="spectral_centroid",
                spectrum_id=spectrum_id,
            ),
            repository=repository,
            artifact_store=store,
        )


def test_tf_width_reports_individual_internal_hole_runs(register_iq):
    repository, source_ref = register_iq(
        "tf_hole_runs",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("tf_hole_runs")
    frequency = np.arange(-4, 5, dtype=float)
    evidence_id, _ = _time_frequency(
        store,
        source_ref,
        frequency,
        np.ones((9, 1)),
    )
    candidate = _candidate_for(
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        mask=((0, 0), (2, 0), (4, 0)),
    )
    candidate_set_id = _candidate(
        store,
        source_ref,
        evidence_id,
        "time_frequency",
        "time_frequency",
        candidate,
    )

    result = estimate_bandwidth(
        EstimateBandwidthArgs(
            source_ref=source_ref,
            method="tf_region_width",
            time_frequency_id=evidence_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert result.quality["frames_with_internal_holes"] == 1
    assert result.quality["internal_hole_run_count"] == 2
    assert result.quality["maximum_internal_hole_bins"] == 1


def test_all_frequency_successes_share_concise_reference_semantics(register_iq):
    repository, source_ref = register_iq(
        "frequency_semantics",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("frequency_semantics")
    spectrum_id = _spectrum(
        store,
        source_ref,
        np.arange(-3, 4, dtype=float),
        np.array([1.0, 2.0, 8.0, 16.0, 8.0, 2.0, 1.0]),
    )
    tf_id, _ = _time_frequency(
        store,
        source_ref,
        np.arange(-2, 3, dtype=float),
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 0.0],
            ]
        ),
        centers=np.array([0.0, 256.0]),
    )
    results = (
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="quadratic_fft_peak",
                spectrum_id=spectrum_id,
            ),
            repository=repository,
            artifact_store=store,
        ),
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="spectral_centroid",
                spectrum_id=spectrum_id,
            ),
            repository=repository,
            artifact_store=store,
        ),
        estimate_frequency(
            EstimateFrequencyArgs(
                source_ref=source_ref,
                method="tf_ridge_fit",
                time_frequency_id=tf_id,
            ),
            repository=repository,
            artifact_store=store,
        ),
    )
    expected_keys = {
        "method",
        "scope",
        "baseband_frequency_hz",
        "frequency_offset_hz",
        "frequency_uncertainty_hz",
        "absolute_rf_hz",
    }

    for result in results:
        assert result.status is ToolStatus.SUCCESS
        assert set(result.semantics) == expected_keys
        assert len(result.semantics) <= 16
        assert "method" in result.semantics["method"].lower()
        assert "scope" in result.semantics["scope"].lower()
        assert "uncertainty" in result.semantics["frequency_uncertainty_hz"].lower()
        assert "baseband" in result.semantics["baseband_frequency_hz"].lower()
        assert "capture center" in result.semantics["absolute_rf_hz"].lower()
        assert "null" in result.semantics["absolute_rf_hz"].lower()


def test_pulse_statistics_preserve_carried_boundary_bit(register_iq):
    repository, source_ref = register_iq("pulse", np.ones(1024, np.complex64))
    result = estimate_pulse_parameters(EstimatePulseParametersArgs(source_ref=source_ref, activity_ranges=[{"start": 0, "stop": 32, "peak_to_noise_db": 11}, {"start": 128, "stop": 160}, {"start": 256, "stop": 288}, {"start": 384, "stop": 416}]), repository=repository)
    assert result.status is ToolStatus.SUCCESS
    assert result.data["pulse_count"] == 4
    assert result.data["width"]["median_samples"] == 32
    assert result.data["pri"]["median_samples"] == 128
    assert result.data["prf_hz"] == 8
    assert result.data["duty_cycle"] == pytest.approx(128 / 1024)
    assert result.data["width"]["median_seconds"] == pytest.approx(32 / 1024)
    assert result.data["width"]["minimum_seconds"] == pytest.approx(32 / 1024)
    assert result.data["width"]["maximum_seconds"] == pytest.approx(32 / 1024)
    assert result.data["pri"]["median_seconds"] == pytest.approx(128 / 1024)
    one = estimate_pulse_parameters(EstimatePulseParametersArgs(source_ref=source_ref, activity_ranges=[{"start": 80, "stop": 112, "boundary_truncated": True}]), repository=repository)
    assert one.data["pri"]["status"] == "not_estimated"


def test_pulse_malformed_order_capture_boundary_and_safe_serialization(register_iq):
    repository, source_ref=register_iq("pbad",np.ones(1024,np.complex64))
    with pytest.raises(ValueError): EstimatePulseParametersArgs(source_ref=source_ref,activity_ranges=[{"start":5,"stop":6,"unexpected":1}])
    with pytest.raises(EstimationBindingError): estimate_pulse_parameters(EstimatePulseParametersArgs(source_ref=source_ref,activity_ranges=[{"start":10,"stop":20},{"start":15,"stop":25}]),repository=repository)
    result=estimate_pulse_parameters(EstimatePulseParametersArgs(source_ref=source_ref,activity_ranges=[{"start":10,"stop":20}]),repository=repository)
    encoded=result.model_dump_json()
    assert "sample_id" not in encoded and "Traceback" not in encoded and len(result.data) <= 14


# Task 7 contracts are intentionally kept in this estimator module: the two
# public tools return bounded scalar evidence, never a classifier label.


def _task7_spectrum(
    store,
    source_ref,
    frequency,
    power,
    *,
    sample_rate_hz=1024.0,
    sample_range=(0, 1024),
    channel_indices=(0,),
    scaling="power",
):
    frequency_array = np.asarray(frequency, dtype=np.float64)
    power_array = np.asarray(power, dtype=np.float64)
    block = SpectrumBlock(
        frequency_hz=frequency_array,
        power_linear=power_array,
        representation=power_array,
        sample_range=sample_range,
        channel_indices=channel_indices,
    )
    payload = SpectrumPayload(
        frequency_hz=frequency_array,
        power_linear=power_array,
        representation=power_array,
        blocks=(block,),
        segment_ranges=(sample_range,),
        method="periodogram",
        scaling=scaling,
        sample_rate_hz=sample_rate_hz,
        frequency_resolution_hz=float(np.min(np.diff(frequency_array))),
    )
    artifact_id = store.put(
        "spectrum",
        source_ref,
        payload,
        {"method": "periodogram"},
        selection=SignalSelection(source_ref=source_ref),
    )
    return artifact_id, block


def _task7_candidate(
    store,
    source_ref,
    artifact_id,
    *,
    artifact_type="spectrum",
    domain="spectral",
    method="spectral_regions",
    sample_range=(0, 1024),
    channel_indices=(0,),
    bounds=(-32.0, 32.0),
    block_index=0,
    mask=(),
    track_frames=(),
    track_bins=(),
):
    candidate = EmissionCandidate(
        candidate_id="emission_0",
        source_ref=source_ref,
        source_artifact_id=artifact_id,
        source_artifact_type=artifact_type,
        method=method,
        domain=domain,
        source_block_index=block_index,
        sample_range=sample_range,
        channel_indices=channel_indices,
        frequency_range=FrequencyRange(
            start_hz=bounds[0], stop_hz=bounds[1], reference="baseband"
        ),
        peak_frequency_hz=(bounds[0] + bounds[1]) / 2.0,
        peak_power_linear=1.0,
        relative_power_db=12.0,
        confidence=1.0,
        lower_truncated=False,
        upper_truncated=False,
        mask_indices=mask,
        track_frame_indices=track_frames,
        track_frequency_bin_indices=track_bins,
    )
    payload = CandidateSetPayload(
        source_ref=source_ref,
        source_artifact_id=artifact_id,
        source_artifact_type=artifact_type,
        method=method,
        domain=domain,
        candidates=(candidate,),
    )
    candidate_set_id = store.put(
        "candidate_set",
        source_ref,
        payload,
        {"candidate_count": 1},
        selection=SignalSelection(source_ref=source_ref),
    )
    return candidate_set_id, candidate


def _assert_task7_public_budget(result):
    def visit(value, depth=0):
        assert depth <= 4
        if isinstance(value, dict):
            assert len(value) <= 24
            for nested in value.values():
                visit(nested, depth + 1)
        elif isinstance(value, list):
            assert len(value) <= 16
            for nested in value:
                visit(nested, depth + 1)
        elif isinstance(value, float):
            assert math.isfinite(value)

    for visible in (
        result.data,
        result.semantics,
        result.quality,
        result.provenance,
    ):
        visit(visible)
    visit(result.warnings)
    encoded = result.model_dump_json().lower()
    for forbidden in (
        "sample_id",
        "ground_truth",
        "traceback",
        "modulation\"",
        "waveform_family",
        "classification",
        "is_blocking",
        "suspected",
        "dbm",
        "dbfs",
    ):
        assert forbidden not in encoded


def test_modulation_and_power_argument_schemas_are_exact_and_strict():
    feature_properties = set(ExtractModulationFeaturesArgs.model_json_schema()["properties"])
    assert feature_properties == {
        "source_ref",
        "sample_ranges",
        "range_mode",
        "channel_indices",
        "channel_mode",
        "frequency_range",
        "candidate_set_id",
        "candidate_id",
        "min_feature_samples",
        "max_lag_samples",
    }
    power_properties = set(MeasurePowerRelationsArgs.model_json_schema()["properties"])
    assert power_properties == {
        "source_ref",
        "spectrum_id",
        "candidate_set_id",
        "candidate_id",
        "noise_quantile",
    }
    assert "repository" not in feature_properties | power_properties
    assert "artifact_store" not in feature_properties | power_properties

    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(source_ref="s", candidate_set_id="c")
    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(
            source_ref="s",
            frequency_range=FrequencyRange(
                start_hz=-1.0, stop_hz=1.0, reference="baseband"
            ),
        )
    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(
            source_ref="s",
            sample_ranges=[{"start": 0, "stop": 64}, {"start": 80, "stop": 144}],
        )
    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(source_ref="s", range_mode="aggregate")
    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(
            source_ref="s",
            candidate_set_id="set",
            candidate_id="candidate",
            sample_ranges=[{"start": 0, "stop": 128}],
        )
    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(
            source_ref="s", min_feature_samples=64, max_lag_samples=64
        )
    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(source_ref="s", channel_indices=list(range(17)))
    with pytest.raises(ValueError):
        ExtractModulationFeaturesArgs(source_ref="s", unexpected=True)
    with pytest.raises(ValueError):
        MeasurePowerRelationsArgs(
            source_ref="s",
            spectrum_id="sp",
            candidate_set_id="cs",
            candidate_id="c",
            noise_quantile=0.0,
        )
    with pytest.raises(ValueError):
        MeasurePowerRelationsArgs(
            source_ref="s",
            spectrum_id="sp",
            candidate_set_id="cs",
            candidate_id="c",
            integration_scope="all",
        )


def test_modulation_core_formulas_tone_and_scale_invariance(register_iq):
    samples = np.arange(512, dtype=np.float64)
    amplitude = 1.0 + 0.25 * np.cos(2.0 * np.pi * samples / 41.0)
    phase = 2.0 * np.pi * (0.125 * samples + 2.0e-5 * samples**2)
    iq = amplitude * np.exp(1j * phase)
    repository, source_ref = register_iq("modulation_formula", iq)
    result = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=source_ref, max_lag_samples=80),
        repository=repository,
        artifact_store=RunArtifactStore("modulation_formula"),
    )
    assert result.status is ToolStatus.SUCCESS
    unit = result.data["channel_results"][0]
    expected_median = float(np.median(amplitude.astype(np.float32)))
    expected_spread = 1.4826 * float(
        np.median(np.abs(amplitude.astype(np.float32) - expected_median))
    )
    assert unit["envelope"]["mean_amplitude"] == pytest.approx(np.mean(amplitude), rel=2e-6)
    assert unit["envelope"]["median_amplitude"] == pytest.approx(expected_median, rel=2e-6)
    assert unit["envelope"]["robust_spread_amplitude"] == pytest.approx(expected_spread, rel=2e-6)
    assert unit["envelope"]["constant_envelope_score"] == pytest.approx(
        1.0 / (1.0 + expected_spread / expected_median), rel=2e-6
    )
    assert unit["envelope"]["papr_db"] == pytest.approx(
        10.0 * np.log10(np.max(amplitude**2) / np.mean(amplitude**2)), rel=2e-6
    )
    assert unit["phase_increment"]["status"] == "success"
    assert unit["instantaneous_frequency"]["status"] == "success"
    assert unit["cumulants"]["status"] == "success"
    assert unit["spectral_shape"]["status"] == "success"
    assert unit["lag_correlation_evidence"]["status"] == "success"

    scaled_repository, scaled_ref = register_iq("modulation_scaled", 7.5 * iq)
    scaled = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=scaled_ref, max_lag_samples=80),
        repository=scaled_repository,
        artifact_store=RunArtifactStore("modulation_scaled"),
    ).data["channel_results"][0]
    for field in ("envelope_cv_robust", "constant_envelope_score", "papr_db"):
        assert scaled["envelope"][field] == pytest.approx(unit["envelope"][field], rel=2e-6)
    for field in (
        "c20_real",
        "c20_imag",
        "c20_magnitude",
        "c21",
        "c40_real",
        "c40_imag",
        "c40_magnitude",
        "c42",
    ):
        assert scaled["cumulants"][field] == pytest.approx(unit["cumulants"][field], abs=2e-6)
    assert scaled["spectral_shape"]["spectral_flatness"] == pytest.approx(
        unit["spectral_shape"]["spectral_flatness"], rel=2e-6
    )

    tone_repository, tone_ref = register_iq(
        "modulation_tone", np.exp(2j * np.pi * 0.125 * samples)
    )
    tone = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=tone_ref),
        repository=tone_repository,
        artifact_store=RunArtifactStore("modulation_tone"),
    )
    assert tone.data["channel_results"][0]["envelope"]["papr_db"] == pytest.approx(
        0.0, abs=0.01
    )
    _assert_task7_public_budget(result)
    _assert_task7_public_budget(tone)


def test_modulation_channel_modes_preserve_noncoherent_power(register_iq):
    samples = np.arange(256, dtype=np.float64)
    tone = np.exp(2j * np.pi * 0.125 * samples)
    iq = np.vstack((tone, -tone))
    repository, source_ref = register_iq("modulation_antiphase", iq)
    store = RunArtifactStore("modulation_antiphase")

    single = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=source_ref, channel_mode="single"),
        repository=repository,
        artifact_store=store,
    )
    per_channel = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=source_ref, channel_mode="per_channel"),
        repository=repository,
        artifact_store=store,
    )
    mean_power = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=source_ref, channel_mode="mean_power"),
        repository=repository,
        artifact_store=store,
    )
    assert [item["channel_index"] for item in per_channel.data["channel_results"]] == [0, 1]
    assert single.data["channel_results"][0]["channel_index"] == 0
    assert single.data["channel_results"][0]["envelope"]["mean_amplitude"] == pytest.approx(1.0)
    mean_unit = mean_power.data["channel_results"][0]
    assert mean_power.status is ToolStatus.PARTIAL
    assert mean_unit["channel_indices"] == [0, 1]
    assert mean_unit["envelope"]["mean_amplitude"] == pytest.approx(1.0)
    assert mean_unit["envelope"]["papr_db"] == pytest.approx(0.0, abs=0.01)
    for feature in (
        "phase_increment",
        "instantaneous_frequency",
        "cumulants",
        "spectral_shape",
        "lag_correlation_evidence",
    ):
        assert mean_unit[feature]["status"] == "not_applicable"
        assert mean_unit[feature]["reason"] == "NONCOHERENT_POWER_HAS_NO_COMPLEX_PHASE"
    _assert_task7_public_budget(mean_power)


def test_modulation_zero_energy_short_input_and_resolved_channel_limit(register_iq):
    zero_repository, zero_ref = register_iq("modulation_zero", np.zeros(256, np.complex64))
    zero = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=zero_ref),
        repository=zero_repository,
        artifact_store=RunArtifactStore("modulation_zero"),
    )
    assert zero.status is ToolStatus.NOT_APPLICABLE
    assert zero.error.code == "ZERO_ENERGY"

    short_repository, short_ref = register_iq("modulation_short", np.ones(127, np.complex64))
    with pytest.raises(InsufficientFeatureSamples):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(source_ref=short_ref),
            repository=short_repository,
            artifact_store=RunArtifactStore("modulation_short"),
        )

    many_repository, many_ref = register_iq(
        "modulation_many_channels", np.ones((17, 256), np.complex64)
    )
    with pytest.raises(InvalidModulationFeatureParameters):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(source_ref=many_ref, channel_mode="per_channel"),
            repository=many_repository,
            artifact_store=RunArtifactStore("modulation_many_channels"),
        )


def test_modulation_spectral_comparison_and_lag_evidence(register_iq):
    rng = np.random.default_rng(440)
    noise = rng.standard_normal(512) + 1j * rng.standard_normal(512)
    samples = np.arange(512, dtype=np.float64)
    tone = np.exp(2j * np.pi * 0.125 * samples)
    noise_repository, noise_ref = register_iq("modulation_noise", noise)
    tone_repository, tone_ref = register_iq("modulation_bin_tone", tone)
    noise_result = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=noise_ref),
        repository=noise_repository,
        artifact_store=RunArtifactStore("modulation_noise"),
    )
    tone_result = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=tone_ref),
        repository=tone_repository,
        artifact_store=RunArtifactStore("modulation_bin_tone"),
    )
    noise_shape = noise_result.data["channel_results"][0]["spectral_shape"]
    tone_shape = tone_result.data["channel_results"][0]["spectral_shape"]
    assert noise_shape["spectral_flatness"] > tone_shape["spectral_flatness"]
    assert tone_shape["strongest_peak_level_db"] > noise_shape["strongest_peak_level_db"]
    assert "peaks" not in tone_shape

    repeated_block = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    repeated = np.tile(repeated_block, 8)
    control_phase = 2.0 * np.pi * (0.03 * np.arange(256) + 0.0007 * np.arange(256) ** 2)
    control = np.exp(1j * control_phase)
    repeated_repository, repeated_ref = register_iq("modulation_repeated", repeated)
    control_repository, control_ref = register_iq("modulation_control", control)
    repeated_lag = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=repeated_ref, max_lag_samples=64),
        repository=repeated_repository,
        artifact_store=RunArtifactStore("modulation_repeated"),
    ).data["channel_results"][0]["lag_correlation_evidence"]
    control_lag = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=control_ref, max_lag_samples=64),
        repository=control_repository,
        artifact_store=RunArtifactStore("modulation_control"),
    ).data["channel_results"][0]["lag_correlation_evidence"]
    assert repeated_lag["peak_lag_samples"] == 32
    assert repeated_lag["peak_normalized_correlation"] == pytest.approx(1.0, abs=1e-6)
    assert repeated_lag["peak_normalized_correlation"] > control_lag["peak_normalized_correlation"]
    assert "cyclic" not in json.dumps(repeated_lag).lower()
    assert "ofdm" not in json.dumps(repeated_lag).lower()


def test_modulation_candidate_preprocessing_and_exact_filter_minimum(register_iq):
    samples = np.arange(1024, dtype=np.float64)
    iq = np.exp(2j * np.pi * 100.0 * samples / 1024.0)
    repository, source_ref = register_iq("modulation_candidate", iq)
    store = RunArtifactStore("modulation_candidate")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    )
    candidate_set_id, _ = _task7_candidate(
        store, source_ref, spectrum_id, bounds=(80.0, 120.0)
    )
    result = extract_modulation_features(
        ExtractModulationFeaturesArgs(
            source_ref=source_ref,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )
    assert result.data["analysis_mode"] == "candidate"
    assert result.data["sample_range"] == {"start": 0, "stop": 1024}
    preprocessing = result.provenance["preprocessing"]
    assert preprocessing["downconversion"] == "candidate_center_to_zero_hz"
    assert preprocessing["filter_name"] == "candidate_bandpass_v1"
    assert preprocessing["filter_taps"] == 33
    assert preprocessing["kaiser_beta"] == pytest.approx(8.6)
    assert preprocessing["zero_phase"] is True
    assert preprocessing["candidate_center_hz"] == pytest.approx(100.0)
    assert abs(
        result.data["channel_results"][0]["instantaneous_frequency"]["median_hz"]
    ) < 0.05

    for length, should_pass in ((99, False), (100, True)):
        local_repository, local_ref = register_iq(
            f"modulation_filter_{length}", iq[:length]
        )
        local_store = RunArtifactStore(f"modulation_filter_{length}")
        local_spectrum_id, _ = _task7_spectrum(
            local_store,
            local_ref,
            [-512.0, -128.0, 128.0, 511.0],
            [0.0, 1.0, 1.0, 0.0],
            sample_range=(0, length),
        )
        local_candidate_set, _ = _task7_candidate(
            local_store,
            local_ref,
            local_spectrum_id,
            sample_range=(0, length),
            bounds=(80.0, 120.0),
        )
        args = ExtractModulationFeaturesArgs(
            source_ref=local_ref,
            candidate_set_id=local_candidate_set,
            candidate_id="emission_0",
            min_feature_samples=64,
            max_lag_samples=32,
        )
        if should_pass:
            assert extract_modulation_features(
                args, repository=local_repository, artifact_store=local_store
            ).status is ToolStatus.SUCCESS
        else:
            with pytest.raises(InsufficientFeatureSamples):
                extract_modulation_features(
                    args, repository=local_repository, artifact_store=local_store
                )


def test_modulation_tf_candidate_uses_mandatory_subrange_without_tf_morphology(register_iq):
    samples = np.arange(512, dtype=np.float64)
    iq = np.exp(2j * np.pi * 0.05 * samples)
    repository, source_ref = register_iq("modulation_tf_candidate", iq)
    store = RunArtifactStore("modulation_tf_candidate")
    frequency = np.array([-512.0, -128.0, 0.0, 128.0, 511.0])
    power = np.ones((frequency.size, 5), dtype=np.float64)
    starts = np.arange(5, dtype=np.int64) * 64
    centers = starts.astype(np.float64) + 49.5
    block = TimeFrequencyBlock(
        frequency_hz=frequency,
        values=power,
        power_linear=power,
        frame_start_samples=starts,
        frame_center_samples=centers,
        sample_range=(0, 512),
        channel_indices=(0,),
    )
    payload = TimeFrequencyPayload(
        blocks=(block,),
        output="power_spectrogram",
        sample_rate_hz=1024.0,
        frequency_resolution_hz=128.0,
        time_resolution_samples=64,
        window_length=100,
        nfft=8,
    )
    tf_id = store.put(
        "time_frequency",
        source_ref,
        payload,
        {"frame_count": 5},
        selection=SignalSelection(source_ref=source_ref),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        tf_id,
        artifact_type="time_frequency",
        domain="time_frequency",
        method="tf_connected_regions",
        sample_range=(64, 228),
        bounds=(-64.0, 64.0),
        mask=((2, 1), (2, 2)),
    )
    result = extract_modulation_features(
        ExtractModulationFeaturesArgs(
            source_ref=source_ref,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )
    assert result.data["sample_range"] == {"start": 64, "stop": 228}
    encoded = result.model_dump_json().lower()
    assert "morphology" not in encoded
    assert "mask_indices" not in encoded

    candidate_record = store.get(candidate_set_id)
    malformed = replace(candidate_record.payload.candidates[0], sample_range=(64, 229))
    malformed_payload = replace(candidate_record.payload, candidates=(malformed,))
    malformed_id = store.put(
        "candidate_set",
        source_ref,
        malformed_payload,
        {"candidate_count": 1},
        selection=SignalSelection(source_ref=source_ref),
    )
    with pytest.raises(CandidateEvidenceMismatch):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=malformed_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def _power_fixture(register_iq, name, *, scaling="power_density", protected=None):
    protected_channel = protected or ProtectedChannel(
        center_offset_hz=-2.5, nominal_bandwidth_hz=3.0
    )
    repository, source_ref = register_iq(
        name,
        np.ones(1024, np.complex64),
        sample_rate_hz=20.0,
        protected_channel=protected_channel,
    )
    store = RunArtifactStore(name)
    frequency = np.array([-8.0, -6.0, -3.0, -1.0, 0.0, 2.0, 5.0, 8.0])
    edges = np.array([-9.0, -7.0, -4.5, -2.0, -0.5, 1.0, 3.5, 6.5, 9.5])
    density = np.ones(frequency.size, dtype=np.float64)
    target = (-4.0, -1.0)
    candidate_band = (2.0, 6.0)
    for index in range(frequency.size):
        if max(edges[index], target[0]) < min(edges[index + 1], target[1]):
            density[index] = 5.0
        if max(edges[index], candidate_band[0]) < min(edges[index + 1], candidate_band[1]):
            density[index] = 9.0
    power = density if scaling == "power_density" else density * np.diff(edges)
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        frequency,
        power,
        sample_rate_hz=20.0,
        scaling=scaling,
    )
    candidate_set_id, _ = _task7_candidate(
        store, source_ref, spectrum_id, bounds=candidate_band
    )
    return repository, source_ref, store, spectrum_id, candidate_set_id


@pytest.mark.parametrize("scaling", ["power_density", "power"])
def test_power_relations_fractional_bin_integration_and_noise_order(register_iq, scaling):
    repository, source_ref, store, spectrum_id, candidate_set_id = _power_fixture(
        register_iq, f"power_fractional_{scaling}", scaling=scaling
    )
    result = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )
    assert result.status is ToolStatus.SUCCESS
    assert result.data["protected_band"]["observed_power"] == pytest.approx(15.0)
    assert result.data["protected_band"]["noise_power"] == pytest.approx(3.0)
    assert result.data["protected_band"]["excess_power"] == pytest.approx(12.0)
    assert result.data["candidate_band"]["observed_power"] == pytest.approx(36.0)
    assert result.data["candidate_band"]["noise_power"] == pytest.approx(4.0)
    assert result.data["candidate_band"]["excess_power"] == pytest.approx(32.0)
    expected_js = 10.0 * math.log10(32.0 / 12.0)
    assert result.data["ratios"]["j_over_s_db"] == pytest.approx(expected_js, abs=0.25)
    assert result.data["ratios"]["sir_db"] == pytest.approx(-expected_js, abs=1e-9)
    assert result.data["protected_band"]["coverage_fraction"] == pytest.approx(1.0)
    assert result.data["candidate_band"]["coverage_fraction"] == pytest.approx(1.0)
    assert "capture-power" in result.semantics["j_over_s_db"]
    _assert_task7_public_budget(result)


def test_power_relations_overlap_clipping_and_no_noise_statuses(register_iq):
    overlap_repository, overlap_ref, overlap_store, overlap_spectrum, _ = _power_fixture(
        register_iq,
        "power_overlap",
        protected=ProtectedChannel(center_offset_hz=0.0, nominal_bandwidth_hz=4.0),
    )
    overlap_candidate_set, _ = _task7_candidate(
        overlap_store, overlap_ref, overlap_spectrum, bounds=(0.0, 4.0)
    )
    overlap = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=overlap_ref,
            spectrum_id=overlap_spectrum,
            candidate_set_id=overlap_candidate_set,
            candidate_id="emission_0",
        ),
        repository=overlap_repository,
        artifact_store=overlap_store,
    )
    assert overlap.status is ToolStatus.PARTIAL
    assert overlap.data["overlap"]["overlap_width_hz"] == pytest.approx(2.0)
    assert overlap.data["overlap"]["protected_overlap_fraction"] == pytest.approx(0.5)
    assert overlap.data["overlap"]["candidate_overlap_fraction"] == pytest.approx(0.5)
    assert overlap.data["ratios"]["reason"] == "BAND_OVERLAP_UNSEPARATED"
    assert overlap.data["ratios"]["j_over_s_db"] is None
    assert overlap.data["ratios"]["sir_db"] is None

    clipped_repository, clipped_ref, clipped_store, clipped_spectrum, _ = _power_fixture(
        register_iq,
        "power_clipped",
        protected=ProtectedChannel(center_offset_hz=-9.0, nominal_bandwidth_hz=4.0),
    )
    clipped_candidate_set, _ = _task7_candidate(
        clipped_store, clipped_ref, clipped_spectrum, bounds=(2.0, 6.0)
    )
    clipped = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=clipped_ref,
            spectrum_id=clipped_spectrum,
            candidate_set_id=clipped_candidate_set,
            candidate_id="emission_0",
        ),
        repository=clipped_repository,
        artifact_store=clipped_store,
    )
    assert clipped.status is ToolStatus.PARTIAL
    assert clipped.data["protected_band"]["coverage_fraction"] == pytest.approx(0.5)
    assert clipped.data["candidate_band"]["coverage_fraction"] == pytest.approx(1.0)
    assert clipped.data["ratios"]["reason"] == "BAND_SUPPORT_CLIPPED"
    assert clipped.data["ratios"]["j_over_s_db"] is None

    no_noise_repository, no_noise_ref = register_iq(
        "power_no_noise",
        np.ones(1024, np.complex64),
        sample_rate_hz=20.0,
        protected_channel=ProtectedChannel(center_offset_hz=-5.0, nominal_bandwidth_hz=10.0),
    )
    no_noise_store = RunArtifactStore("power_no_noise")
    frequency = np.array([-7.5, -2.5, 2.5, 7.5])
    no_noise_spectrum, _ = _task7_spectrum(
        no_noise_store,
        no_noise_ref,
        frequency,
        np.ones(4),
        sample_rate_hz=20.0,
        scaling="power_density",
    )
    no_noise_candidate_set, _ = _task7_candidate(
        no_noise_store, no_noise_ref, no_noise_spectrum, bounds=(0.0, 10.0)
    )
    no_noise = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=no_noise_ref,
            spectrum_id=no_noise_spectrum,
            candidate_set_id=no_noise_candidate_set,
            candidate_id="emission_0",
        ),
        repository=no_noise_repository,
        artifact_store=no_noise_store,
    )
    assert no_noise.status is ToolStatus.PARTIAL
    assert no_noise.data["noise_evidence"]["reason"] == "NO_DISJOINT_NOISE_BINS"
    assert no_noise.data["ratios"]["j_over_s_db"] is None


def test_power_relations_missing_or_unobserved_protected_band(register_iq):
    repository, source_ref = register_iq(
        "power_no_protected", np.ones(1024, np.complex64), sample_rate_hz=20.0
    )
    store = RunArtifactStore("power_no_protected")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-7.5, -2.5, 2.5, 7.5],
        [1.0, 1.0, 2.0, 2.0],
        sample_rate_hz=20.0,
        scaling="power_density",
    )
    candidate_set_id, _ = _task7_candidate(
        store, source_ref, spectrum_id, bounds=(0.0, 5.0)
    )
    missing = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )
    assert missing.status is ToolStatus.NOT_APPLICABLE
    assert missing.error.code == "PROTECTED_CHANNEL_UNAVAILABLE"

    outside_repository, outside_ref = register_iq(
        "power_outside",
        np.ones(1024, np.complex64),
        sample_rate_hz=20.0,
        protected_channel=ProtectedChannel(center_offset_hz=100.0, nominal_bandwidth_hz=2.0),
    )
    outside_store = RunArtifactStore("power_outside")
    outside_spectrum, _ = _task7_spectrum(
        outside_store,
        outside_ref,
        [-7.5, -2.5, 2.5, 7.5],
        [1.0, 1.0, 2.0, 2.0],
        sample_rate_hz=20.0,
        scaling="power_density",
    )
    outside_candidate_set, _ = _task7_candidate(
        outside_store, outside_ref, outside_spectrum, bounds=(0.0, 5.0)
    )
    outside = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=outside_ref,
            spectrum_id=outside_spectrum,
            candidate_set_id=outside_candidate_set,
            candidate_id="emission_0",
        ),
        repository=outside_repository,
        artifact_store=outside_store,
    )
    assert outside.status is ToolStatus.NOT_APPLICABLE
    assert outside.error.code == "BAND_UNOBSERVED"


def test_power_relations_binding_and_malformed_numeric_fail_typed(register_iq):
    repository, source_ref, store, spectrum_id, candidate_set_id = _power_fixture(
        register_iq, "power_binding"
    )
    candidate_record = store.get(candidate_set_id)
    forged_candidate = replace(
        candidate_record.payload.candidates[0], channel_indices=(1,)
    )
    forged_payload = replace(candidate_record.payload, candidates=(forged_candidate,))
    forged_set = store.put(
        "candidate_set",
        source_ref,
        forged_payload,
        {"candidate_count": 1},
        selection=SignalSelection(source_ref=source_ref),
    )
    with pytest.raises(CandidateEvidenceMismatch):
        measure_power_relations(
            MeasurePowerRelationsArgs(
                source_ref=source_ref,
                spectrum_id=spectrum_id,
                candidate_set_id=forged_set,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )

    malformed_store = RunArtifactStore("power_malformed")
    malformed_block = SpectrumBlock(
        frequency_hz=np.array([-2.0, 0.0, 2.0]),
        power_linear=np.array([1.0, -1.0, 1.0]),
        representation=np.ones(3),
        sample_range=(0, 1024),
        channel_indices=(0,),
    )
    malformed_payload = SpectrumPayload(
        frequency_hz=malformed_block.frequency_hz,
        power_linear=malformed_block.power_linear,
        representation=malformed_block.representation,
        blocks=(malformed_block,),
        segment_ranges=((0, 1024),),
        method="periodogram",
        scaling="power_density",
        sample_rate_hz=20.0,
        frequency_resolution_hz=2.0,
    )
    malformed_spectrum = malformed_store.put(
        "spectrum",
        source_ref,
        malformed_payload,
        {"method": "periodogram"},
        selection=SignalSelection(source_ref=source_ref),
    )
    malformed_candidate_set, _ = _task7_candidate(
        malformed_store, source_ref, malformed_spectrum, bounds=(-1.0, 1.0)
    )
    with pytest.raises(UnusablePowerEvidence):
        measure_power_relations(
            MeasurePowerRelationsArgs(
                source_ref=source_ref,
                spectrum_id=malformed_spectrum,
                candidate_set_id=malformed_candidate_set,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=malformed_store,
        )


def test_modulation_nonfinite_derived_result_is_typed_without_warning(register_iq):
    iq = np.full(256, complex(1.0e38, 1.0e38), dtype=np.complex64)
    repository, source_ref = register_iq("modulation_extreme", iq)
    result = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=source_ref),
        repository=repository,
        artifact_store=RunArtifactStore("modulation_extreme"),
    )
    assert result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL}
    _assert_task7_public_budget(result)

    # A forged finite complex128 capture can still exceed representable public
    # envelope arithmetic; the failure must be typed rather than emit a warning.
    frame = repository._frames[source_ref]
    extreme = np.full((1, 256), complex(1.3e308, 1.3e308), dtype=np.complex128)
    extreme.setflags(write=False)
    object.__setattr__(frame, "iq", extreme)
    with pytest.raises(NonFiniteEstimationResult):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(source_ref=source_ref),
            repository=repository,
            artifact_store=RunArtifactStore("modulation_extreme_forged"),
        )


def test_modulation_legal_complex128_subnormal_iq_normalizes_without_warning(
    tmp_path,
):
    smallest_positive = np.nextafter(0.0, 1.0)
    iq_path = tmp_path / "task7_subnormal_complex128.npy"
    np.save(
        iq_path,
        np.full(
            128,
            complex(smallest_positive, 0.0),
            dtype=np.complex128,
        ),
    )
    manifest = IQSampleManifest(
        sample_id="task7_subnormal_complex128",
        iq_path=str(iq_path),
        iq_format="npy",
        dtype="complex128",
        layout="samples",
        sample_rate_hz=1024.0,
        center_frequency_hz=2_400_000_000.0,
    )
    repository = IQRepository([tmp_path])
    source_ref = repository.register(manifest)

    result = extract_modulation_features(
        ExtractModulationFeaturesArgs(source_ref=source_ref),
        repository=repository,
        artifact_store=RunArtifactStore("task7_subnormal_complex128"),
    )

    assert result.status is ToolStatus.PARTIAL
    envelope = result.data["channel_results"][0]["envelope"]
    assert envelope["status"] == "partial"
    assert envelope["reason"] == "ZERO_MEDIAN_ENVELOPE"
    assert envelope["constant_envelope_score"] is None
    assert envelope["zero_envelope_fraction"] == 1.0
    assert envelope["papr_db"] == pytest.approx(0.0, abs=1.0e-12)
    _assert_task7_public_budget(result)


@pytest.mark.parametrize("consumer", ["modulation", "power"])
@pytest.mark.parametrize(
    "malformation",
    [
        "candidate_tuple_contains_none",
        "candidate_collection_is_none",
        "candidate_channels_are_none",
        "source_sample_range_is_none",
        "source_channels_are_none",
        "candidate_frequency_lower_is_none",
        "source_sample_start_is_bool",
        "source_channel_is_float",
        "candidate_set_artifact_type_is_list",
        "candidate_set_artifact_id_is_list",
        "source_sample_rate_is_none",
        "source_blocks_are_none",
    ],
)
def test_modulation_and_power_candidate_binding_rejects_forged_internal_types(
    register_iq,
    consumer,
    malformation,
):
    repository, source_ref = register_iq(
        f"task7_forged_{consumer}_{malformation}",
        np.ones(1024, np.complex64),
        protected_channel=ProtectedChannel(
            center_offset_hz=-128.0,
            nominal_bandwidth_hz=64.0,
        ),
    )
    store = RunArtifactStore(f"task7_forged_{consumer}_{malformation}")
    frequency = np.array([-512.0, -256.0, 0.0, 256.0, 511.0])
    power_values = np.ones(frequency.size)
    block = SpectrumBlock(
        frequency_hz=frequency,
        power_linear=power_values,
        representation=power_values,
        sample_range=(0, 1024),
        channel_indices=(0,),
    )
    if malformation == "source_sample_range_is_none":
        block = replace(block, sample_range=None)
    elif malformation == "source_channels_are_none":
        block = replace(block, channel_indices=None)
    elif malformation == "source_sample_start_is_bool":
        block = replace(block, sample_range=(False, 1024))
    elif malformation == "source_channel_is_float":
        block = replace(block, channel_indices=(0.0,))
    payload = SpectrumPayload(
        frequency_hz=frequency,
        power_linear=power_values,
        representation=power_values,
        blocks=None if malformation == "source_blocks_are_none" else (block,),
        segment_ranges=((0, 1024),),
        method="periodogram",
        scaling="power_density",
        sample_rate_hz=None if malformation == "source_sample_rate_is_none" else 1024.0,
        frequency_resolution_hz=256.0,
    )
    spectrum_id = store.put(
        "spectrum",
        source_ref,
        payload,
        {"method": "periodogram"},
        selection=SignalSelection(source_ref=source_ref),
    )
    candidate = EmissionCandidate(
        candidate_id="emission_0",
        source_ref=source_ref,
        source_artifact_id=spectrum_id,
        source_artifact_type="spectrum",
        method="spectral_regions",
        domain="spectral",
        source_block_index=0,
        sample_range=(0, 1024),
        channel_indices=(0,),
        frequency_range=FrequencyRange(
            start_hz=-32.0,
            stop_hz=32.0,
            reference="baseband",
        ),
        peak_frequency_hz=0.0,
        peak_power_linear=1.0,
        relative_power_db=12.0,
        confidence=1.0,
        lower_truncated=False,
        upper_truncated=False,
    )
    if malformation == "candidate_channels_are_none":
        candidate = replace(candidate, channel_indices=None)
    elif malformation == "candidate_frequency_lower_is_none":
        malformed_range = FrequencyRange.model_construct(
            start_hz=None,
            stop_hz=32.0,
            reference="baseband",
        )
        candidate = replace(candidate, frequency_range=malformed_range)

    candidates = (candidate,)
    if malformation == "candidate_tuple_contains_none":
        candidates = (None,)
    elif malformation == "candidate_collection_is_none":
        candidates = None
    candidate_payload = CandidateSetPayload(
        source_ref=source_ref,
        source_artifact_id=(
            [] if malformation == "candidate_set_artifact_id_is_list" else spectrum_id
        ),
        source_artifact_type=(
            [] if malformation == "candidate_set_artifact_type_is_list" else "spectrum"
        ),
        method="spectral_regions",
        domain="spectral",
        candidates=candidates,
    )
    candidate_set_id = store.put(
        "candidate_set",
        source_ref,
        candidate_payload,
        {"candidate_count": 1},
        selection=SignalSelection(source_ref=source_ref),
    )

    if malformation in {"source_sample_rate_is_none", "source_blocks_are_none"}:
        expected_error = (
            UnusablePowerEvidence
            if consumer == "power"
            else MalformedEstimationEvidence
        )
    else:
        expected_error = CandidateEvidenceMismatch

    with pytest.raises(expected_error):
        if consumer == "modulation":
            extract_modulation_features(
                ExtractModulationFeaturesArgs(
                    source_ref=source_ref,
                    candidate_set_id=candidate_set_id,
                    candidate_id="emission_0",
                ),
                repository=repository,
                artifact_store=store,
            )
        else:
            measure_power_relations(
                MeasurePowerRelationsArgs(
                    source_ref=source_ref,
                    spectrum_id=spectrum_id,
                    candidate_set_id=candidate_set_id,
                    candidate_id="emission_0",
                ),
                repository=repository,
                artifact_store=store,
            )


def test_modulation_phase_frequency_overflow_is_typed_without_runtime_warning(register_iq):
    samples = np.arange(256)
    alternating = np.where(samples % 2 == 0, 1.0 + 0.0j, -1.0 + 0.0j)
    repository, source_ref = register_iq(
        "task7_phase_overflow",
        alternating,
        sample_rate_hz=1.0e308,
    )
    with pytest.raises(NonFiniteEstimationResult):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(source_ref=source_ref),
            repository=repository,
            artifact_store=RunArtifactStore("task7_phase_overflow"),
        )


def test_modulation_candidate_mixing_overflow_is_typed_without_runtime_warning(register_iq):
    repository, source_ref = register_iq(
        "task7_candidate_mix_overflow",
        np.ones(256, np.complex64),
        sample_rate_hz=1.0e308,
    )
    stored_frame = repository._frames[source_ref]
    extreme = np.full(
        (1, 256),
        complex(1.2e308, 1.2e308),
        dtype=np.complex128,
    )
    extreme.setflags(write=False)
    object.__setattr__(stored_frame, "iq", extreme)
    store = RunArtifactStore("task7_candidate_mix_overflow")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-5.0e307, -2.5e307, 0.0, 2.5e307, 4.9e307],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        sample_rate_hz=1.0e308,
        sample_range=(0, 256),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        sample_range=(0, 256),
        bounds=(2.0e307, 3.0e307),
    )
    with pytest.raises(NonFiniteEstimationResult):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_power_protected_band_endpoint_overflow_is_typed(register_iq):
    repository, source_ref = register_iq(
        "task7_protected_endpoint_overflow",
        np.ones(1024, np.complex64),
        sample_rate_hz=1.0e308,
        protected_channel=ProtectedChannel(
            center_offset_hz=1.7e308,
            nominal_bandwidth_hz=1.7e308,
        ),
    )
    store = RunArtifactStore("task7_protected_endpoint_overflow")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-5.0e307, -2.5e307, 0.0, 2.5e307, 4.9e307],
        np.ones(5),
        sample_rate_hz=1.0e308,
        scaling="power_density",
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(-1.0e307, 1.0e307),
    )
    with pytest.raises(NonFiniteEstimationResult):
        measure_power_relations(
            MeasurePowerRelationsArgs(
                source_ref=source_ref,
                spectrum_id=spectrum_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


@pytest.mark.parametrize(
    "malformation",
    [
        "mask_is_none",
        "mask_is_list",
        "duplicate_mask_coordinate",
        "track_frames_are_none",
        "track_bins_are_none",
        "frame_starts_are_strings",
        "frame_starts_are_fractional",
        "frame_starts_are_bool",
        "frame_starts_not_increasing",
        "frame_mapping_exceeds_block",
        "window_length_is_none",
        "window_length_is_bool",
        "window_length_is_float",
    ],
)
def test_modulation_tf_candidate_rejects_malformed_membership_and_frame_mapping(
    register_iq,
    malformation,
):
    repository, source_ref = register_iq(
        f"task7_tf_malformed_{malformation}",
        np.ones(512, np.complex64),
    )
    store = RunArtifactStore(f"task7_tf_malformed_{malformation}")
    frequency = np.array([-512.0, -128.0, 0.0, 128.0, 511.0])
    power_values = np.ones((frequency.size, 3), dtype=np.float64)
    frame_starts = np.array([0, 64, 128], dtype=np.int64)
    candidate_sample_range = (0, 164)
    window_length = 100
    method = "tf_connected_regions"
    mask = ((2, 0), (2, 1))
    track_frames = ()
    track_bins = ()
    if malformation == "mask_is_none":
        mask = None
    elif malformation == "mask_is_list":
        mask = [(2, 0), (2, 1)]
    elif malformation == "duplicate_mask_coordinate":
        mask = ((2, 0), (2, 0), (2, 1))
    elif malformation == "track_frames_are_none":
        method = "tf_tracks"
        mask = ()
        track_frames = None
        track_bins = (2, 2)
    elif malformation == "track_bins_are_none":
        method = "tf_tracks"
        mask = ()
        track_frames = (0, 1)
        track_bins = None
    elif malformation == "frame_starts_are_strings":
        frame_starts = np.array(["0", "64", "128"])
    elif malformation == "frame_starts_are_fractional":
        frame_starts = np.array([0.5, 64.5, 128.5])
    elif malformation == "frame_starts_are_bool":
        frame_starts = np.array([False, True, True])
    elif malformation == "frame_starts_not_increasing":
        frame_starts = np.array([0, 64, 64], dtype=np.int64)
    elif malformation == "frame_mapping_exceeds_block":
        frame_starts = np.array([0, 64, 600], dtype=np.int64)
        mask = ((2, 1), (2, 2))
        candidate_sample_range = (64, 512)
    elif malformation == "window_length_is_none":
        window_length = None
    elif malformation == "window_length_is_bool":
        window_length = True
        candidate_sample_range = (0, 65)
    elif malformation == "window_length_is_float":
        window_length = 100.0

    block = TimeFrequencyBlock(
        frequency_hz=frequency,
        values=power_values,
        power_linear=power_values,
        frame_start_samples=frame_starts,
        frame_center_samples=np.array([49.5, 113.5, 177.5]),
        sample_range=(0, 512),
        channel_indices=(0,),
    )
    payload = TimeFrequencyPayload(
        blocks=(block,),
        output="power_spectrogram",
        sample_rate_hz=1024.0,
        frequency_resolution_hz=128.0,
        time_resolution_samples=64,
        window_length=window_length,
        nfft=8,
    )
    tf_id = store.put(
        "time_frequency",
        source_ref,
        payload,
        {"frame_count": 3},
        selection=SignalSelection(source_ref=source_ref),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        tf_id,
        artifact_type="time_frequency",
        domain="time_frequency",
        method=method,
        sample_range=candidate_sample_range,
        bounds=(-64.0, 64.0),
        mask=mask,
        track_frames=track_frames,
        track_bins=track_bins,
    )
    with pytest.raises(CandidateEvidenceMismatch):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_power_overlap_remains_primary_when_noise_bins_are_also_unavailable(register_iq):
    repository, source_ref = register_iq(
        "task7_overlap_without_noise",
        np.ones(1024, np.complex64),
        sample_rate_hz=20.0,
        protected_channel=ProtectedChannel(
            center_offset_hz=-2.5,
            nominal_bandwidth_hz=15.0,
        ),
    )
    store = RunArtifactStore("task7_overlap_without_noise")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-7.5, -2.5, 2.5, 7.5],
        [2.0, 3.0, 4.0, 5.0],
        sample_rate_hz=20.0,
        scaling="power_density",
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(-5.0, 10.0),
    )
    result = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )
    assert result.status is ToolStatus.PARTIAL
    assert result.data["noise_evidence"]["reason"] == "NO_DISJOINT_NOISE_BINS"
    assert result.data["ratios"]["status"] == "unresolvable"
    assert result.data["ratios"]["reason"] == "BAND_OVERLAP_UNSEPARATED"
    assert result.data["ratios"]["j_over_s_db"] is None
    assert result.data["ratios"]["sir_db"] is None


def test_modulation_spectral_axis_handles_smallest_positive_sample_rate_without_reciprocal(
):
    sample_rate_hz = np.nextafter(0.0, 1.0)
    samples = np.arange(256, dtype=np.float64)
    tone = np.exp(2j * np.pi * 7.0 * samples / samples.size)

    evidence, spectral_bins = _spectral_shape_evidence(tone, sample_rate_hz)

    assert spectral_bins == 256
    assert evidence["status"] == "success"
    strongest_frequency_hz = evidence["strongest_peak_frequency_hz"]
    assert strongest_frequency_hz is not None
    assert math.isfinite(strongest_frequency_hz)


def test_modulation_lag_seconds_overflow_is_a_typed_numerical_failure():
    sample_rate_hz = np.nextafter(0.0, 1.0)
    samples = np.arange(256, dtype=np.float64)
    tone = np.exp(2j * np.pi * samples / 32.0)

    with pytest.raises(NonFiniteEstimationResult):
        _lag_correlation_evidence(tone, sample_rate_hz, 64)


def test_modulation_tool_smallest_positive_sample_rate_has_no_public_nonfinite_leak(
    register_iq,
):
    sample_rate_hz = np.nextafter(0.0, 1.0)
    samples = np.arange(256, dtype=np.float64)
    tone = np.exp(2j * np.pi * samples / 32.0)
    repository, source_ref = register_iq(
        "task7_smallest_positive_sample_rate",
        tone,
        sample_rate_hz=sample_rate_hz,
    )

    with pytest.raises(NonFiniteEstimationResult):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(source_ref=source_ref),
            repository=repository,
            artifact_store=RunArtifactStore("task7_smallest_positive_sample_rate"),
        )


@pytest.mark.parametrize(
    "forged_scaling",
    [[], {}],
    ids=["list", "mapping"],
)
def test_power_unhashable_spectrum_scaling_is_typed_unusable_evidence(
    register_iq,
    forged_scaling,
):
    repository, source_ref = register_iq(
        f"task7_unhashable_scaling_{type(forged_scaling).__name__}",
        np.ones(1024, np.complex64),
        protected_channel=ProtectedChannel(
            center_offset_hz=-128.0,
            nominal_bandwidth_hz=64.0,
        ),
    )
    store = RunArtifactStore(
        f"task7_unhashable_scaling_{type(forged_scaling).__name__}"
    )
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
        scaling=forged_scaling,
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(-32.0, 32.0),
    )

    with pytest.raises(UnusablePowerEvidence):
        measure_power_relations(
            MeasurePowerRelationsArgs(
                source_ref=source_ref,
                spectrum_id=spectrum_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


@pytest.mark.parametrize("consumer", ["modulation", "power"])
@pytest.mark.parametrize("complex_field", ["frequency_hz", "power_linear"])
def test_task7_spectrum_complex_axis_is_rejected_without_complex_warning(
    register_iq,
    consumer,
    complex_field,
):
    repository, source_ref = register_iq(
        f"task7_complex_spectrum_{consumer}_{complex_field}",
        np.ones(1024, np.complex64),
        protected_channel=ProtectedChannel(
            center_offset_hz=-128.0,
            nominal_bandwidth_hz=64.0,
        ),
    )
    store = RunArtifactStore(
        f"task7_complex_spectrum_{consumer}_{complex_field}"
    )
    real_frequency = np.array([-512.0, -256.0, 0.0, 256.0, 511.0])
    real_power = np.ones(real_frequency.size)
    block_frequency = real_frequency
    block_power = real_power
    if complex_field == "frequency_hz":
        block_frequency = real_frequency.astype(np.complex128) + 1.0j
    else:
        block_power = real_power.astype(np.complex128) + 1.0j
    block = SpectrumBlock(
        frequency_hz=block_frequency,
        power_linear=block_power,
        representation=block_power,
        sample_range=(0, 1024),
        channel_indices=(0,),
    )
    payload = SpectrumPayload(
        frequency_hz=block_frequency,
        power_linear=block_power,
        representation=block_power,
        blocks=(block,),
        segment_ranges=((0, 1024),),
        method="periodogram",
        scaling="power_density",
        sample_rate_hz=1024.0,
        frequency_resolution_hz=256.0,
    )
    spectrum_id = store.put(
        "spectrum",
        source_ref,
        payload,
        {"method": "periodogram"},
        selection=SignalSelection(source_ref=source_ref),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(-32.0, 32.0),
    )

    expected_error = (
        UnusablePowerEvidence if consumer == "power" else MalformedEstimationEvidence
    )
    with pytest.raises(expected_error):
        if consumer == "power":
            measure_power_relations(
                MeasurePowerRelationsArgs(
                    source_ref=source_ref,
                    spectrum_id=spectrum_id,
                    candidate_set_id=candidate_set_id,
                    candidate_id="emission_0",
                ),
                repository=repository,
                artifact_store=store,
            )
        else:
            extract_modulation_features(
                ExtractModulationFeaturesArgs(
                    source_ref=source_ref,
                    candidate_set_id=candidate_set_id,
                    candidate_id="emission_0",
                ),
                repository=repository,
                artifact_store=store,
            )


@pytest.mark.parametrize(
    "complex_field",
    ["frequency_hz", "power_linear", "frame_center_samples"],
)
def test_task7_time_frequency_complex_axis_is_rejected_without_complex_warning(
    register_iq,
    complex_field,
):
    repository, source_ref = register_iq(
        f"task7_complex_tf_{complex_field}",
        np.ones(512, np.complex64),
    )
    store = RunArtifactStore(f"task7_complex_tf_{complex_field}")
    real_frequency = np.array([-512.0, -128.0, 0.0, 128.0, 511.0])
    real_power = np.ones((real_frequency.size, 3), dtype=np.float64)
    frame_starts = np.array([0, 64, 128], dtype=np.int64)
    frame_centers = np.array([49.5, 113.5, 177.5])
    block_frequency = real_frequency
    block_power = real_power
    if complex_field == "frequency_hz":
        block_frequency = real_frequency.astype(np.complex128) + 1.0j
    elif complex_field == "power_linear":
        block_power = real_power.astype(np.complex128) + 1.0j
    else:
        frame_centers = frame_centers.astype(np.complex128) + 1.0j
    block = TimeFrequencyBlock(
        frequency_hz=block_frequency,
        values=np.asarray(block_power, dtype=np.complex128),
        power_linear=block_power,
        frame_start_samples=frame_starts,
        frame_center_samples=frame_centers,
        sample_range=(0, 512),
        channel_indices=(0,),
    )
    payload = TimeFrequencyPayload(
        blocks=(block,),
        output="power_spectrogram",
        sample_rate_hz=1024.0,
        frequency_resolution_hz=128.0,
        time_resolution_samples=64,
        window_length=100,
        nfft=8,
    )
    tf_id = store.put(
        "time_frequency",
        source_ref,
        payload,
        {"frame_count": 3},
        selection=SignalSelection(source_ref=source_ref),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        tf_id,
        artifact_type="time_frequency",
        domain="time_frequency",
        method="tf_connected_regions",
        sample_range=(0, 164),
        bounds=(-64.0, 64.0),
        mask=((2, 0), (2, 1)),
    )

    with pytest.raises(MalformedEstimationEvidence):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_task7_finite_real_guard_rejects_unrepresentable_python_integer():
    assert _task7_is_finite_real(10**10000) is False


def test_task7_candidate_scalar_unrepresentable_integer_is_typed_mismatch(register_iq):
    repository, source_ref = register_iq(
        "task7_huge_candidate_scalar",
        np.ones(1024, np.complex64),
    )
    store = RunArtifactStore("task7_huge_candidate_scalar")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(-32.0, 32.0),
    )
    candidate_record = store.get(candidate_set_id)
    forged_candidate = replace(
        candidate_record.payload.candidates[0],
        peak_frequency_hz=10**10000,
    )
    forged_payload = replace(
        candidate_record.payload,
        candidates=(forged_candidate,),
    )
    forged_id = store.put(
        "candidate_set",
        source_ref,
        forged_payload,
        {"candidate_count": 1},
        selection=SignalSelection(source_ref=source_ref),
    )

    with pytest.raises(CandidateEvidenceMismatch):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=forged_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


@pytest.mark.parametrize("consumer", ["modulation", "power"])
def test_task7_upstream_unrepresentable_sample_rate_is_typed_evidence_error(
    register_iq,
    consumer,
):
    repository, source_ref = register_iq(
        f"task7_huge_upstream_fs_{consumer}",
        np.ones(1024, np.complex64),
        protected_channel=ProtectedChannel(
            center_offset_hz=-128.0,
            nominal_bandwidth_hz=64.0,
        ),
    )
    store = RunArtifactStore(f"task7_huge_upstream_fs_{consumer}")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
    )
    spectrum_record = store.get(spectrum_id)
    forged_payload = replace(
        spectrum_record.payload,
        sample_rate_hz=10**10000,
    )
    forged_spectrum_id = store.put(
        "spectrum",
        source_ref,
        forged_payload,
        {"method": "periodogram"},
        selection=SignalSelection(source_ref=source_ref),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        forged_spectrum_id,
        bounds=(-32.0, 32.0),
    )
    expected_error = (
        UnusablePowerEvidence if consumer == "power" else MalformedEstimationEvidence
    )

    with pytest.raises(expected_error):
        if consumer == "power":
            measure_power_relations(
                MeasurePowerRelationsArgs(
                    source_ref=source_ref,
                    spectrum_id=forged_spectrum_id,
                    candidate_set_id=candidate_set_id,
                    candidate_id="emission_0",
                ),
                repository=repository,
                artifact_store=store,
            )
        else:
            extract_modulation_features(
                ExtractModulationFeaturesArgs(
                    source_ref=source_ref,
                    candidate_set_id=candidate_set_id,
                    candidate_id="emission_0",
                ),
                repository=repository,
                artifact_store=store,
            )


@pytest.mark.parametrize("noise_scale", [1.0e-6, 3.0e-6])
def test_modulation_lag_correlation_uses_true_maximum_without_near_tie_tolerance(
    noise_scale,
):
    samples = np.arange(256, dtype=np.float64)
    rng = np.random.default_rng(123)
    iq = np.exp(2j * np.pi * 0.071 * samples) + noise_scale * (
        rng.normal(size=samples.size) + 1j * rng.normal(size=samples.size)
    )
    independent_correlations: dict[int, float] = {}
    for lag in range(4, 65):
        left = iq[lag:]
        right = iq[:-lag]
        independent_correlations[lag] = float(
            abs(np.sum(left * np.conj(right)))
            / math.sqrt(
                float(np.sum(np.abs(left) ** 2))
                * float(np.sum(np.abs(right) ** 2))
            )
        )
    true_maximum_lag = max(
        independent_correlations,
        key=independent_correlations.__getitem__,
    )
    assert true_maximum_lag == 21

    evidence, tested_lags = _lag_correlation_evidence(iq, 1024.0, 64)

    assert tested_lags == 61
    assert evidence["peak_lag_samples"] == true_maximum_lag
    assert evidence["peak_normalized_correlation"] == pytest.approx(
        independent_correlations[true_maximum_lag],
        rel=0.0,
        abs=1.0e-15,
    )


def test_modulation_lag_correlation_exact_tie_keeps_smallest_lag():
    iq = np.ones(256, dtype=np.complex128)

    evidence, tested_lags = _lag_correlation_evidence(iq, 1024.0, 64)

    assert tested_lags == 61
    assert evidence["peak_lag_samples"] == 4
    assert evidence["peak_normalized_correlation"] == 1.0


@pytest.mark.parametrize("tone_frequency_hz", [-128.0, 127.0])
def test_modulation_spectral_shape_detects_shifted_dft_boundary_peak(
    tone_frequency_hz,
):
    sample_rate_hz = 256.0
    samples = np.arange(256, dtype=np.float64)
    iq = np.exp(2j * np.pi * tone_frequency_hz * samples / sample_rate_hz)

    evidence, spectral_bins = _spectral_shape_evidence(iq, sample_rate_hz)

    assert spectral_bins == 256
    assert evidence["peak_count"] == 1
    assert evidence["strongest_peak_frequency_hz"] == pytest.approx(
        tone_frequency_hz,
        abs=1.0e-12,
    )


@pytest.mark.parametrize("tone_frequency_hz", [-127.0, 126.0])
def test_modulation_spectral_shape_boundary_adjacent_control_has_one_peak(
    tone_frequency_hz,
):
    sample_rate_hz = 256.0
    samples = np.arange(256, dtype=np.float64)
    iq = np.exp(2j * np.pi * tone_frequency_hz * samples / sample_rate_hz)

    evidence, _ = _spectral_shape_evidence(iq, sample_rate_hz)

    assert evidence["peak_count"] == 1
    assert evidence["strongest_peak_frequency_hz"] == pytest.approx(
        tone_frequency_hz,
        abs=1.0e-12,
    )


def test_modulation_spectral_shape_wrap_two_bins_apart_can_coexist():
    sample_rate_hz = 256.0
    samples = np.arange(256, dtype=np.float64)
    iq = np.exp(-1j * np.pi * samples) + 0.75 * np.exp(
        2j * np.pi * 127.0 * samples / sample_rate_hz
    )

    evidence, _ = _spectral_shape_evidence(iq, sample_rate_hz)

    assert evidence["peak_count"] == 2
    strongest_frequency_hz = evidence["strongest_peak_frequency_hz"]
    assert strongest_frequency_hz is not None
    assert math.isfinite(strongest_frequency_hz)


def test_modulation_spectral_peak_separation_uses_circular_bin_distance():
    assert estimation_dsp._circular_bin_distance(0, 255, 256) == 1
    assert estimation_dsp._circular_bin_distance(255, 0, 256) == 1
    assert estimation_dsp._circular_bin_distance(0, 254, 256) == 2


def test_power_ratio_uses_log_domain_when_positive_linear_ratio_underflows(register_iq):
    sample_rate_hz = 1.0e308
    repository, source_ref = register_iq(
        "task7_log_domain_ratio",
        np.ones(1024, np.complex64),
        sample_rate_hz=sample_rate_hz,
        protected_channel=ProtectedChannel(
            center_offset_hz=-4.25e307,
            nominal_bandwidth_hz=5.0e306,
        ),
    )
    store = RunArtifactStore("task7_log_domain_ratio")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-5.0e307, -2.5e307, 0.0, 2.5e307],
        [1.0, 0.0, 1.0, 0.0],
        sample_rate_hz=sample_rate_hz,
        scaling="power_density",
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(np.nextafter(0.0, -1.0), np.nextafter(0.0, 1.0)),
    )

    result = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=source_ref,
            spectrum_id=spectrum_id,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
        ),
        repository=repository,
        artifact_store=store,
    )

    protected_excess = result.data["protected_band"]["excess_power"]
    candidate_excess = result.data["candidate_band"]["excess_power"]
    assert protected_excess is not None and protected_excess > 0.0
    assert candidate_excess is not None and candidate_excess > 0.0
    assert candidate_excess / protected_excess == 0.0
    expected_j_over_s_db = 10.0 * (
        math.log10(candidate_excess) - math.log10(protected_excess)
    )
    assert result.status is ToolStatus.SUCCESS
    assert result.data["ratios"]["status"] == "success"
    assert result.data["ratios"]["reason"] is None
    assert result.data["ratios"]["j_over_s_db"] == pytest.approx(
        expected_j_over_s_db,
        rel=0.0,
        abs=1.0e-9,
    )
    assert result.data["ratios"]["sir_db"] == pytest.approx(
        -expected_j_over_s_db,
        rel=0.0,
        abs=1.0e-9,
    )


@pytest.mark.parametrize("sample_count", [64, 99])
def test_modulation_candidate_channel_limit_precedes_filter_length_failures(
    register_iq,
    sample_count,
):
    channel_indices = tuple(range(17))
    repository, source_ref = register_iq(
        f"task7_candidate_channel_priority_{sample_count}",
        np.ones((17, sample_count), np.complex64),
    )
    store = RunArtifactStore(f"task7_candidate_channel_priority_{sample_count}")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
        sample_range=(0, sample_count),
        channel_indices=channel_indices,
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        sample_range=(0, sample_count),
        channel_indices=channel_indices,
        bounds=(-32.0, 32.0),
    )

    with pytest.raises(InvalidModulationFeatureParameters):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


def test_modulation_candidate_channel_limit_is_checked_before_filter_dsp(
    register_iq,
    monkeypatch,
):
    sample_count = 256
    channel_indices = tuple(range(17))
    repository, source_ref = register_iq(
        "task7_candidate_channel_pre_dsp",
        np.ones((17, sample_count), np.complex64),
    )
    store = RunArtifactStore("task7_candidate_channel_pre_dsp")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
        sample_range=(0, sample_count),
        channel_indices=channel_indices,
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        sample_range=(0, sample_count),
        channel_indices=channel_indices,
        bounds=(-32.0, 32.0),
    )
    filter_entered = False

    def fail_if_filter_entered(*args, **kwargs):
        nonlocal filter_entered
        filter_entered = True
        raise AssertionError("candidate filter must not run for an invalid channel count")

    monkeypatch.setattr(
        estimation_tools,
        "_candidate_filtered_iq",
        fail_if_filter_entered,
    )

    with pytest.raises(InvalidModulationFeatureParameters):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )
    assert filter_entered is False


def test_modulation_candidate_allows_ordered_explicit_subset_before_dsp(register_iq):
    sample_count = 256
    candidate_channels = tuple(range(17))
    repository, source_ref = register_iq(
        "task7_candidate_ordered_subset",
        np.ones((17, sample_count), np.complex64),
    )
    store = RunArtifactStore("task7_candidate_ordered_subset")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
        sample_range=(0, sample_count),
        channel_indices=candidate_channels,
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        sample_range=(0, sample_count),
        channel_indices=candidate_channels,
        bounds=(-32.0, 32.0),
    )

    result = extract_modulation_features(
        ExtractModulationFeaturesArgs(
            source_ref=source_ref,
            candidate_set_id=candidate_set_id,
            candidate_id="emission_0",
            channel_indices=[16, 0],
            channel_mode="per_channel",
        ),
        repository=repository,
        artifact_store=store,
    )

    assert result.data["channel_result_count"] == 2
    assert [
        unit["channel_index"] for unit in result.data["channel_results"]
    ] == [16, 0]


def test_modulation_candidate_rejects_explicit_channel_outside_candidate(register_iq):
    sample_count = 256
    repository, source_ref = register_iq(
        "task7_candidate_outside_subset",
        np.ones((3, sample_count), np.complex64),
    )
    store = RunArtifactStore("task7_candidate_outside_subset")
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
        sample_range=(0, sample_count),
        channel_indices=(0, 1),
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        sample_range=(0, sample_count),
        channel_indices=(0, 1),
        bounds=(-32.0, 32.0),
    )

    with pytest.raises(InvalidModulationFeatureParameters):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(
                source_ref=source_ref,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
                channel_indices=[2],
            ),
            repository=repository,
            artifact_store=store,
        )


def test_modulation_lag_correlation_clamps_roundoff_and_uses_computed_maximum():
    samples = np.arange(64, dtype=np.float64)
    iq = np.exp(2j * np.pi * samples / 7.0)

    independent_correlations: dict[int, float] = {}
    for lag in range(4, 33):
        left = iq[lag:]
        right = iq[:-lag]
        raw_correlation = float(
            abs(np.sum(left * np.conj(right)))
            / math.sqrt(
                float(np.sum(np.abs(left) ** 2))
                * float(np.sum(np.abs(right) ** 2))
            )
        )
        independent_correlations[lag] = float(
            np.clip(raw_correlation, 0.0, 1.0)
        )
    expected_lag = max(
        independent_correlations,
        key=independent_correlations.__getitem__,
    )
    assert expected_lag == 5

    evidence, tested_lags = _lag_correlation_evidence(iq, 1024.0, 32)

    assert tested_lags == 29
    assert evidence["peak_lag_samples"] == expected_lag
    assert evidence["peak_normalized_correlation"] == independent_correlations[
        expected_lag
    ]
    assert 0.0 <= evidence["peak_normalized_correlation"] <= 1.0


def test_modulation_lag_correlation_does_not_promote_one_ulp_below_one_to_tie():
    samples = np.arange(64, dtype=np.float64)
    rng = np.random.default_rng(1)
    iq = np.exp(2j * np.pi * 0.071 * samples) + 1.0e-8 * (
        rng.normal(size=samples.size) + 1j * rng.normal(size=samples.size)
    )

    independent_correlations: dict[int, float] = {}
    for lag in range(4, 33):
        left = iq[lag:]
        right = iq[:-lag]
        raw_correlation = float(
            abs(np.sum(left * np.conj(right)))
            / math.sqrt(
                float(np.sum(np.abs(left) ** 2))
                * float(np.sum(np.abs(right) ** 2))
            )
        )
        independent_correlations[lag] = float(
            np.clip(raw_correlation, 0.0, 1.0)
        )
    expected_lag = max(
        independent_correlations,
        key=independent_correlations.__getitem__,
    )
    assert independent_correlations[6] == np.nextafter(1.0, 0.0)
    assert independent_correlations[20] == 1.0
    assert expected_lag == 20

    evidence, tested_lags = _lag_correlation_evidence(iq, 1024.0, 32)

    assert tested_lags == 29
    assert evidence["peak_lag_samples"] == expected_lag
    assert evidence["peak_normalized_correlation"] == independent_correlations[
        expected_lag
    ]


@pytest.mark.parametrize("sample_count", [128, 512])
def test_modulation_spectral_flatness_is_clipped_to_theoretical_unit_interval(
    sample_count,
):
    iq = np.zeros(sample_count, dtype=np.complex128)
    iq[sample_count // 2] = 1.0 + 0.0j

    evidence, spectral_bins = _spectral_shape_evidence(iq, 1024.0)

    assert spectral_bins == sample_count
    assert evidence["status"] == "success"
    assert 0.0 <= evidence["spectral_flatness"] <= 1.0
    assert evidence["spectral_flatness"] == pytest.approx(1.0, abs=1.0e-12)


@pytest.mark.parametrize("field_name", ["center_offset_hz", "nominal_bandwidth_hz"])
@pytest.mark.parametrize(
    ("forged_kind", "forged_value"),
    [
        ("huge_integer", 10**10000),
        ("list", []),
        ("mapping", {}),
        ("complex", 1.0 + 1.0j),
        ("boolean", True),
    ],
    ids=["huge_integer", "list", "mapping", "complex", "boolean"],
)
def test_power_forged_protected_prior_scalar_is_typed_before_float_conversion(
    register_iq,
    field_name,
    forged_kind,
    forged_value,
):
    protected_values = {
        "center_offset_hz": -128.0,
        "nominal_bandwidth_hz": 64.0,
    }
    protected_values[field_name] = forged_value
    protected = ProtectedChannel.model_construct(**protected_values)
    case_name = f"task7_protected_{field_name}_{forged_kind}"
    repository, source_ref = register_iq(
        case_name,
        np.ones(1024, np.complex64),
        protected_channel=protected,
    )
    store = RunArtifactStore(case_name)
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
        scaling="power_density",
    )
    candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(-32.0, 32.0),
    )

    with pytest.raises(UnusablePowerEvidence):
        measure_power_relations(
            MeasurePowerRelationsArgs(
                source_ref=source_ref,
                spectrum_id=spectrum_id,
                candidate_set_id=candidate_set_id,
                candidate_id="emission_0",
            ),
            repository=repository,
            artifact_store=store,
        )


@pytest.mark.parametrize("field_name", ["sample_rate_hz", "center_frequency_hz"])
@pytest.mark.parametrize(
    ("forged_kind", "forged_value"),
    [
        ("huge_integer", 10**10000),
        ("list", []),
        ("mapping", {}),
        ("boolean", True),
    ],
    ids=["huge_integer", "list", "mapping", "boolean"],
)
def test_modulation_forged_manifest_scalar_is_typed_before_float_conversion(
    tmp_path,
    field_name,
    forged_kind,
    forged_value,
):
    iq_path = tmp_path / f"task7_manifest_{field_name}_{forged_kind}.npy"
    np.save(iq_path, np.ones(256, np.complex64))
    manifest = IQSampleManifest(
        sample_id=f"task7_manifest_{field_name}_{forged_kind}",
        iq_path=str(iq_path),
        iq_format="npy",
        dtype="complex64",
        layout="samples",
        sample_rate_hz=1024.0,
        center_frequency_hz=2_400_000_000.0,
    ).model_copy(update={field_name: forged_value})
    repository = IQRepository([tmp_path])
    source_ref = repository.register(manifest)

    with pytest.raises(MalformedEstimationEvidence):
        extract_modulation_features(
            ExtractModulationFeaturesArgs(source_ref=source_ref),
            repository=repository,
            artifact_store=RunArtifactStore(
                f"task7_manifest_{field_name}_{forged_kind}"
            ),
        )


@pytest.mark.parametrize("consumer", ["modulation", "power"])
@pytest.mark.parametrize(
    ("hostile_kind", "hostile_candidate_id"),
    [
        ("overlong", "x" * 513),
        ("path_like", "../../private/candidate.json"),
        ("secret_like", "api_key=not-a-real-secret"),
    ],
    ids=["overlong", "path_like", "secret_like"],
)
def test_task7_bound_candidate_id_requires_safe_task4_format(
    register_iq,
    consumer,
    hostile_kind,
    hostile_candidate_id,
):
    case_name = f"task7_hostile_candidate_{consumer}_{hostile_kind}"
    repository, source_ref = register_iq(
        case_name,
        np.ones(1024, np.complex64),
        protected_channel=ProtectedChannel(
            center_offset_hz=-128.0,
            nominal_bandwidth_hz=64.0,
        ),
    )
    store = RunArtifactStore(case_name)
    spectrum_id, _ = _task7_spectrum(
        store,
        source_ref,
        [-512.0, -256.0, 0.0, 256.0, 511.0],
        np.ones(5),
        scaling="power_density",
    )
    valid_candidate_set_id, _ = _task7_candidate(
        store,
        source_ref,
        spectrum_id,
        bounds=(-32.0, 32.0),
    )
    valid_record = store.get(valid_candidate_set_id)
    forged_candidate = replace(
        valid_record.payload.candidates[0],
        candidate_id=hostile_candidate_id,
    )
    forged_payload = replace(
        valid_record.payload,
        candidates=(forged_candidate,),
    )
    forged_candidate_set_id = store.put(
        "candidate_set",
        source_ref,
        forged_payload,
        {"candidate_count": 1},
        selection=SignalSelection(source_ref=source_ref),
    )

    with pytest.raises(CandidateEvidenceMismatch):
        if consumer == "power":
            measure_power_relations(
                MeasurePowerRelationsArgs(
                    source_ref=source_ref,
                    spectrum_id=spectrum_id,
                    candidate_set_id=forged_candidate_set_id,
                    candidate_id=hostile_candidate_id,
                ),
                repository=repository,
                artifact_store=store,
            )
        else:
            extract_modulation_features(
                ExtractModulationFeaturesArgs(
                    source_ref=source_ref,
                    candidate_set_id=forged_candidate_set_id,
                    candidate_id=hostile_candidate_id,
                ),
                repository=repository,
                artifact_store=store,
            )


@pytest.mark.parametrize(
    ("hostile_kind", "hostile_id"),
    [
        ("overlong", "x" * 513),
        ("path_like", "../../private/artifact.json"),
        ("secret_like", "token=not-a-real-secret"),
    ],
    ids=["overlong", "path_like", "secret_like"],
)
def test_power_missing_protected_prior_does_not_echo_unvalidated_ids(
    register_iq,
    hostile_kind,
    hostile_id,
):
    repository, source_ref = register_iq(
        f"task7_missing_protected_{hostile_kind}",
        np.ones(1024, np.complex64),
        protected_channel=None,
    )

    result = measure_power_relations(
        MeasurePowerRelationsArgs(
            source_ref=source_ref,
            spectrum_id=hostile_id,
            candidate_set_id=hostile_id,
            candidate_id=hostile_id,
        ),
        repository=repository,
        artifact_store=RunArtifactStore(
            f"task7_missing_protected_{hostile_kind}"
        ),
    )

    assert result.status is ToolStatus.NOT_APPLICABLE
    assert result.error is not None
    assert result.error.code == "PROTECTED_CHANNEL_UNAVAILABLE"
    assert result.provenance == {}
    assert hostile_id not in result.model_dump_json()
