"""Public contract and numerical sanity tests for Task 3 representation tools."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from cityrf.legacy.dsp.representations import (
    InsufficientRepresentationSamples,
    InvalidRepresentationParameters,
    SpectrumPayload,
    TimeFrequencyPayload,
)
from cityrf.legacy.runtime import RunArtifactStore
from cityrf.legacy.schemas import ArrayGeometry, FrequencyRange, SampleRange
from cityrf.legacy.tools.inspection import InspectIQArgs, inspect_iq
from cityrf.legacy.tools.representations import (
    ComputeSpectrumArgs,
    ComputeTimeFrequencyArgs,
    compute_spectrum,
    compute_time_frequency,
)


def _store() -> RunArtifactStore:
    return RunArtifactStore(run_id="representation_tests")


def _peak_frequency(payload: SpectrumPayload) -> float:
    return float(payload.frequency_hz[np.argmax(payload.power_linear)])


def test_inspect_iq_schema_exposes_only_source_ref():
    assert set(InspectIQArgs.model_json_schema()["properties"]) == {"source_ref"}


def test_inspect_iq_reports_safe_concise_capabilities(register_iq, complex_tone):
    repository, source_ref = register_iq("inspect", complex_tone)

    result = inspect_iq(InspectIQArgs(source_ref=source_ref), repository=repository)
    rendered = result.model_dump(mode="json")

    assert result.status == "success"
    assert "checks" not in rendered["data"]
    assert rendered["data"]["iq_shape"] == [1, 1024]
    assert rendered["data"]["iq_dtype"] == "complex64"
    assert rendered["data"]["channel_count"] == 1
    assert rendered["data"]["sample_count"] == 1024
    assert rendered["data"]["all_finite"] is True
    assert rendered["data"]["mean_power_linear"] > 0.0
    assert "dc_offset" in rendered["data"]
    assert "papr_db" in rendered["data"]
    assert "clipping" in rendered["data"]
    assert rendered["data"]["capabilities"]["compute_spectrum"] is True
    assert rendered["data"]["capabilities"]["doa"] is False
    assert any("DOA" in item for item in rendered["data"]["missing_requirements"])
    assert all(isinstance(value, bool) for value in rendered["data"]["capabilities"].values())
    assert "reason" not in json.dumps(rendered["data"]["capabilities"])
    serialized = json.dumps(rendered)
    for forbidden in ("fixture_inspect", ".npy", "ground_truth", "raw_iq", "sample_id"):
        assert forbidden not in serialized.lower()


def test_inspect_iq_zero_capture_has_no_papr_or_clipping_warning(register_iq):
    repository, source_ref = register_iq("zero_inspect", np.zeros(256, dtype=np.complex64))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = inspect_iq(InspectIQArgs(source_ref=source_ref), repository=repository)

    assert result.status == "success"
    assert result.data["mean_power_linear"] == 0.0
    assert result.data["papr_db"] is None
    assert result.data["clipping"]["suspected"] is False
    assert result.data["clipping"]["max_magnitude"] == 0.0
    serialized = json.dumps(result.model_dump(mode="json"))
    assert "NaN" not in serialized
    assert "Infinity" not in serialized


def test_inspect_iq_doa_capability_requires_usable_array_sync_and_reference(
    register_iq, ula_four_channel
):
    ula_iq, usable_geometry = ula_four_channel
    repository, source_ref = register_iq(
        "usable_doa",
        ula_iq,
        array_geometry=usable_geometry,
        expected_desired_source_count=1,
    )
    positive = inspect_iq(InspectIQArgs(source_ref=source_ref), repository=repository)
    assert positive.data["capabilities"]["doa"] is True
    assert positive.data["array_availability"]["usable"] is True
    assert positive.data["synchronization_available"] is True
    assert positive.data["reference_frequency_available"] is True
    assert positive.data["acquisition_priors"]["expected_desired_source_count"] == 1

    incomplete_geometry = ArrayGeometry(
        geometry_type="ula", element_count=4, synchronized=True
    )
    incomplete_repository, incomplete_ref = register_iq(
        "incomplete_doa",
        ula_iq,
        center_frequency_hz=None,
        array_geometry=incomplete_geometry,
    )
    negative = inspect_iq(InspectIQArgs(source_ref=incomplete_ref), repository=incomplete_repository)
    assert negative.data["capabilities"]["doa"] is False
    assert negative.data["array_availability"]["usable"] is False
    assert negative.data["reference_frequency_available"] is False
    assert any("usable array geometry" in item for item in negative.data["missing_requirements"])
    assert any("reference/carrier" in item for item in negative.data["missing_requirements"])


def test_inspect_iq_rejects_unmapped_subset_of_a_larger_array_geometry(register_iq, ula_four_channel):
    ula_iq, four_element_geometry = ula_four_channel
    repository, source_ref = register_iq(
        "unmapped_subset_doa",
        ula_iq[:2],
        array_geometry=four_element_geometry,
    )

    result = inspect_iq(InspectIQArgs(source_ref=source_ref), repository=repository)

    assert result.data["array_availability"]["usable"] is False
    assert result.data["capabilities"]["spatial_evidence"] is False
    assert result.data["capabilities"]["doa"] is False
    assert any("channel-to-element mapping" in item for item in result.data["missing_requirements"])


@pytest.mark.parametrize("method", ["fft", "periodogram"])
def test_tone_peak_is_within_one_resolved_bin_for_fft_and_periodogram(
    register_iq, complex_tone, method
):
    repository, source_ref = register_iq(f"tone_{method}", complex_tone)
    store = RunArtifactStore(run_id=f"tone_{method}")
    result = compute_spectrum(
        ComputeSpectrumArgs(source_ref=source_ref, method=method, profile="custom", nfft=256),
        repository=repository,
        artifact_store=store,
    )
    artifact = store.get(result.data["spectrum_id"], expected_type="spectrum")
    spectrum = artifact.payload

    assert isinstance(spectrum, SpectrumPayload)
    assert abs(_peak_frequency(spectrum) - 0.125 * 1_024.0) <= spectrum.frequency_resolution_hz
    assert result.data["resolved_nfft"] == 256
    assert result.data["frequency_resolution_hz"] == pytest.approx(4.0)


def test_welch_psd_is_finite_nonnegative_and_discloses_resolved_parameters(register_iq, complex_tone):
    repository, source_ref = register_iq("welch", complex_tone)
    store = _store()
    result = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=source_ref,
            method="welch_psd",
            profile="custom",
            nfft=256,
            window="hann",
            detrend="constant",
            segment_length=128,
            overlap_fraction=0.5,
            averaging="mean",
            scaling="power_density",
        ),
        repository=repository,
        artifact_store=store,
    )
    spectrum = store.get(result.data["spectrum_id"], expected_type="spectrum").payload

    assert np.isfinite(spectrum.power_linear).all()
    assert (spectrum.power_linear >= 0.0).all()
    assert abs(_peak_frequency(spectrum) - 128.0) <= spectrum.frequency_resolution_hz
    assert result.provenance["resolved_parameters"]["segment_length"] == 128
    assert result.provenance["resolved_parameters"]["overlap_fraction"] == 0.5
    assert result.semantics["frequency_resolution_hz"].startswith("Hz per shifted baseband bin")


def test_welch_allows_zero_padding_when_selection_meets_segment_length(register_iq, complex_tone):
    repository, source_ref = register_iq("welch_zero_padding", complex_tone)
    store = _store()
    result = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=source_ref,
            method="welch_psd",
            profile="custom",
            nfft=256,
            segment_length=128,
            overlap_fraction=0.5,
            averaging="mean",
            sample_ranges=[SampleRange(start=0, stop=128)],
        ),
        repository=repository,
        artifact_store=store,
    )
    payload = store.get(result.data["spectrum_id"], expected_type="spectrum").payload
    assert payload.frequency_hz.size == 256
    assert np.isfinite(payload.power_linear).all()


def test_fft_window_detrend_and_scaling_follow_explicit_window_aware_conventions(register_iq):
    sample_rate_hz = 1_024.0
    samples = np.arange(256, dtype=np.float64)
    constant = np.ones(256, dtype=np.complex64)
    leakage_tone = np.exp(2j * np.pi * 0.131 * samples).astype(np.complex64)
    constant_repository, constant_ref = register_iq(
        "constant_fft", constant, sample_rate_hz=sample_rate_hz
    )
    none_store, constant_store = _store(), _store()
    none = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=constant_ref, profile="custom", nfft=256, window="boxcar", detrend="none"
        ),
        repository=constant_repository,
        artifact_store=none_store,
    )
    detrended = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=constant_ref, profile="custom", nfft=256, window="boxcar", detrend="constant"
        ),
        repository=constant_repository,
        artifact_store=constant_store,
    )
    none_power = none_store.get(none.data["spectrum_id"], expected_type="spectrum").payload.power_linear
    detrended_power = constant_store.get(
        detrended.data["spectrum_id"], expected_type="spectrum"
    ).payload.power_linear
    assert np.max(none_power) > 0.9
    assert np.max(detrended_power) < 1e-12

    leakage_repository, leakage_ref = register_iq("leakage_fft", leakage_tone, sample_rate_hz=sample_rate_hz)
    boxcar_store, hann_store = _store(), _store()
    boxcar = compute_spectrum(
        ComputeSpectrumArgs(source_ref=leakage_ref, profile="custom", nfft=256, window="boxcar", detrend="none"),
        repository=leakage_repository,
        artifact_store=boxcar_store,
    )
    hann = compute_spectrum(
        ComputeSpectrumArgs(source_ref=leakage_ref, profile="custom", nfft=256, window="hann", detrend="none"),
        repository=leakage_repository,
        artifact_store=hann_store,
    )
    boxcar_power = boxcar_store.get(boxcar.data["spectrum_id"], expected_type="spectrum").payload.power_linear
    hann_power = hann_store.get(hann.data["spectrum_id"], expected_type="spectrum").payload.power_linear
    assert not np.allclose(boxcar_power, hann_power)

    density_store = _store()
    density = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=constant_ref,
            profile="custom",
            nfft=256,
            window="boxcar",
            detrend="none",
            scaling="power_density",
        ),
        repository=constant_repository,
        artifact_store=density_store,
    )
    density_payload = density_store.get(
        density.data["spectrum_id"], expected_type="spectrum"
    ).payload
    assert np.sum(density_payload.power_linear) * density_payload.frequency_resolution_hz == pytest.approx(1.0, rel=1e-6)
    assert "noise_floor_db_per_hz" in density.data
    assert "noise_floor_db" not in density.data
    assert density.semantics["noise_floor_db_per_hz"].endswith("dB/Hz.")
    assert all("level_db_per_hz" in peak and "power_db" not in peak for peak in density.data["peaks"])


@pytest.mark.parametrize("method", ["fft", "periodogram", "welch_psd"])
def test_invalid_spectrum_window_is_a_typed_error(register_iq, complex_tone, method):
    repository, source_ref = register_iq(f"bad_window_{method}", complex_tone)
    kwargs = {"source_ref": source_ref, "method": method, "window": "not_a_window"}
    if method == "welch_psd":
        kwargs.update({"segment_length": 128, "overlap_fraction": 0.5, "averaging": "mean"})
    with pytest.raises(InvalidRepresentationParameters):
        compute_spectrum(ComputeSpectrumArgs(**kwargs), repository=repository, artifact_store=_store())


def test_spectrum_rejects_invalid_method_parameters_and_too_short_selection(register_iq, complex_tone):
    repository, source_ref = register_iq("invalid_spectrum", complex_tone)
    with pytest.raises(InvalidRepresentationParameters):
        compute_spectrum(
            ComputeSpectrumArgs(source_ref=source_ref, method="fft", segment_length=128),
            repository=repository,
            artifact_store=_store(),
        )
    with pytest.raises(InvalidRepresentationParameters):
        compute_spectrum(
            ComputeSpectrumArgs(source_ref=source_ref, method="welch_psd", segment_length=128, nfft=64),
            repository=repository,
            artifact_store=_store(),
        )
    with pytest.raises(InsufficientRepresentationSamples):
        compute_spectrum(
            ComputeSpectrumArgs(
                source_ref=source_ref,
                profile="custom",
                nfft=256,
                sample_ranges=[SampleRange(start=0, stop=32)],
            ),
            repository=repository,
            artifact_store=_store(),
        )


def test_spectrum_frequency_ranges_are_hz_explicit_and_bound_to_nyquist(register_iq, complex_tone):
    repository, source_ref = register_iq("frequency_range", complex_tone)
    store = _store()
    result = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=source_ref,
            method="periodogram",
            profile="custom",
            nfft=256,
            sample_ranges=[SampleRange(start=0, stop=256), SampleRange(start=512, stop=768)],
            frequency_range=FrequencyRange(start_hz=120.0, stop_hz=136.0, reference="baseband"),
        ),
        repository=repository,
        artifact_store=store,
    )
    payload = store.get(result.data["spectrum_id"], expected_type="spectrum").payload
    assert payload.frequency_hz.min() >= 120.0
    assert payload.frequency_hz.max() <= 136.0
    assert all(block.frequency_hz.min() >= 120.0 and block.frequency_hz.max() <= 136.0 for block in payload.blocks)
    assert tuple(block.sample_range for block in payload.blocks) == ((0, 256), (512, 768))
    assert result.provenance["frequency_selection"] == {"reference": "baseband", "unit": "Hz"}

    with pytest.raises(InvalidRepresentationParameters):
        compute_spectrum(
            ComputeSpectrumArgs(
                source_ref=source_ref,
                frequency_range=FrequencyRange(start_hz=0.0, stop_hz=600.0, reference="baseband"),
            ),
            repository=repository,
            artifact_store=_store(),
        )


def test_aggregate_ranges_and_mean_power_never_coherently_average_antiphase_channels(
    register_iq, antiphase_two_channel
):
    repository, source_ref = register_iq("antiphase", antiphase_two_channel)
    store = _store()
    result = compute_spectrum(
        ComputeSpectrumArgs(
            source_ref=source_ref,
            method="fft",
            profile="custom",
            nfft=128,
            sample_ranges=[SampleRange(start=0, stop=128), SampleRange(start=512, stop=640)],
            range_mode="aggregate",
            channel_mode="mean_power",
            scaling="power",
        ),
        repository=repository,
        artifact_store=store,
    )
    payload = store.get(result.data["spectrum_id"], expected_type="spectrum").payload

    assert len(payload.blocks) == 4
    assert payload.segment_ranges == ((0, 128), (512, 640))
    assert np.max(payload.power_linear) > 0.5
    assert result.provenance["channel_aggregation"] == "noncoherent_mean_power"
    assert result.provenance["range_aggregation"] == "independent_power_average"
    assert result.artifacts[0].summary["block_count"] == 4
    assert "segment_count" not in result.artifacts[0].summary


def test_spectrum_keeps_arrays_in_artifact_and_exposes_only_bounded_summary(register_iq, complex_tone):
    repository, source_ref = register_iq("spectrum_artifact", complex_tone)
    store = _store()
    result = compute_spectrum(
        ComputeSpectrumArgs(source_ref=source_ref, profile="custom", nfft=256),
        repository=repository,
        artifact_store=store,
    )
    payload = store.get(result.data["spectrum_id"], expected_type="spectrum").payload
    serialized = result.model_dump_json()

    assert isinstance(payload.frequency_hz, np.ndarray)
    assert isinstance(payload.power_linear, np.ndarray)
    assert "frequency_axis_hz" not in serialized
    assert "power_linear" not in serialized
    assert "spectrum_values" not in serialized
    assert result.artifacts[0].artifact_id == result.data["spectrum_id"]
    assert len(result.data["peaks"]) <= 16


def test_stft_uses_shifted_two_sided_axes_and_preserves_original_frame_mapping(
    register_iq, complex_tone
):
    repository, source_ref = register_iq("stft", complex_tone)
    store = _store()
    result = compute_time_frequency(
        ComputeTimeFrequencyArgs(
            source_ref=source_ref,
            output="power_spectrogram",
            profile="custom",
            window_length=128,
            hop_length=64,
            nfft=256,
            window="hann",
            detrend="constant",
        ),
        repository=repository,
        artifact_store=store,
    )
    payload = store.get(result.data["time_frequency_id"], expected_type="time_frequency").payload
    block = payload.blocks[0]

    assert isinstance(payload, TimeFrequencyPayload)
    assert block.frequency_hz.shape == (256,)
    assert block.values.shape == (256, 15)
    assert block.sample_range == (0, 1024)
    assert tuple(block.frame_start_samples) == tuple(range(0, 1024 - 128 + 1, 64))
    assert tuple(block.frame_center_samples) == tuple(start + 63.5 for start in block.frame_start_samples)
    assert np.isfinite(block.values).all()
    assert (block.values >= 0.0).all()
    assert result.data["frame_count"] == 15
    assert result.data["frequency_bin_count"] == 256
    assert result.data["time_resolution_samples"] == 64


def test_stft_mean_power_is_noncoherent_and_chirp_has_variable_peak_trajectory(
    register_iq, antiphase_two_channel, complex_linear_chirp
):
    repository, source_ref = register_iq("antiphase_stft", antiphase_two_channel)
    store = _store()
    result = compute_time_frequency(
        ComputeTimeFrequencyArgs(
            source_ref=source_ref,
            output="power_spectrogram",
            profile="custom",
            window_length=128,
            hop_length=64,
            nfft=128,
            channel_mode="mean_power",
        ),
        repository=repository,
        artifact_store=store,
    )
    antiphase_block = store.get(result.data["time_frequency_id"], expected_type="time_frequency").payload.blocks[0]
    assert np.max(antiphase_block.values) > 0.1

    chirp_repository, chirp_ref = register_iq("chirp_stft", complex_linear_chirp)
    chirp_store = _store()
    chirp_result = compute_time_frequency(
        ComputeTimeFrequencyArgs(
            source_ref=chirp_ref,
            output="power_spectrogram",
            profile="custom",
            window_length=128,
            hop_length=64,
            nfft=256,
        ),
        repository=chirp_repository,
        artifact_store=chirp_store,
    )
    chirp_block = chirp_store.get(chirp_result.data["time_frequency_id"], expected_type="time_frequency").payload.blocks[0]
    ridge_hz = chirp_block.frequency_hz[np.argmax(chirp_block.values, axis=0)]
    assert np.ptp(ridge_hz) > chirp_block.frequency_hz[1] - chirp_block.frequency_hz[0]


def test_stft_aggregate_ranges_store_independent_power_aggregate_without_raw_concatenation(
    register_iq, complex_tone
):
    repository, source_ref = register_iq("aggregate_stft", complex_tone)
    store = _store()
    result = compute_time_frequency(
        ComputeTimeFrequencyArgs(
            source_ref=source_ref,
            output="power_spectrogram",
            profile="custom",
            window_length=128,
            hop_length=64,
            nfft=128,
            sample_ranges=[SampleRange(start=0, stop=256), SampleRange(start=512, stop=768)],
            range_mode="aggregate",
        ),
        repository=repository,
        artifact_store=store,
    )
    payload = store.get(result.data["time_frequency_id"], expected_type="time_frequency").payload

    assert len(payload.blocks) == 2
    assert payload.aggregate_power_linear is not None
    assert payload.aggregate_power_linear.shape == (128,)
    assert np.max(payload.aggregate_power_linear) > 0.1
    assert result.provenance["range_aggregation"] == "independent_power_average"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_length": 1_025, "hop_length": 64, "nfft": 1_025},
        {"window_length": 128, "hop_length": 0, "nfft": 128},
        {"window_length": 128, "hop_length": 64, "nfft": 64},
        {
            "window_length": 128,
            "hop_length": 64,
            "nfft": 128,
            "sample_ranges": [SampleRange(start=0, stop=127)],
        },
    ],
)
def test_stft_rejects_invalid_or_incomplete_frame_requests(register_iq, complex_tone, kwargs):
    repository, source_ref = register_iq("bad_stft", complex_tone)
    with pytest.raises((InvalidRepresentationParameters, InsufficientRepresentationSamples)):
        compute_time_frequency(
            ComputeTimeFrequencyArgs(source_ref=source_ref, profile="custom", **kwargs),
            repository=repository,
            artifact_store=_store(),
        )


def test_stft_keeps_matrices_and_axes_inside_artifact_store(register_iq, complex_tone):
    repository, source_ref = register_iq("stft_artifact", complex_tone)
    store = _store()
    result = compute_time_frequency(
        ComputeTimeFrequencyArgs(
            source_ref=source_ref,
            output="complex_stft",
            profile="custom",
            window_length=128,
            hop_length=64,
            nfft=128,
        ),
        repository=repository,
        artifact_store=store,
    )
    payload = store.get(result.data["time_frequency_id"], expected_type="time_frequency").payload
    serialized = result.model_dump_json()

    assert np.iscomplexobj(payload.blocks[0].values)
    assert np.isfinite(payload.blocks[0].power_linear).all()
    assert (payload.blocks[0].power_linear >= 0.0).all()
    assert "frequency_hz" not in serialized
    assert "frame_start_samples" not in serialized
    assert "stft_matrix" not in serialized
    assert result.artifacts[0].artifact_id == result.data["time_frequency_id"]


def test_invalid_stft_window_is_a_typed_error(register_iq, complex_tone):
    repository, source_ref = register_iq("bad_stft_window", complex_tone)
    with pytest.raises(InvalidRepresentationParameters):
        compute_time_frequency(
            ComputeTimeFrequencyArgs(
                source_ref=source_ref,
                profile="custom",
                window_length=128,
                hop_length=64,
                nfft=128,
                window="not_a_window",
            ),
            repository=repository,
            artifact_store=_store(),
        )
