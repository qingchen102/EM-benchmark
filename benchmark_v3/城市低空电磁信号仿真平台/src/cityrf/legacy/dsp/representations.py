"""Internal, deterministic numerical representations for authorized IQ data.

Numerical arrays in this module are deliberately held by typed payloads only.
They are intended for ``RunArtifactStore`` and must never be placed in a
``ToolResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import signal

from cityrf.legacy.dsp.selection import SelectedSegment


class RepresentationError(ValueError):
    """Base class for typed representation failures that callers can repair."""


class InvalidRepresentationParameters(RepresentationError):
    """The selected method and resolved parameters are incompatible."""


class InsufficientRepresentationSamples(RepresentationError):
    """A selected segment cannot support the requested representation."""


class NonFiniteRepresentationResult(RepresentationError):
    """An algorithm produced a non-finite numerical result."""


@dataclass(frozen=True, repr=False)
class SpectrumBlock:
    """One independently computed spectrum, with its original source interval."""

    frequency_hz: np.ndarray = field(repr=False)
    power_linear: np.ndarray = field(repr=False)
    representation: np.ndarray = field(repr=False)
    sample_range: tuple[int, int]
    channel_indices: tuple[int, ...]

    def __repr__(self) -> str:
        return (
            "SpectrumBlock("
            f"sample_range={self.sample_range!r}, channel_indices={self.channel_indices!r}, "
            f"bin_count={self.frequency_hz.size})"
        )


@dataclass(frozen=True, repr=False)
class SpectrumPayload:
    """Stored shifted spectrum with power retained for later estimators."""

    frequency_hz: np.ndarray = field(repr=False)
    power_linear: np.ndarray = field(repr=False)
    representation: np.ndarray = field(repr=False)
    blocks: tuple[SpectrumBlock, ...] = field(repr=False)
    segment_ranges: tuple[tuple[int, int], ...]
    method: str
    scaling: str
    sample_rate_hz: float
    frequency_resolution_hz: float

    def __repr__(self) -> str:
        return (
            "SpectrumPayload("
            f"method={self.method!r}, scaling={self.scaling!r}, "
            f"bin_count={self.frequency_hz.size}, block_count={len(self.blocks)})"
        )


@dataclass(frozen=True, repr=False)
class TimeFrequencyBlock:
    """One STFT matrix plus its direct mapping to original capture samples."""

    frequency_hz: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    power_linear: np.ndarray = field(repr=False)
    frame_start_samples: np.ndarray = field(repr=False)
    frame_center_samples: np.ndarray = field(repr=False)
    sample_range: tuple[int, int]
    channel_indices: tuple[int, ...]

    def __repr__(self) -> str:
        return (
            "TimeFrequencyBlock("
            f"sample_range={self.sample_range!r}, channel_indices={self.channel_indices!r}, "
            f"shape={self.values.shape!r})"
        )


@dataclass(frozen=True, repr=False)
class TimeFrequencyPayload:
    """Stored two-sided shifted STFT blocks without public numerical summaries."""

    blocks: tuple[TimeFrequencyBlock, ...] = field(repr=False)
    output: str
    sample_rate_hz: float
    frequency_resolution_hz: float
    time_resolution_samples: int
    window_length: int
    nfft: int
    aggregate_power_linear: np.ndarray | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            "TimeFrequencyPayload("
            f"output={self.output!r}, block_count={len(self.blocks)}, nfft={self.nfft})"
        )


SpectrumMethod = Literal["fft", "periodogram", "welch_psd"]
SpectrumScaling = Literal["complex", "magnitude", "power", "power_density"]
StftOutput = Literal["complex_stft", "power_spectrogram"]


def compute_spectrum_blocks(
    segments: list[SelectedSegment],
    *,
    sample_rate_hz: float,
    method: SpectrumMethod,
    nfft: int,
    window: str,
    detrend: str,
    scaling: SpectrumScaling,
    segment_length: int | None,
    overlap_fraction: float | None,
    averaging: str | None,
) -> tuple[SpectrumBlock, ...]:
    """Compute each source interval independently, preserving complex phase."""

    _validate_spectrum_parameters(
        method=method,
        nfft=nfft,
        window=window,
        detrend=detrend,
        scaling=scaling,
        segment_length=segment_length,
        overlap_fraction=overlap_fraction,
        averaging=averaging,
    )
    blocks: list[SpectrumBlock] = []
    for segment in segments:
        selected_channels = _channels_for_mode(segment)
        for channel_index, samples in selected_channels:
            required_length = segment_length if method == "welch_psd" else nfft
            assert required_length is not None
            if samples.size < required_length:
                raise InsufficientRepresentationSamples(
                    "The selected range is shorter than the method's resolved minimum length."
                )
            frequency_hz, representation, power_linear = _one_spectrum(
                samples,
                sample_rate_hz=sample_rate_hz,
                method=method,
                nfft=nfft,
                window=window,
                detrend=detrend,
                scaling=scaling,
                segment_length=segment_length,
                overlap_fraction=overlap_fraction,
                averaging=averaging,
            )
            _require_finite(frequency_hz, representation, power_linear)
            blocks.append(
                SpectrumBlock(
                    frequency_hz=frequency_hz,
                    power_linear=power_linear,
                    representation=representation,
                    sample_range=(segment.sample_range.start, segment.sample_range.stop),
                    channel_indices=(channel_index,),
                )
            )
    return tuple(blocks)


def aggregate_spectrum_blocks(
    blocks: tuple[SpectrumBlock, ...], *, channel_mode: str, range_mode: str, scaling: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the public artifact's aggregate while never phase-averaging IQ."""

    if not blocks:
        raise InsufficientRepresentationSamples("No spectrum blocks were produced.")
    if (channel_mode == "mean_power" or range_mode == "aggregate") and scaling in {
        "complex",
        "magnitude",
    }:
        raise InvalidRepresentationParameters(
            "Complex or magnitude output cannot be non-coherently power-aggregated."
        )
    reference_frequency = blocks[0].frequency_hz
    if any(
        block.frequency_hz.shape != reference_frequency.shape
        or not np.array_equal(block.frequency_hz, reference_frequency)
        for block in blocks[1:]
    ):
        raise InvalidRepresentationParameters("Independent spectra do not share a common frequency grid.")
    powers = np.stack([block.power_linear for block in blocks], axis=0)
    aggregate_power = np.mean(powers, axis=0)
    if scaling in {"power", "power_density"}:
        aggregate_representation = aggregate_power
    else:
        aggregate_representation = blocks[0].representation
    _require_finite(reference_frequency, aggregate_power, aggregate_representation)
    return reference_frequency, aggregate_power, aggregate_representation


def compute_stft_blocks(
    segments: list[SelectedSegment],
    *,
    sample_rate_hz: float,
    output: StftOutput,
    window_length: int,
    hop_length: int,
    nfft: int,
    window: str,
    detrend: str,
) -> tuple[TimeFrequencyBlock, ...]:
    """Compute unpadded two-sided STFT blocks with original sample mapping."""

    _validate_stft_parameters(
        output=output,
        window_length=window_length,
        hop_length=hop_length,
        nfft=nfft,
        window=window,
        detrend=detrend,
    )
    blocks: list[TimeFrequencyBlock] = []
    for segment in segments:
        if segment.iq.shape[1] < window_length:
            raise InsufficientRepresentationSamples(
                "The selected range does not contain one complete STFT window."
            )
        if segment.channel_mode == "mean_power":
            if output == "complex_stft":
                raise InvalidRepresentationParameters(
                    "Complex STFT output cannot use non-coherent mean_power aggregation."
                )
            channel_matrices = [
                _one_stft(
                    channel,
                    sample_rate_hz=sample_rate_hz,
                    window_length=window_length,
                    hop_length=hop_length,
                    nfft=nfft,
                    window=window,
                    detrend=detrend,
                )
                for channel in segment.iq
            ]
            frequency_hz = channel_matrices[0][0]
            matrices = [matrix for _, matrix in channel_matrices]
            if any(not np.array_equal(frequency_hz, freq) for freq, _ in channel_matrices[1:]):
                raise InvalidRepresentationParameters("STFT channels do not share a frequency grid.")
            power_linear = np.mean(np.stack([np.abs(matrix) ** 2 for matrix in matrices]), axis=0)
            blocks.append(
                _make_stft_block(
                    frequency_hz,
                    power_linear,
                    power_linear,
                    segment=segment,
                    channel_indices=segment.channel_indices,
                    window_length=window_length,
                    hop_length=hop_length,
                )
            )
            continue

        for channel_index, samples in _channels_for_mode(segment):
            frequency_hz, complex_matrix = _one_stft(
                samples,
                sample_rate_hz=sample_rate_hz,
                window_length=window_length,
                hop_length=hop_length,
                nfft=nfft,
                window=window,
                detrend=detrend,
            )
            power_linear = np.abs(complex_matrix) ** 2
            values = complex_matrix if output == "complex_stft" else power_linear
            blocks.append(
                _make_stft_block(
                    frequency_hz,
                    values,
                    power_linear,
                    segment=segment,
                    channel_indices=(channel_index,),
                    window_length=window_length,
                    hop_length=hop_length,
                )
            )
    return tuple(blocks)


def _channels_for_mode(segment: SelectedSegment) -> list[tuple[int, np.ndarray]]:
    if segment.channel_mode == "single":
        return [(segment.channel_indices[0], segment.iq[0])]
    return list(zip(segment.channel_indices, segment.iq, strict=True))


def _one_spectrum(
    samples: np.ndarray,
    *,
    sample_rate_hz: float,
    method: SpectrumMethod,
    nfft: int,
    window: str,
    detrend: str,
    scaling: SpectrumScaling,
    segment_length: int | None,
    overlap_fraction: float | None,
    averaging: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "fft":
        prepared = samples[:nfft]
        if detrend != "none":
            prepared = signal.detrend(prepared, type=detrend)
        window_values = signal.get_window(window, nfft, fftbins=True)
        transformed = np.fft.fftshift(np.fft.fft(prepared * window_values, n=nfft))
        frequency_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate_hz))
        coherent_gain = float(np.sum(window_values))
        window_energy = float(np.sum(window_values**2))
        normalized_complex = transformed / coherent_gain
        power = np.abs(normalized_complex) ** 2
        if scaling == "complex":
            representation = normalized_complex
        elif scaling == "magnitude":
            representation = np.abs(normalized_complex)
        elif scaling == "power":
            representation = power
        else:
            power = np.abs(transformed) ** 2 / (sample_rate_hz * window_energy)
            representation = power
        return frequency_hz, representation, power

    detrend_value: str | bool = False if detrend == "none" else detrend
    psd_scaling: Literal["spectrum", "density"] = (
        "density" if scaling == "power_density" else "spectrum"
    )
    if method == "periodogram":
        frequency_hz, power = signal.periodogram(
            samples,
            fs=sample_rate_hz,
            window=window,
            nfft=nfft,
            detrend=detrend_value,
            return_onesided=False,
            scaling=psd_scaling,
        )
    else:
        assert segment_length is not None and overlap_fraction is not None and averaging is not None
        noverlap = int(round(segment_length * overlap_fraction))
        frequency_hz, power = signal.welch(
            samples,
            fs=sample_rate_hz,
            window=window,
            nperseg=segment_length,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend_value,
            return_onesided=False,
            scaling=psd_scaling,
            average=averaging,
        )
    order = np.fft.fftshift(np.arange(frequency_hz.size))
    frequency_hz = frequency_hz[order]
    power = np.real(power[order])
    return frequency_hz, power, power


def _one_stft(
    samples: np.ndarray,
    *,
    sample_rate_hz: float,
    window_length: int,
    hop_length: int,
    nfft: int,
    window: str,
    detrend: str,
) -> tuple[np.ndarray, np.ndarray]:
    detrend_value: str | bool = False if detrend == "none" else detrend
    frequency_hz, _, matrix = signal.stft(
        samples,
        fs=sample_rate_hz,
        window=window,
        nperseg=window_length,
        noverlap=window_length - hop_length,
        nfft=nfft,
        detrend=detrend_value,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    order = np.fft.fftshift(np.arange(frequency_hz.size))
    return frequency_hz[order], matrix[order, :]


def _make_stft_block(
    frequency_hz: np.ndarray,
    values: np.ndarray,
    power_linear: np.ndarray,
    *,
    segment: SelectedSegment,
    channel_indices: tuple[int, ...],
    window_length: int,
    hop_length: int,
) -> TimeFrequencyBlock:
    frame_count = values.shape[1]
    expected_frame_count = 1 + (segment.iq.shape[1] - window_length) // hop_length
    if frame_count != expected_frame_count:
        raise InsufficientRepresentationSamples("STFT did not produce the expected complete frames.")
    starts = segment.sample_range.start + np.arange(frame_count, dtype=np.int64) * hop_length
    centers = starts.astype(np.float64) + (window_length - 1) / 2.0
    _require_finite(frequency_hz, values, power_linear, starts, centers)
    if np.any(power_linear < 0.0):
        raise NonFiniteRepresentationResult("STFT power must be non-negative.")
    return TimeFrequencyBlock(
        frequency_hz=frequency_hz,
        values=values,
        power_linear=power_linear,
        frame_start_samples=starts,
        frame_center_samples=centers,
        sample_range=(segment.sample_range.start, segment.sample_range.stop),
        channel_indices=channel_indices,
    )


def _validate_spectrum_parameters(
    *,
    method: str,
    nfft: int,
    window: str,
    detrend: str,
    scaling: str,
    segment_length: int | None,
    overlap_fraction: float | None,
    averaging: str | None,
) -> None:
    if nfft <= 0:
        raise InvalidRepresentationParameters("The resolved FFT bin count must be positive.")
    if not window:
        raise InvalidRepresentationParameters("A non-empty window is required.")
    _validate_window(window)
    if detrend not in {"none", "constant", "linear"}:
        raise InvalidRepresentationParameters("detrend must be none, constant, or linear.")
    if method == "fft":
        if segment_length is not None or overlap_fraction is not None or averaging is not None:
            raise InvalidRepresentationParameters("Welch parameters are not valid for the FFT method.")
        if scaling not in {"complex", "magnitude", "power", "power_density"}:
            raise InvalidRepresentationParameters("The requested FFT scaling is unsupported.")
        return
    if scaling not in {"power", "power_density"}:
        raise InvalidRepresentationParameters("PSD methods require power or power_density scaling.")
    if method == "periodogram":
        if segment_length is not None or overlap_fraction is not None or averaging is not None:
            raise InvalidRepresentationParameters("Welch parameters are not valid for periodogram.")
        return
    if method != "welch_psd":
        raise InvalidRepresentationParameters("The requested spectrum method is unsupported.")
    if segment_length is None or overlap_fraction is None or averaging is None:
        raise InvalidRepresentationParameters("Welch requires segment_length, overlap_fraction, and averaging.")
    if segment_length <= 0 or nfft < segment_length:
        raise InvalidRepresentationParameters("Welch requires nfft greater than or equal to segment_length.")
    if not 0.0 <= overlap_fraction < 1.0:
        raise InvalidRepresentationParameters("Welch overlap_fraction must be in [0, 1).")
    if averaging not in {"mean", "median"}:
        raise InvalidRepresentationParameters("Welch averaging must be mean or median.")


def _validate_stft_parameters(
    *,
    output: str,
    window_length: int,
    hop_length: int,
    nfft: int,
    window: str,
    detrend: str,
) -> None:
    if output not in {"complex_stft", "power_spectrogram"}:
        raise InvalidRepresentationParameters("The requested STFT output is unsupported.")
    if window_length <= 0 or hop_length <= 0 or nfft <= 0:
        raise InvalidRepresentationParameters("STFT window_length, hop_length, and nfft must be positive.")
    if hop_length > window_length:
        raise InvalidRepresentationParameters("STFT hop_length cannot exceed window_length.")
    if nfft < window_length:
        raise InvalidRepresentationParameters("STFT nfft cannot be smaller than window_length.")
    if not window:
        raise InvalidRepresentationParameters("A non-empty STFT window is required.")
    _validate_window(window)
    if detrend not in {"none", "constant", "linear"}:
        raise InvalidRepresentationParameters("detrend must be none, constant, or linear.")


def _require_finite(*arrays: np.ndarray) -> None:
    if any(not np.isfinite(array).all() for array in arrays):
        raise NonFiniteRepresentationResult("Representation computation produced a non-finite result.")


def _validate_window(window: str) -> None:
    """Convert SciPy window failures to the executor's typed error boundary."""

    try:
        signal.get_window(window, 8, fftbins=True)
    except (TypeError, ValueError) as error:
        raise InvalidRepresentationParameters("The requested window is unsupported.") from error
