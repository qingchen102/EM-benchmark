"""Injected controlled executors for spectrum and time-frequency evidence."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from cityrf.legacy.dsp.representations import (
    SpectrumBlock,
    SpectrumPayload,
    TimeFrequencyPayload,
    aggregate_spectrum_blocks,
    compute_spectrum_blocks,
    compute_stft_blocks,
)
from cityrf.legacy.dsp.selection import select_iq
from cityrf.legacy.runtime import IQRepository, RunArtifactStore
from cityrf.legacy.schemas import SignalSelection, ToolResult, ToolStatus


class ComputeSpectrumArgs(SignalSelection):
    """Flattened selection plus explicit spectrum controls for later wrappers."""

    method: Literal["fft", "periodogram", "welch_psd"] = "fft"
    profile: Literal["fast_scan", "balanced", "high_resolution", "custom"] = "balanced"
    nfft: int | None = None
    window: str = "hann"
    detrend: Literal["none", "constant", "linear"] = "constant"
    scaling: Literal["complex", "magnitude", "power", "power_density"] = "power"
    segment_length: int | None = None
    overlap_fraction: float | None = Field(default=None, ge=0.0, lt=1.0)
    averaging: Literal["mean", "median"] | None = None


class ComputeTimeFrequencyArgs(SignalSelection):
    """Flattened selection plus explicit STFT controls for later wrappers."""

    output: Literal["complex_stft", "power_spectrogram"] = "power_spectrogram"
    profile: Literal["short_transient", "balanced", "frequency_detail", "custom"] = "balanced"
    window_length: int | None = None
    hop_length: int | None = None
    nfft: int | None = None
    window: str = "hann"
    detrend: Literal["none", "constant", "linear"] = "constant"


def compute_spectrum(
    args: ComputeSpectrumArgs, *, repository: IQRepository, artifact_store: RunArtifactStore
) -> ToolResult:
    """Compute a shifted physical-Hz spectrum and place arrays in the artifact store."""

    frame = repository.get(args.source_ref)
    segments = select_iq(frame, args)
    resolved = _resolve_spectrum(args, minimum_length=min(segment.iq.shape[1] for segment in segments))
    blocks = compute_spectrum_blocks(
        segments,
        sample_rate_hz=frame.manifest.sample_rate_hz,
        **resolved,
    )
    blocks = _apply_spectrum_frequency_range(
        blocks,
        args=args,
        sample_rate_hz=frame.manifest.sample_rate_hz,
        center_frequency_hz=frame.manifest.center_frequency_hz,
    )
    frequency_hz, power_linear, representation = aggregate_spectrum_blocks(
        blocks,
        channel_mode=args.channel_mode,
        range_mode=args.range_mode,
        scaling=resolved["scaling"],
    )
    payload = SpectrumPayload(
        frequency_hz=frequency_hz,
        power_linear=power_linear,
        representation=representation,
        blocks=blocks,
        segment_ranges=tuple(
            (segment.sample_range.start, segment.sample_range.stop) for segment in segments
        ),
        method=resolved["method"],
        scaling=resolved["scaling"],
        sample_rate_hz=frame.manifest.sample_rate_hz,
        frequency_resolution_hz=frame.manifest.sample_rate_hz / resolved["nfft"],
    )
    peak_indices = np.argsort(power_linear)[-min(8, power_linear.size) :][::-1]
    density = resolved["scaling"] == "power_density"
    peak_level_key = "level_db_per_hz" if density else "power_db"
    peaks = [
        {
            "frequency_hz": float(frequency_hz[index]),
            peak_level_key: _db(float(power_linear[index])),
        }
        for index in peak_indices
    ]
    summary = {
        "frequency_bin_count": int(frequency_hz.size),
        "frequency_resolution_hz": payload.frequency_resolution_hz,
        "method": resolved["method"],
        "scaling": resolved["scaling"],
        "block_count": len(blocks),
    }
    artifact_id = artifact_store.put(
        "spectrum",
        args.source_ref,
        payload,
        summary,
        selection=args,
        parameters=resolved,
        meaning="Shifted baseband spectrum; numerical arrays are retained internally.",
    )
    return ToolResult(
        tool_name="compute_spectrum",
        status=ToolStatus.SUCCESS,
        data={
            "spectrum_id": artifact_id,
            "resolved_nfft": resolved["nfft"],
            "frequency_resolution_hz": payload.frequency_resolution_hz,
            ("noise_floor_db_per_hz" if density else "noise_floor_db"): _db(
                float(np.median(power_linear))
            ),
            "peaks": peaks,
        },
        semantics={
            "frequency_resolution_hz": "Hz per shifted baseband bin; zero Hz is the capture center.",
            ("noise_floor_db_per_hz" if density else "noise_floor_db"): (
                "Median stored linear power density converted to relative dB/Hz."
                if density
                else "Median stored linear power converted to relative dB with epsilon only for display."
            ),
            "peaks": (
                "Bounded strongest-bin summary in shifted baseband Hz and relative dB/Hz."
                if density
                else "Bounded strongest-bin summary in shifted baseband Hz and relative dB."
            ),
        },
        quality={"finite_power": True, "frequency_reference": "baseband"},
        artifacts=[artifact_store.get(artifact_id).summary],
        provenance={
            "resolved_parameters": resolved,
            "frequency_selection": _frequency_selection_provenance(args),
            "channel_aggregation": _channel_aggregation(args.channel_mode),
            "range_aggregation": _range_aggregation(args.range_mode),
        },
    )


def compute_time_frequency(
    args: ComputeTimeFrequencyArgs,
    *,
    repository: IQRepository,
    artifact_store: RunArtifactStore,
) -> ToolResult:
    """Compute a shifted, unpadded STFT and retain matrices in an artifact."""

    frame = repository.get(args.source_ref)
    segments = select_iq(frame, args)
    resolved = _resolve_stft(args, minimum_length=min(segment.iq.shape[1] for segment in segments))
    if args.range_mode == "aggregate" and resolved["output"] == "complex_stft":
        from cityrf.legacy.dsp.representations import InvalidRepresentationParameters

        raise InvalidRepresentationParameters(
            "Complex STFT output cannot aggregate discontinuous ranges without phase invention."
        )
    blocks = compute_stft_blocks(
        segments,
        sample_rate_hz=frame.manifest.sample_rate_hz,
        **resolved,
    )
    blocks = _apply_stft_frequency_range(
        blocks,
        args=args,
        sample_rate_hz=frame.manifest.sample_rate_hz,
        center_frequency_hz=frame.manifest.center_frequency_hz,
    )
    first_block = blocks[0]
    aggregate_power_linear = _aggregate_stft_power(blocks) if args.range_mode == "aggregate" else None
    payload = TimeFrequencyPayload(
        blocks=blocks,
        aggregate_power_linear=aggregate_power_linear,
        output=resolved["output"],
        sample_rate_hz=frame.manifest.sample_rate_hz,
        frequency_resolution_hz=frame.manifest.sample_rate_hz / resolved["nfft"],
        time_resolution_samples=resolved["hop_length"],
        window_length=resolved["window_length"],
        nfft=resolved["nfft"],
    )
    frame_count = sum(block.values.shape[1] for block in blocks)
    frequency_bin_count = first_block.values.shape[0]
    summary = {
        "frame_count": int(frame_count),
        "frequency_bin_count": int(frequency_bin_count),
        "frequency_resolution_hz": payload.frequency_resolution_hz,
        "time_resolution_samples": payload.time_resolution_samples,
        "block_count": len(blocks),
    }
    artifact_id = artifact_store.put(
        "time_frequency",
        args.source_ref,
        payload,
        summary,
        selection=args,
        parameters=resolved,
        meaning="Shifted two-sided STFT; matrices and axes are retained internally.",
    )
    return ToolResult(
        tool_name="compute_time_frequency",
        status=ToolStatus.SUCCESS,
        data={"time_frequency_id": artifact_id, **summary},
        semantics={
            "frequency_resolution_hz": "Hz per shifted two-sided baseband bin; zero Hz is capture center.",
            "time_resolution_samples": "Hop between adjacent complete STFT frame starts in original sample indices.",
            "frame_count": "Count of complete unpadded frames across independently retained source blocks.",
        },
        quality={"finite_values": True, "frequency_reference": "baseband"},
        artifacts=[artifact_store.get(artifact_id).summary],
        provenance={
            "resolved_parameters": resolved,
            "frequency_selection": _frequency_selection_provenance(args),
            "channel_aggregation": _channel_aggregation(args.channel_mode),
            "range_aggregation": _range_aggregation(args.range_mode),
            "frame_mapping": "frame starts and centers map directly to original capture sample indices.",
        },
    )


def _resolve_spectrum(args: ComputeSpectrumArgs, *, minimum_length: int) -> dict[str, object]:
    if args.profile == "custom" and args.nfft is None:
        from cityrf.legacy.dsp.representations import InvalidRepresentationParameters

        raise InvalidRepresentationParameters("custom spectrum profile requires explicit nfft.")
    profile_nfft = {
        "fast_scan": min(256, minimum_length),
        "balanced": min(1024, minimum_length),
        "high_resolution": minimum_length,
    }
    nfft = args.nfft if args.nfft is not None else profile_nfft.get(args.profile)
    assert nfft is not None
    if args.method == "welch_psd":
        segment_length = args.segment_length if args.segment_length is not None else min(nfft, minimum_length)
        overlap_fraction = args.overlap_fraction if args.overlap_fraction is not None else 0.5
        averaging = args.averaging if args.averaging is not None else "mean"
    else:
        segment_length = args.segment_length
        overlap_fraction = args.overlap_fraction
        averaging = args.averaging
    return {
        "method": args.method,
        "nfft": nfft,
        "window": args.window,
        "detrend": args.detrend,
        "scaling": args.scaling,
        "segment_length": segment_length,
        "overlap_fraction": overlap_fraction,
        "averaging": averaging,
    }


def _resolve_stft(args: ComputeTimeFrequencyArgs, *, minimum_length: int) -> dict[str, object]:
    if args.profile == "custom" and any(
        value is None for value in (args.window_length, args.hop_length, args.nfft)
    ):
        from cityrf.legacy.dsp.representations import InvalidRepresentationParameters

        raise InvalidRepresentationParameters(
            "custom STFT profile requires window_length, hop_length, and nfft."
        )
    defaults = {
        "short_transient": (min(128, minimum_length), min(32, max(1, minimum_length // 4))),
        "balanced": (min(256, minimum_length), min(64, max(1, minimum_length // 4))),
        "frequency_detail": (minimum_length, max(1, minimum_length // 4)),
    }
    default_window, default_hop = defaults.get(args.profile, (None, None))
    window_length = args.window_length if args.window_length is not None else default_window
    hop_length = args.hop_length if args.hop_length is not None else default_hop
    nfft = args.nfft if args.nfft is not None else window_length
    assert window_length is not None and hop_length is not None and nfft is not None
    return {
        "output": args.output,
        "window_length": window_length,
        "hop_length": hop_length,
        "nfft": nfft,
        "window": args.window,
        "detrend": args.detrend,
    }


def _apply_spectrum_frequency_range(
    blocks: tuple[SpectrumBlock, ...],
    *,
    args: SignalSelection,
    sample_rate_hz: float,
    center_frequency_hz: float | None,
) -> tuple[SpectrumBlock, ...]:
    lower, upper = _baseband_frequency_bounds(
        args, sample_rate_hz=sample_rate_hz, center_frequency_hz=center_frequency_hz
    )
    if lower is None:
        return blocks
    selected_blocks: list[SpectrumBlock] = []
    for block in blocks:
        mask = (block.frequency_hz >= lower) & (block.frequency_hz <= upper)
        if not np.any(mask):
            from cityrf.legacy.dsp.representations import InvalidRepresentationParameters

            raise InvalidRepresentationParameters("The requested frequency range contains no resolved bins.")
        selected_blocks.append(
            SpectrumBlock(
                frequency_hz=block.frequency_hz[mask],
                power_linear=block.power_linear[mask],
                representation=block.representation[mask],
                sample_range=block.sample_range,
                channel_indices=block.channel_indices,
            )
        )
    return tuple(selected_blocks)


def _apply_stft_frequency_range(
    blocks, *, args: SignalSelection, sample_rate_hz: float, center_frequency_hz: float | None
):
    lower, upper = _baseband_frequency_bounds(
        args, sample_rate_hz=sample_rate_hz, center_frequency_hz=center_frequency_hz
    )
    if lower is None:
        return blocks
    from cityrf.legacy.dsp.representations import InvalidRepresentationParameters, TimeFrequencyBlock

    selected_blocks = []
    for block in blocks:
        mask = (block.frequency_hz >= lower) & (block.frequency_hz <= upper)
        if not np.any(mask):
            raise InvalidRepresentationParameters("The requested frequency range contains no resolved bins.")
        selected_blocks.append(
            TimeFrequencyBlock(
                frequency_hz=block.frequency_hz[mask],
                values=block.values[mask, :],
                power_linear=block.power_linear[mask, :],
                frame_start_samples=block.frame_start_samples,
                frame_center_samples=block.frame_center_samples,
                sample_range=block.sample_range,
                channel_indices=block.channel_indices,
            )
        )
    return tuple(selected_blocks)


def _aggregate_stft_power(blocks) -> np.ndarray:
    """Average independently computed block powers without joining time samples."""

    from cityrf.legacy.dsp.representations import InvalidRepresentationParameters

    reference_frequency = blocks[0].frequency_hz
    if any(
        block.frequency_hz.shape != reference_frequency.shape
        or not np.array_equal(block.frequency_hz, reference_frequency)
        for block in blocks[1:]
    ):
        raise InvalidRepresentationParameters("STFT blocks do not share a common frequency grid.")
    aggregate = np.mean(np.stack([np.mean(block.power_linear, axis=1) for block in blocks]), axis=0)
    if not np.isfinite(aggregate).all() or np.any(aggregate < 0.0):
        raise InvalidRepresentationParameters("STFT aggregate power is invalid.")
    return aggregate


def _baseband_frequency_bounds(
    args: SignalSelection, *, sample_rate_hz: float, center_frequency_hz: float | None
) -> tuple[float | None, float | None]:
    if args.frequency_range is None:
        return None, None
    requested = args.frequency_range
    nyquist = sample_rate_hz / 2.0
    if requested.reference == "absolute_rf":
        if center_frequency_hz is None:
            from cityrf.legacy.dsp.representations import InvalidRepresentationParameters

            raise InvalidRepresentationParameters("absolute_rf frequency ranges require a known capture center frequency.")
        lower = requested.start_hz - center_frequency_hz
        upper = requested.stop_hz - center_frequency_hz
    else:
        lower, upper = requested.start_hz, requested.stop_hz
    if lower < -nyquist or upper > nyquist:
        from cityrf.legacy.dsp.representations import InvalidRepresentationParameters

        raise InvalidRepresentationParameters("The requested Hz range exceeds observable baseband Nyquist limits.")
    return lower, upper


def _frequency_selection_provenance(args: SignalSelection) -> dict[str, str]:
    if args.frequency_range is None:
        return {"reference": "baseband", "unit": "Hz"}
    return {"reference": args.frequency_range.reference, "unit": "Hz"}


def _channel_aggregation(channel_mode: str) -> str:
    return {
        "single": "single_selected_channel",
        "per_channel": "independent_channel_results",
        "mean_power": "noncoherent_mean_power",
    }[channel_mode]


def _range_aggregation(range_mode: str) -> str:
    return (
        "independent_power_average"
        if range_mode == "aggregate"
        else "independent_source_segments"
    )


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(value, np.finfo(np.float64).tiny)))
