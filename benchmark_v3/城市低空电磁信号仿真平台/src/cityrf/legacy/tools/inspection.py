"""Safe inspection executor for one opaque, repository-owned IQ capture."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cityrf.legacy.runtime import IQRepository
from cityrf.legacy.schemas import ToolResult, ToolStatus


class InspectIQArgs(BaseModel):
    """Arguments for inspection; dependencies remain injected and non-public."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)


def inspect_iq(args: InspectIQArgs, *, repository: IQRepository) -> ToolResult:
    """Return concise acquisition facts and tool availability without raw IQ."""

    frame = repository.get(args.source_ref)
    iq = frame.iq
    channel_power = np.mean(np.abs(iq) ** 2, axis=1)
    dc = np.mean(iq, axis=1)
    peak_power = np.max(np.abs(iq) ** 2, axis=1)
    valid_power = channel_power > 0.0
    papr_db = (
        float(10.0 * np.log10(np.max(peak_power[valid_power] / channel_power[valid_power])))
        if np.any(valid_power)
        else None
    )
    magnitudes = np.abs(iq)
    global_maximum = float(np.max(magnitudes))
    repeated_peak_count = int(np.count_nonzero(np.isclose(magnitudes, global_maximum, rtol=0.0, atol=0.0)))
    clipping_suspected = global_maximum > 0.0 and repeated_peak_count >= max(4, iq.shape[1] // 100)

    geometry = frame.manifest.array_geometry
    observed_channel_count = int(iq.shape[0])
    geometry_supports_channels = bool(
        geometry
        and (
            len(geometry.element_positions_m) == observed_channel_count
            or (
                geometry.geometry_type == "ula"
                and geometry.element_spacing_m is not None
                and geometry.element_count is not None
                and geometry.element_count == observed_channel_count
            )
        )
    )
    synchronized_value = (
        geometry.synchronized
        if geometry is not None and geometry.synchronized is not None
        else frame.manifest.synchronized
    )
    synchronized = synchronized_value is True
    reference_frequency_available = any(
        value is not None
        for value in (
            geometry.reference_frequency_hz if geometry is not None else None,
            frame.manifest.reference_frequency_hz,
            frame.manifest.center_frequency_hz,
        )
    )
    doa_available = bool(
        observed_channel_count >= 2
        and geometry_supports_channels
        and synchronized
        and reference_frequency_available
    )
    missing_requirements: list[str] = []
    if observed_channel_count < 2:
        missing_requirements.append("DOA requires at least two channels.")
    if not geometry_supports_channels:
        missing_requirements.append(
            "DOA requires usable array geometry with a channel-to-element mapping for the observed channel count."
        )
    if not synchronized:
        missing_requirements.append("DOA requires synchronized channels.")
    if not reference_frequency_available:
        missing_requirements.append("DOA requires a usable reference/carrier frequency.")
    protected = frame.manifest.protected_channel
    protected_data = (
        {
            "available": protected is not None,
            "center_offset_hz": protected.center_offset_hz if protected else None,
            "nominal_bandwidth_hz": protected.nominal_bandwidth_hz if protected else None,
            "expected_modulation": protected.expected_modulation if protected else None,
        }
    )
    return ToolResult(
        tool_name="inspect_iq",
        status=ToolStatus.SUCCESS,
        data={
            "iq_shape": [int(iq.shape[0]), int(iq.shape[1])],
            "iq_dtype": str(iq.dtype),
            "channel_count": int(iq.shape[0]),
            "sample_count": int(iq.shape[1]),
            "all_finite": bool(np.isfinite(iq.real).all() and np.isfinite(iq.imag).all()),
            "mean_power_linear": float(np.mean(channel_power)),
            "dc_offset": {"real": float(np.mean(dc.real)), "imag": float(np.mean(dc.imag))},
            "papr_db": papr_db,
            "clipping": {
                "suspected": clipping_suspected,
                "max_magnitude": global_maximum,
                "repeated_peak_count": repeated_peak_count,
            },
            "acquisition_priors": {
                "sample_rate_hz": float(frame.manifest.sample_rate_hz),
                "center_frequency_hz": frame.manifest.center_frequency_hz,
                "expected_desired_source_count": frame.manifest.expected_desired_source_count,
            },
            "protected_channel": protected_data,
            "array_availability": {
                "present": geometry is not None,
                "usable": geometry_supports_channels,
                "supports_observed_channel_count": geometry_supports_channels,
            },
            "synchronization_available": synchronized,
            "reference_frequency_available": reference_frequency_available,
            "capabilities": {
                "compute_spectrum": True,
                "compute_time_frequency": True,
                "spatial_evidence": doa_available,
                "doa": doa_available,
            },
            "missing_requirements": missing_requirements,
        },
        semantics={
            "mean_power_linear": "Mean squared complex-envelope magnitude in capture units.",
            "dc_offset": "Capture-wide complex mean in capture units.",
            "papr_db": "Maximum per-channel peak-to-average power ratio in dB.",
            "clipping": "Repeated exact peak magnitudes are an uncalibrated clipping indicator.",
        },
        quality={"finite_input": True, "channel_power_count": int(channel_power.size)},
        provenance={"inspection": "fixed_basic_checks_v1"},
    )
