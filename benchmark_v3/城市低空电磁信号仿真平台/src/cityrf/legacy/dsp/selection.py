"""Validated IQ selection without altering channel phase or raw range boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cityrf.legacy.runtime.iq_repository import IQFrame
from cityrf.legacy.schemas import SampleRange, SignalSelection


class InvalidSelection(ValueError):
    """The requested source, sample range, or channels are not available."""


@dataclass(frozen=True, repr=False)
class SelectedSegment:
    """One selected IQ interval, retaining its original half-open sample range.

    ``mean_power`` is intentionally only a policy marker.  The original complex
    channel IQ is retained so later spectral tools can aggregate channel powers
    non-coherently rather than phase-averaging samples here.
    """

    source_ref: str
    iq: np.ndarray = field(repr=False)
    sample_range: SampleRange
    channel_indices: tuple[int, ...]
    range_mode: Literal["independent", "aggregate"]
    channel_mode: Literal["single", "per_channel", "mean_power"]
    aggregation_policy: Literal[
        "independent_results", "aggregate_results_without_raw_iq_concatenation"
    ]

    def __repr__(self) -> str:
        return (
            "SelectedSegment("
            f"source_ref={self.source_ref!r}, sample_range={self.sample_range!r}, "
            f"channel_indices={self.channel_indices!r}, range_mode={self.range_mode!r}, "
            f"channel_mode={self.channel_mode!r})"
        )


def select_iq(frame: IQFrame, selection: SignalSelection) -> list[SelectedSegment]:
    """Return one typed segment per requested range without raw-IQ concatenation."""

    if selection.source_ref != frame.source_ref:
        raise InvalidSelection("The selection source reference does not match the IQ frame.")

    channel_count, sample_count = frame.iq.shape
    channel_indices = tuple(selection.channel_indices) if selection.channel_indices else tuple(range(channel_count))
    if any(index >= channel_count for index in channel_indices):
        raise InvalidSelection("A requested channel index is out of bounds.")

    ranges = selection.sample_ranges or [SampleRange(start=0, stop=sample_count)]
    if any(sample_range.stop > sample_count for sample_range in ranges):
        raise InvalidSelection("A requested sample range is out of bounds.")

    aggregation_policy: Literal[
        "independent_results", "aggregate_results_without_raw_iq_concatenation"
    ] = (
        "aggregate_results_without_raw_iq_concatenation"
        if selection.range_mode == "aggregate"
        else "independent_results"
    )
    return [
        SelectedSegment(
            source_ref=frame.source_ref,
            iq=frame.iq[np.asarray(channel_indices), sample_range.start : sample_range.stop],
            sample_range=sample_range,
            channel_indices=channel_indices,
            range_mode=selection.range_mode,
            channel_mode=selection.channel_mode,
            aggregation_policy=aggregation_policy,
        )
        for sample_range in ranges
    ]
