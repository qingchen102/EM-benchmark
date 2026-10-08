"""DSP contracts shared by later controlled electromagnetic tools."""

from .representations import (
    InsufficientRepresentationSamples,
    InvalidRepresentationParameters,
    NonFiniteRepresentationResult,
    SpectrumPayload,
    TimeFrequencyPayload,
)
from .selection import InvalidSelection, SelectedSegment, select_iq

__all__ = [
    "InsufficientRepresentationSamples",
    "InvalidRepresentationParameters",
    "InvalidSelection",
    "NonFiniteRepresentationResult",
    "SelectedSegment",
    "SpectrumPayload",
    "TimeFrequencyPayload",
    "select_iq",
]
