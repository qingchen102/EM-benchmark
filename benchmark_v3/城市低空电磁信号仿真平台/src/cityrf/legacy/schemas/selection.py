"""Controlled sample, channel, and frequency selection schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SampleRange(BaseModel):
    """A non-empty half-open sample interval ``[start, stop)``."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    stop: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_non_empty(self) -> "SampleRange":
        if self.stop <= self.start:
            raise ValueError("stop must be greater than start for a non-empty [start, stop) range")
        return self

    @property
    def length(self) -> int:
        return self.stop - self.start


class FrequencyRange(BaseModel):
    """A non-empty frequency interval with an explicit reference frame."""

    model_config = ConfigDict(extra="forbid")

    start_hz: float
    stop_hz: float
    reference: Literal["baseband", "capture_center", "absolute_rf"]

    @model_validator(mode="after")
    def validate_non_empty(self) -> "FrequencyRange":
        if self.stop_hz <= self.start_hz:
            raise ValueError("stop_hz must be greater than start_hz")
        return self

    @property
    def bandwidth_hz(self) -> float:
        return self.stop_hz - self.start_hz


class SignalSelection(BaseModel):
    """Selection shared by controlled DSP tools for one registered source."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)
    sample_ranges: list[SampleRange] = Field(default_factory=list)
    range_mode: Literal["independent", "aggregate"] = "independent"
    channel_indices: list[int] = Field(default_factory=list)
    channel_mode: Literal["single", "per_channel", "mean_power"] = "single"
    frequency_range: FrequencyRange | None = None

    @model_validator(mode="after")
    def validate_channels(self) -> "SignalSelection":
        if any(index < 0 for index in self.channel_indices):
            raise ValueError("channel_indices must be non-negative")
        if len(set(self.channel_indices)) != len(self.channel_indices):
            raise ValueError("channel_indices must not contain duplicates")
        return self
