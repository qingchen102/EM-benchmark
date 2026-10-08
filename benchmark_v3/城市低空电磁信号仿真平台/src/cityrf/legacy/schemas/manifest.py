"""External IQ manifest and acquisition-prior schemas."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class ProtectedChannel(BaseModel):
    """Public prior for the protected channel, never interference ground truth."""

    model_config = ConfigDict(extra="forbid")

    center_offset_hz: float
    nominal_bandwidth_hz: float = Field(gt=0)
    expected_modulation: str | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class ArrayGeometry(BaseModel):
    """Array geometry supplied with an IQ sample when spatial analysis is available."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    geometry_type: Literal["ula", "custom"] = Field(
        default="ula", validation_alias=AliasChoices("geometry_type", "type")
    )
    element_count: int | None = Field(default=None, gt=0)
    element_positions_m: list[float] = Field(
        default_factory=list,
        validation_alias=AliasChoices("element_positions_m", "positions_m"),
    )
    element_spacing_m: float | None = Field(default=None, gt=0)
    reference_frequency_hz: float | None = Field(default=None, gt=0)
    synchronized: bool | None = None
    calibration: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("element_spacing_m", "reference_frequency_hz")
    @classmethod
    def validate_finite_positive_scalars(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("array geometry scalar values must be finite positive values")
        return value

    @field_validator("element_positions_m")
    @classmethod
    def validate_finite_positions(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("array geometry positions must be finite")
        return value

    @model_validator(mode="after")
    def validate_element_count(self) -> "ArrayGeometry":
        if self.element_count is not None and self.element_positions_m:
            if self.element_count != len(self.element_positions_m):
                raise ValueError("element_count must match element_positions_m")
        if self.geometry_type == "ula" and self.element_positions_m:
            positions = self.element_positions_m
            if any(right <= left for left, right in zip(positions, positions[1:])):
                raise ValueError("ULA element_positions_m must be strictly increasing")
        return self


class IQSampleManifest(BaseModel):
    """Authorized external description of a single IQ capture."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_id: str = Field(min_length=1)
    iq_path: str = Field(min_length=1)
    iq_format: str = Field(
        default="npy", validation_alias=AliasChoices("iq_format", "format")
    )
    dtype: str = Field(min_length=1)
    layout: str = Field(min_length=1)
    sample_rate_hz: float = Field(gt=0)
    center_frequency_hz: float | None = None
    protected_channel: ProtectedChannel | None = None
    expected_desired_source_count: int | None = Field(default=None, ge=0)
    array_geometry: ArrayGeometry | None = None
    reference_frequency_hz: float | None = Field(default=None, gt=0)
    synchronized: bool | None = None
    calibration: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("sample_rate_hz", "reference_frequency_hz")
    @classmethod
    def validate_finite_positive_frequencies(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("capture frequency values must be finite positive values")
        return value

    @field_validator("center_frequency_hz")
    @classmethod
    def validate_finite_center_frequency(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("capture center frequency must be finite")
        return value
