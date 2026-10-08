"""Safe, JSON-serializable result envelopes for controlled tool calls."""

from __future__ import annotations

from enum import StrEnum
import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


_MAX_TEXT_LENGTH = 512
_MAX_MAPPING_ENTRIES = 24
_MAX_SEQUENCE_ITEMS = 16
_MAX_NESTING_DEPTH = 4
_SENSITIVE_KEY_MARKERS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
    "cookie",
)
_RAW_PAYLOAD_KEYS = {
    "array",
    "covariance",
    "covariancematrix",
    "iq",
    "iqdata",
    "iqsamples",
    "matrix",
    "payload",
    "payloaddata",
    "psd",
    "psdvalues",
    "rawiq",
    "rawsamples",
    "samples",
    "signal",
    "signaldata",
    "spectrum",
    "spectrumvalues",
    "stft",
    "stftvalues",
    "waveform",
    "waveformdata",
}
_FILESYSTEM_KEY_SUFFIXES = {
    "directory",
    "file",
    "filename",
    "folder",
    "location",
    "path",
    "root",
    "uri",
    "url",
}
_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:\b[A-Za-z]:[\\/]|(?:^|[\s'\"(])/(?:[^\s/]+/)+[^\s]+|(?:^|[\s'\"(])\\\\[^\s\\/]+[\\/])"
)
_FILE_EXTENSION_PATTERN = r"(?:bin|csv|h5|hdf5|iq|json|jsonl|mat|npy|npz|png|tsv|txt|wav)"
_RELATIVE_FILE_PATTERN = re.compile(
    rf"^(?![A-Za-z][A-Za-z0-9+.-]*://)(?:\.{1,2}[\\/])?(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.{_FILE_EXTENSION_PATTERN}$",
    re.IGNORECASE,
)
_BARE_FILE_PATTERN = re.compile(
    rf"^[A-Za-z0-9_.-]+\.{_FILE_EXTENSION_PATTERN}$", re.IGNORECASE
)
_LOCAL_FILE_URI_PATTERN = re.compile(r"^file://", re.IGNORECASE)
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:\b(?:[a-z][a-z0-9_-]*_)?(?:api[_-]?key|token|secret|password|credential)\s*[:=]|\bauthorization\s*[:=]|\bbearer\s+\S+|\bsk-[a-z0-9_-]{8,})"
)
_TRACEBACK_PATTERN = re.compile(
    r"(?i)(?:\btraceback\b|\bstack\s+trace\b|\bfile\s+['\"][^'\"]+['\"],\s*line\s+\d+)"
)


def _reject_unsafe_content() -> None:
    """Fail closed without placing rejected content in a validation message."""

    raise ValueError("unsafe LLM-visible content")


def _validate_safe_text(value: str) -> None:
    if (
        len(value) > _MAX_TEXT_LENGTH
        or _ABSOLUTE_PATH_PATTERN.search(value)
        or _RELATIVE_FILE_PATTERN.fullmatch(value)
        or _BARE_FILE_PATTERN.fullmatch(value)
        or _LOCAL_FILE_URI_PATTERN.match(value)
        or _SECRET_VALUE_PATTERN.search(value)
        or _TRACEBACK_PATTERN.search(value)
    ):
        _reject_unsafe_content()


def _validate_safe_key(value: str) -> None:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if not value or any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
        _reject_unsafe_content()
    tokenized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value).lower()
    key_tokens = re.findall(r"[a-z0-9]+", tokenized)
    if key_tokens and key_tokens[-1] in _FILESYSTEM_KEY_SUFFIXES:
        _reject_unsafe_content()
    if normalized in _RAW_PAYLOAD_KEYS or normalized.startswith("raw"):
        _reject_unsafe_content()


def _validate_llm_visible_value(value: Any, depth: int = 0) -> None:
    """Allow concise JSON summaries while rejecting sensitive or raw payload content."""

    if depth > _MAX_NESTING_DEPTH:
        _reject_unsafe_content()
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _reject_unsafe_content()
        return
    if isinstance(value, str):
        _validate_safe_text(value)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_MAPPING_ENTRIES:
            _reject_unsafe_content()
        for key, nested_value in value.items():
            if not isinstance(key, str):
                _reject_unsafe_content()
            _validate_safe_key(key)
            _validate_llm_visible_value(nested_value, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_SEQUENCE_ITEMS:
            _reject_unsafe_content()
        for nested_value in value:
            _validate_llm_visible_value(nested_value, depth + 1)
        return
    _reject_unsafe_content()


def _validate_llm_visible_mapping(value: Any) -> Any:
    if not isinstance(value, dict):
        _reject_unsafe_content()
    _validate_llm_visible_value(value)
    return value


def _validate_llm_visible_provenance(value: Any) -> Any:
    """Reserve one bounded registry-owned entry without widening other mappings."""

    if not isinstance(value, dict):
        _reject_unsafe_content()
    if len(value) <= _MAX_MAPPING_ENTRIES:
        _validate_llm_visible_value(value)
        return value
    if (
        len(value) != _MAX_MAPPING_ENTRIES + 1
        or type(value.get("cache_hit")) is not bool
    ):
        _reject_unsafe_content()
    for key, nested_value in value.items():
        if not isinstance(key, str):
            _reject_unsafe_content()
        _validate_safe_key(key)
        _validate_llm_visible_value(nested_value, 1)
    return value


def _validate_llm_visible_text(value: Any) -> Any:
    if not isinstance(value, str):
        _reject_unsafe_content()
    _validate_safe_text(value)
    return value


class ToolStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class ToolError(BaseModel):
    """Structured, safe-to-display error details without tracebacks or raw paths."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    repair_hint: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("code", "message", "repair_hint", mode="before")
    @classmethod
    def validate_safe_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return _validate_llm_visible_text(value)

    @field_validator("details", mode="before")
    @classmethod
    def validate_safe_details(cls, value: Any) -> Any:
        return _validate_llm_visible_mapping(value)


class ArtifactSummary(BaseModel):
    """LLM-visible artifact metadata; numerical payloads remain in the artifact store."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    source_ref: str | None = None
    summary: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("artifact_id", "artifact_type", "meaning", "source_ref", mode="before")
    @classmethod
    def validate_safe_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return _validate_llm_visible_text(value)

    @field_validator("summary", mode="before")
    @classmethod
    def validate_safe_summary(cls, value: Any) -> Any:
        return _validate_llm_visible_mapping(value)


class ToolResult(BaseModel):
    """Uniform, JSON-safe response emitted by every controlled electromagnetic tool."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    schema_version: str = "1.0"
    tool_name: str = Field(min_length=1)
    status: ToolStatus
    data: dict[str, JsonValue] = Field(default_factory=dict)
    semantics: dict[str, JsonValue] = Field(default_factory=dict)
    quality: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    error: ToolError | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("schema_version", "tool_name", mode="before")
    @classmethod
    def validate_safe_text(cls, value: Any) -> Any:
        return _validate_llm_visible_text(value)

    @field_validator("data", "semantics", "quality", mode="before")
    @classmethod
    def validate_safe_mappings(cls, value: Any) -> Any:
        return _validate_llm_visible_mapping(value)

    @field_validator("provenance", mode="before")
    @classmethod
    def validate_safe_provenance(cls, value: Any) -> Any:
        return _validate_llm_visible_provenance(value)

    @field_validator("warnings", mode="before")
    @classmethod
    def validate_safe_warnings(cls, value: Any) -> Any:
        if not isinstance(value, list):
            _reject_unsafe_content()
        _validate_llm_visible_value(value)
        return value
