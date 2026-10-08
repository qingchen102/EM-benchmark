"""Task-specific output definitions and stable agent-run envelopes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class TaskSpec(BaseModel):
    """Caller-defined instruction and output contract for an agent run."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    output_mode: Literal["text", "json"] = "text"
    output_schema: dict[str, JsonValue] | None = None
    output_semantics: dict[str, JsonValue] = Field(default_factory=dict)
    benchmark_policy: dict[str, JsonValue] = Field(default_factory=dict)
    validation: Literal["json_schema", "none"] | None = None

    @model_validator(mode="after")
    def validate_json_schema(self) -> "TaskSpec":
        if self.output_mode == "json" and self.output_schema is None:
            raise ValueError("output_schema is required when output_mode is 'json'")
        if self.output_mode == "json" and self.validation == "none":
            raise ValueError("JSON tasks must use json_schema validation")
        if self.output_mode == "json":
            self.validation = "json_schema"
        return self


class AgentRunResult(BaseModel):
    """Stable harness envelope independent of the task-specific output shape."""

    model_config = ConfigDict(extra="forbid")

    run_status: Literal["success", "partial", "error"]
    final_content: str = ""
    parsed_output: JsonValue | None = None
    validation_errors: list[dict[str, JsonValue]] = Field(default_factory=list)
    trace_path: str | None = None
