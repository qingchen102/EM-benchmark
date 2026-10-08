"""Authoritative, shared schemas for the interference-agent MVP."""

from .manifest import ArrayGeometry, IQSampleManifest, ProtectedChannel
from .selection import FrequencyRange, SampleRange, SignalSelection
from .task import AgentRunResult, TaskSpec
from .tool_result import ArtifactSummary, ToolError, ToolResult, ToolStatus

__all__ = [
    "AgentRunResult",
    "ArrayGeometry",
    "ArtifactSummary",
    "FrequencyRange",
    "IQSampleManifest",
    "ProtectedChannel",
    "SampleRange",
    "SignalSelection",
    "TaskSpec",
    "ToolError",
    "ToolResult",
    "ToolStatus",
]
