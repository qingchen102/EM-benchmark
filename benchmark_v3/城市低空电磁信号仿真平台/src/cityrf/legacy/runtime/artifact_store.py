"""Typed, run-scoped storage for numerical artifacts kept out of ToolMessages."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
import secrets
from typing import Any, Mapping

import numpy as np
from pydantic import BaseModel

from cityrf.legacy.schemas import ArtifactSummary, SignalSelection


class ArtifactStoreError(RuntimeError):
    """Base error for internal artifact-store lifecycle failures."""


class ArtifactNotFound(ArtifactStoreError):
    """The supplied handle was never issued by any currently known store."""


class ArtifactExpired(ArtifactStoreError):
    """The supplied handle was invalidated by ``clear()``."""


class CrossRunArtifact(ArtifactStoreError):
    """The supplied handle belongs to a different run-scoped store."""


class ArtifactTypeMismatch(ArtifactStoreError):
    """The stored artifact does not have the requested type."""


class ArtifactSourceMismatch(ArtifactStoreError):
    """The requested source does not match the artifact's bound source."""


@dataclass(frozen=True, repr=False)
class StoredArtifact:
    """Internal artifact record; its numerical payload is never serialized publicly."""

    artifact_id: str
    run_id: str
    artifact_type: str
    source_ref: str
    selection: SignalSelection | None
    metadata: dict[str, Any] = field(repr=False)
    parameter_hash: str
    payload: Any = field(repr=False)
    summary: ArtifactSummary
    created_at: datetime

    def __repr__(self) -> str:
        return (
            "StoredArtifact("
            f"artifact_id={self.artifact_id!r}, artifact_type={self.artifact_type!r}, "
            f"source_ref={self.source_ref!r}, parameter_hash={self.parameter_hash!r}, "
            f"created_at={self.created_at.isoformat()!r})"
        )


class RunArtifactStore:
    """Store immutable opaque, typed artifact snapshots for one run."""

    _ARTIFACT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
    _issued_store_tokens: set[str] = set()

    def __init__(self, run_id: str):
        if not run_id:
            raise ValueError("run_id is required.")
        self.run_id = run_id
        self._store_token = secrets.token_hex(8)
        self._issued_store_tokens.add(self._store_token)
        self._records: dict[str, StoredArtifact] = {}
        self._expired_ids: set[str] = set()

    def put(
        self,
        artifact_type: str,
        source_ref: str,
        payload: Any,
        metadata: Mapping[str, Any],
        *,
        selection: SignalSelection | None = None,
        parameters: Mapping[str, Any] | None = None,
        meaning: str | None = None,
    ) -> str:
        """Store one payload and return an opaque artifact handle.

        ``metadata`` is validated through ``ArtifactSummary`` before it becomes
        visible to a future ToolMessage.  The payload itself is retained only in
        the returned internal ``StoredArtifact``.
        """

        if not self._ARTIFACT_TYPE_PATTERN.fullmatch(artifact_type):
            raise ArtifactStoreError("Artifact type is invalid.")
        if selection is not None and selection.source_ref != source_ref:
            raise ArtifactSourceMismatch("Artifact selection source does not match the artifact source.")
        payload_snapshot = _owned_snapshot(payload)
        metadata_snapshot = _owned_snapshot(dict(metadata))
        selection_snapshot = _owned_snapshot(selection)
        artifact_id = self._new_artifact_id(artifact_type)
        summary = ArtifactSummary(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            meaning=meaning or f"{artifact_type} artifact.",
            source_ref=source_ref,
            summary=_detached_view(metadata_snapshot),
        )
        record = StoredArtifact(
            artifact_id=artifact_id,
            run_id=self.run_id,
            artifact_type=artifact_type,
            source_ref=source_ref,
            selection=selection_snapshot,
            metadata=metadata_snapshot,
            parameter_hash=self.parameter_hash(parameters if parameters is not None else metadata_snapshot),
            payload=payload_snapshot,
            summary=_owned_snapshot(summary),
            created_at=datetime.now(timezone.utc),
        )
        self._records[artifact_id] = record
        return artifact_id

    def get(
        self,
        artifact_id: str,
        *,
        expected_type: str | None = None,
        expected_source_ref: str | None = None,
    ) -> StoredArtifact:
        """Get a detached record view without serializing its typed payload."""

        record = self._records.get(artifact_id)
        if record is None:
            if artifact_id in self._expired_ids:
                raise ArtifactExpired("The artifact handle has expired.")
            if self._belongs_to_another_store(artifact_id):
                raise CrossRunArtifact("The artifact handle belongs to a different run.")
            raise ArtifactNotFound("The artifact handle is unknown.")
        if expected_type is not None and record.artifact_type != expected_type:
            raise ArtifactTypeMismatch("The artifact type does not match the requested type.")
        if expected_source_ref is not None and record.source_ref != expected_source_ref:
            raise ArtifactSourceMismatch("The artifact source does not match the requested source.")
        detached = _detached_view(record)
        if not isinstance(detached, StoredArtifact):
            raise ArtifactStoreError("Stored artifact state is invalid.")
        return detached

    def clear(self) -> None:
        """Invalidate every handle issued by this store and discard its payloads."""

        self._expired_ids.update(self._records)
        self._records.clear()

    @staticmethod
    def parameter_hash(parameters: Mapping[str, Any]) -> str:
        """Return a stable SHA-256 hash of JSON-safe resolved parameters."""

        try:
            canonical = json.dumps(
                parameters,
                default=RunArtifactStore._json_default,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ArtifactStoreError("Artifact parameters must be finite JSON-safe values.") from error
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite parameter value")
        raise TypeError("Unsupported artifact parameter")

    def _new_artifact_id(self, artifact_type: str) -> str:
        while True:
            artifact_id = f"{artifact_type}_{self._store_token}_{secrets.token_hex(12)}"
            if artifact_id not in self._records and artifact_id not in self._expired_ids:
                return artifact_id

    def _belongs_to_another_store(self, artifact_id: str) -> bool:
        parts = artifact_id.rsplit("_", 2)
        return len(parts) == 3 and parts[1] in self._issued_store_tokens and parts[1] != self._store_token


def _owned_snapshot(value: Any, *, active_ids: set[int] | None = None) -> Any:
    """Detach accepted payload graphs; NumPy arrays become private read-only owners."""

    return _snapshot_value(value, detached=False, active_ids=active_ids)


def _detached_view(value: Any, *, active_ids: set[int] | None = None) -> Any:
    """Return caller-safe views/copies without exposing a mutable stored owner."""

    return _snapshot_value(value, detached=True, active_ids=active_ids)


def _snapshot_value(value: Any, *, detached: bool, active_ids: set[int] | None) -> Any:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, Enum, datetime)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ArtifactStoreError("Artifact arrays must not contain object references.")
        if detached:
            view = value.view()
            view.setflags(write=False)
            return view
        return _immutable_bytes_backed_array(value)

    active = active_ids if active_ids is not None else set()
    object_id = id(value)
    if object_id in active:
        raise ArtifactStoreError("Cyclic artifact payloads are not supported.")

    if isinstance(value, BaseModel):
        active.add(object_id)
        try:
            snapshot = value.model_copy(deep=False)
            field_values = _snapshot_value(value.__dict__, detached=detached, active_ids=active)
            extra_values = _snapshot_value(
                value.__pydantic_extra__, detached=detached, active_ids=active
            )
            private_values = _snapshot_value(
                value.__pydantic_private__, detached=detached, active_ids=active
            )
            snapshot.__dict__.clear()
            snapshot.__dict__.update(field_values)
            object.__setattr__(snapshot, "__pydantic_extra__", extra_values)
            object.__setattr__(snapshot, "__pydantic_private__", private_values)
            object.__setattr__(snapshot, "__pydantic_fields_set__", set(value.model_fields_set))
            return snapshot
        finally:
            active.remove(object_id)

    if is_dataclass(value) and not isinstance(value, type):
        active.add(object_id)
        try:
            try:
                snapshot = object.__new__(type(value))
                for item in fields(value):
                    field_value = _snapshot_value(
                        getattr(value, item.name), detached=detached, active_ids=active
                    )
                    object.__setattr__(snapshot, item.name, field_value)
            except ArtifactStoreError:
                raise
            except Exception as error:
                raise ArtifactStoreError("Artifact value could not be snapshotted safely.") from error
            return snapshot
        finally:
            active.remove(object_id)

    if isinstance(value, tuple):
        active.add(object_id)
        try:
            items = tuple(
                _snapshot_value(item, detached=detached, active_ids=active) for item in value
            )
            if type(value) is tuple:
                return items
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return type(value)(items)
        finally:
            active.remove(object_id)

    if isinstance(value, frozenset):
        active.add(object_id)
        try:
            return type(value)(
                _snapshot_value(item, detached=detached, active_ids=active) for item in value
            )
        finally:
            active.remove(object_id)

    if isinstance(value, MappingABC):
        active.add(object_id)
        try:
            return {
                _snapshot_value(key, detached=detached, active_ids=active): _snapshot_value(
                    item, detached=detached, active_ids=active
                )
                for key, item in value.items()
            }
        finally:
            active.remove(object_id)

    if isinstance(value, list):
        active.add(object_id)
        try:
            return [
                _snapshot_value(item, detached=detached, active_ids=active) for item in value
            ]
        finally:
            active.remove(object_id)

    if isinstance(value, set):
        active.add(object_id)
        try:
            return {
                _snapshot_value(item, detached=detached, active_ids=active) for item in value
            }
        finally:
            active.remove(object_id)

    if type(value) is object:
        return object()
    raise ArtifactStoreError("Artifact value could not be snapshotted safely.")


def _immutable_bytes_backed_array(array: np.ndarray) -> np.ndarray:
    """Copy numerical values behind an immutable buffer with no stored-owner alias."""

    order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
    buffer = array.tobytes(order=order)
    flat = np.frombuffer(buffer, dtype=array.dtype, count=array.size)
    return flat.reshape(array.shape, order=order)
