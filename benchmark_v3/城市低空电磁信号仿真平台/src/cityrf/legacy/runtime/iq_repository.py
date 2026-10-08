"""Authorized, in-memory loading of validated IQ captures.

The supported layout vocabulary is deliberately small and explicit:

* ``samples``: a one-dimensional capture containing samples only;
* ``channel_first``: a two-dimensional capture already shaped ``[channel, sample]``;
* ``sample_first``: a two-dimensional capture shaped ``[sample, channel]``.

No dimension-based orientation inference is performed.  Filesystem paths and raw
arrays remain internal to this module; callers receive only an opaque source ref.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import secrets
from typing import Iterable

import numpy as np

from cityrf.legacy.schemas import IQSampleManifest


class IQRepositoryError(RuntimeError):
    """Base error for authorized IQ repository operations."""


class UnsupportedIQFormat(IQRepositoryError):
    """The manifest or filename is outside the MVP's ``.npy`` format."""


class UnauthorizedIQPath(IQRepositoryError):
    """The resolved capture path is outside every configured allowed root."""


class IQLayoutError(IQRepositoryError):
    """The declared layout is incompatible with the IQ array rank."""


class IQDataValidationError(IQRepositoryError):
    """The loaded capture is not finite, complex IQ data."""


class IQDtypeError(IQRepositoryError):
    """The manifest dtype is invalid or differs from the loaded IQ dtype."""


class UnknownSourceReference(IQRepositoryError):
    """The opaque source reference was never issued by this repository."""


class SourceReferenceExpired(IQRepositoryError):
    """The opaque source reference was invalidated by repository cleanup."""


@dataclass(frozen=True, repr=False)
class IQFrame:
    """Internal normalized IQ capture in canonical ``[channel, sample]`` order."""

    source_ref: str
    iq: np.ndarray = field(repr=False)
    manifest: IQSampleManifest = field(repr=False)
    public_metadata: dict[str, object]

    def __repr__(self) -> str:
        return f"IQFrame(source_ref={self.source_ref!r}, shape={self.iq.shape!r}, dtype={self.iq.dtype!s})"


class IQRepository:
    """Own only explicitly allowlisted, validated IQ captures for one run."""

    _LAYOUT_SAMPLES = "samples"
    _LAYOUT_CHANNEL_FIRST = "channel_first"
    _LAYOUT_SAMPLE_FIRST = "sample_first"

    def __init__(self, allowed_roots: Iterable[str | Path]):
        roots = tuple(Path(root).resolve() for root in allowed_roots)
        if not roots:
            raise ValueError("At least one allowed IQ root is required.")
        self._allowed_roots = roots
        self._frames: dict[str, IQFrame] = {}
        self._expired_refs: set[str] = set()

    def register(self, manifest: IQSampleManifest) -> str:
        """Validate and register a manifest, returning an opaque source reference."""

        declared_dtype = self._parse_declared_dtype(manifest.dtype)
        path = self._authorized_npy_path(manifest)
        try:
            loaded = np.load(path, allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise IQDataValidationError("IQ data could not be loaded safely.") from error

        if not isinstance(loaded, np.ndarray):
            raise IQDataValidationError("IQ data must be a NumPy array.")
        iq = self._normalize_iq(loaded, manifest.layout, declared_dtype)
        manifest_snapshot = manifest.model_copy(deep=True)
        source_ref = self._new_source_ref()
        self._frames[source_ref] = IQFrame(
            source_ref=source_ref,
            iq=iq,
            manifest=manifest_snapshot,
            public_metadata=self._public_metadata(source_ref, iq, manifest_snapshot),
        )
        return source_ref

    def get(self, source_ref: str) -> IQFrame:
        """Return a detached view for a currently registered opaque reference."""

        frame = self._frames.get(source_ref)
        if frame is not None:
            iq_view = frame.iq.view()
            iq_view.setflags(write=False)
            return IQFrame(
                source_ref=frame.source_ref,
                iq=iq_view,
                manifest=frame.manifest.model_copy(deep=True),
                public_metadata=deepcopy(frame.public_metadata),
            )
        if source_ref in self._expired_refs:
            raise SourceReferenceExpired("The IQ source reference has expired.")
        raise UnknownSourceReference("The IQ source reference is unknown.")

    def clear(self) -> None:
        """Invalidate every source reference issued by this repository."""

        self._expired_refs.update(self._frames)
        self._frames.clear()

    def _authorized_npy_path(self, manifest: IQSampleManifest) -> Path:
        if manifest.iq_format != "npy":
            raise UnsupportedIQFormat("Only the npy IQ format is supported.")
        path = Path(manifest.iq_path).resolve()
        if path.suffix != ".npy":
            raise UnsupportedIQFormat("Only .npy IQ files are supported.")
        if not any(path == root or root in path.parents for root in self._allowed_roots):
            raise UnauthorizedIQPath("The IQ capture is outside the authorized roots.")
        return path

    @classmethod
    def _normalize_iq(cls, array: np.ndarray, layout: str, declared_dtype: np.dtype) -> np.ndarray:
        if not np.issubdtype(array.dtype, np.complexfloating):
            raise IQDataValidationError("IQ data must use a complex dtype.")
        if array.dtype != declared_dtype:
            raise IQDtypeError("Loaded IQ dtype does not match the manifest declaration.")
        if array.ndim == 1:
            if layout != cls._LAYOUT_SAMPLES:
                raise IQLayoutError("One-dimensional IQ must declare the samples layout.")
            iq = array[np.newaxis, :]
        elif array.ndim == 2:
            if layout == cls._LAYOUT_CHANNEL_FIRST:
                iq = array
            elif layout == cls._LAYOUT_SAMPLE_FIRST:
                iq = array.T
            else:
                raise IQLayoutError("Two-dimensional IQ must declare channel_first or sample_first.")
        else:
            raise IQLayoutError("IQ data must be one- or two-dimensional.")

        if iq.shape[0] == 0 or iq.shape[1] == 0:
            raise IQDataValidationError("IQ data must contain channels and samples.")
        if not np.isfinite(iq.real).all() or not np.isfinite(iq.imag).all():
            raise IQDataValidationError("IQ data must contain only finite values.")
        return _immutable_bytes_backed_array(iq, declared_dtype)

    @staticmethod
    def _parse_declared_dtype(value: str) -> np.dtype:
        try:
            declared_dtype = np.dtype(value)
        except (TypeError, ValueError) as error:
            raise IQDtypeError("Manifest IQ dtype is invalid.") from error
        if not np.issubdtype(declared_dtype, np.complexfloating):
            raise IQDtypeError("Manifest IQ dtype must be complex.")
        return declared_dtype

    @staticmethod
    def _public_metadata(
        source_ref: str, iq: np.ndarray, manifest: IQSampleManifest
    ) -> dict[str, object]:
        """Return concise metadata permitted for later Agent-visible summaries."""

        metadata: dict[str, object] = {
            "source_ref": source_ref,
            "iq_shape": [int(iq.shape[0]), int(iq.shape[1])],
            "iq_dtype": str(iq.dtype),
            "sample_rate_hz": manifest.sample_rate_hz,
        }
        if manifest.center_frequency_hz is not None:
            metadata["center_frequency_hz"] = manifest.center_frequency_hz
        return metadata

    def _new_source_ref(self) -> str:
        while True:
            source_ref = f"iq_{secrets.token_hex(12)}"
            if source_ref not in self._frames and source_ref not in self._expired_refs:
                return source_ref


def _immutable_bytes_backed_array(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Create one C-order snapshot whose complete base chain ends in bytes."""

    buffer = array.tobytes(order="C")
    flat = np.frombuffer(buffer, dtype=dtype, count=array.size)
    return flat.reshape(array.shape, order="C")
