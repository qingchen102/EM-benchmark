"""Deterministic IQ fixtures shared by representation-tool tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from cityrf.legacy.runtime import IQRepository
from cityrf.legacy.schemas import ArrayGeometry, IQSampleManifest, ProtectedChannel


SAMPLE_RATE_HZ = 1_024.0


@pytest.fixture
def register_iq(tmp_path: Path) -> Callable[..., tuple[IQRepository, str]]:
    """Persist a declared capture and return its repository plus opaque handle."""

    def register(
        name: str,
        iq: np.ndarray,
        *,
        sample_rate_hz: float = SAMPLE_RATE_HZ,
        center_frequency_hz: float | None = 2_400_000_000.0,
        protected_channel: ProtectedChannel | None = None,
        array_geometry: ArrayGeometry | None = None,
        expected_desired_source_count: int | None = None,
        synchronized: bool | None = None,
        reference_frequency_hz: float | None = None,
    ) -> tuple[IQRepository, str]:
        path = tmp_path / f"{name}.npy"
        stored = np.asarray(iq, dtype=np.complex64)
        np.save(path, stored)
        layout = "samples" if stored.ndim == 1 else "channel_first"
        manifest = IQSampleManifest(
            sample_id=f"fixture_{name}",
            iq_path=str(path),
            iq_format="npy",
            dtype="complex64",
            layout=layout,
            sample_rate_hz=sample_rate_hz,
            center_frequency_hz=center_frequency_hz,
            protected_channel=protected_channel,
            array_geometry=array_geometry,
            expected_desired_source_count=expected_desired_source_count,
            synchronized=synchronized,
            reference_frequency_hz=reference_frequency_hz,
        )
        repository = IQRepository([tmp_path])
        return repository, repository.register(manifest)

    return register


@pytest.fixture
def complex_tone() -> np.ndarray:
    samples = np.arange(1_024, dtype=np.float64)
    return np.exp(2j * np.pi * 0.125 * samples).astype(np.complex64)


@pytest.fixture
def gated_pulse_train() -> np.ndarray:
    rng = np.random.default_rng(13)
    samples = np.arange(1_024)
    gate = (samples % 128) < 32
    noise = 0.01 * (rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size))
    return (gate * np.exp(2j * np.pi * 0.08 * samples) + noise).astype(np.complex64)


@pytest.fixture
def complex_linear_chirp() -> np.ndarray:
    samples = np.arange(1_024, dtype=np.float64)
    instantaneous_frequency = -0.20 + 0.40 * samples / samples.size
    phase = 2.0 * np.pi * np.cumsum(instantaneous_frequency)
    return np.exp(1j * phase).astype(np.complex64)


@pytest.fixture
def ula_four_channel() -> tuple[np.ndarray, ArrayGeometry]:
    rng = np.random.default_rng(29)
    samples = np.arange(1_024, dtype=np.float64)
    base = np.exp(2j * np.pi * 0.125 * samples)
    steering = np.exp(1j * np.arange(4) * 0.35)
    noise = 0.005 * (
        rng.standard_normal((4, samples.size)) + 1j * rng.standard_normal((4, samples.size))
    )
    geometry = ArrayGeometry(
        geometry_type="ula",
        element_count=4,
        element_spacing_m=0.0625,
        reference_frequency_hz=2_400_000_000.0,
        synchronized=True,
    )
    return (steering[:, np.newaxis] * base + noise).astype(np.complex64), geometry


@pytest.fixture
def antiphase_two_channel(complex_tone: np.ndarray) -> np.ndarray:
    return np.vstack((complex_tone, -complex_tone)).astype(np.complex64)
