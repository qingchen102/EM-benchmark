from __future__ import annotations
import numpy as np

def apply_channel(signal, snr_db=10.0, cfo=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(signal, dtype=np.complex128); n = len(x)
    if cfo: x = x * np.exp(1j * 2 * np.pi * float(cfo) * np.arange(n))
    power = np.mean(np.abs(x)**2); noise_power = power / (10**(float(snr_db)/10))
    noise = (rng.normal(size=n)+1j*rng.normal(size=n)) * np.sqrt(noise_power/2)
    y = x + noise; p = np.sqrt(np.mean(np.abs(y)**2))
    return y / p if p > 0 else y
