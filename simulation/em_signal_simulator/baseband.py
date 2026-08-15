from __future__ import annotations
import numpy as np

MODULATIONS = {"BPSK", "QPSK", "16QAM", "64QAM", "GFSK", "OOK", "OFDM", "FHSS", "LFM"}

def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    p = np.sqrt(np.mean(np.abs(x) ** 2))
    return x / p if p > 0 else x

def _qam(n, order, rng):
    m = int(np.sqrt(order)); levels = np.arange(-(m - 1), m, 2)
    z = levels[rng.integers(0, m, n)] + 1j * levels[rng.integers(0, m, n)]
    return z / np.sqrt(np.mean(np.abs(z) ** 2))

def generate_baseband(mod_type="QPSK", num_samples=1024, rng=None, **kw):
    rng = np.random.default_rng() if rng is None else rng
    mod = str(mod_type).upper()
    n = int(num_samples)
    if n <= 0: raise ValueError("num_samples must be positive")
    if mod not in MODULATIONS: raise ValueError(f"Unsupported modulation: {mod_type}")
    if mod == "BPSK": x = 2 * rng.integers(0, 2, n) - 1
    elif mod == "QPSK": x = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n)))
    elif mod == "16QAM": x = _qam(n, 16, rng)
    elif mod == "64QAM": x = _qam(n, 64, rng)
    elif mod == "OOK": x = rng.integers(0, 2, n).astype(float)
    elif mod == "GFSK":
        bits = 2 * rng.integers(0, 2, n) - 1
        bt = float(kw.get("bt", .35)); span = max(3, int(4 / max(bt, .05)))
        kernel = np.ones(span) / span; freq = np.convolve(bits, kernel, mode="same")
        x = np.exp(1j * np.pi * np.cumsum(freq) / 2)
    elif mod == "LFM":
        f0, f1 = kw.get("f0", -.4), kw.get("f1", .4); t = np.arange(n) / n
        x = np.exp(1j * 2 * np.pi * (f0 * t + .5 * (f1 - f0) * t * t) * n)
    elif mod == "FHSS":
        hops = int(kw.get("hops", max(2, n // 128))); hop_len = max(1, n // hops)
        freqs = rng.uniform(-.45, .45, hops); x = np.zeros(n, complex)
        for i in range(hops):
            sl = slice(i * hop_len, min(n, (i + 1) * hop_len)); x[sl] = np.exp(1j * 2 * np.pi * freqs[i] * np.arange(sl.stop-sl.start))
    elif mod == "OFDM":
        carriers = int(kw.get("subcarriers", 32)); cp = int(kw.get("cyclic_prefix", 8)); out=[]
        while len(out) < n:
            q = _qam(carriers, 4, rng); symbol = np.fft.ifft(q); out.extend(np.r_[symbol[-cp:], symbol])
        x = np.asarray(out[:n])
    return _norm(x)
