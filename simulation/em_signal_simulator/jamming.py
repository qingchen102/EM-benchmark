from __future__ import annotations
import numpy as np

JAMMING_TYPES = {"none", "single_tone", "swept", "pulse", "broadband"}

def inject_jamming(signal, jamming_type="none", jsr_db=3.0, rng=None, **kw):
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(signal, dtype=np.complex128); n = len(x); typ = str(jamming_type).lower()
    if typ not in JAMMING_TYPES: raise ValueError(f"Unsupported jamming type: {jamming_type}")
    if typ == "none": return x.copy()
    t = np.arange(n)
    if typ == "single_tone":
        f = float(kw.get("frequency", kw.get("center_frequency", .15))); j = np.exp(1j * 2 * np.pi * f * t)
    elif typ == "swept":
        f0, f1 = kw.get("f_start", -.4), kw.get("f_stop", .4); phase = 2*np.pi*(f0*t + (f1-f0)*t*t/(2*max(n,1))); j = np.exp(1j*phase)
    elif typ == "pulse":
        duty = float(kw.get("duty_cycle", .1)); mask = rng.random(n) < duty; j = mask * (rng.normal(size=n)+1j*rng.normal(size=n)) / np.sqrt(2)
    else: j = (rng.normal(size=n) + 1j*rng.normal(size=n)) / np.sqrt(2)
    target = np.mean(np.abs(x)**2) * 10**(float(jsr_db)/10)
    jp = np.mean(np.abs(j)**2)
    return x + j * np.sqrt(target / jp) if jp > 0 else x.copy()
