"""零成本冒烟：data_gen16.gen_wave16 改走 bandwidth_normalized 入口后，
19 类的实测 99% 占用带宽是否落在 BW_RANGE 内（否则 CNN 仍是“训练视角 ≠ 评测视角”）。

用法：python -u _smoke_gen16.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen16 import (  # noqa: E402
    BW_RANGE,
    CLASSES16,
    gen_wave16,
    simulate_residual,
)


def obw_norm(iq: np.ndarray, fs: float = 4e6, q: float = 0.99) -> float:
    """99% 功率带宽 / fs（频域累积口径，与 measure_obw99 同义）。"""
    n = len(iq)
    p = np.abs(np.fft.fft(iq)) ** 2
    f = np.fft.fftfreq(n, 1 / fs)
    order = np.argsort(f)
    f, p = f[order], p[order]
    c = np.cumsum(p) / (p.sum() + 1e-30)
    lo = f[min(np.searchsorted(c, (1 - q) / 2), n - 1)]
    hi = f[min(np.searchsorted(c, 1 - (1 - q) / 2), n - 1)]
    return float(hi - lo) / fs


def main() -> None:
    rng = np.random.default_rng(12345)
    print(f"BW_RANGE={BW_RANGE}  越界判据=[0.5*lo, 1.5*hi]", flush=True)
    bad = 0
    for name in CLASSES16:
        vals = [obw_norm(gen_wave16(name, rng)) for _ in range(3)]
        med = float(np.median(vals))
        ok = BW_RANGE[0] * 0.5 <= med <= BW_RANGE[1] * 1.5
        if not ok:
            bad += 1
        print(
            f"{name:18s} obw={med:.3f}  "
            f"({', '.join(f'{v:.3f}' for v in vals)})"
            f"{'' if ok else '   <-- 越界'}",
            flush=True,
        )
    print(f"越界 {bad}/{len(CLASSES16)}", flush=True)

    x = gen_wave16("Noise_jam", rng).astype(np.complex128)
    y = simulate_residual(x, rng, n_sib=1)
    print(
        f"simulate_residual ok: shape={np.shape(y)} "
        f"finite={bool(np.isfinite(y).all())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
