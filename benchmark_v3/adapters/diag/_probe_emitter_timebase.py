"""时基体检：辐射体的事件数/跳速随窗长的变化（零 API，只调生成器）。

问：把窗从 10 ms 拉到 10 s，事件数会按物理速率增长，还是保持"每窗固定个数"？
用法：python -X utf8 _tmp_timebase_probe.py
"""
import sys
from pathlib import Path

import numpy as np

ADP = Path(__file__).resolve().parents[1]          # benchmark_v3/adapters（本文件在 adapters/diag/ 下）
sys.path.insert(0, str(ADP.parent / "simulation" / "em_signal_simulator_v3"))
import emitters as em  # noqa: E402

FS = 4e6


def _bursts(x, k):
    env = np.abs(x)
    ker = np.ones(k) / k
    env_s = np.convolve(env, ker, mode="same")
    thr = 0.25 * float(np.max(env_s))
    on = env_s > thr
    return int(np.sum(np.diff(on.astype(np.int8)) == 1) + (1 if on[0] else 0))


def _hops(x, blk=64):
    nb = len(x) // blk
    if nb < 2:
        return 0
    d = np.abs(np.fft.fft(x[: nb * blk].reshape(nb, blk), axis=1))
    f = np.argmax(d, axis=1)
    return int(np.sum(np.diff(f) != 0))


HOP = {"Bluetooth_FHSS", "UAV_RC_hopping"}
BURST = ["WiFi_burst", "DECT", "Welder", "Corona", "VFD_noise", "Microwave", "Tone_jam"]

print(f"fs={FS/1e6:g} MHz  （40 000 样点 = 10 ms，4 000 000 = 1 s）")
print(f"{'类':18s} {'10 ms 事件':>10s} {'100 ms 事件':>11s} {'1 s 事件':>9s} {'每窗固定?':>10s}")
for name in ["Bluetooth_FHSS", "UAV_RC_hopping"] + BURST:
    row = []
    for length in (40000, 400000, 4000000):
        rng = np.random.default_rng(7)
        try:
            x = np.asarray(em.generate_emitter(name, rng, length=length)["iq"], dtype=complex)
        except Exception as exc:  # noqa: BLE001
            row.append(f"ERR({type(exc).__name__})")
            continue
        n_ev = _hops(x) if name in HOP else _bursts(x, max(8, length // 2000))
        row.append(str(n_ev))
    try:
        fixed = "是（每窗）" if row[0].isdigit() and row[2].isdigit() and int(row[2]) < 3 * int(row[0]) else "否（随窗）"
    except ValueError:
        fixed = "-"
    print(f"{name:18s} {row[0]:>10s} {row[1]:>11s} {row[2]:>9s} {fixed:>10s}")
