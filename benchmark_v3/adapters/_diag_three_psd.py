"""只读：一次看穿——原始 IQ / 目标参考 / 残差 三者的绝对 PSD 剖面（零 token）。

判据：
  残差在 [500k,810k] 的台阶若与"目标参考"同形 → 剥离模板带限导致的目标残留；
  若为平底抬升 → 另有机制。
打印 100 kHz 一格的绝对 dB（同一 1e-30 参考），三列并排。

用法：python _diag_three_psd.py d1_00001
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

D1 = _V3 / "dataset_d1"


def prof(x: np.ndarray, fs: float, edges) -> list[float]:
    n = x.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    order = np.argsort(f)
    fs_s = f[order]
    psd = (np.abs(np.fft.fft(x, axis=-1)) ** 2).mean(axis=0)[order]
    db = 10 * np.log10(psd + 1e-30)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fs_s >= lo * 1e3) & (fs_s < hi * 1e3)
        out.append(float(np.median(db[m])) if m.any() else float("nan"))
    return out


def main(sid: str) -> None:
    cap = D1 / sid / "capture" / "public"
    priv = D1 / sid / "capture" / "private"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    refs = rca._build_target_reference(priv, n)
    ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                         allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                         fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                         target_bw_normalized=0.25, aoi_center=[0, 0],
                         target_reference=refs)
    ct.set_context(ctx)
    occ = ct.estimate_band_occupancy_by_correlation("station_a")
    iq = ct._station_iq_raw("station_a")
    ref = np.asarray(refs["station_a"])
    resid = np.load(ctx.cache_dir / "resid_station_a.npy")
    edges = [-1800, -900, -810, -700, -600, -500, -400, -200, 200, 400, 500, 600, 700, 810, 900, 1800]
    a = prof(iq, fs, edges)
    b = prof(ref, fs, edges)
    c = prof(resid, fs, edges)
    print(f"===== {sid}  station_a  (peak over floor = "
          f"{[p['over_floor_db'] for p in occ['residual_peaks']]}, floor="
          f"{occ['residual_noise_floor_db']} dB) =====")
    print(f"  {'band(kHz)':<14}{'raw dB':>10}{'ref dB':>10}{'resid dB':>10}{'ref-raw':>9}")
    for (lo, hi), ra, rb, rc in zip(zip(edges[:-1], edges[1:]), a, b, c):
        print(f"  {lo:>6}~{hi:<6}  {ra:10.1f}{rb:10.1f}{rc:10.1f}{rb-ra:9.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d1_00001")
