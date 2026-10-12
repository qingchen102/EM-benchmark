"""只读诊断：D1 某样本的门控判据为何报不出真源（峰值/局部门限/空间相干逐项打印）。

用法：python -X utf8 diag\\_probe_d1_peakmiss.py <dataset_root> <sample_id>

背景：新版数据（配额 v2，100/档）D1 仪表侧覆盖率 80.4%（旧代 92.6%），
需要分清是「真源能量不够」还是「判据把真源滤掉了」。本脚本对样本做三重取证：
  1. 逐 5 kHz 打印真值频偏附近的 (相对局部门限 dB, 窄窗空间相干 λ1/Σλ)；
  2. 打印门控判据 `_residual_bands` 实际返回的带（含 over/coh/带宽）；
  3. 打印旧判据 `_residual_peaks_legacy` 的带（做对照）。
"""
from __future__ import annotations

import json as _json
import pathlib as _pl
import shutil
import sys
import tempfile

import numpy as np

_HERE = _pl.Path(__file__).resolve().parent
_V3 = _HERE.parent
for _p in (str(_V3), str(_V3 / "simulation"), str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = _pl.Path(argv[0]).resolve()
    sid = argv[1]
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = _json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    tr = _json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    gts = list(tr.get("interferers") or [])
    n = int(obs["receivers"][0]["shape"][1])
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    tmp = _pl.Path(tempfile.mkdtemp(prefix=f"miss_{sid}_"))
    try:
        refs = rca._build_target_reference(priv, n)
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        rx_half = float(occ["band_hz"][1])
        resid = np.asarray(ct._strip_target_at_station("station_a"), np.complex128)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"== {root.parent.name}/{sid}  GT={len(gts)}  matched={occ['matched_energy_fraction']}"
          f"  floor={occ['residual_noise_floor_db']} dB  rx_half={rx_half/1e3:.0f}k ==")
    for g in gts:
        print(f"  GT {g['kind']} f={g['freq_offset_hz']/1e3:9.1f}k bw={g.get('bandwidth_hz', 0)/1e3:8.1f}k"
              f" power={g.get('power_dbm')}dBm")

    f = np.fft.fftfreq(resid.shape[1], 1 / fs)
    inb = np.where((f >= -rx_half) & (f <= rx_half))[0]
    db = 10 * np.log10((np.abs(np.fft.fft(resid, axis=-1)) ** 2).mean(axis=0)[inb] + 1e-30)
    from scipy.ndimage import median_filter
    k = max(int(round(ct.PEAK_FLOOR_WIN_HZ / (fs / resid.shape[1]))) | 1, 5)
    over = db - median_filter(db, size=k, mode="nearest")
    f_ib = f[inb]

    for g in gts:
        c0 = g["freq_offset_hz"]
        print(f"  -- 真源 {c0/1e3:.1f}k 附近（步长 5k，窗口 {ct.PEAK_FLOOR_WIN_HZ/1e3:.0f}k）--")
        for d in range(-30_000, 30_001, 5_000):
            fc = c0 + d
            m = int(np.argmin(np.abs(f_ib - fc)))
            coh = ct._band_coherence(resid, fs, float(f_ib[m]), ct.PEAK_COH_WIN_HZ)
            flag = ""
            if abs(f_ib[m] - c0) < 3e3:
                flag = "  <= 真源"
            print(f"     {f_ib[m]/1e3:9.1f}k  over={over[m]:6.1f} dB  coh={coh:.3f}"
                  f"  门限 over≥{ct.PEAK_OVER_DB} / coh≥{ct.PEAK_COH_MIN}{flag}")

    print("  -- 门控判据返回 --")
    for b in ct._residual_bands(resid, fs, rx_half):
        print(f"     {b}")
    print("  -- 旧判据返回 --")
    for b in ct._residual_peaks_legacy(resid, fs, rx_half):
        print(f"     {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
