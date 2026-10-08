"""只读：D1 原始 PSD / 剥目标后残差 PSD 的分段剖面（零 token）。

目的：搞清"剥目标后 residual_peaks 里的强峰（±850k，46~49dB）"到底是
   (a) 真源（生成器漏标 / 工具口径错）
   (b) 目标自身在 [±500k, ±810k] 的带外泄漏（剥离模板带限 → 物理上剥不掉）
   (c) 接收滤波器过渡带形状
输出：逐段（每 100 kHz 一格）的中位功率，原始 vs 残差，单位 dB（相对各自中位）。

用法：python _diag_psd_shape.py d1_00001 [d1_00000 ...]
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


def band_medians(psd_db: np.ndarray, freqs: np.ndarray, edges_khz) -> list[tuple[str, float]]:
    out = []
    for lo, hi in zip(edges_khz[:-1], edges_khz[1:]):
        m = (freqs >= lo * 1e3) & (freqs < hi * 1e3)
        out.append((f"{lo:>6.0f}~{hi:>6.0f}k", float(np.median(psd_db[m])) if m.any() else float("nan")))
    return out


def main(sids: list[str]) -> None:
    edges = [-1800, -1200, -900, -810, -700, -600, -500, -300, -100, 100, 300,
             500, 600, 700, 810, 900, 1200, 1800]
    for sid in sids:
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                    # noqa: BLE001
            refs = {}
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        iq = ct._station_iq_raw("station_a")
        f = np.fft.fftfreq(iq.shape[1], 1 / fs)
        order = np.argsort(f)
        f = f[order]
        raw = np.abs(np.fft.fft(iq, axis=1)) ** 2
        raw = 10 * np.log10(raw.mean(axis=0)[order] + 1e-30)
        ct.estimate_band_occupancy_by_correlation("station_a")     # 生成缓存
        resid = np.load(ctx.cache_dir / "resid_station_a.npy")
        fr = np.fft.fftfreq(resid.shape[1], 1 / fs)
        ro = np.argsort(fr)
        fr = fr[ro]
        rp = np.abs(np.fft.fft(resid, axis=1)) ** 2
        rp = 10 * np.log10(rp.mean(axis=0)[ro] + 1e-30)

        print(f"\n===== {sid}  fs={fs/1e6:.1f}M  GT {len(ifw)} 个 "
              f"{[(g.get('kind'), round(g.get('freq_offset_hz',0)/1e3)) for g in ifw]} =====")
        print("  频段(kHz)        原始dB(相对原始中位)   残差dB(相对残差中位)")
        r0 = np.median(raw)
        p0 = np.median(rp)
        for (lab, rv), (_, pv) in zip(band_medians(raw, f, edges),
                                      band_medians(rp, fr, edges)):
            print(f"  {lab}   {rv - r0:8.1f}            {pv - p0:8.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["d1_00001", "d1_00007"])
