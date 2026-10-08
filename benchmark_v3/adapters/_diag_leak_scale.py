"""只读：验证"残差 = 目标参考的固定比例泄漏"这一机制（零 token）。

在干净 D1 样本上，residual ≈ iq − ref 里仍有 -87 dB 的准单音台阶，而热噪声底是
-131 dB。若这是"参考与实测目标之间的固定比例失配"（增益/定标误差），则：
  对每个子带算  r(f) = resid(f) / ref(f)  的复数中位
  干净样本上 |r| 应**跨频带基本恒定**（同一比例）；
  有干扰样本上，干扰所在子带 |r| 应显著抬高。
并且  ref 乘上最佳复标量后相减，残差应塌到噪声底。

用法：python _diag_leak_scale.py d1_00001 d1_00003
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


def main(sids: list[str]) -> None:
    for sid in sids:
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        refs = rca._build_target_reference(priv, n)
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        iq = ct._station_iq_raw("station_a")
        ref = np.fft.ifft(np.fft.fft(np.asarray(refs["station_a"]), axis=1) *
                          ((np.fft.fftfreq(n, 1 / fs) >= -0.45 * fs) &
                           (np.fft.fftfreq(n, 1 / fs) <= 0.45 * fs)), axis=1)
        resid = np.load(ctx.cache_dir / "resid_station_a.npy") if (ctx.cache_dir / "resid_station_a.npy").is_file() else None
        if resid is None:
            ct.estimate_band_occupancy_by_correlation("station_a")
            resid = np.load(ctx.cache_dir / "resid_station_a.npy")
        el = 0                                     # 单阵元看比例
        R = np.fft.fft(resid[el])
        F = np.fft.fft(ref[el])
        I = np.fft.fft(iq[el])
        f = np.fft.fftfreq(n, 1 / fs)
        edges = [-1800, -900, -810, -700, -600, -500, -400, -200, 200, 400, 500, 600, 700, 810, 900, 1800]
        print(f"\n===== {sid} =====")
        print(f"  GT 干扰: {[(g.get('kind'), round(g.get('freq_offset_hz', 0)/1e3)) for g in ifw]}")
        print(f"  {'band(kHz)':<14}{'|resid|dB':>11}{'|ref|dB':>10}{'|r| dB':>9}"
              f"{'|resid-a*ref|dB':>17}")
        # 全局最佳复标量 a = <ref, res>/<ref, ref>（带内）
        m = (f >= -810e3) & (f <= 810e3)
        a = float(np.vdot(F[m], R[m]).real / (np.vdot(F[m], F[m]).real + 1e-30))
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (f >= lo * 1e3) & (f < hi * 1e3)
            if not mm.any():
                continue
            rp = float(np.median(np.abs(R[mm]) ** 2))
            fp = float(np.median(np.abs(F[mm]) ** 2))
            resid2 = R - a * F
            r2 = float(np.median(np.abs(resid2[mm]) ** 2))
            print(f"  {lo:>6}~{hi:<6}{10*np.log10(rp+1e-30):11.1f}"
                  f"{10*np.log10(fp+1e-30):10.1f}"
                  f"{10*np.log10(rp/(fp+1e-30)+1e-30):9.1f}"
                  f"{10*np.log10(r2+1e-30):17.1f}")
        print(f"  复数最佳标量 (实部, 虚部) = ({a:.5f}, "
              f"{float(np.vdot(F[m], R[m]).imag/(np.vdot(F[m], F[m]).real+1e-30)):.5f})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["d1_00001", "d1_00003"])
