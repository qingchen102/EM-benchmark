"""只读：真源峰的相干性为何偏低——检查峰窗内是否有**多个物理信号**（零 token）。

做法：在峰频率附近扫 10 kHz 细窗，打印每个窗的功率与 λ1 占比，看
  (a) 单点源（coherent 且功率同量级）
  (b) 两个源（区间内 λ1 掉到 0.5 附近、窗口一挪就跳）
  (c) 失配噪声（λ1 ~0.14 全平）
用法：python _diag_coh_detail.py d1_00000 d1_00007 d1_00009
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


def coh_pow(resid, fs, center, win):
    n = resid.shape[1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = np.abs(f - center) <= win
    X = np.fft.fft(resid, axis=1)[:, m]
    if X.shape[1] < 2:
        return float("nan"), float("nan")
    p = float(np.mean(np.abs(X) ** 2))
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-30)
    R = Xn @ Xn.conj().T / Xn.shape[1]
    w = np.linalg.eigvalsh(R)[::-1].real
    return float(w[0] / (w.sum() + 1e-30)), 10 * np.log10(p + 1e-30)


def main(sids):
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
        ct.estimate_band_occupancy_by_correlation("station_a")
        resid = np.load(ctx.cache_dir / "resid_station_a.npy")
        print(f"\n===== {sid}  GT={[(g['kind'], round(g['freq_offset_hz']/1e3), round(g['bandwidth_hz']/1e3)) for g in ifw]} =====")
        print(f"  {'center(k)':>10}" + "".join(f"{w/1e3:>8.0f}k" for w in (5e3, 10e3, 20e3, 40e3, 80e3)))
        print(f"  {'':>10}(λ1 占比)")
        centers = [g["freq_offset_hz"] for g in ifw]
        centers += [g["freq_offset_hz"] + 200e3 for g in ifw]
        centers += [g["freq_offset_hz"] - 200e3 for g in ifw]
        centers += [0.0, 300e3, -300e3, 700e3, -700e3, 830e3, -830e3]
        for c in sorted(set(centers)):
            row = f"  {c/1e3:10.1f}"
            for w in (5e3, 10e3, 20e3, 40e3, 80e3):
                lam, _ = coh_pow(resid, fs, c, w)
                row += f"{lam:8.2f}"
            print(row)


if __name__ == "__main__":
    main(sys.argv[1:] or ["d1_00000", "d1_00007"])
