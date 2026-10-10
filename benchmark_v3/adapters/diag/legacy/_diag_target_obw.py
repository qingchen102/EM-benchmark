"""只读：测"合作目标参考"的真实占用带宽 vs 名义 1 MHz / 剥离模板带（零 token）。

背景：capture_tools 的剥离模板按 ctx.target_bw_normalized=0.25（=1 MHz）取 ±500k，
但接收链是 firwin(taps=129, cutoff=rx_bw/2=810k)，目标实际占满到 ±810k 也完全可能。
若真实占用 > 模板带，则 [500k, 810k] 区间的目标能量**剥不掉**，会以 46~49 dB 的
"强峰"出现在 residual_peaks 里被 agent 当干扰报出来——这正是 D1/S0 虚警。

输出：逐样本 per-station 的参考 99% 占用带宽、-20dB 带宽、以及台阶边缘位置。

用法：python _diag_target_obw.py [n_samples] [difficulty]
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

import run_capture_agent as rca   # noqa: E402

ROOT = _V3 / "dataset_d1"


def obw99(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    """返回 (99% 占用带宽, 上/下沿不对称度描述)。psd 已排序。"""
    tot = psd.sum()
    c = np.cumsum(psd)
    lo = freqs[np.searchsorted(c, 0.005 * tot)]
    hi = freqs[np.searchsorted(c, 0.995 * tot)]
    return float(hi - lo), float(lo), float(hi)


def main(n_samples: int) -> None:
    idx = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
    print(f"{'sid':<10} {'st':<9} {'OBW99(kHz)':>10} {'lo(k)':>8} {'hi(k)':>8} "
          f"{'-20dB-edge(k)':>14}")
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = ROOT / sid / "capture" / "public"
        priv = ROOT / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception as exc:                              # noqa: BLE001
            print(f"{sid:<10} ref failed: {str(exc)[:50]}")
            continue
        f = np.fft.fftfreq(n, 1 / fs)
        order = np.argsort(f)
        fs_sorted = f[order]
        for st, ref in refs.items():
            if ref is None:
                continue
            a = np.asarray(ref)
            psd = (np.abs(np.fft.fft(a, axis=1)) ** 2).mean(axis=0)[order]
            bw, lo, hi = obw99(fs_sorted, psd)
            psd_db = 10 * np.log10(psd + 1e-30)
            peak = float(psd_db.max())
            above = fs_sorted[psd_db >= peak - 20]
            edge = float(max(abs(above.min()), abs(above.max()))) if above.size else float("nan")
            print(f"{sid:<10} {st:<9} {bw/1e3:10.1f} {lo/1e3:8.1f} {hi/1e3:8.1f} "
                  f"{edge/1e3:14.1f}")
        print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
