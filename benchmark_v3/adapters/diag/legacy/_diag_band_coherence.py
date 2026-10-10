"""只读：验证判别量——"子带空间相干性"能否分开真干扰与目标残差（零 token）。

机制（本轮已实测确认）：
  干净 D1 样本剥离后，残差在 ±810k 内是**平坦 -88 dB**（不是热噪声 -131 dB，
  也不是目标频谱的缩放副本）：这是"参考回放"里那 0.15% 失配能量，宽频带、
  跨阵元不相关（像噪声），却被现有峰判据（≥8 dB over 中位底）报成 46~49 dB 的"强峰"。
  真干扰则是一个点源 → 其子带协方差的第一特征值占比 λ1/Σλ 高。

判据：对每个候选子带算 λ1/Σλ（8 元阵）。
   真源应接近 1；失配噪声应接近 1/8。
用法：python _diag_band_coherence.py d1_00001 d1_00003 d1_00007
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


def band_coherence(resid: np.ndarray, fs: float, center: float, bw: float,
                   win_hz: float | None = None):
    """子带协方差第一特征值占比 + 子带功率。win_hz 给定时用固定窄窗（更纯）。"""
    n = resid.shape[1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = (np.abs(f - center) <= win_hz) if win_hz else (np.abs(f - center) <= max(bw / 2, 20e3))
    X = np.fft.fft(resid, axis=1)[:, m]
    if X.shape[1] < 2:
        return float("nan"), float("nan")
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-30)
    R = X @ X.conj().T / X.shape[1]
    w = np.linalg.eigvalsh(R)[::-1].real
    return float(w[0] / (w.sum() + 1e-30)), float(10 * np.log10(np.mean(np.abs(X) ** 2) + 1e-30))


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
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        resid = np.load(ctx.cache_dir / "resid_station_a.npy")
        # 自身参考（无噪）也过同一判别量，作为"纯失配"对照
        ref = np.asarray(refs["station_a"])
        print(f"\n===== {sid}  GT {[(g.get('kind'), round(g.get('freq_offset_hz',0)/1e3)) for g in ifw]} =====")
        print(f"  {'peak(kHz)':>10}{'over_dB':>9}{'bw(k)':>8}"
              f"{'lam1@±60k':>11}{'lam1@±20k':>11}   |  {'ref@±60k':>9}")
        for p in occ["residual_peaks"]:
            lam60, _ = band_coherence(resid, fs, p["freq_offset_hz"], p["bandwidth_hz"], 60e3)
            lam20, _ = band_coherence(resid, fs, p["freq_offset_hz"], p["bandwidth_hz"], 20e3)
            lamr, _ = band_coherence(ref, fs, p["freq_offset_hz"], p["bandwidth_hz"], 60e3)
            print(f"  {p['freq_offset_hz']/1e3:10.1f}{p['over_floor_db']:9.1f}"
                  f"{p['bandwidth_hz']/1e3:8.1f}{lam60:11.4f}{lam20:11.4f}   |  {lamr:9.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["d1_00001", "d1_00003"])
