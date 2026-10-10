"""只读验证：空间相干性判据在 D1 全部样本上的可分性（零 token，判据定阈用）。

背景（本轮实测）：剥目标后的残差在通带内是**平坦 -88 dB**（热噪声底 -131 dB），
即参考回放那 ~0.15% 的失配能量，宽频带且**跨阵元不相干**；而真干扰是点源。
于是  峰值子带协方差第一特征值占比 λ1/Σλ  →  真源 ≈ 1，失配噪声 ≈ 0.13。

本脚本对每个 residual_peak 打标（与 GT 同带=真 / 否则=假），输出可分性与
各阈值下的漏报/虚警，供"定阈 + 写进工具契约"用。

用法：python _diag_coherence_auc.py [n_samples]
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


def coherence(resid: np.ndarray, fs: float, center: float, win_hz: float = 40e3) -> float:
    n = resid.shape[1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = np.abs(f - center) <= win_hz
    X = np.fft.fft(resid, axis=1)[:, m]
    if X.shape[1] < 2:
        return float("nan")
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-30)
    R = X @ X.conj().T / X.shape[1]
    w = np.linalg.eigvalsh(R)[::-1].real
    return float(w[0] / (w.sum() + 1e-30))


def main(n_samples: int) -> None:
    idx = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    wins = (5e3, 10e3, 20e3, 40e3, 80e3)
    real: dict[float, list] = {w: [] for w in wins}
    fake: dict[float, list] = {w: [] for w in wins}
    misses = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
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
        peaks = occ["residual_peaks"]
        matched_gt = set()
        for p in peaks:
            f0 = p["freq_offset_hz"]
            hit = None
            for gi, g in enumerate(ifw):
                if abs(f0 - g["freq_offset_hz"]) <= 200e3:
                    hit = gi
                    break
            for w in wins:
                (real if hit is not None else fake)[w].append(
                    (coherence(resid, fs, f0, w), p["over_floor_db"],
                     p["bandwidth_hz"], sid))
            if hit is not None:
                matched_gt.add(hit)
        for gi in range(len(ifw)):
            if gi not in matched_gt:
                misses.append((sid, ifw[gi]["kind"], ifw[gi]["freq_offset_hz"]))
        gts = [g["freq_offset_hz"] for g in ifw]
        print(f"{sid}  GT={[round(v/1e3) for v in gts]}  peaks={len(peaks)}")

    print(f"\n===== 汇总（n={n_samples} 样本）=====")
    print(f"  真峰 {len(real[wins[0]])} 个 / 假峰 {len(fake[wins[0]])} 个"
          f" / 未产生峰的 GT 源 {len(misses)}")

    def stat(name, rows):
        c = np.array([r[0] for r in rows])
        o = np.array([r[1] for r in rows])
        print(f"  {name}: coh 中位 {np.median(c):.3f}  P10 {np.percentile(c,10):.3f}"
              f"  P90 {np.percentile(c,90):.3f} | over_floor 中位 {np.median(o):.1f}dB")

    stat("真峰", real[wins[0]])
    stat("假峰", fake[wins[0]])
    print("\n  窗宽 × 阈值 的留真/留假：")
    for w in wins:
        R = np.array([r[0] for r in real[w]])
        F = np.array([r[0] for r in fake[w]])
        print(f"   窗 ±{w/1e3:.0f}k: 真峰中位 {np.median(R):.2f} / 假峰P90 {np.percentile(F,90):.2f}"
              f"  最大假峰 {F.max():.2f}")
        for thr in (0.50, 0.60, 0.70, 0.80, 0.90):
            tp = int((R >= thr).sum())
            fp = int((F >= thr).sum())
            print(f"      thr={thr:.2f}: 留真 {tp}/{len(R)}（漏 {len(R)-tp}）"
                  f"  留假 {fp}/{len(F)}")
    if misses:
        print("  完全未报的 GT 源:")
        for m in misses:
            print("   ", m)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
