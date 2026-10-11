"""临时探针（零 token）：全 50 条 D2 样本的"目标参考直减"是否被加噪破坏。

gate A/B 显示加噪后 D2 tool-view 召回 0.561→0.367，但抽查的 4 个样本
IQ 总功率几乎不变。真正的嫌疑是**相位噪声**：它是公共乘性过程
（receiver.py:50-54 的 exp(1j*cumsum(step))，每样本一步），而
`_build_target_reference` 给的是"干净"综合波形 —— 一旦窗内累积相位
随机化，iq 与 ref 就失去逐样本一致性，直减后残差里会**残留整个目标**。

判据：matched_frac = |<iq,ref>|^2/(||iq||^2||ref||^2)
  old 期望 ≈ 0.98（残差 -16 dB）；若 new 有一批样本掉到 <0.9 ⇒ 坐实。
"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))

import run_capture_agent as rca  # noqa: E402

ROOTS = {
    "old": _AD.parent / "_pre_noise_20261010" / "dataset_d2",
    "new": _AD.parent / "dataset_d2",
}
d = lambda x: 10 * np.log10(x + 1e-300)  # noqa: E731


def measure(root: Path, sid: str) -> tuple[float, float] | None:
    cap = root / sid / "capture"
    priv, pub = cap / "private", cap / "public"
    iq_p = pub / "iq_station_a.npy"
    if not iq_p.exists():
        return None
    obs = json.loads((pub / "observations.json").read_text(encoding="utf-8"))
    fs = obs["receivers"][0]["sample_rate_hz"]
    iq = np.asarray(np.load(iq_p), complex)
    ns = iq.shape[1]
    refs = rca._build_target_reference(priv, ns)
    key = "station_a" if "station_a" in refs else sorted(refs)[0]
    ref = np.asarray(refs[key], complex)
    fr = np.fft.fftfreq(ns, 1 / fs)
    m = np.abs(fr) <= fs * 0.45
    iqb = np.fft.ifft(np.fft.fft(iq, axis=1) * m, axis=1)
    refb = np.fft.ifft(np.fft.fft(ref, axis=1) * m, axis=1)
    num = abs(np.sum(iqb * np.conj(refb))) ** 2
    den = float(np.sum(np.abs(iqb) ** 2)) * float(np.sum(np.abs(refb) ** 2))
    frac = num / den if den else 0.0
    pe = float(np.mean(np.abs(iqb - refb) ** 2))
    pi_ = float(np.mean(np.abs(iqb) ** 2))
    return frac, d(pe / pi_)


def main() -> None:
    sids = sorted(p.name for p in ROOTS["new"].iterdir() if p.name.startswith("d2_"))
    print(f"n={len(sids)} 样本（station_a）\n")
    print(f"{'sid':12s} {'old_frac':>9s} {'new_frac':>9s} {'old_res/iq':>11s} {'new_res/iq':>11s}")
    rows = []
    for sid in sids:
        a = measure(ROOTS["old"], sid)
        b = measure(ROOTS["new"], sid)
        if a is None or b is None:
            print(f"{sid:12s}  (缺文件)")
            continue
        rows.append((sid, a[0], b[0], a[1], b[1]))
        flag = "  <== 直减被破坏" if b[0] < 0.90 else ""
        print(f"{sid:12s} {a[0]:9.6f} {b[0]:9.6f} {a[1]:11.2f} {b[1]:11.2f}{flag}")
    if not rows:
        return
    fo = np.array([r[1] for r in rows]); fn = np.array([r[2] for r in rows])
    ro = np.array([r[3] for r in rows]); rn = np.array([r[4] for r in rows])
    print(f"\nmatched_frac  中位 old={np.median(fo):.6f} new={np.median(fn):.6f}   "
          f"min old={fo.min():.6f} new={fn.min():.6f}")
    print(f"resid/iq dB   中位 old={np.median(ro):.2f} new={np.median(rn):.2f}   "
          f"最大 old={ro.max():.2f} new={rn.max():.2f}")
    bad = [(r[0], r[2]) for r in rows if r[2] < 0.90]
    print(f"new 掉破 0.90 的样本：{len(bad)}/{len(rows)}" + (f" → {bad[:10]}" if bad else ""))
    inc = [(r[0], r[4] - r[3]) for r in rows]
    inc.sort(key=lambda t: -t[1])
    print("残差抬升最大的 8 个样本：" + ", ".join(f"{s}({v:+.1f}dB)" for s, v in inc[:8]))


if __name__ == "__main__":
    main()
