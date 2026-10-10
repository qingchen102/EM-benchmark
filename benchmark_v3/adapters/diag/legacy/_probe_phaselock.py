"""临时探针（零 token）：加噪是否破坏了"合作目标参考直减"？

背景：2026-10-10 加噪换数据后，D2 的 gate tool-view 召回 0.561 → 0.367。
在 3 个 D2 样本上对比"加噪前 / 加噪后"的：
  - |iq|^2 / |ref|^2 / |resid|^2（残差 = 带限 iq − 带限 ref）
  - matched_energy_fraction = |<iq,ref>|^2 / (||iq||^2 ||ref||^2)
    —— 旧数据实测 0.998（直减有效）；若该值崩塌 ⇒ 参考与接收之间失去了
       逐样本相位一致性（相位噪声是公共乘性过程，而参考是"干净"综合）。
判据：resid/iq 只掉几个 dB（旧）vs 几乎不掉（新）。
"""
from __future__ import annotations

import json
import sys
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


def probe(sid: str) -> None:
    print(f"=== {sid} ===")
    for tag, root in ROOTS.items():
        cap = root / sid / "capture"
        priv, pub = cap / "private", cap / "public"
        if not (pub / "iq_station_a.npy").exists():
            print(f"  {tag:3s} 缺失：{pub}")
            continue
        obs = json.loads((pub / "observations.json").read_text(encoding="utf-8"))
        fs = obs["receivers"][0]["sample_rate_hz"]
        ns = obs["receivers"][0]["shape"][1]
        refs = rca._build_target_reference(priv, ns)
        iq = np.asarray(np.load(pub / "iq_station_a.npy"), complex)
        ref = np.asarray(refs["station_a"], complex)
        n = iq.shape[1]
        fr = np.fft.fftfreq(n, 1 / fs)
        m = np.abs(fr) <= fs * 0.45
        iqb = np.fft.ifft(np.fft.fft(iq, axis=1) * m, axis=1)
        refb = np.fft.ifft(np.fft.fft(ref, axis=1) * m, axis=1)
        resid = iqb - refb
        pi_ = float(np.mean(np.abs(iqb) ** 2))
        pr = float(np.mean(np.abs(refb) ** 2))
        pe = float(np.mean(np.abs(resid) ** 2))
        num = abs(np.sum(iqb * np.conj(refb))) ** 2
        den = float(np.sum(np.abs(iqb) ** 2)) * float(np.sum(np.abs(refb) ** 2))
        frac = num / den if den else 0.0
        print(f"  {tag:3s} |iq|^2={d(pi_):8.2f} dB  |ref|^2={d(pr):8.2f} dB  "
              f"|resid|^2={d(pe):8.2f} dB  resid/iq={d(pe / pi_):7.2f} dB  "
              f"matched_frac={frac:.6f}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["d2_00001", "d2_00006", "d2_00010"]):
        probe(s)
