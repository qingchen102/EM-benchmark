"""只读探针：把「真值里的载频/带宽」与「观测 IQ 的实际频谱」放在一起看。

动机：查 `_an_lna_obw.py` 发现多个辐射体的生成器**忽略 bw 参数**（Noise_jam 直接
返回满带白噪），怀疑真值的 freq_offset_hz / bandwidth_hz 对部分 kind 没有可观测对应物。
本脚本只用已落盘的 public IQ，不需要任何随机种子。
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent.parent.parent
FS = 4.0e6

CASES = [("d3", "d3_00005"), ("d4", "d4_00000"), ("d2", "d2_00007"), ("d1", "d1_00000")]

for d, sid in CASES:
    root = B / f"dataset_{d}" / sid
    tr = json.loads((root / "capture" / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    print("=" * 84)
    print(f"### {sid}  target={tr['target'].get('kind')}  干扰 {len(tr['interferers'])} 个")
    for f in tr["interferers"]:
        ps = f.get("per_station") or {}
        inr = {k: round(float(v.get("inr_db", 0)), 1) for k, v in ps.items()}
        print(f"   {f.get('role')}/{f.get('kind'):<16} f0={float(f.get('freq_offset_hz', 0))/1e3:8.1f} kHz  "
              f"BW={float(f.get('bandwidth_hz', 0))/1e3:7.1f} kHz  INR(逐站)={inr}")
    iq = np.load(root / "capture" / "public" / "iq_station_a.npy")
    x = iq[0] if iq.ndim > 1 else iq
    x = x - x.mean()
    x = x * np.hanning(len(x))
    p = np.abs(np.fft.fft(x)) ** 2
    fr = np.fft.fftfreq(len(x), 1.0 / FS)
    o = np.argsort(fr)
    fr, p = fr[o], p[o]
    pdb = 10 * np.log10(p / p.max() + 1e-30)
    print("   --- station_a 观测谱（相对峰，16 段平均）")
    edges = np.linspace(-FS / 2, FS / 2, 17)
    for i in range(16):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        print(f"      {edges[i]/1e3:7.0f}~{edges[i+1]/1e3:7.0f} kHz  {pdb[m].mean():7.1f} dB")
    top = np.argsort(p)[-6:][::-1]
    print("   --- 最强 6 条谱线")
    for t in top:
        print(f"      {fr[t]/1e3:8.1f} kHz   {pdb[t]:7.1f} dB")
