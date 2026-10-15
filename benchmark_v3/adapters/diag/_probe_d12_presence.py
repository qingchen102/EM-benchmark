"""d12（卡 8）窗内源存在性实测：窗外强源开着 vs 压到 −100 dBm，做同一个 scenario 的 A/B。

回答的问题（数据有效性，不是难度）：
  窗外强源把前端压住之后，**窗内的源还在不在**？
    - 在（频带功率/峰底比基本守恒，只是被压制）⇒ 数据自洽，题目极难但有效；
    - 不在（源带功率塌到噪底）⇒ 该档的答案键在物理上就不可答，要写清楚。

口径：station_a 复基带 IQ 的 PSD（Welch）。对真值里**每个窗内源**取其设计的
`freq_offset_hz ± bandwidth_hz/2` 为源带，报：
  峰底比 A / 峰底比 B / 源带功率 Δ(A−B) / 该源带的噪底抬升 Δ(A−B)
其中「底」= 全窗（去掉所有源带后）PSD 中位。

用法：python -X utf8 diag/_probe_d12_presence.py [样本目录] [station]
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ADP = HERE.parents[1]
sys.path.insert(0, str(ADP))
PLAT = ADP / "城市低空电磁信号仿真平台"   # ADP = benchmark_v3
sys.path.insert(0, str(PLAT / "src"))

from cityrf.config import Scenario          # noqa: E402
from cityrf import runner                   # noqa: E402

_DEFAULT = ADP / "dataset_d12" / "d12_00000"
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
if SRC.is_dir():
    SRC = SRC / "capture" / "private" / "scenario.json"
STATION = sys.argv[2] if len(sys.argv) > 2 else "station_a"
TRUTH = SRC.parent / "d1_truth.json"
ROOT = ADP.parent / "_tmp_d12_presence"

sc = Scenario.model_validate_json(SRC.read_text(encoding="utf-8"))
truth = json.loads(TRUTH.read_text(encoding="utf-8"))
blocker = next(e for e in sc.emitters if e.carrier_hz)
fs = sc.sample_rate_hz

inwin = [g for g in truth["interferers"] if not g.get("out_of_window")]
if not inwin:
    print("该样本没有窗内源（答案键为空）")
    raise SystemExit(0)


def run(power_dbm: float, tag: str) -> np.ndarray:
    s = sc.model_copy(deep=True)
    for e in s.emitters:
        if e.carrier_hz:
            e.power_dbm = power_dbm
    out = ROOT / tag
    if out.exists():
        shutil.rmtree(out)
    runner.run_scenario(s, out)
    return np.load(out / "public" / f"iq_{STATION}.npy")


if ROOT.exists():
    shutil.rmtree(ROOT)
iq_a = run(float(blocker.power_dbm), "with")       # 窗外强源在场
iq_b = run(-100.0, "without")                      # 窗外强源压掉

n = iq_a.shape[-1]
f = np.fft.fftfreq(n, 1 / fs)
win = np.abs(f) <= 0.45 * fs


def psd(x: np.ndarray) -> np.ndarray:
    """Welch PSD；输入可以是 (天线, N) 或 (N,)，多天线取各天线 PSD 的均值。"""
    if x.ndim > 1:
        return np.mean([psd(x[i]) for i in range(x.shape[0])], axis=0)
    seg = 2048
    nseg = n // seg
    w = np.hanning(seg)
    acc = np.zeros(seg)
    for k in range(nseg):
        blk = x[k * seg:(k + 1) * seg] * w
        acc += np.abs(np.fft.fft(blk)) ** 2
    return acc / max(nseg, 1) / (np.sum(w ** 2))


P_a, P_b = psd(iq_a), psd(iq_b)
fseg = np.fft.fftfreq(P_a.size, 1 / fs)
inb_a, inb_b = P_a[np.abs(fseg) <= 0.45 * fs], P_b[np.abs(fseg) <= 0.45 * fs]
fin = fseg[np.abs(fseg) <= 0.45 * fs]


def bands_of(g):
    f0 = float(g.get("freq_offset_hz", 0.0))
    bw = max(float(g.get("bandwidth_hz", 0.0)), 60e3)
    return f0 - bw / 2.0, f0 + bw / 2.0, bw


bands = [bands_of(g) for g in inwin]
mask_src = np.zeros_like(fin, dtype=bool)
for lo, hi, _ in bands:
    mask_src |= (fin >= lo) & (fin <= hi)
floor_a = float(np.median(inb_a[~mask_src]))
floor_b = float(np.median(inb_b[~mask_src]))

print(f"样本 {SRC.parents[1].name}  站 {STATION}  窗内源 {len(inwin)} 个  "
      f"全窗噪底 A(强源开)={10*np.log10(floor_a):.1f} dB  B(强源关)={10*np.log10(floor_b):.1f} dB")
print(f"{'体制':16s} {'设计偏移':>10s} {'带宽':>9s} {'峰底比A':>9s} {'峰底比B':>9s} {'源带Δ(A-B)':>11s} {'底噪Δ(A-B)':>11s}")
print("-" * 88)
for g, (lo, hi, bw) in zip(inwin, bands):
    m = (fin >= lo) & (fin <= hi)
    if not m.any():
        continue
    pk_a, pk_b = float(inb_a[m].max()), float(inb_b[m].max())
    pb_a, pb_b = float(inb_a[m].mean()), float(inb_b[m].mean())
    print(f"{g['kind']:16s} {float(g.get('freq_offset_hz', 0.0))/1e3:+9.1f}k {bw/1e3:8.1f}k "
          f"{10*np.log10(pk_a/floor_a):8.1f}  {10*np.log10(pk_b/floor_b):8.1f}  "
          f"{10*np.log10(pb_a/pb_b):+10.1f}  {10*np.log10(floor_a/floor_b):+10.1f}")
print(f"\n全窗噪底抬升（强源开 − 强源关）= {10*np.log10(floor_a/floor_b):+.1f} dB")
