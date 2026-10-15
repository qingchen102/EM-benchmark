"""卡 5 决定性检查：1090 MHz 窗外强源到底有没有改变窗内信号？

做法：拿 d11_00000 的私有 scenario.json，跑两遍完整接收链——
  A：原样（雷达 power_dbm = 73.9 ⇒ 到站端口约 −20 dBm）
  B：把雷达功率降到 −200 dBm（等效移除）
然后比较 station_a 的复基带 IQ：
  · 总功率 / 带内底噪（目标带外区域的 PSD 中位）
  · 目标信号功率与 SINR（目标带内能量 / 带外底噪）
若 A 的底噪被抬起、SINR 下降 ⇒ 「窗外强源 → 窗内灵敏度塌方/互调」在本模型里真实成立。
"""
import json
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ADP = HERE.parents[1]                      # benchmark_v3/adapters
sys.path.insert(0, str(ADP))
PLAT = ADP.parent / "城市低空电磁信号仿真平台"
sys.path.insert(0, str(PLAT / "src"))

from cityrf.config import Scenario          # noqa: E402
from cityrf import runner                   # noqa: E402
from cityrf.receiver import receive         # noqa: E402

_DEFAULT = ADP.parent / "dataset_d11" / "d11_00000" / "capture" / "private" / "scenario.json"
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
if SRC.is_dir():
    SRC = SRC / "capture" / "private" / "scenario.json"
ROOT = ADP.parent / "_tmp_d11_ab"

sc = Scenario.model_validate_json(SRC.read_text(encoding="utf-8"))
target = next(e for e in sc.emitters if e.role == "communication")
blocker = next(e for e in sc.emitters if e.carrier_hz)
print(f"场景载频 {sc.carrier_frequency_hz/1e9:.2f} GHz   目标 {target.preset} @{target.power_dbm:.1f} dBm"
      f"   窗外源 {blocker.preset} {blocker.carrier_hz/1e9:.3f} GHz @{blocker.power_dbm:.1f} dBm")
print(f"前端：imd3={sc.receivers[0].frontend.imd3_enabled} iip3={sc.receivers[0].frontend.iip3_dbm:.1f} dBm "
      f"soft_limit={sc.receivers[0].frontend.soft_limit_input_dbm}")


def run(power_dbm, tag):
    s = sc.model_copy(deep=True)
    for e in s.emitters:
        if e.carrier_hz:
            e.power_dbm = power_dbm
    out = ROOT / tag
    if out.exists():
        shutil.rmtree(out)
    runner.run_scenario(s, out)
    iq = np.load(out / "public" / "iq_station_a.npy")
    return iq


if ROOT.exists():
    shutil.rmtree(ROOT)
iq_a = run(float(blocker.power_dbm), "with")
iq_b = run(-100.0, "without")

fs = sc.sample_rate_hz
n = iq_a.shape[-1]
f = np.fft.fftfreq(n, 1 / fs)
psd_a = np.abs(np.fft.fft(iq_a[0].astype(np.complex128))) ** 2
psd_b = np.abs(np.fft.fft(iq_b[0].astype(np.complex128))) ** 2
band = np.abs(f) <= 0.5e6          # 目标带内（UAV_video_OFDM ≈1 MHz）
outb = (np.abs(f) > 1.0e6)         # 目标带外、仍在接收通带内
med = lambda p, m: float(np.median(p[m]))
print()
print(f"{'':22s}{'总功率(dB)':>12s}{'带外底噪中位(dB)':>18s}{'带内能量(dB)':>14s}{'带内/带外(dB)':>15s}")
for tag, iq, psd in (("A 有窗外强源", iq_a, psd_a), ("B 移除窗外源", iq_b, psd_b)):
    tot = 10 * np.log10(float(np.mean(np.abs(iq[0]) ** 2)) + 1e-30)
    fl = 10 * np.log10(med(psd, outb) + 1e-30)
    sig = 10 * np.log10(float(np.sum(psd[band])) + 1e-30)
    print(f"{tag:22s}{tot:12.1f}{fl:18.1f}{sig:14.1f}{sig - fl:15.1f}")
d_floor = 10 * np.log10(med(psd_a, outb) + 1e-30) - 10 * np.log10(med(psd_b, outb) + 1e-30)
print()
print(f"结论：窗内底噪抬升 {d_floor:+.1f} dB（A − B）")
