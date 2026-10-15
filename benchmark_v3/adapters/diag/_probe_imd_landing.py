"""卡 8（d12）决定性检查：强基站下行（窗外强源）到底在带内留下了什么？

判据分两条，必须分开看：
  · **阻塞/灵敏度塌方**：带内各段的底噪整体被抬起（A 相对 B 均匀升高）。
  · **IMD 产物落窗内**：与外源频差取 fs/3 ⇒ 三阶产物 3Δ 折叠到 DC，
    所以 **0–150 kHz 这一段的抬升应显著高于其它段**（不是整体平移）。

做法：读样本的私有 scenario.json，跑两遍完整接收链——
  A：原样（基站下行 power_dbm 原值）；B：把它压到 −100 dBm（等效移除）。
比较 station_a 复基带 IQ 的分段 PSD。
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

_DEFAULT = ADP.parent / "dataset_d12" / "d12_00000" / "capture" / "private" / "scenario.json"
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
if SRC.is_dir():
    SRC = SRC / "capture" / "private" / "scenario.json"
STATION = sys.argv[2] if len(sys.argv) > 2 else "station_a"
# argv[3]（可选）= 窗外强源的**到站端口功率** dBm。生成器把端口功率按最远站距离换算成自由空间
# 参考的 power_dbm（`power_dbm = port + fspl_db(d_max, fc_tx)`），这里用同一公式反推，
# 用来扫“阻塞多深才既明显又不把窗内源压死”。
PORT_DBM = float(sys.argv[3]) if len(sys.argv) > 3 else None
ROOT = ADP.parent / "_tmp_d12_ab"

# 分段（Hz）：DC 段 = IMD 产物落点；780k 以上 = 窗外（但仍在采样带内）
SEGS = [(0, 150e3, "DC 0–150k（IMD 落点）"), (150e3, 400e3, "150–400k"),
        (400e3, 780e3, "400–780k"), (780e3, 1.2e6, "780k–1.2M（窗外）"),
        (1.2e6, 2.0e6, "1.2M–2M（窗外）")]

sc = Scenario.model_validate_json(SRC.read_text(encoding="utf-8"))
blocker = next(e for e in sc.emitters if e.carrier_hz)
print(f"场景载频 {sc.carrier_frequency_hz/1e9:.4f} GHz   窗外源 {blocker.preset} "
      f"{blocker.carrier_hz/1e9:.6f} GHz @{blocker.power_dbm:.1f} dBm "
      f"(Δ = {(blocker.carrier_hz - sc.carrier_frequency_hz)/1e6:+.4f} MHz, fs = {sc.sample_rate_hz/1e6:.1f} MHz)")
print(f"前端 imd3={sc.receivers[0].frontend.imd3_enabled} iip3={sc.receivers[0].frontend.iip3_dbm:.1f} dBm "
      f"soft_limit={sc.receivers[0].frontend.soft_limit_input_dbm}   站 = {STATION}")
if PORT_DBM is not None:
    _idx = [i for i, e in enumerate(sc.emitters) if e.carrier_hz][0]
    _bl = sc.emitters[_idx]
    _d = max(float(np.linalg.norm(np.asarray(r.position_m, float) - np.asarray(_bl.position_m, float)))
             for r in sc.receivers)
    _fspl = 20.0 * float(np.log10(4.0 * np.pi * max(_d, 1.0) * float(_bl.carrier_hz) / 299792458.0))
    _bl.power_dbm = PORT_DBM + _fspl
    print(f"[扫参] 把窗外源端口功率设为 {PORT_DBM:+.1f} dBm ⇒ power_dbm = {_bl.power_dbm:.1f} "
          f"(最远站 {_d:.0f} m, FSPL {_fspl:.1f} dB)")

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
iq_a, iq_b = run(float(blocker.power_dbm), "with"), run(-100.0, "without")

fs = sc.sample_rate_hz
n = iq_a.shape[-1]
f = np.fft.fftfreq(n, 1 / fs)
pa = np.abs(np.fft.fft(iq_a[0].astype(np.complex128))) ** 2
pb = np.abs(np.fft.fft(iq_b[0].astype(np.complex128))) ** 2

db = lambda x: 10 * np.log10(float(x) + 1e-30)
print()
print(f"{'段':<24}{'A 有强源':>11}{'B 移除':>11}{'Δ(A−B)':>11}")
rows = []
for lo, hi, name in SEGS:
    m = (np.abs(f) >= lo) & (np.abs(f) < hi)
    a, b = db(np.median(pa[m])), db(np.median(pb[m]))
    rows.append((name, a, b, a - b))
    print(f"{name:<24}{a:11.1f}{b:11.1f}{a - b:+11.1f}")

dc = next(r for r in rows if r[0].startswith("DC"))
others = [r for r in rows if not r[0].startswith("DC")]
mean_other = float(np.mean([r[3] for r in others]))
print()
print(f"结论：全窗平均抬升 {mean_other:+.1f} dB（阻塞/灵敏度塌方），"
      f"其中 DC 段抬升 {dc[3]:+.1f} dB（比其它段高 {dc[3] - mean_other:+.1f} dB ⇒ IMD 产物落带内）")
print(f"提示：判定「IMD 真落窗内」看的是 DC 段相对其它段的**差值**；只看整段抬升分不清阻塞与互调。")

# —— 可检测性口径（真正决定题目能不能答）：目标带内峰值 vs 目标带外邻近底噪 ——
m_sig = np.abs(f) <= 0.45e6          # 目标（UAV_video_OFDM ≈1 MHz）占据的带
m_fl = (np.abs(f) > 0.55e6) & (np.abs(f) <= 0.78e6)   # 目标外、通带内的邻近底噪
print()
print(f"{'':16}{'目标峰(dB)':>12}{'邻近底噪(dB)':>14}{'SNR(dB)':>10}")
snr = {}
for tag, pa_ in (("A 有强源", pa), ("B 移除", pb)):
    pk = db(float(np.max(pa_[m_sig])))
    fl = db(float(np.median(pa_[m_fl])))
    snr[tag] = pk - fl
    print(f"{tag:16}{pk:12.1f}{fl:14.1f}{pk - fl:10.1f}")
d_snr = snr["B 移除"] - snr["A 有强源"]
print(f"⇒ 窗外强源造成的目标 SNR 损失：**{d_snr:+.1f} dB**（正=变差）")
