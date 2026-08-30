"""低空场景辐射体池（v3）。

每个辐射体 = 生成函数 + 元数据（display_name / 调制 / 典型带宽 / 场景说明）。
相对 v2 干扰池的关键升级：**真实时域行为**——蓝牙/遥控的跳频信道序列、
WiFi 的突发包间隙、雷达的 PRI 脉冲串、GPS 的扩频——这些时域结构是
辐射体溯源任务的区分度来源（v2 只有连续波形，溯源无从下手）。

全部生成器只依赖 numpy + v2 baseband 原语（generate_baseband）。
状态：草案——辐射体清单与参数范围待师兄确认后定稿。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 复用 v2 已验证的基带原语（monorepo 内相对导入）
_V2SIM = Path(__file__).resolve().parents[3] / "benchmark_v2" / "simulation"
for _p in (str(_V2SIM), str(_V2SIM.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from em_signal_simulator.baseband import generate_baseband  # noqa: E402


def gen_bluetooth_fhss(rng, length, n_channels=12, dwell=(25, 60), ch_bw=0.045,
                       band=0.55, gap=(3, 12)):
    """蓝牙：GFSK 突发在 n_channels 个信道间伪随机跳频（跳驻留 + 包间隙）。"""
    x = np.zeros(length, dtype=complex)
    pos, ch = 0, int(rng.integers(0, n_channels))
    while pos < length:
        d = min(int(rng.integers(*dwell)), length - pos)
        seg = generate_baseband("GFSK", d, rng=rng, bandwidth_normalized=ch_bw)
        f = (ch - (n_channels - 1) / 2) / max(n_channels - 1, 1) * band
        t = np.arange(d)
        x[pos:pos + d] += seg * np.exp(1j * 2.0 * np.pi * f * t)
        pos += d + int(rng.integers(*gap))
        ch = (ch + int(rng.integers(1, n_channels))) % n_channels
    return x


def gen_wifi_burst(rng, length, n_packets=(2, 4), pkt=(120, 320), gap=(40, 140),
                   pkt_bw=0.55):
    """WiFi：OFDM 突发包——包内连续、包间有静默间隙（真实 802.11 是分发的）。"""
    x = np.zeros(length, dtype=complex)
    pos = 0
    for _ in range(int(rng.integers(*n_packets))):
        d = min(int(rng.integers(*pkt)), length - pos)
        if d < 40:
            break
        seg = generate_baseband("OFDM", d, rng=rng, bandwidth_normalized=pkt_bw)
        x[pos:pos + d] += seg
        pos += d + int(rng.integers(*gap))
        if pos >= length:
            break
    return x


def gen_uav_video_ofdm(rng, length, bw=0.60):
    """无人机图传：连续宽带 OFDM（视频下行，占用持续、带宽宽——与突发 WiFi 区分）。"""
    return generate_baseband("OFDM", length, rng=rng, bandwidth_normalized=bw)


def gen_uav_rc_hopping(rng, length, n_channels=4, dwell=(12, 30), ch_bw=0.02,
                       band=0.30, gap=(8, 25)):
    """无人机遥控：窄带 GFSK 跳频，信道少/跳得快/带宽窄（与蓝牙区分：信道数与带宽）。"""
    x = np.zeros(length, dtype=complex)
    pos, ch = 0, int(rng.integers(0, n_channels))
    while pos < length:
        d = min(int(rng.integers(*dwell)), length - pos)
        seg = generate_baseband("GFSK", d, rng=rng, bandwidth_normalized=ch_bw)
        f = (ch - (n_channels - 1) / 2) / max(n_channels - 1, 1) * band
        t = np.arange(d)
        x[pos:pos + d] += seg * np.exp(1j * 2.0 * np.pi * f * t)
        pos += d + int(rng.integers(*gap))
        ch = (ch + int(rng.integers(1, n_channels))) % n_channels
    return x


def gen_radar_pulsed_lfm(rng, length, pri=(25, 60), duty=0.35, bw=0.45):
    """低空雷达：脉冲化 LFM，固定 PRI + 随机起始（脉冲内线性调频）。"""
    x = np.zeros(length, dtype=complex)
    pri_n = int(rng.integers(*pri))
    pulse_n = max(int(pri_n * duty), 8)
    start = int(rng.integers(0, pri_n))
    pos = start
    while pos < length:
        d = min(pulse_n, length - pos)
        seg = generate_baseband("LFM", d, rng=rng, bandwidth_normalized=bw)
        x[pos:pos + d] += seg
        pos += pri_n
    return x


def gen_gps_dsss(rng, length, chip_rate=0.08):
    """GPS：DSSS/BPSK——随机 ±1 码片序列扩频（带宽 ≈ chip_rate 的 2 倍主瓣），功率极低。"""
    n_chips = int(np.ceil(length * chip_rate))
    chips = rng.choice([-1.0, 1.0], size=n_chips)
    seq = np.repeat(chips, int(round(1.0 / chip_rate)))[:length]
    return seq.astype(np.complex128)


# 辐射体注册表：名称 → (生成函数, 调制标签, 典型占用带宽, 场景说明)
EMITTERS = {
    "Bluetooth_FHSS":  (gen_bluetooth_fhss,  "GFSK", 0.55, "2.4GHz 跳频通信（蓝牙）"),
    "WiFi_burst":      (gen_wifi_burst,      "OFDM", 0.55, "802.11 突发包（WiFi）"),
    "UAV_video_OFDM":  (gen_uav_video_ofdm,  "OFDM", 0.60, "无人机图传下行（连续宽带）"),
    "UAV_RC_hopping":  (gen_uav_rc_hopping,  "GFSK", 0.30, "无人机遥控上行（窄带跳频）"),
    "Radar_pulsed_LFM":(gen_radar_pulsed_lfm,"LFM",  0.45, "低空雷达脉冲串"),
    "GPS_DSSS":        (gen_gps_dsss,        "BPSK", 0.16, "GPS L1 C/A（DSSS 扩频，弱信号）"),
}


def generate_emitter(name: str, rng: np.random.Generator, length: int = 1024,
                     **kw) -> dict:
    """按辐射体名生成一段波形 + 元数据。

    bw 仅对以 bw 为带宽参数的生成器生效（GPS_DSSS 等）；其余生成器使用
    自身的默认带宽参数（ch_bw/pkt_bw 等）。
    """
    gen, mod, typ_bw, scene = EMITTERS[name]
    bw = kw.pop("bw", typ_bw)
    try:
        iq = gen(rng, length, bw=bw, **kw)
    except TypeError:
        iq = gen(rng, length, **kw)
    return {"iq": iq, "emitter": name, "modulation": mod,
            "bandwidth_normalized": bw, "scene": scene}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name in EMITTERS:
        out = generate_emitter(name, rng)
        p = float(np.mean(np.abs(out["iq"]) ** 2) + 1e-30)
        print(f"{name:18s} power={p:8.3f}  mod={out['modulation']:5s}  {out['scene']}")
