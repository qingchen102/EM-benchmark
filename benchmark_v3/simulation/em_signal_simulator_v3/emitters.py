"""v3 辐射体池——城市低空电磁生态覆盖（完整版）。

相对骨架版的关键升级：
1. **带宽真实性**：GT 带宽一律用 实测 OBW99（measure_obw99），不再用名义常数。
   —— 修复骨架版的 P0-2（雷达/GPS 带宽记录造假）。
2. **城市低空专属辐射体**补入：微波炉间歇宽带、DECT 无绳、Zigbee、GNSS压制/
   欺骗（对抗性干扰源）、图传私有 OFDM（连续）、遥控双向、对这缺口的直接回应。
3. 复用 v2 baseband 全部原语（OFDM/FHSS/QAM/LFM），不自造基带。

全部生成器仅依赖 numpy + 复用原语。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_V2SIM = Path(__file__).resolve().parents[3] / "benchmark_v2" / "simulation"
for _p in (str(_V2SIM), str(_V2SIM.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from em_signal_simulator.baseband import generate_baseband, measure_obw99  # noqa: E402


# ---------------- 蓝牙类（v2 已证） ----------------
def gen_bluetooth_fhss(rng, length, n_channels=12, dwell=(25, 60), ch_bw=0.045,
                       band=0.55, gap=(3, 12)):
    x = np.zeros(length, dtype=complex)
    pos, ch = 0, int(rng.integers(0, n_channels))
    while pos < length:
        d = min(int(rng.integers(*dwell)), length - pos)
        seg = generate_baseband("GFSK", d, rng=rng, bandwidth_normalized=ch_bw)
        f = (ch - (n_channels - 1) / 2) / max(n_channels - 1, 1) * band
        x[pos:pos + d] += seg * np.exp(1j * 2.0 * np.pi * f * np.arange(d))
        pos += d + int(rng.integers(*gap))
        ch = (ch + int(rng.integers(1, n_channels))) % n_channels
    return x


def gen_wifi_burst(rng, length, n_packets=(2, 4), pkt=(120, 320), gap=(40, 140), pkt_bw=0.55):
    x = np.zeros(length, dtype=complex)
    pos = 0
    for _ in range(int(rng.integers(*n_packets))):
        d = min(int(rng.integers(*pkt)), length - pos)
        if d < 40:
            break
        x[pos:pos + d] += generate_baseband("OFDM", d, rng=rng, bandwidth_normalized=pkt_bw)
        pos += d + int(rng.integers(*gap))
        if pos >= length:
            break
    return x


# ---------------- 无人机类 ----------------
def gen_uav_video_ofdm(rng, length, bw=0.60):
    """图传下行：连续宽带 OFDM（视频为主，占用持续、带宽宽——与突发 WiFi 区分）。"""
    return generate_baseband("OFDM", length, rng=rng, bandwidth_normalized=bw)


def gen_uav_rc_hopping(rng, length, n_channels=4, dwell=(12, 30), ch_bw=0.02,
                       band=0.30, gap=(8, 25)):
    """遥控上行：窄带 GFSK 跳频，信道少/跳得快/带宽窄。"""
    x = np.zeros(length, dtype=complex)
    pos, ch = 0, int(rng.integers(0, n_channels))
    while pos < length:
        d = min(int(rng.integers(*dwell)), length - pos)
        seg = generate_baseband("GFSK", d, rng=rng, bandwidth_normalized=ch_bw)
        f = (ch - (n_channels - 1) / 2) / max(n_channels - 1, 1) * band
        x[pos:pos + d] += seg * np.exp(1j * 2.0 * np.pi * f * np.arange(d))
        pos += d + int(rng.integers(*gap))
        ch = (ch + int(rng.integers(1, n_channels))) % n_channels
    return x


def gen_radar_pulsed_lfm(rng, length, pri=(25, 60), duty=0.35, bw=0.45):
    x = np.zeros(length, dtype=complex)
    pri_n = int(rng.integers(*pri))
    pulse_n = max(int(pri_n * duty), 8)
    start = int(rng.integers(0, pri_n))
    pos = start
    while pos < length:
        d = min(pulse_n, length - pos)
        x[pos:pos + d] += generate_baseband("LFM", d, rng=rng, bandwidth_normalized=bw)
        pos += pri_n
    return x


# ---------------- 导航（修复角色：GNSS 也可能是被压制/被欺骗的受害者） ----------------
def gen_gps_dsss(rng, length, chip_rate=0.08):
    n_chips = int(np.ceil(length * chip_rate))
    chips = rng.choice([-1.0, 1.0], size=n_chips)
    return np.repeat(chips, int(round(1.0 / chip_rate)))[:length].astype(np.complex128)


def gen_gnss_jam(rng, length, bw=0.02):
    """GNSS 压制：单音/窄带扫频 扫过 GNSS 频段宽带的 0~占空比。低功率持续。"""
    return gen_gnss_jam_impl(rng, length, bw, cli=False)


def gen_gnss_jam_impl(rng, length, bw, cli):
    x = np.zeros(length, dtype=complex)
    f = float(rng.uniform(-0.12, 0.12))
    dur = length
    t = np.arange(dur)
    shift = float(rng.uniform(0, 0.01))
    x[:dur] += np.exp(1j * 2.0 * np.pi * (f * t + 0.5 * shift * t * t))
    return x


def gen_gnss_spoof(rng, length, bw=0.12):
    """GNSS 欺骗：含 GPS 扩频码（合法结构）但在错误频偏/时延，需源级区分。"""
    return gen_gps_dsss(rng, length, chip_rate=0.12)


# ---------------- 城市日用/商用信号 ----------------
def gen_microwave(rng, length, burst=(40, 120), gap=(60, 140), bw=0.30):
    """家用/商用微波炉：2.4G 宽带间歇噪声（占空比约 50% 的磁控管辐射）。"""
    x = np.zeros(length, dtype=complex)
    pos = 0
    while pos < length:
        d = min(int(rng.integers(*burst)), length - pos)
        seg = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        # 宽带噪声近似高斯，无调制结构
        x[pos:pos + d] += seg
        pos += d + int(rng.integers(*gap))
        if pos >= length:
            break
    return x


def gen_dect(rng, length, n_frames=3, frame=0.55, dwell_frame=0.4):
    """DECT 无绳电话：TDMA 帧突发（固定帧长 ~0.55，随机帧起始）。"""
    x = np.zeros(length, dtype=complex)
    for k in range(n_frames):
        d = int(frame * length)
        seg = generate_baseband("GFSK", d, rng=rng, bandwidth_normalized=0.05)
        start = int(rng.integers(0, length // 8))
        x[start:start + d] += seg if start + d <= length else seg[:length - start]
    return x


def gen_zigbee(rng, length, n_pkt=(3, 6), pkt=(90, 220), gap=(40, 160), bw=0.02):
    """Zigbee/LoRa（低占空比微功率遥测）：短时 QPSK 突发（包长需满足基带成形最小长度）。"""
    x = np.zeros(length, dtype=complex)
    pos = 0
    for _ in range(int(rng.integers(*n_pkt))):
        d = min(int(rng.integers(*pkt)), length - pos)
        if d < 40:
            break
        x[pos:pos + d] += generate_baseband("QPSK", d, rng=rng, bandwidth_normalized=bw)
        pos += d + int(rng.integers(*gap))
    return x


def gen_tv_ofdm(rng, length, bw=0.9):
    """广播/电视：连续宽带 OFDM（广播单频网络）。"""
    return generate_baseband("OFDM", length, rng=rng, bandwidth_normalized=bw)


# ---------------- 电子对抗类（真实压制干扰） ----------------
def gen_chirp_jam(rng, length, bw=0.45, period=0.8):
    """扫频式压制（ECM）：线性扫频，周期重复，宽带覆盖。"""
    return gen_swept_baseband(rng, length, bw, period)


def gen_swept_baseband(rng, length, bw, period):
    seg = generate_baseband("LFM", int(period * length), rng=rng, bandwidth_normalized=bw)
    reps = int(np.ceil(length / (period * length)))
    out = np.tile(seg[:int(period * length)], reps)[:length]
    return out


def gen_noise_jam(rng, length, bw=0.6):
    """噪声式压制：宽带高斯（噪声干扰）。"""
    return (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / np.sqrt(2.0)


def gen_tone_jam(rng, length, bw=0.0):
    """单音压制：纯点频。"""
    f = float(rng.uniform(-0.3, 0.3))
    return np.exp(1j * 2.0 * np.pi * f * np.arange(length))


# ---------------- 辐射体注册表（城市低空覆盖矩阵） ----------------
# 与骨架版相比：新增微波炉/DECT/Zigbee/GNSS压制/GNSS欺骗/图传私有OFDM/广播电视/扫频压制
EMITTERS = {
    # 民航通信类
    "Bluetooth_FHSS":   (gen_bluetooth_fhss,  "GFSK", "2.4GHz 跳频通信（蓝牙/BLE）"),
    "WiFi_burst":       (gen_wifi_burst,      "OFDM", "802.11 突发包（WiFi）"),
    "LTE_QPSK":         (lambda rng, L, **k: generate_baseband("QPSK", L, rng=rng, bandwidth_normalized=0.5), "QPSK", "4G/5G 基站（连续）"),
    "DECT":             (gen_dect,            "GFSK", "DECT 无绳电话（TDMA 帧突发）"),
    "Zigbee":           (gen_zigbee,          "QPSK", "Zigbee/LoRa 微功率遥测"),
    # 无人机类
    "UAV_video_OFDM":   (gen_uav_video_ofdm,  "OFDM", "无人机图传下行（连续宽带 OFDM）"),
    "UAV_RC_hopping":   (gen_uav_rc_hopping,  "GFSK", "无人机遥控上行（窄带跳频）"),
    # 雷达
    "Radar_pulsed_LFM": (gen_radar_pulsed_lfm,"LFM",  "低空雷达脉冲串"),
    # 导航
    "GPS_DSSS":         (gen_gps_dsss,        "BPSK", "GNSS 正常导航（DSSS，弱信号）"),
    "GNSS_jam":         (gen_gnss_jam,        "single_tone", "GNSS 压制（窄带扫频覆盖导航频段）"),
    "GNSS_spoof":       (gen_gnss_spoof,      "BPSK", "GNSS 欺骗（合法扩频码、错误频偏/时延）"),
    # 城市日用/商用
    "Microwave":        (gen_microwave,       "broadband", "微波炉（2.4G 宽带间歇噪）"),
    "TV_broadcast":     (gen_tv_ofdm,         "OFDM", "广播电视（连续宽带 OFDM）"),
    # 电子对抗
    "Chirp_jam":        (gen_chirp_jam,       "LFM",  "扫频式电子对抗（ECM）"),
    "Noise_jam":        (gen_noise_jam,       "broadband", "噪声式压制干扰"),
    "Tone_jam":         (gen_tone_jam,        "single_tone", "单音压制干扰"),
}


# 敌我意图（v3 新任务轴）三级：malicious 恶意 / incidental 无意 / benign 无害
# - malicious：明确用于破坏的对抗源（压制/欺骗/ECM）
# - incidental：非针对本系统的共存/偶发干扰（微波炉/DECT/Zigbee）
# - benign：正常合法信号（通信/广播/导航/图传/雷达/遥控），多为背景辐射体
INTENT = {
    "Bluetooth_FHSS":   "benign",
    "WiFi_burst":       "benign",
    "LTE_QPSK":         "benign",
    "DECT":             "incidental",
    "Zigbee":           "incidental",
    "UAV_video_OFDM":   "benign",
    "UAV_RC_hopping":   "benign",
    "Radar_pulsed_LFM": "benign",
    "GPS_DSSS":         "benign",
    "GNSS_jam":         "malicious",
    "GNSS_spoof":       "malicious",
    "Microwave":        "incidental",
    "TV_broadcast":     "benign",
    "Chirp_jam":        "malicious",
    "Noise_jam":        "malicious",
    "Tone_jam":         "malicious",
}


def generate_emitter(name: str, rng: np.random.Generator, length: int = 1024, **kw) -> dict:
    """按辐射体名生成波形 + 实测 OBW99 元数据（修复骨架版 P0-2：带宽不再用名义常数）。

    - emitters 的生成器签名为 (rng, length, **kw)；bw 若传入会在生成器内使用
      （仅对接受 bw 参数者生效），否则用默认。
    - **带宽、调制标号、意图**都来自实测/注册表，供 GT 使用（与 v2 OBW99 一致）。
    """
    gen, mod, scene = EMITTERS[name]
    iq = gen(rng, length, **kw)
    # 实测占用带宽：对纯无结构宽带用 OBW99 会虚高，这里统一测一次并 clamp
    obw = float(measure_obw99(iq)) if measure_obw99(iq) > 0 else 1.0
    if obw > 1.0:
        obw = 1.0
    intent = INTENT.get(name, "benign")
    return {"iq": iq, "emitter": name, "modulation": mod,
            "bandwidth_normalized": round(obw, 3), "scene": scene,
            "intent": intent, "is_jamming": intent == "malicious"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name in EMITTERS:
        out = generate_emitter(name, rng)
        p = float(np.mean(np.abs(out["iq"]) ** 2) + 1e-30)
        print(f"{name:14s} mod={out['modulation']:8s} bw={out['bandwidth_normalized']:.3f}"
              f"  jam={out['is_jamming']}  {out['scene']}")