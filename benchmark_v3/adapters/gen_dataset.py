"""D1–D4 数据集生成器：场景=城市 | 难度=场景包络 | 观测=S0/S1/S2(同采集分站评分)。

D1–D4 统一采集口径（2026-09-28）：
    - 采样率 4 MHz、通带 ±810 kHz——目标 1 MHz 完整在窗口内（旧 2 MHz 口径下
      D2–D4 目标被拦腰截断 50/50、干扰外缘 52–65% 越界的坑已修）；
    - D1 干扰：Hz 域与目标分离 ≥50 kHz（"看得见"）；D2–D4 干扰：共道重叠保留
      （难度本身），Hz 偏置 40–200 kHz / 带宽 400–800 kHz，外缘 ≤600 kHz 不越界；
    - **可辨识性约束**：任何源对不允许"深度重叠 × 全站 ΔDOA<15° × 时域同类"
      三维全糊（物理无解题）——违反重抽干扰布点，保证难而可解；
    - RT 实测可见性（achieved）对全部 RT 难度回填（答案键只含物理可观测的源）；
    - 合作目标先验写入 public observations.cooperative_sources。

用法：
    python gen_dataset.py --difficulty d1..d4 --count 50 --out ../dataset_dX
    （D2+ 需 WSL .venv-rt）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_PLATFORM = _V3 / "城市低空电磁信号仿真平台"
for _p in (str(_PLATFORM / "src"), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2" / "simulation"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scene_occupancy import SceneOccupancy  # noqa: E402
from cityrf.config import (Antenna, Component, Emitter, Frontend,  # noqa: E402
                           Propagation, Receiver, Scenario)
from cityrf.runner import run_scenario  # noqa: E402
from em_signal_simulator_v3.emitters import INTENT  # noqa: E402  # 敌我意图映射（T3 答案轴）
import evidence_layer  # noqa: E402  # 受害者层公开视图（规范 v1.0 §7）

FS_HZ = 4e6                          # 全难度统一采集带宽（2026-09-28：D2–D4 对齐 D1，
                                     # 目标 1 MHz 不再被通带拦腰截断；Hz 域物理量不变）
RX_FRAC = 0.45 * 0.9                 # 接收通带占 fs 比例（±810 kHz）
TGT_BANDWIDTH_HZ = 1e6               # 目标组件带宽（物理量，不随数据集变）
SEP_MARGIN_HZ = 50e3                 # 干扰-目标 / 干扰-通带边 保护间隔
DURATION = 0.01                      # 10 ms 窗（4 MHz = 40k 采样/站）
CARRIER = 2.4e9
NF_DB = 5.0
C = 299792458.0

# 时域粗类别（与 audit_separability 同表）：可辨识性约束的"时域维"
COARSE = {
    "LTE_QPSK": "cont", "UAV_video_OFDM": "cont", "TV_broadcast": "cont",
    "GNSS_spoof": "cont", "GNSS_jam": "cont", "Noise_jam": "cont",
    "Bluetooth_FHSS": "hop", "UAV_RC_hopping": "hop",
    "DECT": "tdma",
    "WiFi_burst": "burst", "Zigbee": "burst", "Microwave": "burst",
    "Chirp_jam": "sweep", "Radar_pulsed_LFM": "pulse",
    "Tone_jam": "tone", "tone": "tone", "noise": "noise",
}
# QA 约束（2026-10-03 拍板，三条）：
#   1) 每干扰 ≥3 站 INR≥3dB 可见（LoS+功率近似，生成期硬约束）——治 47% 硬上限
#   2) 干扰对在有源共享站上 |Δu| ≥ 2/M（有符号方向余弦；整条替换 Δazimuth——
#      Δazimuth 在端射区失效、|sin(doa)| 是互补变量都不是阵列分辨的量）
#   3) z 不可观测 → 定位按 2D 计（规范口径）
U_SEP_MIN = 2.0 / 8.0                # 8 元 0.5λ ULA 的 Rayleigh 限（2/M）
STATION_AXIS = (0.0, 1.0, 0.0)       # 平行水平阵轴（全数据集统一约定）


def _u_signed(src, station_pos) -> float:
    """有符号方向余弦 u = axis·unit(p−o)（阵列实际分辨的量；axis=(0,1,0)）."""
    d = np.asarray(src[:2], float) - np.asarray(station_pos[:2], float)
    return float(d[1] / (np.linalg.norm(d) + 1e-12))


def _los_ok(src, station_pos, occ) -> bool:
    return not occ.los_blocked(src, station_pos)


def _n_visible_stations(src, stations, occ) -> int:
    """LoS 可见站数（INR 终检由 achieved 承担；此处 LoS 近似是生成期闸门）。"""
    return sum(1 for s in stations if _los_ok(src, s, occ))


def _unidentifiable_pair(interferers, tgt, tgt_name, stations, occ=None) -> str | None:
    """可辨识性检查（仅**干扰×干扰对**，QA 约束 2）。

    深度重叠(>50%) × 时域同类的干扰对，若在**所有共享可见站**上
    |Δu| < 2/M（有符号方向余弦，阵列 Rayleigh 限）→ 该对在任何可见站都
    分辨不开（物理无解），返回描述；否则 None。

    目标×干扰对豁免：合作剥离走第四维路径，不依赖频/角/时域三维。
    频率重叠本身保留（D2 难度），只保证干扰间至少一维可分——难而可解。"""
    srcs = [(ifw["kind"], ifw["pos"], ifw["offset_hz"], ifw["bandwidth_hz"])
            for ifw in interferers]

    def ov_frac(f1, b1, f2, b2):
        o = min(f1 + b1 / 2, f2 + b2 / 2) - max(f1 - b1 / 2, f2 - b2 / 2)
        return max(0.0, o) / max(min(b1, b2), 1e-9)

    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            k1, p1, f1, b1 = srcs[i]
            k2, p2, f2, b2 = srcs[j]
            if ov_frac(f1, b1, f2, b2) <= 0.5:
                continue
            if COARSE.get(k1) != COARSE.get(k2):
                continue
            shared = [s for s in stations
                      if occ is None or (_los_ok(p1, s, occ) and _los_ok(p2, s, occ))]
            if not shared:
                return f"{k1} × {k2} (overlap>50%, 无共享可见站, 同为{COARSE.get(k1)})"
            du_min = min(abs(_u_signed(p1, s) - _u_signed(p2, s)) for s in shared)
            if du_min < U_SEP_MIN:
                return (f"{k1} × {k2} (overlap>50%, min|Δu|={du_min:.3f}"
                        f"<{U_SEP_MIN:.2f} @共享站{len(shared)}, 同为{COARSE.get(k1)})")
    return None


def _single_source_visible_ok(interferers, stations, occ, min_vis=3) -> str | None:
    """QA 约束 1：每个干扰 ≥3 站 LoS 可见（INR 终检由 achieved 承担）。
    返回违规描述或 None。"""
    for ifw in interferers:
        nv = _n_visible_stations(ifw["pos"], stations, occ)
        if nv < min_vis:
            return (f"{ifw['kind']} 仅 {nv} 站 LoS 可见 (<{min_vis})"
                    f"—— trio 类方法硬上限")
    return None

TARGETS = ["LTE_QPSK", "UAV_video_OFDM"]
INTERFERERS = ["tone", "noise"]      # D1 简单体制（自由模式，频偏 GT 精确）
COEX = ["Bluetooth_FHSS", "WiFi_burst", "Microwave", "Zigbee", "DECT",
        "VFD_noise", "Corona", "Welder"]  # 场景 v1.0 第一批扩展：+家电变频/电晕/电焊脉冲
STATION_SUBSETS = {"S0": ["station_a"],
                   "S1": ["station_a", "station_b", "station_c"],
                   "S2": ["station_a", "station_b", "station_c", "station_d"]}


def fspl_db(d_m: float, f_hz: float = CARRIER) -> float:
    return 20.0 * np.log10(max(d_m, 1.0)) + 20.0 * np.log10(f_hz) - 147.55


def noise_dbm(bw_hz: float) -> float:
    return -174.0 + 10.0 * np.log10(bw_hz) + NF_DB


def dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def az_deg(src, rx) -> float:
    """到达角（从站指向源），与 FreeSpaceBackend 约定一致。"""
    return float(np.degrees(np.arctan2(src[1] - rx[1], src[0] - rx[0])))


class Placer:
    """受约束随机布点：楼宇占用拒绝采样 + 几何包络 + 场景边界钳制。"""

    def __init__(self, occ: SceneOccupancy, inset: float = 60.0):
        self.occ = occ
        lo_x, lo_y, hi_x, hi_y = occ.extent
        self.bounds = (lo_x + inset, lo_y + inset, hi_x - inset, hi_y - inset)

    def _clamp(self, x, y):
        return (min(max(x, self.bounds[0]), self.bounds[2]),
                min(max(y, self.bounds[1]), self.bounds[3]))

    def los_blocked(self, p1, p2) -> bool:
        return self.occ.los_blocked(p1, p2)

    def target_nlos(self, rng, stations, **kw) -> tuple | None:
        return self.occ.target_nlos(rng, stations, **kw)

    def target(self, rng) -> tuple:
        p = self.occ.free_point(rng, z_low=30, z_high=80)
        if p is None:
            raise RuntimeError("找不到楼外目标点")
        x, y = self._clamp(p[0], p[1])
        return (x, y, p[2])

    def interferer(self, rng, tgt, d_low=400, d_high=900) -> tuple | None:
        p = self.occ.free_point_near(rng, tgt[0], tgt[1], d_low, d_high,
                                     z_low=10, z_high=50)
        if p is None:
            return None
        x, y = self._clamp(p[0], p[1])
        if not self.occ.is_free(x, y) or dist((x, y), tgt[:2]) < 120:
            return None
        return (x, y, p[2])

    def stations(self, rng, tgt, n=4) -> list[tuple]:
        base = rng.uniform(0, 360)
        out = []
        r = rng.uniform(350, 700)
        for k in range(n):
            ang = np.radians(base + k * 360 / n + rng.uniform(-25, 25))
            rr = r * rng.uniform(0.85, 1.2)
            cand = None
            for _ in range(300):
                x = tgt[0] + rr * np.cos(ang) * rng.uniform(0.8, 1.2)
                y = tgt[1] + rr * np.sin(ang) * rng.uniform(0.8, 1.2)
                x, y = self._clamp(x, y)
                if self.occ.is_free(x, y):
                    cand = (x, y, float(rng.uniform(5, 15)))
                    break
            if cand is None:                     # 兜底：全域随机找
                cand = self.occ.free_point(rng, z_low=5, z_high=15)
            if cand is None or dist(cand, tgt) < 120:
                cand = self.occ.free_point(rng, z_low=5, z_high=15)
            out.append(cand)
        return out

    def stations_center(self, rng, n=4) -> list[tuple]:
        """D3/D4：监测网固定环绕 AOI 中心（基础设施视角），目标漫游进入覆盖区。"""
        cx = (self.bounds[0] + self.bounds[2]) / 2
        cy = (self.bounds[1] + self.bounds[3]) / 2
        span = min(self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1])
        base = rng.uniform(0, 360)
        out = []
        for k in range(n):
            ang = np.radians(base + k * 360 / n + rng.uniform(-20, 20))
            rr = span * rng.uniform(0.18, 0.26)
            cand = None
            for _ in range(300):
                x = cx + rr * np.cos(ang) * rng.uniform(0.85, 1.15)
                y = cy + rr * np.sin(ang) * rng.uniform(0.85, 1.15)
                x, y = self._clamp(x, y)
                if self.occ.is_free(x, y):
                    cand = (x, y, float(rng.uniform(5, 15)))
                    break
            if cand is None:
                cand = self.occ.free_point(rng, z_low=5, z_high=15)
            out.append(cand)
        return out


def build_sample(idx: int, seed: int, occ: Placer, difficulty: str = "d1") -> dict:
    rng = np.random.default_rng(seed)
    fs = FS_HZ
    rx_bw = fs * RX_FRAC
    if difficulty in ("d3", "d4"):
        stations = occ.stations_center(rng, n=4)     # 监测网固定环布覆盖区
        tgt = occ.target_nlos(rng, stations) or occ.target(rng)  # 低 SNR 物理来源：楼后 NLoS
    else:
        tgt = occ.target(rng)
        stations = None                              # 依赖目标位置，稍后布设
    tgt_name = TARGETS[int(rng.integers(len(TARGETS)))]
    tgt_power = float(rng.uniform(10, 20) if difficulty in ("d1", "d2")
                      else rng.uniform(0, 10))       # D3/D4：低功率上行/远距

    interferers = []
    if difficulty == "d1":
        has_if = bool(rng.random() < 0.5)
        if has_if:
            ifp = occ.interferer(rng, tgt)
            if ifp is not None:
                interferers.append({"kind": INTERFERERS[int(rng.integers(len(INTERFERERS)))],
                                    "pos": ifp, "mode": "free"})
    elif difficulty == "d2":              # D2：固定 2 个共存辐射体（QA 布点四步
        # 落地后 3 源拥挤需 3 对约束同时成立、生成效率崩——留高难子档或移 D3）
        for kind in rng.choice(COEX, size=2, replace=False):
            ifp = None
            for _try in range(20):        # free_point 偶发失败重试——兜底不退 1 源
                ifp = occ.interferer(rng, tgt, d_low=250, d_high=800)
                if ifp is not None:
                    break
            if ifp is not None:
                interferers.append({"kind": str(kind), "pos": ifp, "mode": "preset"})
            else:
                raise RuntimeError(f"干扰 {kind} 布点 20 次失败（采样域过挤）")
    else:                                  # D3/D4：恶意干扰入场（电子对抗压制）
        mal = ["Chirp_jam", "Noise_jam", "Tone_jam"]
        kinds = [str(k) for k in rng.choice(mal, size=1, replace=False)]
        if difficulty == "d4":
            kinds.append("GNSS_spoof")     # D4：欺骗类（合法结构+错误参数）
        if rng.random() < 0.5:             # D3/D4：50% 再混一个共存源
            kinds.append(str(rng.choice(COEX)))
        for kind in kinds:
            ifp = occ.interferer(rng, tgt, d_low=250, d_high=800)
            if ifp is not None:
                interferers.append({"kind": kind, "pos": ifp, "mode": "preset"})

    if difficulty not in ("d3", "d4"):
        stations = occ.stations(rng, tgt, n=4)

    # 频偏/带宽 + 功率反推 + 可辨识性约束（2026-09-28）：源对不允许
    # "深度重叠 × 全站 ΔDOA<15° × 时域同类"三维全糊（物理无解题）——
    # 违反则重抽干扰布点与频偏，重新反推功率，保证题目"难而可解"。
    def _draw_ifw_freq(ifw_in, diff_in, rng_in):
        """按难度抽 (频偏, 带宽)。D1：Hz 域显式分离；D2–D4：混合重叠度三档
        （场景 v1.0 定死：深/部分/分离各约 1/3，模拟现实 ISM 全频段散布）。"""
        if diff_in != "d1":
            ov_mode = int(rng_in.integers(0, 3))
            if ov_mode == 0:      # 深重叠：中心 40–200k（目标带内），bw 400–800k
                ifw_in["offset_hz"] = float(rng_in.choice([-1, 1]) * rng_in.uniform(40e3, 200e3))
                ifw_in["bandwidth_hz"] = float(rng_in.uniform(400e3, 800e3))
            elif ov_mode == 1:    # 部分重叠：中心 200–500k（目标裙边），bw 300–500k
                ifw_in["offset_hz"] = float(rng_in.choice([-1, 1]) * rng_in.uniform(200e3, 500e3))
                ifw_in["bandwidth_hz"] = float(rng_in.uniform(300e3, 500e3))
            else:                 # 频谱分离：内缘≥目标半宽+50k，中心 600–700k，bw 100–200k
                ifw_in["offset_hz"] = float(rng_in.choice([-1, 1]) * rng.uniform(600e3, 700e3))
                ifw_in["bandwidth_hz"] = float(rng_in.uniform(100e3, 200e3))
            return
        # D1：Hz 域显式分离（目标半宽 500k，通带半宽 810k，两侧保护 50k）
        tgt_half = TGT_BANDWIDTH_HZ / 2
        rx_half = rx_bw / 2
        if ifw_in["kind"] == "tone":
            bw_eff, gt_bw = 50e3, 100.0
        else:
            bw_eff = gt_bw = float(rng_in.uniform(80e3, 180e3))
        lo = tgt_half + bw_eff / 2 + SEP_MARGIN_HZ
        hi = rx_half - bw_eff / 2 - SEP_MARGIN_HZ
        if lo >= hi:
            raise RuntimeError(f"D1 频率计划无分离空间: [{lo:.0f}, {hi:.0f}] Hz")
        ifw_in["offset_hz"] = float(rng_in.choice([-1, 1]) * rng_in.uniform(lo, hi))
        ifw_in["bandwidth_hz"] = gt_bw

    # QA 布点四步（2026-10-03 拍板）：LoS 可见掩码 → 域内采样 → 共享站
    # |Δu|≥2/M → 信道 INR 终检（achieved，由 runner 落盘）。
    occ_core = getattr(occ, "occ", occ)      # Placer 包装 or 裸 SceneOccupancy
    last_bad = "未检查"
    # 逐干扰独立重抽可见性（单点 ≥3 站 LoS 21%，整组重抽 4.4%/轮太低——
    # 60 轮有 6.6% 全败概率；独立重抽单干扰 5 次内收敛）；对间 |Δu| 整组判。
    for ifw in interferers:
        for _try in range(80):
            if _n_visible_stations(ifw["pos"], stations, occ_core) >= 3:
                break
            ifp = occ.interferer(rng, tgt, d_low=250, d_high=800)
            if ifp is not None:
                ifw["pos"] = ifp
        else:
            raise RuntimeError(f"QA 约束 1: {ifw['kind']} 80 次重抽仍 <3 站 LoS 可见")
    last_bad = "未检查"
    for _ in range(60):
        for ifw in interferers:
            d_max = max(dist(ifw["pos"], s) for s in stations)
            desired_inr = float(rng.uniform(8, 15) if difficulty == "d2" else rng.uniform(8, 20))
            ifw["power_dbm"] = round(min(desired_inr + noise_dbm(rx_bw) + fspl_db(d_max), 33.0), 2)
            _draw_ifw_freq(ifw, difficulty, rng)
        bad = _unidentifiable_pair(interferers, tgt, tgt_name, stations, occ=occ_core)
        if bad is None:
            last_bad = None
            break
        last_bad = bad
        for ifw in interferers:
            if ifw["mode"] == "preset":
                # 位置+频偏同抽：固定 AOI 站几何下（D4 三连续源）仅重抽位置
                # 自由度不足，拉开频偏可把深度重叠降为浅重叠。
                # 位置重抽后需复检可见性（QA 约束 1），失败则保留原位只动频偏。
                newp = occ.interferer(rng, tgt, d_low=250, d_high=800)
                if newp is not None and _n_visible_stations(newp, stations, occ_core) >= 3:
                    ifw["pos"] = newp
                _draw_ifw_freq(ifw, difficulty, rng)
    else:
        raise RuntimeError(f"QA 约束 2 (|Δu|≥2/M) 60 次重抽仍不满足: {last_bad}")

    ems = [Emitter(id="target", role="communication", position_m=tgt,
                   power_dbm=tgt_power, preset=tgt_name,
                   components=[Component(kind="noise", bandwidth_hz=1e6)])]
    for j, ifw in enumerate(interferers):
        if ifw["mode"] == "free":
            comps = [Component(kind=ifw["kind"], offset_hz=ifw["offset_hz"],
                               bandwidth_hz=ifw["bandwidth_hz"])]
        else:                              # D2 预设辐射体
            comps = [Component(kind="noise", bandwidth_hz=ifw["bandwidth_hz"],
                               offset_hz=ifw["offset_hz"])]
        ems.append(Emitter(id=f"interferer_{j + 1}", role="interference",
                           position_m=ifw["pos"], power_dbm=ifw["power_dbm"],
                           preset=ifw["kind"] if ifw["mode"] == "preset" else None,
                           components=comps))
    rxs = [Receiver(id=f"station_{chr(ord('a') + i)}", position_m=s,
                    bandwidth_hz=rx_bw, frontend=Frontend(),
                    antenna=Antenna(geometry_type="rectangular", num_rows=1, num_cols=8))
           for i, s in enumerate(stations)]

    if difficulty == "d1":
        prop = Propagation(backend="free_space")
    else:                                  # D2+：城市真实建筑 RT（多径+遮挡）
        prop = Propagation(backend="sionna_rt",
                           scene="priors/hangzhou_geometry/scene.xml")
    if difficulty == "d4":
        # D4：强阻塞触发非线性 + 阵元相关噪声 + 机动多普勒
        for r in rxs:
            r.frontend.imd3_enabled = True
            r.frontend.iip3_dbm = float(rng.uniform(5, 15))
            r.frontend.correlated_noise_enabled = True
            r.frontend.noise_correlation = 0.3
        ems[0].velocity_mps = (float(rng.uniform(-30, 30)), float(rng.uniform(-30, 30)), 0.0)
    sc = Scenario(name=f"{difficulty}_{idx:05d}", seed=seed, duration_s=DURATION,
                  sample_rate_hz=fs, carrier_frequency_hz=CARRIER,
                  max_working_mb=4096, emitters=ems, receivers=rxs, propagation=prop)

    # ---- GT：几何 + 物理推导功率 ----
    nz = noise_dbm(rx_bw)
    tgt_gt = {"emitter": tgt_name, "role": "target", "power_dbm": tgt_power,
              "position_m": list(tgt), "freq_offset_hz": 0.0,
              "per_station": {}}
    for s in stations:
        d = dist(tgt, s)
        pr = tgt_power - fspl_db(d)
        tgt_gt["per_station"][f"s{stations.index(s)}"] = {
            "dist_m": round(d, 1), "doa_deg": round(az_deg(tgt, s), 2),
            "pr_dbm": round(pr, 2), "snr_db": round(pr - nz, 2),
            "los_blocked": bool(occ.los_blocked(tgt, s)) if difficulty != "d1" else False}
    ifw_gts = []
    for ifw in interferers:
        ps = {}
        for i, s in enumerate(stations):
            d = dist(ifw["pos"], s)
            pr = ifw["power_dbm"] - fspl_db(d)
            ps[f"s{i}"] = {"dist_m": round(d, 1), "doa_deg": round(az_deg(ifw["pos"], s), 2),
                           "pr_dbm": round(pr, 2), "inr_db": round(pr - nz, 2)}
        rec = {"kind": ifw["kind"], "role": "interferer", "mode": ifw["mode"],
               "power_dbm": ifw["power_dbm"], "position_m": list(ifw["pos"]),
               "freq_offset_hz": ifw["offset_hz"],
               "bandwidth_hz": ifw["bandwidth_hz"], "per_station": ps}
        if ifw["mode"] == "preset":
            rec["intent"] = INTENT[ifw["kind"]]   # T3 敌我意图答案轴（16 辐射体确定性映射）
        ifw_gts.append(rec)

    envelope = ("近距合作目标+0/1简单干扰+视距+仅热噪声" if difficulty == "d1" else
                ("中距合作目标+2共存辐射体(深重叠配额,3源留高难子档)+城市RT多径/遮挡+仅热噪声" if difficulty == "d2" else
                 ("楼后NLoS低SNR+恶意压制干扰+共存混合+城市RT" if difficulty == "d3" else
                  "欺骗+强阻塞非线性+机动多普勒+相关噪声+NLoS+城市RT")))
    truth = {"schema_version": "d1-0.1", "sample_id": f"{difficulty}_{idx:05d}",
             "seed": seed, "difficulty": difficulty.upper(), "scene_type": "urban",
             "envelope": envelope,
             "target": tgt_gt, "interferers": ifw_gts,
             "receivers": [{"id": f"station_{chr(ord('a') + i)}", "position_m": list(s)}
                           for i, s in enumerate(stations)],
             "station_subsets": STATION_SUBSETS,
             "noise": {"model": "thermal", "nf_db": NF_DB,
                       "noise_floor_dbm": round(nz, 2)}}
    # 受害者层（场景 v1.0 §7）：干扰事件表——恶意源的瞄准关系与影响分级。
    # D2 共存源无干扰事件（背景辐射体）；D3/D4 恶意源 aimed=true 朝向目标，
    # power_at_victim=干扰到达目标无人机接收机处的功率，impact 按绝对压制电平分级
    # （≥-60dBm severe=接收机被压死 / -60~-85 moderate / <-85 negligible）。
    if difficulty in ("d3", "d4"):
        events = []
        for j, ifw in enumerate(interferers):
            if ifw["mode"] != "preset" or ifw["kind"] not in ("Chirp_jam", "Noise_jam",
                                                              "Tone_jam", "GNSS_spoof", "GNSS_jam"):
                continue
            d_v = dist(ifw["pos"], tgt)
            p_at_victim = ifw["power_dbm"] - fspl_db(d_v)
            env_db = p_at_victim - nz     # 目标处干扰环境强度（超噪底）
            impact = ("severe" if env_db >= 20.0 else
                      "moderate" if env_db >= 10.0 else "negligible")
            events.append({"event_id": f"E{j + 1}", "source_id": f"interferer_{j + 1}",
                           "mechanism": "spoof" if ifw["kind"] == "GNSS_spoof" else "jam",
                           "victim_id": "target_uav_rx", "aimed": True,
                           "power_at_victim_dbm": round(p_at_victim, 2),
                           "env_over_noise_db": round(env_db, 1),
                           "impact_class": impact})
        truth["interference_events"] = events
    return {"scenario": sc, "truth": truth, "out_name": f"{difficulty}_{idx:05d}"}


def _link_quality_telemetry(truth: dict) -> dict:
    """合作图传链路质量遥测——图传接收端（无人机自身 + 地面站）的自测值。

    物理口径：各监测站收到的图传信号功率 vs 同站同频段的干扰+噪声 → SINR →
    期望未编码 QPSK 符号误码率。这是监测网络对合作链路的质量监控（合法 public
    证据），直接回答"链路质量是否退化"。注意：D2 场景暂无"受害者接收机"
    （无人机收遥控/GNSS 的链路未建模），受害者层的 impact 用"目标处干扰环境
    强度"表达（见 interference_events）。遥测不暴露干扰身份。"""
    fspl = globals()["fspl_db"]
    tgt = np.asarray(truth["target"]["position_m"], float)
    tgt_p = truth["target"]["power_dbm"]
    noise_p = 10 ** ((truth["noise"]["noise_floor_dbm"] - 30) / 10)   # W
    out = {}
    # 各监测站视角：图传信号 vs 干扰+噪声在该站（监测站收图传的质量）
    for r in truth["receivers"]:
        sid = r["id"]
        st = np.asarray(r["position_m"], float)
        sig_dbm = tgt_p - fspl_db(float(np.linalg.norm(tgt - st)))
        ifw_lin = 0.0
        for j, g in enumerate(truth["interferers"]):
            ip = g["power_dbm"] - fspl_db(float(np.linalg.norm(
                np.asarray(g["position_m"], float) - st)))
            ifw_lin += 10 ** ((ip - 30) / 10)
        sinr_db = sig_dbm - 10 * np.log10(ifw_lin + noise_p)
        sinr_lin = 10 ** (sinr_db / 10)
        ser = 0.5 * _math_erfc(np.sqrt(sinr_lin))
        out[sid] = {"sinr_db": round(sinr_db, 2),
                    "expected_ser": float(f"{ser:.3e}"),
                    "expected_packet_loss": round(float(1 - (1 - ser) ** 1000), 4)}
    return out


def _math_erfc(x: float) -> float:
    import math
    return math.erfc(x / np.sqrt(2))


def cooperative_sources_prior(truth: dict) -> dict:
    """合作目标先验（考卷上印的合法先验）：型号 + 名义带宽 + 合作链路质量遥测。

    平台 runner 每次 replay 都会整体重写 observations.json（`cityrf/runner.py:102`），
    抹掉本函数写入的字段；因此 `benchmark_v3/adapters/expose_evidence.py --coop`
    负责在 replay 之后把它恢复回 public。两条路径的 `_target_bw_normalized` 已用
    `_probe_prior.py` 对全部 200 个样本核对逐位相同（等价的理由：目标 preset 的
    组件带宽恒为 TGT_BANDWIDTH_HZ，或该 preset 不受理带宽参数）。
    """
    return {
        "source_id": "target", "role": "cooperative_target",
        "preset": truth["target"]["emitter"],
        "nominal_bandwidth_hz": TGT_BANDWIDTH_HZ,
        "link_quality_per_station": _link_quality_telemetry(truth),
        "note": ("cooperative UAV link monitored by this network (public prior). "
                 "link_quality_per_station = the cooperative ground-station's OWN "
                 "measurements of its video-downlink reception quality (expected "
                 "uncoded QPSK symbol-error-rate and per-station SINR estimate). "
                 "Degradation indicates co-channel interference on the link."),
    }


def _index_entry(t: dict, out_name: str) -> dict:
    """index.json 的单样本条目（正常生成与 --resume 跳过共用同一口径）。

    seed 一律取**真值里的实际 seed**（= seed0 + 候选序号 [+ 1000*QA 重试次数]）。
    旧四档数据集里 index 记的是名义 seed0+i，当时 QA 不重试、两者恰好相等；
    一旦 QA 重试或有 --qa-skip 跳号，名义值就无法复现该样本，故以真值为准。
    """
    snrs = [v["snr_db"] for v in t["target"]["per_station"].values()]
    return {"id": out_name, "seed": t["seed"],
            "target": {"emitter": t["target"]["emitter"],
                       "snr_db_range": [min(snrs), max(snrs)]},
            "interferers": [{"kind": g["kind"],
                             "inr_db_range": [min(v["inr_db"] for v in g["per_station"].values()),
                                              max(v["inr_db"] for v in g["per_station"].values())]}
                            for g in t["interferers"]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--difficulty", choices=["d1", "d2", "d3", "d4"], default="d1")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed0", type=int, default=31000)
    ap.add_argument("--resume", action="store_true",
                    help="跳过已完整生成的样本（断点续跑；不改变任何 RNG 语义）")
    ap.add_argument("--qa-skip", action="store_true",
                    help="单个候选连续 6 次 QA/RT 失败时不中断，跳过该候选继续下一个；"
                         "配合 --count 用一个大候选池凑满 --want 条（与 --resume 互斥）")
    ap.add_argument("--want", type=int, default=None,
                    help="需要凑满的样本数（默认 = --count）")
    args = ap.parse_args()
    if args.qa_skip and args.resume:
        # 跳过候选会让 seed↔槽位映射不可复原（候选序号从 0 重来、已消费的 seed 会被重复使用）
        ap.error("--qa-skip 与 --resume 互斥：跳过候选后候选序号↔槽位的映射不可复原")
    if args.want is not None and args.want > args.count:
        ap.error(f"--want ({args.want}) 不能大于 --count ({args.count})：候选池不够大")
    if args.out is None:
        args.out = str(_V3 / f"dataset_{args.difficulty}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    occ = SceneOccupancy.load(_PLATFORM / "priors" / "hangzhou_geometry" / "occupancy.npz")
    placer = Placer(occ)

    index = {"dataset": f"EM-interference {args.difficulty.upper()}",
             "difficulty": args.difficulty.upper() + {"d1": "(看得见)", "d2": "(挤得慌)",
                                                      "d3": "(有人捣乱)", "d4": "(极限藏匿)"}[args.difficulty],
             "scene_type": "urban", "scene": "hangzhou_downtown(footprint-validated)",
             "observation_subsets": STATION_SUBSETS,
             "envelope": {"target": ("合作无人机 10-20dBm @350-700m, 高30-80m"
                                     if args.difficulty in ("d1", "d2") else
                                     "合作无人机 0-10dBm 楼后NLoS, 高30-80m, D4含机动多普勒"),
                          "interferer": ("50%无/50%单音或窄带噪声, 全站INR≥3dB(最远站反推), "
                                         "Hz域分离: |f|-bw/2≥目标半宽+50kHz 且 通带内裕量≥50kHz"
                                         if args.difficulty == "d1" else
                                         ("2共存辐射体(蓝牙/WiFi/微波炉/Zigbee/DECT), 全站INR≥3dB, "
                                          "共道重叠+可辨识性约束(深度重叠源对必须角度可分)"
                                          if args.difficulty == "d2" else
                                          ("1-2恶意压制(扫频/噪声/单音)+50%共存源, 全站INR≥3dB, "
                                           "可辨识性约束+RT实测可见性回填"
                                           if args.difficulty == "d3" else
                                           "GNSS欺骗+恶意压制+50%共存源, 可辨识性约束+RT实测可见性回填"))),
                          "propagation": ("free_space(视距)" if args.difficulty == "d1"
                                          else "sionna_rt 城市真实建筑(多径/遮挡)"),
                          "noise": "thermal only"},
             "samples": []}
    want = args.want or args.count
    accepted = 0
    cand = 0
    while cand < args.count and accepted < want:
        seed = args.seed0 + cand
        cand += 1
        # 输出槽位按**已接受**数命名（连续无空洞），候选序号单独推进：
        # 某个候选的 QA 几何达不到就跳过去用下一个 seed0 候选，而不是整个重跑换 seed0。
        out = out_root / f"{args.difficulty}_{accepted:05d}" / "capture"
        if args.resume and (out / "private" / "d1_truth.json").exists():
            # 断点续跑：直接复用磁盘上已完成的真值重建 index 条目（不动 RNG）
            t_done = json.loads((out / "private" / "d1_truth.json").read_text(encoding="utf-8"))
            index["samples"].append(_index_entry(t_done, f"{args.difficulty}_{accepted:05d}"))
            print(f"  [{accepted + 1}/{want}] {args.difficulty}_{accepted:05d} 已存在，跳过")
            accepted += 1
            continue
        spec = None
        for attempt in range(6):             # RT 偶发越界/遮挡/QA坏几何 → 整样本重采样
            try:
                spec = build_sample(accepted, seed + 1000 * attempt, placer, args.difficulty)
            except RuntimeError as e:        # QA 约束不满足=该目标几何下可见区稀缺
                if attempt == 5:
                    spec = None
                    break
                print(f"    [{args.difficulty}_{accepted:05d}] 第{attempt+1}次 QA 布点失败"
                      f"（{str(e)[:50]}），换 seed 重试 …")
                continue
            if out.exists():
                shutil.rmtree(out)
            try:
                run_scenario(spec["scenario"], out)
                break
            except (ValueError, RuntimeError) as e:
                if attempt == 5:
                    spec = None
                    break
                print(f"    [{spec['out_name']}] 第{attempt+1}次布设失败（{str(e)[:60]}），重采样 …")
        if spec is None:
            if not args.qa_skip:
                raise RuntimeError(
                    f"{args.difficulty} 候选 seed={seed} 连续 6 次失败（QA 或 RT 布设）；"
                    f"加 --qa-skip 可放弃该候选继续凑数")
            print(f"  !! 放弃候选 {cand}/{args.count}（seed={seed}，QA/RT 连续 6 次失败），"
                  f"当前 {accepted}/{want}")
            continue
        truth = spec["truth"]
        # 合作目标先验写入 public（平台 schema 预留字段）：Agent 只读 public 即可拿到
        # 合法先验（型号+名义带宽），信息隔离边界=考卷上印什么，而非运行器私下读 private。
        obs_p = out / "public" / "observations.json"
        obs = json.loads(obs_p.read_text(encoding="utf-8"))
        obs["cooperative_sources"] = [cooperative_sources_prior(truth)]
        # 受害者层公开视图（场景设计规范 v1.0 §7）：机理 + 影响等级可见，
        # aimed / victim / 链路预算屏蔽（取舍与理由见 evidence_layer.py）。
        # 无事件（D1/D2）时不写 key，历史 observations.json 保持逐字节一致。
        n_ev = evidence_layer.attach_public_events(obs, truth)
        if n_ev:
            print(f"    [{truth['sample_id']}] 受害者层公开事件 {n_ev} 条 -> observations.json")
        obs_p.write_text(json.dumps(obs, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.difficulty != "d1":
            # RT 实测可见性回填（2026-09-28 起对全部 RT 难度生效；此前只对 d2 回填是
            # 实现疏漏，D3/D4 答案键曾含大量 0 径=物理收不到的幽灵干扰）
            pt = json.loads((out / "private" / "truth.json").read_text(encoding="utf-8"))
            kmap = {r["id"]: f"s{i}" for i, r in enumerate(truth["receivers"])}
            ach: dict[str, dict] = {}
            for st in pt["stations"]:
                sk = kmap[st["receiver_id"]]
                for src in st["sources"]:
                    pw = float(np.mean(src["mean_port_power_w"]))
                    e = ach.setdefault(src["emitter_id"], {})
                    e[sk] = {"pr_dbm": round(10 * np.log10(pw + 1e-30) + 30, 2),
                             "path_count": src["path_count"],
                             "path_azimuths_deg": [round(np.degrees(a), 1)
                                                   for a in src["geometry"]["arrival_azimuth_rad"][:6]]}
            truth["achieved"] = ach
        (out.parent / "capture" / "private" / "d1_truth.json").write_text(
            json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
        t = truth
        snrs = [v["snr_db"] for v in t["target"]["per_station"].values()]
        index["samples"].append(_index_entry(t, spec["out_name"]))
        print(f"  [{accepted + 1}/{want}] {spec['out_name']}  "
              f"SNR {min(snrs):.0f}~{max(snrs):.0f}dB  "
              f"干扰={len(t['interferers'])}个:"
              + (",".join(g["kind"] for g in t["interferers"]) or "无"))
        accepted += 1
    (out_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    snr_all = [s["target"]["snr_db_range"] for s in index["samples"]]
    n_if = sum(1 for s in index["samples"] if s["interferers"])
    print(f"完成 {accepted} 条（候选池 {cand}）："
          f"SNR {min(a for a,_ in snr_all):.0f}~{max(b for _,b in snr_all):.0f} dB，"
          f"含干扰 {n_if} 条 / 无干扰 {accepted-n_if} 条；S0/S1/S2 由 station_subsets 切分。")
    if accepted < want:
        raise RuntimeError(f"候选池用尽（{args.count} 个）只凑到 {accepted}/{want} 条；"
                           f"加大 --count 重跑")


if __name__ == "__main__":
    main()