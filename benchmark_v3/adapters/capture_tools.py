"""平台 capture 格式 → Agent 工具适配层（benchmark_v3 D1–D4 数据集考生侧仪器）。

角色边界（沿 benchmark 概念图）：工具=仪器，只返回测量值/候选证据，不做最终结论。

- v2 五件工具的算法核心原样复用（谱分析/源数/时域/调制切片）。输入经单位 RMS
  归一化缓存：v2 工具的全部特征（PAPR/平坦度/峰度/占空比）与 CNN 后端（内部自带
  RMS 归一化）都是尺度不变的，全局归一化只影响可读性、不影响测量有效性。
- 角度与定位用**方向余弦（u 空间）仪器**：水平 ULA 的阵列流形只依赖
  u = sin(az)·cos(el)（阵轴方向余弦），(az, el) 的表示整条脊退化——3D 网格
  MUSIC 会把同一源报成多个"伪峰"。1D u 扫描无简并；源方位角由 asin(u) 给出，
  天生带 ULA 前后瓣镜像对（az 与 180°−az 响应相同，单站原理上不可分辨，
  多站锥面交会按残差消解）。
- 定位 = 各站锥面约束 unit(p−o_i)·a_i = u_i 的网格最小二乘（a_i 为该站阵轴
  世界方向），对 (x, y, h) 粗网格 + 局部细化；干扰带与目标带重叠时目标会作为
  另一假设出现，由 Agent 用"先定位目标再排除"链路消歧。
- 站点白名单按 S 级强制；工具结果在同一样本内按参数缓存（CNN/空间扫描开销大
  且确定性）。频率轴：v2 工具内部归一化到 fs，包装层并列输出 Hz 值（fs=2 MHz）。

信息隔离：工具只读 capture/public/（observations.json + 各站 IQ），不接触 private。
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_PLATFORM_SRC = _V3 / "城市低空电磁信号仿真平台" / "src"
_V2 = _V3.parent / "benchmark_v2"
for _p in (str(_PLATFORM_SRC), str(_V2), str(_V2 / "simulation"),
           str(_V3 / "simulation"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools_v2  # noqa: E402  v2 五件工具核心（零改复用）
from cityrf.baselines.spatial import band_covariance, mdl_rank  # noqa: E402
from cityrf.baselines.checked_signal import signal_block, select_band  # noqa: E402
from cityrf.baselines.checked_features import template_residual  # noqa: E402
from em_signal_simulator_v3.emitters import (  # noqa: E402
    EMITTERS, INTENT as INTENT_OF)   # 意图表唯一来源：与答案键（gen_dataset.py）同一张

# ---------------- 残差峰判据（2026-10-08 重修，D1/S0 虚警根治） ----------------
# 机理（本轮实测）：target_reference 直减后的残差在通带内有一层**平坦 ~-88 dB** 的
# 能量（热噪声底仅 -131 dB）——即参考回放里 ~0.15% 的失配能量，宽频带、**跨阵元
# 不相干**。旧判据用"整通带 PSD 中位"当噪声底 + ≥5 bin 连段，会把它与目标滚降段
# 并成一个 ~850 kHz 宽的"强峰"（46~49 dB）→ 干净场景虚警、真源带宽/高度全失真。
# 修法两条：①噪声底改**局部中位滤波**（不被目标滚降带偏）；②每个候选带加
# **窄窗空间相干门控** λ1/Σλ（真源=点源≈1，失配能量≈0.15，8 元阵白噪≈0.125）。
# 定阈依据 `_diag_bands_v2.py`：D1 12 样本上 0 虚警、GT 全覆盖（真 band coh 中位 0.97，
# 假 band 一个都没有；旧判据在干净样本上产生 50 个假峰）。
PEAK_OVER_DB = 8.0          # 候选带需高于局部底的门限（dB）
PEAK_COH_MIN = 0.6          # 候选带中心 ±20k 的空间相干门限
PEAK_COH_WIN_HZ = 20e3      # 相干量测窗（±）
PEAK_FLOOR_WIN_HZ = 300e3   # 局部底的中位滤波窗
PEAK_STEP_HZ = 10e3         # 占用曲线的扫描步长
PEAK_MAX_N = 6              # 单站最多返回带数


def _band_coherence(iq: np.ndarray, fs: float, center_hz: float,
                    win_hz: float) -> float:
    """中心 ±win 窄窗内的空间相干度 = 子带协方差第一特征值占比 λ1/Σλ。

    点源（含相干多径）→ 秩 1 → 接近 1；跨阵元不相关的失配能量/热噪声 → 接近 1/M。
    量测窗内先按列归一化，避免带内功率沿频起伏影响特征值占比。"""
    n = iq.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = np.abs(f - center_hz) <= win_hz
    if int(m.sum()) < 2:
        return float("nan")
    x = np.fft.fft(iq, axis=-1)[:, m]
    x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-30)
    r = x @ x.conj().T / x.shape[1]
    w = np.linalg.eigvalsh(r)[::-1].real
    return float(w[0] / (w.sum() + 1e-30))


def _residual_peaks_legacy(resid: np.ndarray, fs: float, rx_half: float) -> list[dict]:
    """2026-10-08 前的旧峰判据（全局中位底 + >=5 bin 连段），仅供 A/B 复核。

    已知缺陷：把参考失配能量与目标滚降段并成 ~850 kHz 宽的假"强峰"（D1 干净样本
    虚警的根因）。设 DSH_PEAK_LEGACY=1 可切回，用于对比历史数字。"""
    n = resid.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    inband = np.where((f >= -rx_half) & (f <= rx_half))[0]
    psd = np.abs(np.fft.fft(resid, axis=-1)) ** 2
    psd_ib = psd[:, inband].mean(axis=0)
    db = 10 * np.log10(psd_ib + 1e-30)
    floor = float(np.median(db))
    over = db - floor
    hot = over >= 8.0
    peaks: list[dict] = []
    k = 0
    while k < len(hot):
        if hot[k]:
            k2 = k
            while k2 + 1 < len(hot) and hot[k2 + 1]:
                k2 += 1
            if k2 - k + 1 >= 5:
                seg = over[k:k2 + 1]
                kc = k + int(np.argmax(seg))
                peaks.append({"freq_offset_hz": round(float(f[inband[kc]]), 0),
                              "over_floor_db": round(float(seg.max()), 1),
                              "bandwidth_hz": round((k2 - k + 1) * fs / n, 0),
                              "band_hz": [round(float(f[inband[k]]), 0),
                                          round(float(f[inband[k2]]), 0)]})
            k = k2 + 1
        else:
            k += 1
    peaks.sort(key=lambda p: -p["over_floor_db"])
    return peaks[:PEAK_MAX_N]


def _residual_bands(resid: np.ndarray, fs: float, rx_half: float) -> list[dict]:
    """残差 → 候选干扰带清单（局部底 + 相干门控，见上方机理注释）。"""
    n = resid.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    inb = np.where((f >= -rx_half) & (f <= rx_half))[0]
    if inb.size == 0:
        return []
    psd = (np.abs(np.fft.fft(resid, axis=-1)) ** 2).mean(axis=0)
    db = 10 * np.log10(psd[inb] + 1e-30)
    fs_bin = fs / n
    k = max(int(round(PEAK_FLOOR_WIN_HZ / fs_bin)) | 1, 5)
    if k >= db.size:
        k = max((db.size // 2) | 1, 5)
    from scipy.ndimage import median_filter
    floor = median_filter(db, size=k, mode="nearest")
    over = db - floor
    step = max(int(round(PEAK_STEP_HZ / fs_bin)), 1)
    marks: list[tuple[float, float, float]] = []      # (freq, over_db, coh)
    for ci in range(0, inb.size, step):
        fc = float(f[inb[ci]])
        o = float(over[ci])
        if o < PEAK_OVER_DB:
            continue
        c = _band_coherence(resid, fs, fc, PEAK_COH_WIN_HZ)
        if not np.isfinite(c) or c < PEAK_COH_MIN:
            continue
        marks.append((fc, o, c))
    if not marks:
        return []
    # 相邻占用中心（间隔 ≤ 2 步）合并成带；峰取带内 over 最大者
    groups: list[list[tuple[float, float, float]]] = []
    cur = [marks[0]]
    for prev, item in zip(marks, marks[1:]):
        if abs(item[0] - prev[0]) <= 2 * step * fs_bin + 1.0:
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]
    groups.append(cur)
    bands: list[dict] = []
    for g in groups:
        best = max(g, key=lambda t: t[1])
        lo = min(t[0] for t in g) - PEAK_COH_WIN_HZ
        hi = max(t[0] for t in g) + PEAK_COH_WIN_HZ
        bands.append({"freq_offset_hz": round(best[0], 0),
                      "over_floor_db": round(best[1], 1),
                      "bandwidth_hz": round(hi - lo, 0),
                      "band_hz": [round(lo, 0), round(hi, 0)],
                      "coherence": round(best[2], 3)})
    bands.sort(key=lambda p: -p["over_floor_db"])
    return bands[:PEAK_MAX_N]


def _peak_selector():
    """峰判据选择：默认新判据；DSH_PEAK_LEGACY=1 切回旧判据做 A/B。"""
    import os
    return (_residual_peaks_legacy if os.environ.get("DSH_PEAK_LEGACY") == "1"
            else _residual_bands)


class ToolContext:
    """当前样本的工具运行环境（run_capture_agent 每样本设置）。"""

    def __init__(self, capture_dir: Path, cache_dir: Path, allowed_stations: list[str],
                 fs: float, fc: float, target_bw_normalized: float | None = None,
                 aoi_center: list[float] | None = None,
                 target_reference: dict | None = None):
        self.capture_dir = Path(capture_dir)
        self.cache_dir = Path(cache_dir)
        self.allowed_stations = list(allowed_stations)
        self.fs = float(fs)
        self.fc = float(fc)
        self.target_bw_normalized = target_bw_normalized
        self.aoi_center = aoi_center
        self.target_reference = target_reference or {}   # {station: 目标接收参考复数波形}
        self._npy_cache: dict[str, Path] = {}
        self._result_cache: dict[tuple, dict] = {}
        self._observations: dict | None = None

    @property
    def observations(self) -> dict:
        if self._observations is None:
            self._observations = json.loads(
                (self.capture_dir / "observations.json").read_text(encoding="utf-8"))
        return self._observations


_CURRENT: ToolContext | None = None


def set_context(ctx: ToolContext) -> None:
    global _CURRENT
    _CURRENT = ctx


def _ctx() -> ToolContext:
    if _CURRENT is None:
        raise RuntimeError("capture_tools context not set (run_capture_agent sets it per sample)")
    return _CURRENT


def _check_station(station: str) -> str | None:
    allowed = _ctx().allowed_stations
    if station not in allowed:
        return (f"station '{station}' is not available at this observation level; "
                f"allowed stations: {allowed}")
    return None


def _station_iq_raw(station: str, on_residual: bool = False) -> np.ndarray:
    """读某站原始 IQ；on_residual=True 时读目标剥离残差缓存（由
    estimate_band_occupancy_by_correlation 生成，供 Agent 对残差重跑分析）。"""
    ctx = _ctx()
    if on_residual:
        resid = ctx.cache_dir / f"resid_{station}.npy"
        if not resid.is_file():
            raise RuntimeError(f"residual cache {resid.name} not found; run "
                               "estimate_band_occupancy_by_correlation first")
        return np.load(resid).astype(np.complex128)
    return np.load(ctx.capture_dir / f"iq_{station}.npy").astype(np.complex128)


def _station_npy(station: str, on_residual: bool = False) -> Path:
    """v2 频谱/调制工具的输入缓存。

    2026-09-29 重大修复：改存**阵元 0 单通道**。原缓存存 8 阵元全阵——v2 的
    _combine_channels 对多阵元做"阵元 0 相位 × 平均幅度"重构，多源混合时该
    非线性相位调制会摧毁频谱结构（D2 全重叠下干扰谱峰被搅碎，Agent 频谱
    全盲的总根因）。单通道下 _combine_channels 恒等，v2 工具直接看到真实谱。
    （尺度无关：RMS 归一化仅影响可读性；DOA/定位/源数工具仍用 _station_iq_raw
    的全阵数据，不受影响。）"""
    ctx = _ctx()
    key = f"{'resid_' if on_residual else 'norm_'}{station}"
    if key not in ctx._npy_cache:
        ctx.cache_dir.mkdir(parents=True, exist_ok=True)
        iq = _station_iq_raw(station, on_residual=on_residual)
        single = iq[0] if iq.ndim == 2 else iq                # 阵元 0 单通道
        rms = float(np.sqrt(np.mean(np.abs(single) ** 2))) + 1e-12
        out = ctx.cache_dir / f"{key}.npy"
        np.save(out, (single / rms).astype(np.complex64))
        ctx._npy_cache[key] = out
    return ctx._npy_cache[key]


def _station_geometry(station: str) -> tuple[np.ndarray, np.ndarray]:
    """(阵列中心世界坐标, 阵轴世界方向单位向量)——来自公开 observations。

    阵轴 = local col 方向（元素排布方向）旋转到世界系；元素沿该轴 0.5λ 均匀排布。
    """
    rec = next(r for r in _ctx().observations["receivers"]
               if r["receiver_id"] == station)
    pos = np.asarray(rec["receiver_position_m"], float)
    rot = np.asarray(rec["array_pose"]["rotation_local_to_world"], float)
    axis = rot @ np.asarray(rec["array_geometry"]["col_direction_local"], float)
    return pos, axis / np.linalg.norm(axis)


def _band_edges(f_center_hz: float | None, bandwidth_hz: float | None) -> tuple[float, float]:
    ctx = _ctx()
    if f_center_hz is None:
        return -0.45 * ctx.fs, 0.45 * ctx.fs              # 全接收带
    half = max((bandwidth_hz or 60e3) / 2.0, 25e3) + 15e3  # 吸收测频误差
    return (max(f_center_hz - half, -0.45 * ctx.fs),
            min(f_center_hz + half, 0.45 * ctx.fs))


def _cached_result(key: tuple, fn):
    """同一样本内相同参数的工具调用直接返回缓存（CNN/空间分析开销大且确定性）。"""
    ctx = _ctx()
    if key not in ctx._result_cache:
        ctx._result_cache[key] = fn()
    return ctx._result_cache[key]


# ---------------- u 空间空间谱仪器（1D MUSIC，无脊简并） ----------------

# 每站 u 峰上限（_u_scan 出峰数与聚类取数共用同一参数，2026-10-03 统一）
U_PEAK_MAX = 4
CLAIM_TOL = 0.05        # _support 的 u 匹配窗（"交会点在峰上"的严格判定；
                        # 2026-10-03 曾在重写中丢失定义，闭包引用抛 NameError
                        # 被 gate 的 per-sample except 吞成"无簇"——11/12 样本）


def _u_scan(station: str, f_center_hz: float | None, bandwidth_hz: float | None,
            on_residual: bool = False) -> dict:
    """带内空间协方差 + 阵轴方向余弦 u∈[−1,1] 的 1D MUSIC 扫描。

    返回 {u_peaks: [(u, rel_db)], rank}；u 无简并，但 az = asin(u) 保留 ULA
    前后瓣镜像对（asin 分支与 180°−asin 分支）。on_residual=True 时对目标剥离
    残差扫描（共道下先剥目标再看剩余方向）。
    """
    ctx = _ctx()
    lo, hi = _band_edges(f_center_hz, bandwidth_hz)
    cov, snaps = band_covariance(_station_iq_raw(station, on_residual=on_residual),
                                 ctx.fs, lo, hi)
    evals, evecs = np.linalg.eigh(cov)
    evals, evecs = evals[::-1], evecs[:, ::-1]
    rank, _ = mdl_rank(evals, snaps)
    used = max(1, int(rank))
    noise = evecs[:, used:]
    m = cov.shape[0]
    u = np.linspace(-0.999, 0.999, 801)
    steer = np.exp(2j * np.pi * 0.5 * np.outer(np.arange(m), u))
    proj = np.abs(noise.conj().T @ steer) ** 2
    music = 1.0 / np.maximum(proj.sum(axis=0), 1e-20)
    music_db = 10.0 * np.log10(music / music.max() + 1e-30)
    # 绝对显著性：峰高相对谱本底（中位数）——跨子带可比（rel 是子带内相对，
    # 跨子带无意义）。脊线图用它过滤噪声峰。
    music_abs = 10.0 * np.log10(music / (np.median(music) + 1e-30) + 1e-30)
    # 局部极大取峰（3 点窗），按强度排序
    idx = np.where((music_db[1:-1] >= music_db[:-2]) & (music_db[1:-1] >= music_db[2:]))[0] + 1
    idx = idx[np.argsort(music_db[idx])[::-1]][:U_PEAK_MAX]
    # 2026-10-03：曾放宽到 -16 dB 试图救弱源峰，实测新增峰全是假的
    # （d2_00008 station_a 的 -0.939 与该站真源 u 全正矛盾）→ 回退 -8；
    # 弱源召回的正路是分带测向（弱源在自己带内 SIR 高），不是压阈值。
    peaks = [(round(float(u[i]), 3), round(float(music_db[i]), 1)) for i in idx
             if music_db[i] >= -8.0]
    peaks_abs = [(round(float(u[i]), 3), round(float(music_abs[i]), 1)) for i in idx
                 if music_db[i] >= -8.0]
    return {"u_peaks": peaks or [(round(float(u[int(np.argmax(music))]), 3), 0.0)],
            "u_peaks_abs": peaks_abs,
            "eig1": float(evals[0]),
            "rank": int(used), "band_hz": [round(lo), round(hi)]}


def _u_scan_cached(station: str, f_center_hz: float | None, bandwidth_hz: float | None,
                   on_residual: bool = False) -> dict:
    ctx = _ctx()
    key = ("uscan", station, None if f_center_hz is None else round(f_center_hz / 1e3),
           None if bandwidth_hz is None else round(bandwidth_hz / 1e3), bool(on_residual))
    if key not in ctx._result_cache:
        ctx._result_cache[key] = _u_scan(station, f_center_hz, bandwidth_hz,
                                         on_residual=on_residual)
    return ctx._result_cache[key]


def _ridge_pool(station: str, on_residual: bool = True,
                sub_bw: float = 200e3, hop: float = 100e3,
                du_link: float = 0.05, min_span: int = 2,
                min_strength: float = 20.0,
                sig_ratio: float = 3.0,
                max_width_hz: float = 1.75e6) -> list:
    """(u,f) 脊线图：窄子带滑窗 MUSIC，按 u 连续性连成脊（2026-10-03 规格）。

    给峰加频率维，把"u 一维归属无解"变成"(u,f) 二维归属"。

    显著性判据（2026-10-03 三代迭代定稿）：**λ1 / mean(λ2..8) ≥ sig_ratio**——
    子带内自带噪声参考，与带宽/滤波器/跨子带归一无关，物理含义 = 1 + M·SNR_elem。
    前两代（MUSIC 谱相对峰 / 相对子带中位 / λ1 相对全带中位）跨子带不可比，
    且 D2"能量地毯"下 λ1_max/λ1_median 只有 3.9~5.5，恒够不着绝对阈值 6。

    **v1 共同分量预剔除**：原始 IQ 在"信号空区"就有 λ1/λ2=5e5 的宽带空间相干
    分量（疑似目标经剥离后残留的相干成分，剥离仅压 ~4000×），它主导每个子带
    → 贯穿全带的"噪声走廊"假脊（span=34 实测）。先取全带残差协方差主特征向
    量 v1 投影掉再建脊。风险：与目标同向的真实干扰会被一并剔除（生成约束
    ΔDOA≥15° 有界；v1 方向可由诊断脚本对照真值核验）。"""
    ctx = _ctx()
    rx = ctx.fs * 0.45
    n = _station_iq_raw(station, on_residual=on_residual).shape[1]
    iq = _station_iq_raw(station, on_residual=on_residual)
    # ---- 预剔除：逐阵元去均值（DC/极低频公共分量）----
    # 上一版用全带主特征向量 v1 投影——实测在 D1 会把最强干扰本身剔掉
    # （v1=最强分量≠共同分量），且假脊全在通带边缘（滚降区）。
    # "空区相干分量"更可能是 DC 偏置（各阵元全同→空间全 1 向量→λ1/λ2 巨大），
    # 去均值物理清晰且不误伤真源。滑窗同时避开滚降区（|f_c| ≤ 1.35MHz）。
    iq = iq - iq.mean(axis=1, keepdims=True)
    # ---- 条件化 v1 投影（2026-10-03 修正判别）：剥离残余的宽带相干分量贯穿
    # 通带（每子带 λ1/med(λ4..8)≈200），显著性判据对它失明。v1 的等效 u 若与
    # "目标带残余峰"方向一致（|Δu|<0.1）→ 是目标剥离残余 → 投影剔除；
    # 若 v1 是其他强源（如 D1 的 tone）→ 不剔（那是要找的真源）。
    try:
        R_all = iq @ iq.conj().T / iq.shape[1]
        _ev, _evec = np.linalg.eigh(R_all)
        v1 = _evec[:, -1] / (np.linalg.norm(_evec[:, -1]) + 1e-12)
        uu = np.linspace(-1.0, 1.0, 401)
        A_u = np.exp(2j * np.pi * 0.5 * np.outer(np.arange(8), uu))
        u_v1 = float(uu[int(np.argmax(np.abs(A_u.conj().T @ v1)))])
        tgt_half_bw = 0.5 * (ctx.target_bw_normalized or 0.25) * ctx.fs
        # 参照方向 = **原始 IQ** 目标带的最强峰（剥离前的目标，公开数据可得；
        # on_residual 的谱上带外强源也会出峰，不能当"目标残余方向"）
        scan_raw = _u_scan_cached(station, 0.0, 2 * tgt_half_bw, on_residual=False)
        u_tgt = max(scan_raw["u_peaks"], key=lambda t: t[1])[0] if scan_raw["u_peaks"] else None
        if u_tgt is not None and abs(u_v1 - u_tgt) < 0.1:
            iq = iq - np.outer(v1, v1.conj() @ iq)
    except Exception:
        pass
    # 扫描上限 = 接收机通带半宽 + 50k（firmwin 截止 ±810k 之外是滚降区，
    # 公共分量在滚降区相对突起会造成假脊；接收带宽来自公开 observations）
    _bw_hz = ctx.observations["receivers"][0].get("bandwidth_hz", 1.62e6)
    scan_edge = float(_bw_hz) / 2 + 50e3
    n_sub = int(2 * (min(rx, scan_edge) - sub_bw / 2) // hop)
    raw: list[tuple[float, float, float]] = []
    for i in range(n_sub):
        f_c = -scan_edge + sub_bw / 2 + i * hop
        lo = max(f_c - sub_bw / 2, -rx)
        hi = min(f_c + sub_bw / 2, rx)
        try:
            R, snaps = band_covariance(iq, ctx.fs, lo, hi)
        except Exception:
            continue
        evals = np.linalg.eigvalsh(R)[::-1]
        lam1 = float(np.real(evals[0]))
        # 噪声参考用 median(λ4..8)：共道两源时 λ2（甚至 λ3）也是信号，
        # mean(λ2..8) 会把信号当噪声、比值减半（d2_00002 的 Welder/Microwave
        # 混叠对因此卡在阈值下）；median(λ4..8) 容忍至多 3 个共道源。
        noise_ref = float(np.median(np.real(evals[3:])))
        if noise_ref <= 0:
            continue
        ratio = lam1 / noise_ref
        scan = _u_scan_cached(station, f_c, sub_bw, on_residual=on_residual)
        # top2 峰都收：次源在被强源占据的子带里对应 λ2 方向（λ1 已被强源拿走），
        # 只收 top 会漏掉所有"强源压制下的次源"（d2_00002 的 Welder 教训）。
        for top_u, _rel in (scan["u_peaks"][:2] if scan["u_peaks"] else []):
            raw.append((f_c, float(top_u), float(ratio)))
    # ---- 连线 ----
    ridges: list[dict] = []
    for f_c, u, ratio in sorted(raw, key=lambda t: t[0]):
        best = None
        for r in ridges:
            if abs(r["u_last"] - u) <= du_link and r["f_last"] < f_c - hop / 2:
                if best is None or r["f_last"] > best["f_last"]:
                    best = r
        if best is None:
            ridges.append({"u_last": u, "f_last": f_c, "span": 1,
                           "ratios": [ratio], "us": [u], "fcs": [f_c]})
        else:
            best["u_last"] = u; best["f_last"] = f_c; best["span"] += 1
            best["ratios"].append(ratio); best["us"].append(u); best["fcs"].append(f_c)
    out = []
    for r in ridges:
        if r["span"] < min_span:
            continue
        if r["span"] * np.mean(r["ratios"]) < min_strength:
            continue                        # 低强度脊=噪声（窄带真源 strength 巨大）
        us = np.asarray(r["us"]); rats = np.asarray(r["ratios"])
        out.append({"u_mean": float(np.sum(us * rats) / np.sum(rats)),
                    "f_lo": float(min(r["fcs"]) - sub_bw / 2),
                    "f_hi": float(max(r["fcs"]) + sub_bw / 2),
                    "span": int(r["span"]),
                    "strength": float(r["span"] * np.mean(rats))})
    # ---- 贪心合并主瓣分裂 ----
    out.sort(key=lambda r: -r["strength"])
    merged: list[dict] = []
    for r in out:
        tgt = None
        for m in merged:
            if (abs(r["u_mean"] - m["u_mean"]) <= du_link
                    and not (r["f_hi"] < m["f_lo"] or r["f_lo"] > m["f_hi"])):
                tgt = m
                break
        if tgt is None:
            merged.append(dict(r))
        else:
            tw = tgt["strength"] + r["strength"]
            tgt["u_mean"] = (tgt["u_mean"] * tgt["strength"]
                             + r["u_mean"] * r["strength"]) / tw
            tgt["f_lo"] = min(tgt["f_lo"], r["f_lo"])
            tgt["f_hi"] = max(tgt["f_hi"], r["f_hi"])
            tgt["span"] += r["span"]
            tgt["strength"] = tw
    merged = [r for r in merged if r["f_hi"] - r["f_lo"] <= max_width_hz]
    merged.sort(key=lambda r: -r["strength"])
    return merged


def _deflation_drop(station: str, f_center_hz: float, bandwidth_hz: float,
                    pos_xy) -> float:
    """deflation（源抵消）检验：把候选源的导向从带内协方差里投影掉后重扫
    MUSIC，返回该源 u 处峰高的下降量（dB）。

    真源 → 该方向的信号子空间分量被移除，峰大幅塌陷；假源 → 本来就不在
    谱里，投影几乎不改变谱。这是与峰表**独立**的去假判据（回投检验的峰表
    与射线池同源，拦不住假峰；2026-10-03 d2_00008 教训）。"""
    ctx = _ctx()
    rx = ctx.fs * 0.45
    lo, hi = _band_edges(f_center_hz, bandwidth_hz)
    lo, hi = max(lo, -rx), min(hi, rx)
    resid = _station_iq_raw(station, on_residual=True)
    R, snaps = band_covariance(resid, ctx.fs, lo, hi)
    rec = next(r for r in ctx.observations["receivers"] if r["receiver_id"] == station)
    off = np.asarray(rec["element_offsets_world_m"], float)
    rp3 = np.asarray(rec["receiver_position_m"], float)
    d_xy = np.array([pos_xy[0] - rp3[0], pos_xy[1] - rp3[1], 0.0])
    u_src = d_xy / (np.linalg.norm(d_xy) + 1e-12)
    a = np.exp(2j * np.pi * ctx.fc / 299792458.0 * (off @ u_src))
    pa = (a.conj() @ (R @ a)) / (a.conj() @ a)          # a 方向功率
    R_proj = R - (pa / (a.conj() @ a)) * np.outer(a, a.conj())
    axis = off[-1] - off[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    u0 = float(axis @ d_xy) / (np.linalg.norm(d_xy) + 1e-12)

    def _peak_db(R_in):
        evals, evecs = np.linalg.eigh(R_in)
        evals, evecs = evals[::-1], evecs[:, ::-1]
        rank, _ = mdl_rank(evals, snaps)
        noise = evecs[:, max(1, int(rank)):]
        m = R_in.shape[0]
        uu = np.linspace(-0.999, 0.999, 801)
        steer = np.exp(2j * np.pi * 0.5 * np.outer(np.arange(m), uu))
        proj = np.abs(noise.conj().T @ steer) ** 2
        music = 1.0 / np.maximum(proj.sum(axis=0), 1e-20)
        i0 = int(np.argmin(np.abs(uu - u0)))
        w = 10                                          # ±0.025 u 窗
        seg = music[max(0, i0 - w):i0 + w + 1]
        # 相对谱本底（中位数）而非全局最大峰——投影后整条伪谱会重标定，全局
        # 最大可能换峰，相对口径的 drop 被峰移动污染（2026-10-03 修正）。
        return 10.0 * np.log10(seg.max() / (np.median(music) + 1e-30))

    return float(_peak_db(R) - _peak_db(R_proj))


def _u_to_azimuths(u: float) -> list[float]:
    """u → 方位角候选对（仰角≈0 近似；仰角使真值落在两候选之间的小偏差内）。"""
    a = float(np.degrees(np.arcsin(np.clip(u, -1.0, 1.0))))
    mir = (180.0 - a + 180.0) % 360.0 - 180.0
    return [round(a, 1), round(mir, 1)]


# ---------------- 工具 1：频谱分析（v2 核心 + Hz 换算） ----------------

def analyze_spectrum_station(station: str = "station_a", use_spatial: bool = True,
                             on_residual: bool = False, **_) -> dict:
    err = _check_station(station)
    if err:
        return {"error": err}
    ctx = _ctx()

    def _run():
        res = tools_v2.analyze_spectrum(str(_station_npy(station, on_residual=on_residual)),
                                        target_bandwidth_normalized=ctx.target_bw_normalized,
                                        use_spatial=use_spatial)
        res["station"] = station
        res["on_residual"] = bool(on_residual)
        res["obw99_hz"] = round(res.get("obw99_normalized", 0.0) * ctx.fs, 1)
        for c in res.get("sources_candidates", []):
            c["freq_offset_hz"] = round(c["freq"] * ctx.fs, 1)
            c["bandwidth_hz"] = round(c.get("bandwidth", 0.0) * ctx.fs, 1)
            if c.get("angle_deg") is not None:
                # v2 局部角 ≡ 阵轴方向余弦角（el≈0），给世界方位角候选对
                a = float(c.pop("angle_deg"))
                c["world_azimuth_candidates_deg"] = _u_to_azimuths(
                    float(np.sin(np.deg2rad(a))))
        return res

    return _cached_result(("spectrum", station, bool(use_spatial), bool(on_residual)), _run)


# ---------------- 工具 2：源数估计（u 空间全带 MDL 秩） ----------------

def estimate_num_sources_station(station: str = "station_a", max_sources: int = 4,
                                 on_residual: bool = False, **_) -> dict:
    """全接收带 u 空间 MDL 秩 = 空间分量数（每分量=一个传播源或强多径回波）。

    不用 v2 的 final_suggestion 修正树：它按合并谱峰数上/下修 MDL，标定于
    v2 的 4 阵元/1024 点口径，在 8 阵元/长窗 capture 上会被噪声伪峰带偏
    （干净样本 raw MDL=1 被修成 4）。MDL 秩无简并、对多径诚实（RT 场景秩
    可大于辐射体数，由 Agent 结合谱候选裁决）。on_residual=True 时对目标剥离
    残差计数（共道下先剥目标，看还剩几个空间分量）。
    """
    err = _check_station(station)
    if err:
        return {"error": err}

    def _run():
        scan = _u_scan(station, None, None, on_residual=on_residual)   # 全接收带
        rank = min(int(scan["rank"]), int(max_sources) + 1, 8)
        return {"station": station,
                "on_residual": bool(on_residual),
                "num_spatial_components": rank,
                "u_peaks": scan["u_peaks"],
                "note": ("MDL spatial rank over the full receiver band; each component is "
                         "one propagating source or a strong multipath echo. RT scenes "
                         "can exceed the emitter count — cross-check with spectral "
                         "candidates and per-band DOA scans before answering.")}

    return _cached_result(("num_sources", station, int(max_sources), bool(on_residual)), _run)


# ---------------- 工具 3：单站 DOA（u 空间 + 方位角候选对） ----------------

def estimate_doa_station(station: str = "station_a", freq_offset_hz: float | None = None,
                         bandwidth_hz: float | None = None, on_residual: bool = False,
                         **_) -> dict:
    err = _check_station(station)
    if err:
        return {"error": err}
    ctx = _ctx()
    scan = _u_scan_cached(station, freq_offset_hz, bandwidth_hz, on_residual=on_residual)
    out = {
        "station": station,
        "on_residual": bool(on_residual),
        "band_hz": scan["band_hz"] if freq_offset_hz is not None else None,
        "direction_cosine_u_peaks": scan["u_peaks"],
        "azimuth_candidates_deg": [_u_to_azimuths(u) for u, _ in scan["u_peaks"]],
        "mdl_rank": scan["rank"],
        "mirror_note": ("azimuth_candidates_deg = [asin(u), 180-asin(u)]: a horizontal ULA "
                        "cannot distinguish a source at az from its mirror at 180-az "
                        "(identical response) — a single station cannot resolve the pair; "
                        "multi-station estimate_position resolves it by cone intersection."),
    }
    if ctx.aoi_center:
        pos, _ = _station_geometry(station)
        to_aoi = np.asarray(ctx.aoi_center[:2]) - pos[:2]
        out["aoi_bearing_deg"] = round(float(np.degrees(np.arctan2(to_aoi[1], to_aoi[0]))), 1)
    return out


# ---------------- 工具 4：多站锥面交会定位（S1/S2 核心能力） ----------------

def _cone_localize(stations: list[dict], u_choice: list[float]) -> list[dict]:
    """网格最小二乘：Σ_i (unit(p−o_i)·a_i − u_i)²，对 (x,y,h) 粗扫 + 局部细化。"""
    origins = np.array([s["origin"] for s in stations])
    axes = np.array([s["axis"] for s in stations])
    center = origins[:, :2].mean(axis=0)
    span, coarse = 2500.0, 25.0
    xs = np.arange(center[0] - span, center[0] + span, coarse)
    ys = np.arange(center[1] - span, center[1] + span, coarse)
    hs = np.arange(2.0, 260.0, 16.0)
    X, Y, H = np.meshgrid(xs, ys, hs, indexing="ij")
    pts = np.stack([X, Y, H], axis=-1).reshape(-1, 3)

    def cost(p):
        d = p[None, :, :] - origins[:, None, :]           # (n_st, n_pts, 3)
        u_pred = np.einsum("npk,nk->np", d / np.linalg.norm(d, axis=2)[..., None], axes)
        return np.sum((u_pred - np.asarray(u_choice)[:, None]) ** 2, axis=0)

    c = cost(pts)
    order = np.argsort(c)[:64]
    refined = []
    for i in order:
        cx, cy, ch = pts[i]
        xs2 = np.arange(cx - coarse, cx + coarse, 5.0)
        ys2 = np.arange(cy - coarse, cy + coarse, 5.0)
        hs2 = np.arange(max(2.0, ch - 16.0), ch + 16.0, 4.0)
        X2, Y2, H2 = np.meshgrid(xs2, ys2, hs2, indexing="ij")
        p2 = np.stack([X2, Y2, H2], axis=-1).reshape(-1, 3)
        c2 = cost(p2)
        j = int(np.argmin(c2))
        refined.append((float(c2[j]), p2[j]))
    refined.sort(key=lambda t: t[0])
    out, seen = [], []
    for cval, p in refined:
        xy = [round(float(p[0]), 1), round(float(p[1]), 1)]
        if any(np.hypot(xy[0] - q["position_xy_m"][0], xy[1] - q["position_xy_m"][1]) < 40.0
               for q in out):
            continue
        out.append({"position_xy_m": xy, "height_m": round(float(p[2]), 1),
                    "u_residual_rms": round(float(np.sqrt(cval / len(u_choice))), 4)})
        if len(out) >= 5:
            break
    return out


def estimate_position(freq_offset_hz: float, bandwidth_hz: float | None = None,
                      on_residual: bool = False, **_) -> dict:
    ctx = _ctx()
    stations = sorted(ctx.allowed_stations)
    if len(stations) < 2:
        return {"error": ("multi-station positioning requires >= 2 stations; "
                          "at S0 only one station is available")}
    per_station = []
    for st in stations:
        scan = _u_scan_cached(st, freq_offset_hz, bandwidth_hz, on_residual=on_residual)
        pos, axis = _station_geometry(st)
        us = [u for u, _ in scan["u_peaks"][:2]]          # 每站前 2 个 u 候选
        per_station.append({"station": st, "origin": pos.tolist(), "axis": axis.tolist(),
                            "u_candidates": us})
    # 软指派：每站取与其几何最一致的 u 候选（免组合枚举，且容多径伪峰）
    stations_geo = [{"origin": np.array(s["origin"]), "axis": np.array(s["axis"])}
                    for s in per_station]
    u_choice = [s["u_candidates"][0] for s in per_station]
    hyp = _cone_localize(stations_geo, u_choice)
    # 用粗定位残差给每站挑更好的 u 候选后重解一次
    if hyp and any(len(s["u_candidates"]) > 1 for s in per_station):
        best = hyp[0]
        p0 = np.array([best["position_xy_m"][0], best["position_xy_m"][1], best["height_m"]])
        for s, g in zip(per_station, stations_geo):
            d = p0 - g["origin"]
            u_pred = float(np.dot(d / np.linalg.norm(d), g["axis"]))
            s["u_choice"] = min(s["u_candidates"], key=lambda u: (u - u_pred) ** 2)
        u_choice = [s.get("u_choice", s["u_candidates"][0]) for s in per_station]
        hyp = _cone_localize(stations_geo, u_choice)
    return {"band_center_hz": round(freq_offset_hz, 0), "num_stations": len(stations),
            "on_residual": bool(on_residual),
            "per_station_u": [{"station": s["station"], "u_candidates": s["u_candidates"]}
                              for s in per_station],
            "hypotheses": hyp,
            "note": ("ranked by direction-cosine residual (u_residual_rms; smaller = rays "
                     "more consistent). Co-band contamination: the target may appear as a "
                     "separate hypothesis — locate the target first (narrow band at 0 Hz) "
                     "and exclude it. NLoS/multipath biases u.")}


# ---------------- 工具 5：时域特征（v2 核心） ----------------

def detect_time_domain_station(station: str = "station_a", **_) -> dict:
    err = _check_station(station)
    if err:
        return {"error": err}

    def _run():
        res = tools_v2.detect_time_domain(str(_station_npy(station)))
        res["station"] = station
        return res

    return _cached_result(("time_domain", station), _run)


# ---------------- 工具 6：调制/波形切片特征（v2 核心，CNN 后端） ----------------

def estimate_modulation_features_station(station: str = "station_a",
                                         freq_offset_hz: float | None = None,
                                         on_residual: bool = False, **_) -> dict:
    err = _check_station(station)
    if err:
        return {"error": err}

    def _run():
        res = tools_v2.estimate_modulation_features(
            str(_station_npy(station, on_residual=on_residual)),
            target_bandwidth_normalized=_ctx().target_bw_normalized)
        res["station"] = station
        res["on_residual"] = bool(on_residual)
        res["note_angle"] = ("per_peak angle_deg (if present) is the legacy array-local "
                             "convention; use estimate_doa_station for world azimuth")
        return res

    res = _cached_result(("mod_features", station, bool(on_residual)), _run)
    if freq_offset_hz is not None:
        res["requested_freq_offset_hz"] = round(float(freq_offset_hz), 0)
    return res


# ---------------- 工具 7：辐射体手册（公开规格，T2 溯源证据参照） ----------------

RADIATOR_CATALOG: dict[str, dict] = {
    "tone": {"modulation": "single_tone",
             "description": "Continuous unmodulated carrier (free-mode simple source)"},
    "noise": {"modulation": "bandlimited_noise",
              "description": "Band-limited white noise (free-mode simple source)"},
    **{name: {"modulation": mod, "description": desc}
       for name, (_, mod, desc) in EMITTERS.items()},
}


_CNN_CACHE: dict = {}


def _load_cnn():
    """延迟加载 19 类体制分类器（rad_cls，checkpoint 由 train16 产出）。"""
    if not _CNN_CACHE.get("model"):
        import torch
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here / "rad_cls"))
        from rad_cls.model16 import RadCNN  # noqa: E402
        ck = torch.load(here / "rad_cls" / "rad_classifier" / "checkpoint.pt",
                        map_location="cpu", weights_only=False)
        model = RadCNN()
        model.load_state_dict(ck["model"])
        model.eval()
        _CNN_CACHE.update(model=model, classes=ck["classes"])
    return _CNN_CACHE


# 意图表：统一到 emitters.INTENT（见文件头 import），本文件不再自带副本。
# 历史教训：这份副本曾与答案键不一致（Bluetooth_FHSS / WiFi_burst：副本 incidental、
# 真值 benign）——工具代填的 intent 会与评分口径冲突。单一来源后二者恒等。


def classify_extracted_waveform(station: str = "station_a", **_) -> dict:
    """对 extract_source_waveforms 落盘的每候选 CNN-ready 波形做体制分类，
    返回可直接拼进最终答案的 JSON 片段（freq/kind/position/confidence）。
    注意：intent 与 is_spoof 仍在本函数内部计算（下游/诊断要用），但**对外工具表**
    经 _agent_facing 剥掉——它们是 kind 的确定性查表值，给出去就等于把 T3 意图轴与
    欺骗轴变成“照抄工具”。Agent 侧的题面要求就是自己从 kind + 行为推出来。
    2026-10-03 新增：agent 实测 10 轮全耗在测量上、一次都没作答——本工具把
    "R05-R10 逐源分类"压成一轮，且把候选板→答案字段的映射在工具内完成。"""
    ctx = _ctx()
    # 先确保提取已跑（bfw 缓存由 extract 落盘；未跑则此处补——消除调用顺序
    # 依赖：agent 若跳过 R4 直接调本工具，items 会全空 → num_interferers=0
    # 欠报，D1/S0 回归 11/12 欠报的根因）。
    try:
        extract_source_waveforms(freq_offset_hz=None, bandwidth_hz=None)
    except Exception:
        pass
    clu = estimate_sources_by_spatial_clustering(on_residual=True)
    candidates = clu.get("candidates") or []
    if not candidates:
        return {"error": "no candidates from spatial clustering",
                "num_interferers": 0, "interferers": []}
    cnn = _load_cnn()
    import torch
    items = []
    for k, c in enumerate(candidates):
        wf = ctx.cache_dir / f"bfw_{station}_{k}.npy"
        if not wf.is_file():
            continue
        w = np.load(wf).astype(np.complex64)
        xb = torch.from_numpy(np.stack([w.real, w.imag]).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pr = torch.softmax(cnn["model"](xb)[0], dim=0)
        kind = cnn["classes"][int(torch.argmax(pr))]
        items.append({
            "freq_offset_hz": round(0.5 * (c["band_hz"][0] + c["band_hz"][1])),
            "kind": kind,
            "intent": INTENT_OF.get(kind, "incidental"),
            "is_spoof": kind == "GNSS_spoof",
            "position_xy_m": c["position_xy_m"],
            "confidence": round(float(pr.max()), 2),
        })
    return {"num_interferers": len(items), "interferers": items,
            "note": ("each item maps 1:1 to a clustering candidate; freq_offset_hz "
                     "= candidate band center (CNN-ready shift already applied); "
                     "position_xy_m is 2D. Copy these fields into the final JSON "
                     "answer, replacing nothing else.")}


def list_radiator_catalog(**_) -> dict:
    """16 种城市低空辐射体 + 2 种简单体制的公开波形规格（不含意图标签）。"""
    return {"radiators": RADIATOR_CATALOG,
            "note": "public spec sheet; identify the radiator kind from measured "
                    "waveform features, not by guessing"}


# ---------------- 工具 8：合作目标剥离（共道解耦核心） ----------------

def _strip_target_at_station(station: str, lo: float | None = None,
                             hi: float | None = None,
                             max_delay_samples: int = 64) -> np.ndarray:
    """对单站做目标剥离，返回**整通带**残差 IQ，并缓存 resid_{station}.npy。

    2026-10-02 改为**参考直减**：target_reference 是运行器注入的"无噪无干扰
    目标接收回放"（与 IQ 的目标分量逐位一致——receive 线性），residual = iq-ref
    数学上完美剥离。原 FIR 模板拟合（template_residual 16 taps）在带限 OFDM
    参考上病态，拟合伪影会污染残差频谱（CNN 体制识别对之敏感）。

    2026-10-03 P0：处理带从"目标带 ±500k"放宽为**整通带 ±0.45fs**——旧口径把
    带外干扰整段删掉（d2_00008 的 WiFi -681k / Zigbee +682k 在进 MUSIC 之前
    就没了；全 D2 67% 干扰受损：35% 完全删除 + 32% 跨界削边）。lo/hi 参数
    废弃保留（兼容旧调用），内部一律整通带；目标照减（ref 同 mask 带限）。"""
    ctx = _ctx()
    ref = ctx.target_reference.get(station)
    if ref is None:
        raise RuntimeError(f"target reference not available for station '{station}'")
    ref = np.asarray(ref, complex)          # (8, N) 全阵参考，逐阵元直减——
    iq = _station_iq_raw(station)           # 目标导向相位各阵元不同，绝不能
    n = iq.shape[1]                         # 用单阵元参考广播减全阵（否则 7/8
    f = np.fft.fftfreq(n, 1 / ctx.fs)       # 阵元目标残留，LS 提取被污染）
    rx_half = ctx.fs * 0.45
    mask = (f >= -rx_half) & (f <= rx_half)
    ref_sel = np.fft.ifft(np.fft.fft(ref, axis=1) * mask, axis=1).astype(complex)
    residual_iq = np.fft.ifft(np.fft.fft(iq, axis=1) * mask, axis=1) - ref_sel
    ctx.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ctx.cache_dir / f"resid_{station}.npy", residual_iq.astype(np.complex64))
    return residual_iq


def strip_target_all_stations(lo: float, hi: float, max_delay_samples: int = 64) -> dict:
    """对全部 allowed 站剥离目标并缓存残差（多站空间聚类的预处理）。

    对"剥不了目标"的站（目标参考缺失/剥离失败，通常是该站目标 0 径=被楼完全
    遮挡、根本看不到目标），直接用带限原始 IQ 当残差缓存——该站本来就不含目标，
    原始 ≈ 残差。保证多站 RANSAC 能凑齐足够站数（2026-09-29 修）。
    返回 {station: resid_iq}。"""
    ctx = _ctx()
    out = {}
    for st in ctx.allowed_stations:
        try:
            out[st] = _strip_target_at_station(st, lo, hi, max_delay_samples)
            continue
        except Exception:
            pass
        # 剥离失败：该站目标不可见 → 直接用带限原始 IQ 当残差
        try:
            iq = _station_iq_raw(st)
            n = iq.shape[1]
            fq = np.fft.fftfreq(n, 1 / ctx.fs)
            mask = (fq >= lo) & (fq <= hi)
            resid = np.fft.ifft(np.fft.fft(iq, axis=1) * mask, axis=1)
            ctx.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(ctx.cache_dir / f"resid_{st}.npy", resid.astype(np.complex64))
            out[st] = resid
        except Exception:
            continue
    return out


def _beamform_extract_station(station: str, src_positions: list,
                              lo: float, hi: float,
                              diag_load: float = 0.05) -> dict:
    """多源 LS 波束成形：已知 K 个源位置，从该站 8 阵元残差中同时提取 K 路源波形。

    残差（目标已剥）= K 个源经各自多径信道的叠加 + 噪声。源位置已知（多站交会
    米级），导向矢量可精确构造；K 源同时 LS 分离比逐源 MVDR 稳（互泄漏被显式建模）。

    返回 {'waveforms': (K,N) complex64, 'cond': 条件数, 'energy_frac': 每源能量占比,
          'ok': bool, 'reason': 失败原因}。失败时 ok=False。
    """
    ctx = _ctx()
    c = 299792458.0
    obs = ctx.observations
    rec = next(r for r in obs["receivers"] if r["receiver_id"] == station)
    off = np.asarray(rec["element_offsets_world_m"], float)      # 8×3 世界偏移
    resid = _station_iq_raw(station, on_residual=True)           # (8, N) 带限残差
    n = resid.shape[1]
    # K 源导向矢量（平台约定：a = exp(2jπ·fc/c·(off·u_src))，u_src=站→源单位向量）
    # 2026-10-03：u_src **水平化**（z 分量置 0 再归一化）。4 站阵轴同为世界 +y、
    # 站高差仅 8m——z 在该几何下不可观测，交会估计的 z 在 2~66m 乱跳，通过
    # |Δ| 归一化污染 u_y（Δz 30m @ 500m 水平距 → u 误差 ~0.05，恰是提取劣化
    # 量级）。水平阵的导向只依赖水平角，水平化后 z 误差不再进入导向。
    A = np.zeros((8, len(src_positions)), complex)
    rp3 = np.asarray(rec["receiver_position_m"], float)
    for k, sp in enumerate(src_positions):
        sp3 = np.asarray(sp[:3], float) if len(sp) >= 3 else np.array([sp[0], sp[1], rp3[2]])
        d_xy = np.array([sp3[0] - rp3[0], sp3[1] - rp3[1], 0.0])
        u_src = d_xy / (np.linalg.norm(d_xy) + 1e-12)
        A[:, k] = np.exp(2j * np.pi * ctx.fc / c * (off @ u_src))
    # 2026-10-03：按带分组提取后，lo/hi=组频带（组内源带的并集）——输入带限
    # 到组频带，滤掉组外源（它们不在 A 里，会直接漏进提取波形）。同带组内
    # 共道串扰仍由联合 LS 显式建模。此前的"全带不做二次切"决策针对的是
    # "全部候选塞一个 A"的旧语境；分组后带限必要且正确（CNN 训练分布含
    # 带限切片增强，口径兼容）。残差缓存现为整通带（P0 后），组频带内内容完整。
    f = np.fft.fftfreq(n, 1 / ctx.fs)
    mask = (f >= max(lo, -ctx.fs * 0.45)) & (f <= min(hi, ctx.fs * 0.45))
    X = np.fft.ifft(np.fft.fft(resid, axis=1) * mask, axis=1) if mask.any() else resid
    # 多源 LS：Ŝ = pinv(A) @ X；条件数筛查（角距过近时病态）
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > 1e6:
        return {"ok": False, "reason": f"steering matrix ill-conditioned (cond={cond:.1e})"}
    S = np.linalg.pinv(A) @ X                                    # (K, N)
    # 能量占比（各源分离出的相对能量）
    p = np.mean(np.abs(S) ** 2, axis=1)
    energy_frac = [round(float(v) / (float(p.sum()) + 1e-30), 3) for v in p]
    return {"ok": True, "waveforms": S.astype(np.complex64),
            "cond": round(cond, 1), "energy_frac": energy_frac,
            "diag_load": diag_load}


def estimate_band_occupancy_by_correlation(station: str = "station_a",
                                           band_low_hz: float | None = None,
                                           band_high_hz: float | None = None,
                                           max_delay_samples: int = 64, **_) -> dict:
    """合作目标带相关剥离：用运行器注入的目标参考（含多径、无噪声无干扰）对目标带做
    FIR 信道拟合，量"匹配（目标）能量 vs 失配残差能量 vs 噪声底"，并缓存目标剥离后的
    残差带限 IQ 供对残差重跑频谱/DOA/定位。共道场景（D2 目标与干扰频谱重叠）下，
    先剥目标再看剩余能量与方向，是解耦共道的正路。
    """
    err = _check_station(station)
    if err:
        return {"error": err}
    ctx = _ctx()
    rx_half = ctx.fs * 0.45
    tgt_half = (ctx.target_bw_normalized * ctx.fs / 2) if ctx.target_bw_normalized else 0.0
    tgt_half = min(max(float(tgt_half), 60e3), rx_half)
    lo = -tgt_half if band_low_hz is None else float(band_low_hz)
    hi = +tgt_half if band_high_hz is None else float(band_high_hz)
    lo, hi = max(lo, -rx_half), min(hi, rx_half)
    iq = _station_iq_raw(station)
    n = iq.shape[1]
    if hi - lo < 2 * ctx.fs / n:
        return {"error": "target band narrower than two FFT bins"}
    try:
        residual_iq = _strip_target_at_station(station, lo, hi, max_delay_samples)
    except Exception as exc:
        return {"error": str(exc)}
    used_tap = min(max(int(max_delay_samples), 1), 256)
    resid_path = ctx.cache_dir / f"resid_{station}.npy"
    # matched = 1 - 残差能量/原始能量（目标被剥离的比例）。
    # 2026-10-03 P0：residual 与分母都用整通带口径（residual 现含带外干扰，
    # 若分母留在目标带会造成口径错位、matched 虚低）。
    _iq0 = _station_iq_raw(station)
    n0 = _iq0.shape[1]
    _f = np.fft.fftfreq(n0, 1 / ctx.fs)
    _m = (_f >= -rx_half) & (_f <= rx_half)
    _orig = np.fft.ifft(np.fft.fft(_iq0, axis=1) * _m, axis=1)
    matched = round(max(1.0 - float(np.mean(np.abs(residual_iq) ** 2)) /
                        (float(np.mean(np.abs(_orig) ** 2)) + 1e-30), 0.0), 3)
    n_res = residual_iq.shape[1]
    f_res = np.fft.fftfreq(n_res, 1 / ctx.fs)
    # 峰搜索范围 = 整通带（2026-10-03 P0）：旧口径只在目标带内找峰，带外干扰
    # 连"定带"这一步都发现不了（与残差带限构成双重锁死）。
    inband = np.where((f_res >= -rx_half) & (f_res <= rx_half))[0]
    psd_res = np.abs(np.fft.fft(residual_iq, axis=1)) ** 2
    psd_ib = psd_res[:, inband].mean(axis=0)
    psd_db = 10 * np.log10(psd_ib + 1e-30)
    floor_db = float(np.median(psd_db))
    # 2026-10-08：峰判据重修（局部底 + 窄窗空间相干门控，见模块级注释）。
    # 旧口径 = 全局中位底 + ≥5 bin 连段，会把参考失配能量与目标滚降段并成 ~850 kHz
    # 宽的假"强峰"（46~49 dB），是 D1/S0 干净场景虚警的直接来源。
    peaks = _peak_selector()(residual_iq, ctx.fs, rx_half)
    # 宽带"共道支持区域"版本（同一残差，旧口径）：仅供**聚类内部定带**使用——
    # 下游 per-candidate 提取集的"同组"判据依赖 band_hz 宽度（频带重叠=共道伙伴），
    # 0.6 相干门控后的 40~60 kHz 窄带会让真实共道伙伴判不出重叠（D2 召回实测
    # 54%→36%）。对 agent 的**对外峰清单**只暴露门控后的 residual_peaks。
    legacy_peaks = (_residual_peaks_legacy(residual_iq, ctx.fs, rx_half)
                    if peaks is not None else [])
    noise_floor_db = round(floor_db, 1)
    band_occupied = len(peaks) > 0
    return {
        "station": station,
        "band_hz": [round(-rx_half), round(rx_half)],
        "target_band_hz": [round(lo), round(hi)],
        "max_delay_samples": used_tap,
        "matched_energy_fraction": matched,
        "residual_noise_floor_db": noise_floor_db,
        "residual_peaks": peaks,
        "residual_peaks_legacy": legacy_peaks,
        "band_occupied_beyond_target": band_occupied,
        "residual_iq_cached": resid_path.name,
        "note": ("residual = cooperative target removed (reference subtraction; the "
                 "residual floor is dominated by ~0.15%% reference mismatch energy that "
                 "is spatially INCOHERENT). residual_peaks are already gated: each band "
                 "is >= %.0f dB over a LOCAL median floor AND has spatial coherence "
                 "lambda1/sum >= %.2f in a +-%.0fk window, so spatially-white mismatch "
                 "energy is excluded. coherence ~1 = point source (real emitter), "
                 "~0.13 = noise-like (not a source). If band_occupied_beyond_target is "
                 "true, re-run analyze_spectrum_station / estimate_doa_station / "
                 "estimate_position with on_residual=true to isolate co-channel "
                 "interferers from the stripped band."
                 % (PEAK_OVER_DB, PEAK_COH_MIN, PEAK_COH_WIN_HZ / 1e3)),
    }


# ---------------- 工具 9：时频结构测量（占空/突发/跳速/扫频，D2 共存体区分钥匙） ----------------
def analyze_time_frequency_station(station: str = "station_a", freq_offset_hz: float = 0.0,
                                   bandwidth_hz: float | None = None,
                                   on_residual: bool = False, **_) -> dict:
    """对指定源带做带限切片 + STFT，测量时域结构证据：占空比、突发率、
    跳频信道数、扫频斜率。D2 的 5 种共存体主要靠时域结构区分
    （WiFi 突发 / 微波炉间歇 / Zigbee 短包 / DECT 时隙 / 蓝牙跳频）——
    本工具把 audit_separability 里"仅时域理论可分"的源对变成仪器可分。
    只返回测量值，类别判定由 Agent 对照辐射体手册完成。"""
    err = _check_station(station)
    if err:
        return {"error": err}
    ctx = _ctx()
    lo, hi = _band_edges(freq_offset_hz, bandwidth_hz)
    iq = _station_iq_raw(station, on_residual=on_residual)
    n = iq.shape[1]
    f = np.fft.fftfreq(n, 1 / ctx.fs)
    mask = (f >= lo) & (f <= hi)
    if mask.sum() < 2:
        return {"error": "band narrower than two FFT bins"}
    x = np.fft.ifft(np.fft.fft(iq, axis=1) * mask, axis=1).mean(axis=0)
    # 多尺度标定（2026-09-28）：win 64/hop 32（16µs 帧）可分辨 WiFi 包（30-80µs）、
    # 蓝牙驻留（6-15µs）、微波炉间歇、DECT 时隙。活动检测用 p10 噪声参考
    # （"中位×4"对高占空比/连续信号失效）；连续性用帧功率变异系数 CV；
    # 跳频用全帧峰频直方图（不筛活动帧——蓝牙宏占空连续）。
    # 6 体制指纹（标定实测）：WiFi/Zigbee duty 低+CV 高；微波炉 duty~0.78+bursts~180；
    # DECT duty~0.6+CV~1.3；蓝牙 CV~0.13+hop_ch~12；连续噪声 CV~0.16+hop_ch 0。
    win, hop = 64, 32
    nframes = max((n - win) // hop + 1, 1)
    w = np.hanning(win)
    frames = np.stack([x[i * hop:i * hop + win] * w for i in range(nframes)])
    pfr = np.abs(np.fft.fft(frames, axis=1)) ** 2
    frame_pow = pfr.sum(axis=1)
    ref = np.percentile(frame_pow, 10) + 1e-30
    active = frame_pow > ref * 10 ** (6 / 10)
    duty = round(float(active.mean()), 3)
    edges = int(np.sum(np.diff(active.astype(int)) == 1))
    seg_ms = round(float(active.sum() * hop / ctx.fs * 1e3), 2)
    cv = round(float(np.std(frame_pow) / (np.mean(frame_pow) + 1e-30)), 2)
    fgrid = np.fft.fftfreq(win, 1 / ctx.fs)
    peaks = fgrid[np.argmax(pfr, axis=1)]          # 全帧峰频（不筛活动）
    hist, _ = np.histogram(peaks, bins=64, range=(-2e6, 2e6))
    hop_ch = int(np.sum(hist > max(5, 0.05 * len(peaks))))
    return {
        "station": station, "on_residual": bool(on_residual),
        "band_hz": [round(lo), round(hi)],
        "duty_cycle": duty,
        "burst_count_per_window": edges,
        "mean_active_ms": seg_ms,
        "frame_power_cv": cv,
        "hop_channel_count": hop_ch,
        "note": ("measurements only — match against the radiator catalog's time-domain "
                 "structure. Calibrated fingerprints: WiFi/Zigbee low duty + high CV; "
                 "Microwave duty~0.78 + ~180 bursts; DECT duty~0.6 + CV~1.3; Bluetooth "
                 "CV~0.13 + hop_ch~12; continuous noise CV~0.16 + hop_ch 0. WiFi vs "
                 "Zigbee additionally need occupied bandwidth from spectrum analysis."),
    }


# ---------------- 工具 11：波束成形提取单源波形（源位置 → 独立波形 → 识别输入） ----------------

def extract_source_waveforms(freq_offset_hz: float | None = None,
                             bandwidth_hz: float | None = None,
                             max_delay_samples: int = 64, **_) -> dict:
    """波束成形提取单源波形：先空间聚类定源（源位置已知），再对每站做多源 LS
    波束成形，把共道重叠里的每个源**各自的独立波形**提取出来（互泄漏被显式建模）。
    输出每源波形的能量占比/缓存路径/分离质量（条件数）——Agent 拿到的是"每个源的
    独立波形缓存"，可送体制识别（CNN/拟合）或对照辐射体手册逐源分析。
    需 >=3 站（S1/S2）；S0 无空间分离能力，报错说明。
    """
    err_len = None
    ctx = _ctx()
    stations = sorted(ctx.allowed_stations)
    if len(stations) < 3:
        return {"error": "waveform extraction requires >= 3 stations (S1/S2 only)"}
    rx_half = ctx.fs * 0.45
    # 1. 源位置：空间聚类（内部含多站联合定带+宽带回退）。
    # 2026-10-03：读 **candidates** 栏——提取是测量动作，不该被支持度门槛预筛
    # （门槛栏空时硬报错曾把 28/50 样本整条链路清零）。candidates 为空才算
    # 聚类失败。
    clu = estimate_sources_by_spatial_clustering(freq_offset_hz, bandwidth_hz,
                                                 on_residual=True)
    sources = clu.get("candidates") or []
    if not sources:
        return {"error": "spatial clustering found no separable sources",
                "clustering": {k: clu.get(k) for k in ("n_intersections", "bands")}}
    src_positions = [s["position_xy_m"] for s in sources]
    # 同源散簇合并（2026-10-03 P1）：聚类 40m 合并半径小于多径散射扩展，同一
    # 物理源会拆成多个近簇——近共线重复列让 LS 病态、能量分配乱（tool-view
    # 命中源分类 3/12 的根因之一）。两簇在**所有站**的导向矢量相关都 >0.85
    # 才合并（跨站一致保证是同一物理源而非相邻的两个源），位置按 triads 加权。
    C_LIGHT = 299792458.0
    def _steers(pos2):
        cols = {}
        for r in ctx.observations["receivers"]:
            off_r = np.asarray(r["element_offsets_world_m"], float)
            rp3 = np.asarray(r["receiver_position_m"], float)
            d_xy = np.array([pos2[0] - rp3[0], pos2[1] - rp3[1], 0.0])
            u_r = d_xy / (np.linalg.norm(d_xy) + 1e-12)   # 水平化：z 不进导向
            cols[r["receiver_id"]] = np.exp(2j * np.pi * ctx.fc / C_LIGHT * (off_r @ u_r))
        return cols
    merged = []                          # [pos_xy(np.ndarray), weight, steers(dict)]
    for pos, wt in ((s["position_xy_m"], s["n_agreeing_triads"]) for s in sources):
        s_new = _steers(pos)
        tgt_m = None
        for m in merged:
            if all(abs(np.vdot(s_new[st], m[2][st])) / 8 > 0.85 for st in s_new):
                tgt_m = m
                break
        if tgt_m is None:
            merged.append([np.asarray(pos, float), float(wt), s_new])
        else:
            tw = tgt_m[1] + float(wt)
            tgt_m[0] = (tgt_m[0] * tgt_m[1] + np.asarray(pos, float) * float(wt)) / tw
            tgt_m[1] = tw
    src_positions = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p, _, _ in merged]
    # 频带：取聚类带（残差峰或回退全带）
    bands = clu.get("bands") or []
    if freq_offset_hz is not None:
        lo, hi = _band_edges(freq_offset_hz, bandwidth_hz)
    elif bands:
        f0 = bands[0]["band_center_hz"]
        lo, hi = max(f0 - 500e3, -rx_half), min(f0 + 500e3, rx_half)
    else:
        lo, hi = -min(0.25 * ctx.fs / 2, rx_half), min(0.25 * ctx.fs / 2, rx_half)
    lo, hi = max(lo, -rx_half), min(hi, rx_half)
    # 2. 每站分组 LS 提取。入组判据（2026-10-03 修正）：**导向去相关**——
    # 新候选与组内全部已有候选的站均导向互相关 <0.8 才同组（等价 max|Δu|
    # >~0.12，对应阵列 2/M=0.25 的真实分辨力）。原"频带重叠即同组"失效：
    # 聚类候选的 band_hz 全是 ~858kHz 宽且互相重叠（67% 样本只有 1 组），
    # 而候选间导向互相关中位 1.00（GT 真源只有 0.53）——近共线候选让 A 秩亏、
    # pinv 行范数爆炸，提取波形灌满噪声（条件准确率被压在 28.6%）。
    def _st_corr(pos1, pos2) -> float:
        s1, s2 = _steers(pos1), _steers(pos2)
        return float(np.mean([abs(np.vdot(s1[st], s2[st])) / 8 for st in s1]))

    # ---- per-candidate 提取集（2026-10-03 动作 A）：替换贪心分组 ----
    # 漏损分解（分母 32 个有共道伙伴的认领）：50% 伙伴未进候选集（召回问题，
    # 提取侧修不了）/12% 双条件不过/6% 未同组/31% 同组——而同组的 10 组里
    # 8 组混了假候选（贪心顺序依赖+组公共成员）→ oracle +12.5pp 被稀释为 0。
    # per-candidate 提取集 = {k} ∪ {频带重叠 且 导向去相关(<0.8) 的候选}——
    # 无顺序依赖、不会被别的组挡住；成员筛选 top-K=2（自己+perp_rms 最低的
    # 1 个伙伴，贴 oracle 的纯净真值对）。相同集合只提取一次。
    extract_sets: list[list[int]] = []
    for k, c in enumerate(sources):
        blo, bhi = c["band_hz"]
        members = [k]
        partners = []
        for m, cm in enumerate(sources):
            if m == k:
                continue
            mlo, mhi = cm["band_hz"]
            if (not (mhi < blo or mlo > bhi)
                    and _st_corr(c["position_xy_m"], cm["position_xy_m"]) < 0.8):
                partners.append(m)
        partners.sort(key=lambda m: sources[m].get("perp_rms_m", 999.0))
        members += partners[:1]              # top-K=2：自己+最强 1 个伙伴
        extract_sets.append(sorted(set(members)))
    uniq_sets: dict[tuple, list[int]] = {}
    for k, ms in enumerate(extract_sets):
        uniq_sets.setdefault(tuple(ms), []).append(k)
    per_station = {}
    extracted = {}                       # 全局候选序号 -> {station: path}
    for st in stations:
        conds = []
        for ms, ks in uniq_sets.items():
            poses = [sources[m]["position_xy_m"] for m in ms]
            lo_g = min(sources[m]["band_hz"][0] for m in ms)
            hi_g = max(sources[m]["band_hz"][1] for m in ms)
            try:
                r = _beamform_extract_station(st, poses, lo_g, hi_g)
            except Exception as exc:
                per_station[st] = {"error": str(exc)[:80]}
                break
            if not r.get("ok"):
                per_station[st] = {"error": r.get("reason")}
                break
            conds.append(r["cond"])
            W = r["waveforms"]
            ctx.cache_dir.mkdir(parents=True, exist_ok=True)
            for j, gk in enumerate(ms):
                pth = ctx.cache_dir / f"bfw_{st}_{gk}.npy"
                # 落盘即 CNN-ready（2026-10-03 口径下沉）：变频到候选频带中心
                # + RMS 归一化。切片 2 曾试"带内谱峰精测"——深重叠下串扰谱峰
                # 会带偏 argmax，条件准确率反而 -1.9pp（34.0→32.1），已回退带
                # 中心；"裸波形不能直接喂 CNN"的口径原先只活在 gate 里，下一
                # 个消费者必踩（漏做掉到 7.3%）。
                w = W[j].astype(np.complex128)
                cand_c = 0.5 * (sources[gk]["band_hz"][0] + sources[gk]["band_hz"][1])
                w = w * np.exp(-1j * 2 * np.pi * cand_c *
                               np.arange(w.shape[0]) / ctx.fs)
                w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)
                np.save(pth, w.astype(np.complex64))
                extracted.setdefault(gk, {})[st] = pth.name
        if not conds:
            continue
        per_station[st] = {"cond": max(conds),
                           "waveforms_cached": [f"bfw_{st}_{k}.npy"
                                                for k in sorted(extracted)]}
    # 3. 每源摘要：站间波形一致性（同源在不同站的波形经信道不同，但能量量级可对比）
    src_summary = []
    for k in range(len(src_positions)):
        st_have = list(extracted.get(k, {}).keys())
        ens = []
        for st in st_have:
            w = np.load(ctx.cache_dir / f"bfw_{st}_{k}.npy")
            e = float(np.mean(np.abs(w) ** 2))
            ens.append(round(10 * np.log10(e + 1e-30), 1))   # dB，未归一化小数值直接 round 会变 0
        src_summary.append({"source_index": k, "position_xy_m": src_positions[k],
                            "stations_extracted": st_have,
                            "energy_per_station": dict(zip(st_have, ens))})
    return {"n_sources": len(src_positions),
            "band_hz": [round(lo), round(hi)],
            "sources": src_summary,
            "per_station": per_station,
            "waveform_files": extracted,
            "note": ("each source's INDEPENDENT waveform extracted via multi-source LS "
                     "beamforming (co-channel mutual leakage explicitly modeled). Waveforms "
                     "cached as bfw_{station}_{k}.npy — feed each to radiator identification "
                     "(CNN/classification) per source. Stations with cond>1e6 or extraction "
                     "errors are reported per-station.")}


# ---------------- 工具 10：多站空间聚类分离（S1/S2 共道源分离核心） ----------------

def _u_to_world_rays(u: float, pos: np.ndarray, axis: np.ndarray) -> list[np.ndarray]:
    """u（方向余弦，阵轴投影）→ 世界 3D 单位方向（ULA 锥面，含镜象两分支）。

    单臂 ULA 只测阵轴方向余弦 u=cos(θ)，θ 为相对阵轴夹角；方位在其锥面任意。
    取两个正交参考构造两个候选世界方向（真实源取其一，也可用于交会）。"""
    v_ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1 = np.cross(axis, np.cross(axis, v_ref))
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    sin_t = np.sqrt(max(1.0 - u * u, 0.0))
    d = axis * u + v1 * sin_t
    return [d, axis * u - v1 * sin_t]


def estimate_sources_by_spatial_clustering(freq_offset_hz: float | None = None,
                                           bandwidth_hz: float | None = None,
                                           max_sources: int = 6,
                                           on_residual: bool = True,
                                           n_trials: int = 2000, **_) -> dict:
    """多站空间聚类分离共道源（S1/S2 核心能力；S0 单站退化不可用）。

    流程（2026-10-03 整合版）：
      1. 入口补齐各站残差缓存（strip_target_all_stations 内置缺参考→原始 IQ 回退）；
      2. 定带：多站 occupancy 残差峰合并（宽带地毯回退全带）；
      3. 每站 (u,f) 脊线池 _ridge_pool（窄子带滑窗 MUSIC→脊，带频率标签）；
      4. RANSAC：各带 _u_scan 峰射线交会，回投一致性对脊 u_mean 检验；
      5. 带内密度聚类（40m）→ 跨带位置合并 → 候选打分
         （perp_rms/deflation/triads 优先——n_explained 在共线组内无区分力）
         → 40m NMS → top-K；
      6. 输出双栏：candidates（全量打分榜）+ sources（过支持度门槛，
         K=len(sources) 仅对此栏成立）。
    高度不可观测（4 站阵轴平行 +y、站高差 <8m），定位按 2D 计。
    """
    ctx = _ctx()
    stations = sorted(ctx.allowed_stations)
    if len(stations) < 3:
        return {"error": "spatial clustering requires >= 3 stations (S1/S2 only)"}
    try:
        strip_target_all_stations(-ctx.fs * 0.45, ctx.fs * 0.45)
    except Exception:
        pass
    from cityrf.baselines.localization import intersect_rays
    rx_half = ctx.fs * 0.45
    # ---- 定带 ----
    # 2026-10-08：此处**刻意用旧宽带判据**（_residual_peaks_legacy）而不是
    # 对外导出的相干门控峰。原因：下游 per-candidate 提取集的"同组"判据依赖
    # band_hz **宽度**（频带重叠 → 共道伙伴），门控后的 40~60 kHz 窄带会让
    # 真实共道伙伴判不出重叠，D2 召回实测 54%→36%。聚类侧的带语义是"共道支持
    # 区域"，与 agent 侧"这在哪几个频点上有源"是两件事，故分开取用。
    # D1 的虚警问题只出现在**对外报告路径**（agent 读 residual_peaks），
    # 由对外峰清单的相干门控负责，聚类路径不受影响。
    if freq_offset_hz is None:
        band_map: dict[int, dict] = {}
        for st in stations:
            try:
                occ_st = estimate_band_occupancy_by_correlation(st)
            except Exception:
                continue
            for p in (occ_st.get("residual_peaks_legacy")
                      or occ_st.get("residual_peaks") or []):
                key = int(round(p["freq_offset_hz"] / 50e3))
                if key not in band_map or p["over_floor_db"] > band_map[key]["over_floor_db"]:
                    band_map[key] = p
        peak_bands = sorted(band_map.values(), key=lambda p: -p["over_floor_db"])[:4]
        if not peak_bands:
            peak_bands = [{"freq_offset_hz": 0.0, "bandwidth_hz": 2 * rx_half * 0.98}]
    else:
        peak_bands = [{"freq_offset_hz": float(freq_offset_hz),
                       "bandwidth_hz": float(bandwidth_hz or 60e3)}]
    # ---- 脊池 + 峰池 ----
    # 脊池落盘缓存（2026-10-03）：脊池确定性（同 resid 同结果），但构建要
    # 200 次 _u_scan/样本——gate 一轮 50 样本 >4 分钟的主开销。按
    # (resid mtime vs cache mtime) 失效，resid 重生成后自动过期。
    ridge_pool: dict[str, list] = {s: [] for s in stations}
    for s in stations:
        resid_p = ctx.cache_dir / f"resid_{s}.npy"
        cache_p = ctx.cache_dir / f"ridge_{s}.json"
        hit = (cache_p.is_file() and resid_p.is_file()
               and cache_p.stat().st_mtime >= resid_p.stat().st_mtime)
        if hit:
            try:
                ridge_pool[s] = json.loads(cache_p.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        try:
            ridge_pool[s] = _ridge_pool(s, on_residual=on_residual)
        except (NameError, AttributeError, TypeError, UnboundLocalError) as exc:
            raise RuntimeError(f"ridge pool build failed: {exc}") from exc
        except Exception:
            continue
        try:
            cache_p.write_text(json.dumps(ridge_pool[s]), encoding="utf-8")
        except Exception:
            pass
    peak_pool: dict[str, list] = {
        s: [(r["u_mean"], r["f_lo"], r["f_hi"], r["span"], r["strength"])
            for r in ridge_pool[s]] for s in stations}
    pos_s_cache = {s: _station_geometry(s) for s in stations}
    u_tol = 0.4 / _station_iq_raw(stations[0]).shape[0]
    # ---- RANSAC：**2 站俯视方位线交会**（2026-10-03 主杠杆）----
    # z 不可观测（4 站阵轴平行 +y、站高差 <8m）、定位按 2D 计 → 2 站的两条
    # 锥面射线俯视投影成 2 条方位线，在 2D 平面交于一点（2 方程 2 未知，
    # 数学成立）。≥2 站覆盖 62.5% vs ≥3 站 32%——召回天花板翻倍。
    # 分支歧义（锥面镜象 × 射线选择的多解）交给 40m 密度聚类消化：真解会被
    # 多组配对重复命中，假分支不会。回投检验降级为**加分项**（验证站命中数
    # 记入 triads 权重），不再一票否决（覆盖本来就只有 32%，否决会踢掉真源）。
    def _azimuth_cross(p1, d1, p2, d2):
        """两条俯视方位线的 2D 交点 + 交角（deg）。平行返回 None。"""
        d1v = np.array([d1[0], d1[1]]); d2v = np.array([d2[0], d2[1]])
        n1 = np.linalg.norm(d1v); n2 = np.linalg.norm(d2v)
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        d1v, d2v = d1v / n1, d2v / n2
        cross = d1v[0] * d2v[1] - d1v[1] * d2v[0]
        if abs(cross) < np.sin(np.radians(12.0)):    # 交角 <12° = 病态几何
            return None
        diff = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        t = (diff[0] * d2v[1] - diff[1] * d2v[0]) / cross
        return np.array([p1[0] + t * d1v[0], p1[1] + t * d1v[1],
                         float(np.degrees(abs(np.arcsin(np.clip(cross, -1, 1)))))])

    band_report: list[dict] = []
    n_hits_total = 0
    band_cands: list[dict] = []

    def _bands_to_cands(bands: list[dict], report: list[dict],
                        duo: bool = False) -> tuple[list[dict], int]:
        """把一批候选带跑 trio RANSAC，返回 (band_cands, 本轮命中数)。

        2026-10-08 从内联循环提为函数：对空结果做一次"整通带回退"重试
        （见下方 _run_band_ransac），修 d2_00000 这类"定带偏窄/几何不过 →
        直接报错"的 50 样本级联失败。逻辑与提函数前逐行一致。
        """
        out: list[dict] = []
        hits_n = 0
        for pb in bands:
            f0, bw = pb["freq_offset_hz"], pb["bandwidth_hz"]
            per_st: dict[str, list] = {}
            for st in stations:
                try:
                    scan = _u_scan_cached(st, f0, bw, on_residual=on_residual)
                except Exception:
                    continue
                pos_s, axis_s = pos_s_cache[st]
                for u, rel in scan["u_peaks"][:U_PEAK_MAX]:
                    for d in _u_to_world_rays(u, pos_s, axis_s):
                        per_st.setdefault(st, []).append((pos_s, d))
            usable = [s for s in stations if s in per_st]
            report.append({"band_center_hz": round(f0, 0),
                           "n_stations_with_peaks": len(usable)})
            if len(usable) < 2:
                continue
            rng = np.random.default_rng(7)
            band_hits: list[tuple] = []
            # 2026-10-03 实测定案：duo 交会（2 站）召回 25.8% 低于 trio 基线 29.2%——
            # 2 站覆盖 62.5% 的理论天花板被"分支歧义×密度竞争"稀释（4 分支组合
            # 1 对 3 错，假交点在密度上不输真解）。解锁前提：分支消解（如站间
            # u 一致性预筛分支）或加权密度聚类。当前默认回退 trio。
            DUO_RANSAC = bool(duo)
            if DUO_RANSAC:
                trials_here = (n_trials * 3) // max(len(bands), 1)
            else:
                trials_here = n_trials // max(len(bands), 1)
            for _ in range(trials_here):
                if DUO_RANSAC:
                    pair = rng.choice(usable, 2, replace=False)
                else:
                    pair = rng.choice(usable, 3, replace=False)
                rays = [per_st[s][int(rng.integers(len(per_st[s])))] for s in pair]
                if DUO_RANSAC:
                    xing = _azimuth_cross(rays[0][0], rays[0][1], rays[1][0], rays[1][1])
                    if xing is None:
                        continue
                    p2d = xing[:2]
                    if not (np.linalg.norm(p2d) < 3000):
                        continue
                    weight = 1.0
                    for s in usable:
                        if s in pair:
                            continue
                        pos_s, axis_s = pos_s_cache[s]
                        dvec = np.array([p2d[0] - pos_s[0], p2d[1] - pos_s[1], 0.0])
                        u_pred = float(axis_s @ dvec) / (np.linalg.norm(dvec) + 1e-12)
                        upk = _u_scan_cached(s, f0, bw, on_residual=on_residual)["u_peaks"]
                        if any(abs(u_pred - up[0]) < u_tol for up in upk):
                            weight += 1.0
                    angle = float(xing[2])
                    perp_proxy = 90.0 / max(angle, 5.0)
                    band_hits.append((np.array([p2d[0], p2d[1], 10.0]),
                                      perp_proxy * (2.0 / weight)))
                else:
                    hit = intersect_rays([r[0] for r in rays], [r[1] for r in rays])
                    if not (hit and 0 < hit["position_m"][2] <= 300
                            and max(hit["ranges_m"]) < 2000):
                        continue
                    p3 = np.asarray(hit["position_m"], float)
                    n_inlier = 0
                    for s in usable:
                        pos_s, axis_s = pos_s_cache[s]
                        dvec = p3 - pos_s
                        u_pred = float(axis_s @ dvec) / (np.linalg.norm(dvec) + 1e-12)
                        upk = _u_scan_cached(s, f0, bw, on_residual=on_residual)["u_peaks"]
                        if any(abs(u_pred - up[0]) < u_tol for up in upk):
                            n_inlier += 1
                    if n_inlier >= 3:
                        band_hits.append((p3, float(hit.get("perpendicular_rms_m", 999.0))))
            hits_n += len(band_hits)
            if not band_hits:
                continue
            clusters_b: list[dict] = []
            for (p, perp) in band_hits:
                near = [c for c in clusters_b if np.linalg.norm(p - c["mu"]) < 40]
                if near:
                    c = near[0]
                    c["pts"].append(p); c["perps"].append(perp)
                    c["mu"] = np.mean(c["pts"], axis=0)
                else:
                    clusters_b.append({"mu": p, "pts": [p], "perps": [perp]})
            clusters_b.sort(key=lambda c: -len(c["pts"]))
            for c in clusters_b[:3]:
                if len(c["pts"]) < 3:
                    continue
                out.append({"pos": c["mu"], "triads": len(c["pts"]),
                            "perp_rms": float(np.median(c["perps"])),
                            "f0": f0, "bw": bw})
        return out, hits_n

    band_cands, n_hits_total = _bands_to_cands(peak_bands, band_report)
    if not band_cands and freq_offset_hz is None and peak_bands:
        # 2026-10-08 回退：相干门控定带给出的是**窄带**候选（bandwidth_hz 40~150k），
        # 而 trio RANSAC 的 u_scan 在窄带上只取到 1 个峰，3D 射线交点很容易
        # 被镜像分支带到 z 越界（d2_00000 实测：4 站各有峰，但一条都不过）。
        # 旧口径在"无峰"时会回退**整通带**（1.76 MHz 宽），宽带上 u_scan 的
        # 脊结构不同、几何反而过得去。这里保留同一层保险：用整通带再跑一次。
        retry_band = [{"freq_offset_hz": 0.0,
                       "bandwidth_hz": 2 * rx_half * 0.98}]
        band_cands, n_hits_total = _bands_to_cands(retry_band, band_report)
        if not band_cands:
            # 仍无解 → 最后一级保险：**duo（2 站俯视方位线交会）**。trio 要求
            # 3D 射线交点的 z 落在 (0,300] 且最大斜距 <2000，而端射区（|u|→1）
            # 的 u 估计误差会让 z 爆掉；duo 走 2D 方位线、只要求交角 ≥12°，
            # 配合 40 m 密度聚类 + n_inlier≥3 判据筛掉镜像分支（d2_00000 实测
            # 真源 u 在 3 站为 0.10/-0.39/-1.00，第三站落在端射区）。
            band_cands, n_hits_total = _bands_to_cands(retry_band, band_report,
                                                       duo=True)
    if not band_cands:
        return {"error": "no valid intersections in any residual band",
                "candidates": [], "sources": [], "bands": band_report}
    # ---- 跨带位置合并 ----
    band_cands.sort(key=lambda c: -c["triads"])
    merged_cands: list[dict] = []
    for cand in band_cands:
        near = [f for f in merged_cands if np.linalg.norm(cand["pos"][:2] - f["pos_xy"]) < 40]
        if near:
            near[0]["triads"] += cand["triads"]
            near[0]["perps"].append(cand["perp_rms"])
            continue
        merged_cands.append({"pos_xy": cand["pos"][:2].copy(), "triads": cand["triads"],
                             "perps": [cand["perp_rms"]],
                             "f0": cand["f0"], "bw": cand["bw"]})

    def _support(pos_xy, band=None) -> set:
        sup = set()
        deltas = []
        for s in stations:
            pos_s, axis_s = pos_s_cache[s]
            dvec = np.array([pos_xy[0] - pos_s[0], pos_xy[1] - pos_s[1], 0.0])
            u_pred = float(axis_s @ dvec) / (np.linalg.norm(dvec) + 1e-12)
            hit = []
            for (um, flo, fhi, _sp, _st) in peak_pool[s]:
                if abs(u_pred - um) >= CLAIM_TOL:
                    continue
                if band is not None and (band[1] < flo or band[0] > fhi):
                    continue
                hit.append((um, abs(u_pred - um)))
            if hit:
                deltas.append(min(d for _, d in hit))
                sup.add((s, round(float(min(hit, key=lambda t: t[1])[0]), 3)))
        if not deltas:
            return sup, 1.0, 1.0
        return sup, float(np.mean(deltas)), float(max(deltas))

    # ---- 候选打分（perp_rms/deflation/triads 优先）+ NMS + top-K ----
    # ④ deflation 低通（2026-10-03 D1 虚警课题）：deflation_db=投影掉候选方向
    #    后 MUSIC 峰塌陷量——真源塌陷巨大（D2 真候选中位 46.6dB）、"无中生有"
    #    的假候选塌陷小（D1 干净样本假候选仅 8.7dB）——defl < 15dB 的候选
    #    直接滤（D1 干净场景虚警 38% 的主修；D2 真候选全部 >>15dB 不受损）。
    MIN_DEFLATION_DB = 15.0
    scored: list[dict] = []
    for c in merged_cands:
        band = [c["f0"] - c["bw"] / 2, c["f0"] + c["bw"] / 2]
        sup, _md, max_du = _support(c["pos_xy"], band)
        drops = []
        for st in stations:
            try:
                drops.append(_deflation_drop(st, c["f0"], c["bw"],
                                             [float(c["pos_xy"][0]), float(c["pos_xy"][1])]))
            except Exception:
                continue
        med = float(np.median(drops)) if drops else 0.0
        scored.append({"pos_xy": c["pos_xy"], "triads": c["triads"],
                       "f0": c["f0"], "bw": c["bw"],
                       "band_hz": [c["f0"] - c["bw"] / 2, c["f0"] + c["bw"] / 2],
                       "n_explained": len(sup), "max_du": max_du,
                       "perp_rms": float(np.median(c["perps"])),
                       "deflation_db": round(med, 1)})
    scored = [c for c in scored if c["deflation_db"] >= MIN_DEFLATION_DB]
    scored.sort(key=lambda c: (c["perp_rms"], -c["deflation_db"], -c["triads"],
                               c["max_du"], -c["n_explained"]))
    # NMS 双层（2026-10-03 QA2 重测后升级）：
    # ① **u 空间 NMS**——53% 的假候选与真源同方向（max|Δu|<0.25 Rayleigh 限、
    #    65% 在 300m 内）是"同方向错距离解"，40m 位置 NMS 拦不住；站均导向
    #    互相关 >0.85（等效 max|Δu|<0.15）合并。QA2 强制共道对 |Δu|≥2/M →
    #    真源对互相关≈0 → 不会误合真源（旧数据集上此刀掉召回的问题已消失）。
    # ② 位置 NMS（40m）保留（远距离幻影）。
    def _steers_nms(pos2):
        cols = {}
        for s in stations:
            pos_s, _axis = pos_s_cache[s]
            d_xy = np.array([pos2[0] - pos_s[0], pos2[1] - pos_s[1], 0.0])
            u_r = d_xy[1] / (np.linalg.norm(d_xy) + 1e-12)   # 阵轴 (+y) 投影标量
            cols[s] = np.exp(2j * np.pi * 0.5 * u_r * np.arange(8))
        return cols

    def _st_corr_nms(pos1, pos2) -> float:
        s1 = _steers_nms(pos1)
        s2 = _steers_nms(pos2)
        return float(np.mean([abs(np.vdot(s1[st], s2[st])) / 8 for st in s1]))

    # ③ 双字段门限**已证伪回退**（2026-10-03 配方第②步实测）：假簇 63.0→53.6%
    #    但条件准确率 48.1→46.2 反降——"两字段近似独立叠加"不成立：confidence
    #    本身 = 1/(1+max_du+perp_rms/100)，与 perp_rms 共享项、非独立字段，
    #    双差门只丢"双差"候选、单差假候选保留、真候选却被误丢。保留 u-NMS +
    #    位置 NMS 即当前最优组合。
    final: list[dict] = []
    for c in scored:                          # 已按 perp_rms 升序——先者留
        for f in final:
            # u 空间 NMS 加**频带重叠**条件（2026-10-03 实测修正）：48 组真源对
            # 中 3 组（6%）互相关>0.85 会被误合，且全部是频带不重叠的**非共道**
            # 对——QA 约束只保证共道对可分，非共道对角度不受限。同方向且同频带
            # 才合并（近失幻影必然同频带，这一刀不损失其收益）。
            fb = (not (c["band_hz"][1] < f["band_hz"][0] or
                       c["band_hz"][0] > f["band_hz"][1]))
            if fb and _st_corr_nms(c["pos_xy"], f["pos_xy"]) > 0.85:
                break
        else:
            if any(np.linalg.norm(c["pos_xy"] - f["pos_xy"]) < 40 for f in final):
                continue
            final.append(c)

    def _fmt(f):
        return {"position_xy_m": [round(float(f["pos_xy"][0]), 1), round(float(f["pos_xy"][1]), 1)],
                "band_hz": [round(f["f0"] - f["bw"] / 2), round(f["f0"] + f["bw"] / 2)],
                "n_agreeing_triads": int(f["triads"]),
                "n_explained_stations": int(f["n_explained"]),
                "max_du": round(f["max_du"], 3),
                "perp_rms_m": round(f["perp_rms"], 1),
                "deflation_db": f["deflation_db"],
                "confidence": round(1.0 / (1 + f["max_du"] + f["perp_rms"] / 100.0), 2)}

    n_usable = sum(1 for s in stations if peak_pool[s])
    candidates = [_fmt(f) for f in final[:max_sources]]
    gated = [f for f in final
             if f["n_explained"] >= max(n_usable - 1, 1) and f["max_du"] <= 0.3]
    sources = [_fmt(f) for f in gated[:max_sources]]
    return {"n_stations": len(stations), "n_intersections": n_hits_total,
            "n_candidates": len(final),
            "bands": band_report,
            "candidates": candidates,
            "sources": sources,
            "support_threshold": {"n_explained_min": max(n_usable - 1, 1),
                                  "max_du_max": 0.3, "n_usable_stations": n_usable},
            "note": ("candidates = ALL scored cone-intersection candidates ranked by "
                     "(perp_rms asc, deflation desc, triads desc) after 40m NMS — do NOT "
                     "use len(candidates) as the interferer count. sources = candidates "
                     "passing the support gate; K = len(sources) is the spatial "
                     "source-count evidence. Height unobservable (parallel horizontal "
                     "arrays), localization is 2D by design.")}
# ---------------- 对外工具表：剥离仅供内部使用的字段 ----------------
# `estimate_band_occupancy_by_correlation` 必须同时算两条峰清单：
#   residual_peaks        —— 门控后的对外清单（agent 读这条）
#   residual_peaks_legacy —— 旧宽带口径，**只给聚类内部定带**用（见本文件 1309 行注释）
# 但工具返回值是 `json.dumps(result)` 原样回灌给模型的（benchmark_v2/evaluator_v2.py:367-372），
# 于是 2026-10-08 起 agent 其实**同时**看到两条互相矛盾的清单：一条 3~6 条窄峰（8~22 dB），
# 一条 850 kHz 宽、46~80 dB 的旧口径宽带。实测后果：D2/S2 的 detection 0.86 → 0.48。
# 这里在**对外工具表**这一层剥掉内部字段，函数本身与聚类内部调用都不变。
AGENT_HIDDEN_KEYS = ("residual_peaks_legacy",)
# 逐条目（interferers[*]）要剥掉的内部字段：intent / is_spoof 都不是测量结果，
# 而是 kind 的确定性查表值（INTENT_OF[kind]；is_spoof = (kind == "GNSS_spoof")）。
# 实测（diag/_an_intent_axis.py）：不剥时 Agent 的 intent 就是照抄工具
# （D2/D3/D4 = 98.9% / 88.1% / 91.6%），T3 两轴退化成“体制轴的下游换算”。
AGENT_HIDDEN_ITEM_KEYS = ("intent", "is_spoof")


def _agent_facing(fn):
    """包一层：从对外工具返回值里剥掉 AGENT_HIDDEN_KEYS 指定的内部字段（原地 pop）。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, dict):
            for key in AGENT_HIDDEN_KEYS:
                out.pop(key, None)
            items = out.get("interferers")
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        for key in AGENT_HIDDEN_ITEM_KEYS:
                            it.pop(key, None)
        return out

    return wrapper

TOOL_FUNCTIONS = {
    "analyze_spectrum_station": analyze_spectrum_station,
    "estimate_num_sources_station": estimate_num_sources_station,
    "estimate_doa_station": estimate_doa_station,
    "estimate_position": estimate_position,
    "detect_time_domain_station": detect_time_domain_station,
    "estimate_modulation_features_station": estimate_modulation_features_station,
    "list_radiator_catalog": list_radiator_catalog,
    "estimate_band_occupancy_by_correlation":
        _agent_facing(estimate_band_occupancy_by_correlation),
    "analyze_time_frequency_station": analyze_time_frequency_station,
    "estimate_sources_by_spatial_clustering": estimate_sources_by_spatial_clustering,
    "extract_source_waveforms": extract_source_waveforms,
    "classify_extracted_waveform":
        _agent_facing(classify_extracted_waveform),
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "analyze_spectrum_station",
        "description": ("Per-station spectrum analysis (v2 core): merged PSD peaks, per-source "
                        "candidates {freq, bandwidth (normalized to fs; *_hz copies in Hz), "
                        "power ratio, world_azimuth_candidates_deg}, OBW99. freq_offset_hz is "
                        "relative to the receiver center (0 Hz = target band center)."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string", "description": "Station id, e.g. station_a"},
            "use_spatial": {"type": "boolean",
                            "description": "MVDR joint freq-angle candidates (slower). Default true."},
            "on_residual": {"type": "boolean",
                            "description": "Analyze the cooperative-target-stripped residual band IQ instead of the raw capture (set true after estimate_band_occupancy_by_correlation reports band_occupied_beyond_target). Default false."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "estimate_num_sources_station",
        "description": ("Estimate the number of spatial components at one station over the "
                        "full receiver band (MDL rank; each component = one source or a strong "
                        "multipath echo, so RT scenes may exceed the true emitter count). "
                        "Cross-check with spectral candidates and per-band DOA."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string"},
            "max_sources": {"type": "integer", "description": "Upper bound, default 4"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "estimate_doa_station",
        "description": ("Per-station DOA for the band around freq_offset_hz in direction-cosine "
                        "u space (no ridge degeneracy): u peaks + azimuth candidate PAIRS "
                        "[asin(u), 180-asin(u)] (ULA front/back mirror — single station cannot "
                        "resolve; use estimate_position). Omit freq_offset_hz for full band."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string"},
            "freq_offset_hz": {"type": "number",
                               "description": "Source band center in Hz (from spectrum analysis)"},
            "bandwidth_hz": {"type": "number", "description": "Approx source bandwidth in Hz"},
            "on_residual": {"type": "boolean",
                            "description": "Scan the target-stripped residual band instead of raw capture. Default false."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "estimate_position",
        "description": ("Multi-station cone-intersection localization for the band around "
                        "freq_offset_hz: per-station direction-cosine measurements -> least-"
                        "squares grid over (x, y, height) -> ranked hypotheses {position_xy_m, "
                        "u_residual_rms}. Needs >= 2 stations (S1/S2). World frame, meters."),
        "parameters": {"type": "object", "properties": {
            "freq_offset_hz": {"type": "number"},
            "bandwidth_hz": {"type": "number"},
            "on_residual": {"type": "boolean",
                            "description": "Localize from the target-stripped residual band. Default false."}},
            "required": ["freq_offset_hz"]}}},
    {"type": "function", "function": {
        "name": "extract_source_waveforms",
        "description": ("Beamform-extract each co-channel candidate's INDEPENDENT waveform: "
                        "after spatial clustering, own-projection LS per candidate (co-channel "
                        "crosstalk left to the classifier, which is trained with siblings). "
                        "Waveforms are cached CNN-ready (shifted to candidate band center + "
                        "RMS normalized) as bfw_{station}_{k}.npy - feed directly to radiator "
                        "identification. Returns per-source energy per station + extraction "
                        "quality (cond). Needs >= 3 stations."),
        "parameters": {"type": "object", "properties": {
            "freq_offset_hz": {"type": "number", "description": "Band center; omit for full residual clustering"},
            "bandwidth_hz": {"type": "number", "description": "Band width"},
            "max_delay_samples": {"type": "integer", "description": "Legacy FIR taps (unused; direct subtraction)"},
            "on_residual": {"type": "boolean", "description": "Unused; always on the stripped residual"}},
            "required": []}}},
{"type": "function", "function": {
        "name": "classify_extracted_waveform",
        "description": ("Classify each candidate's extracted waveform (CNN, 19 radiators) "
                        "and return READY-TO-COPY answer fields: {freq_offset_hz, kind, "
                        "intent, is_spoof, position_xy_m, confidence} per candidate. Call "
                        "this AFTER extract_source_waveforms; then write the final JSON "
                        "answer directly from these items — no further measurement needed."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string", "description": "Station whose cached waveforms to classify. Default station_a"}},
            "required": []}}},
        {"type": "function", "function": {
        "name": "detect_time_domain_station",
        "description": ("Per-station time-domain features: PAPR, pulse duty cycle, pulse edge "
                        "count, clipping fraction (burst/pulse and saturation evidence)."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "estimate_modulation_features_station",
        "description": ("Per-station modulation/waveform features with per-peak band slices and "
                        "template distances (CNN backend when available). Compare slice features "
                        "against the radiator catalog to identify the kind."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string"},
            "freq_offset_hz": {"type": "number", "description": "Optional, for your own notes"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "list_radiator_catalog",
        "description": ("Public spec sheet of the 16 urban low-altitude radiators + 2 simple "
                        "kinds (tone/noise): modulation class and waveform description. Use it "
                        "to map measured features to a radiator kind."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "estimate_band_occupancy_by_correlation",
        "description": ("Cooperative-target correlation strip on the target band: the monitoring "
                        "network provides the cooperative target's reference waveform (with "
                        "multipath, noise/interference-free). Subtracts it from the capture, "
                        "reports matched (target) vs unmatched residual energy, and caches the "
                        "target-stripped residual band IQ. residual_peaks are GATED measurements "
                        "{freq_offset_hz, over_floor_db (vs a LOCAL median floor), bandwidth_hz, "
                        "band_hz, coherence}: every entry already cleared >= 8 dB over floor AND "
                        "spatial coherence lambda1/sum >= 0.6 in a +-20 kHz window, so "
                        "spatially-white residual energy is excluded. coherence ~1 = point-source "
                        "emitter, ~0.13 = noise-like. If band_occupied_beyond_target is true, the "
                        "target band carries co-channel interferers — then re-run "
                        "analyze_spectrum_station / estimate_doa_station / estimate_position with "
                        "on_residual=true to isolate them."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string"},
            "band_low_hz": {"type": "number", "description": "Target band lower edge; omit to use the measured 0 Hz target OBW"},
            "band_high_hz": {"type": "number", "description": "Target band upper edge"},
            "max_delay_samples": {"type": "integer", "description": "FIR channel taps for target multipath fit (default 64, max 256)"}},
            "required": ["station"]}}},
    {"type": "function", "function": {
        "name": "analyze_time_frequency_station",
        "description": ("Time-frequency structure of the band around freq_offset_hz: duty "
                        "cycle, burst count, mean active duration, hop-channel count / dwell, "
                        "and frequency-sweep slope. This is the key discriminator for "
                        "spectrally-overlapping radiators (burst vs intermittent vs hopping "
                        "vs TDMA vs continuous)."),
        "parameters": {"type": "object", "properties": {
            "station": {"type": "string"},
            "freq_offset_hz": {"type": "number", "description": "Source band center in Hz"},
            "bandwidth_hz": {"type": "number", "description": "Approx source bandwidth in Hz"},
            "on_residual": {"type": "boolean", "description": "Analyze the target-stripped residual. Default false."}},
            "required": ["freq_offset_hz"]}}},
    {"type": "function", "function": {
        "name": "estimate_sources_by_spatial_clustering",
        "description": ("Multi-station spatial clustering to SEPARATE co-channel interferers "
                        "(needs >= 3 stations; S1/S2). Each station measures direction-cosine "
                        "peaks in the band, then RANSAC cone-intersection across stations finds "
                        "spatially-separable sources (density peaks >= 40m apart). Returns "
                        "ranked sources {position_xy_m, height_m, confidence} — each is a "
                        "distinct spatial source that single-channel slicing cannot resolve. "
                        "K = len(sources) is the spatial source-count evidence. Use on "
                        "the residual band (default on_residual=true) after correlation strip."),
        "parameters": {"type": "object", "properties": {
            "freq_offset_hz": {"type": "number", "description": "Band center; omit to use station_a residual peak"},
            "bandwidth_hz": {"type": "number", "description": "Band width"},
            "max_sources": {"type": "integer", "description": "Upper bound, default 6"},
            "on_residual": {"type": "boolean", "description": "Use target-stripped residual. Default true"}},
            "required": []}}},
]
