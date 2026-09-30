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
from em_signal_simulator_v3.emitters import EMITTERS  # noqa: E402


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
    """单位 RMS 归一化缓存（v2 工具输入口径；尺度对全部 v2 特征无影响）。"""
    ctx = _ctx()
    key = f"{'resid_' if on_residual else 'norm_'}{station}"
    if key not in ctx._npy_cache:
        ctx.cache_dir.mkdir(parents=True, exist_ok=True)
        iq = _station_iq_raw(station, on_residual=on_residual)
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) + 1e-12
        out = ctx.cache_dir / f"{key}.npy"
        np.save(out, (iq / rms).astype(np.complex64))
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
    # 局部极大取峰（3 点窗），按强度排序
    idx = np.where((music_db[1:-1] >= music_db[:-2]) & (music_db[1:-1] >= music_db[2:]))[0] + 1
    idx = idx[np.argsort(music_db[idx])[::-1]][:4]
    peaks = [(round(float(u[i]), 3), round(float(music_db[i]), 1)) for i in idx
             if music_db[i] >= -8.0]
    return {"u_peaks": peaks or [(round(float(u[int(np.argmax(music))]), 3), 0.0)],
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
                                         freq_offset_hz: float | None = None, **_) -> dict:
    err = _check_station(station)
    if err:
        return {"error": err}

    def _run():
        res = tools_v2.estimate_modulation_features(
            str(_station_npy(station)),
            target_bandwidth_normalized=_ctx().target_bw_normalized)
        res["station"] = station
        res["note_angle"] = ("per_peak angle_deg (if present) is the legacy array-local "
                             "convention; use estimate_doa_station for world azimuth")
        return res

    res = _cached_result(("mod_features", station), _run)
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


def list_radiator_catalog(**_) -> dict:
    """16 种城市低空辐射体 + 2 种简单体制的公开波形规格（不含意图标签）。"""
    return {"radiators": RADIATOR_CATALOG,
            "note": "public spec sheet; identify the radiator kind from measured "
                    "waveform features, not by guessing"}


# ---------------- 工具 8：合作目标剥离（共道解耦核心） ----------------

def _strip_target_at_station(station: str, lo: float, hi: float,
                             max_delay_samples: int = 64) -> np.ndarray:
    """对单站做目标带相关性剥离，返回残差带限 IQ，并缓存 resid_{station}.npy。
    供 estimate_band_occupancy_by_correlation 与多站空间聚类共用。"""
    ctx = _ctx()
    ref = ctx.target_reference.get(station)
    if ref is None:
        raise RuntimeError(f"target reference not available for station '{station}'")
    ref = np.asarray(ref, complex)
    if ref.ndim == 2:
        ref = ref[0]
    iq = _station_iq_raw(station)
    n = iq.shape[1]
    f = np.fft.fftfreq(n, 1 / ctx.fs)
    mask = (f >= lo) & (f <= hi)
    s = signal_block(iq, ctx.fs, ctx.fc, receiver_id=station)
    s_sel = select_band(s, lo, hi)
    ref_sel = np.fft.ifft(np.fft.fft(ref) * mask).astype(complex)
    requested = min(max(int(max_delay_samples), 1), 256)
    res, used_tap = None, None
    for tap in sorted({requested, 32, 16, 8, 4, 2}, reverse=True):
        if tap > requested:
            continue
        try:
            res = template_residual(s_sel, ref_sel, max_delay_samples=tap)
            used_tap = tap
            break
        except ValueError:
            continue
    if res is None:
        raise RuntimeError("target strip rank-deficient at all FIR sizes")
    residual_iq = np.asarray(res["residual"].iq, complex)
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
    # matched = 1 - 残差能量/带内原始能量（目标带被剥离的比例）
    _iq0 = _station_iq_raw(station)
    n0 = _iq0.shape[1]
    _f = np.fft.fftfreq(n0, 1 / ctx.fs)
    _m = (_f >= lo) & (_f <= hi)
    _orig = np.fft.ifft(np.fft.fft(_iq0, axis=1) * _m, axis=1)
    matched = round(max(1.0 - float(np.mean(np.abs(residual_iq) ** 2)) /
                        (float(np.mean(np.abs(_orig) ** 2)) + 1e-30), 0.0), 3)
    n_res = residual_iq.shape[1]
    f_res = np.fft.fftfreq(n_res, 1 / ctx.fs)
    inband = np.where((f_res >= lo) & (f_res <= hi))[0]
    psd_res = np.abs(np.fft.fft(residual_iq, axis=1)) ** 2
    psd_ib = psd_res[:, inband].mean(axis=0)
    psd_db = 10 * np.log10(psd_ib + 1e-30)
    floor_db = float(np.median(psd_db))
    over = psd_db - floor_db
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
                peaks.append({"freq_offset_hz": round(float(f_res[inband[kc]]), 0),
                              "over_floor_db": round(float(seg.max()), 1),
                              "bandwidth_hz": round((k2 - k + 1) * ctx.fs / n_res, 0)})
            k = k2 + 1
        else:
            k += 1
    noise_floor_db = round(floor_db, 1)
    band_occupied = len(peaks) > 0
    return {
        "station": station,
        "band_hz": [round(lo), round(hi)],
        "max_delay_samples": used_tap,
        "matched_energy_fraction": matched,
        "residual_noise_floor_db": noise_floor_db,
        "residual_peaks": peaks,
        "band_occupied_beyond_target": band_occupied,
        "residual_iq_cached": resid_path.name,
        "note": ("residual = cooperative target removed (FIR multipath fit trained on first "
                 "half, validated on second half; multipath spread beyond max_delay_samples "
                 "raises residual). If band_occupied_beyond_target is true, re-run "
                 "analyze_spectrum_station / estimate_doa_station / estimate_position with "
                 "on_residual=true to isolate co-channel interferers from the stripped band."),
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
    """多站空间聚类分离共道源（S1/S2 核心能力；S0 因单站退化不可用）。

    问题：单通道切片下共道干扰在信息论上不可分（盲源分离）。但各干扰在不同站
    看到不同世界 DOA → 多站锥面交会可在空间上分离。流程：
      1. 对目标的残差带内，每站 1D-MUSIC 测多个 u 峰（含镜象分支）；
      2. RANSAC：随机 ≥3 站各抽 1 候选 → intersect_rays 交会（病态/背向/超高自动拒）；
      3. 空间聚类（<40m 合并）→ K 个位置簇 = K 个源的候选位置；
      4. 返回每簇：位置/交会命中数（密度=置信）+ per_station 指派方向。
    返回源簇列表，供 Agent 判断"该残差带内有几个空间可分源"与定位。
    """
    ctx = _ctx()
    stations = sorted(ctx.allowed_stations)
    if len(stations) < 3:
        return {"error": "spatial clustering requires >= 3 stations (S1/S2 only)"}
    from cityrf.baselines.localization import intersect_rays
    # 目标带（若未给频偏，用 station_a 残差谱峰定带）
    rx_half = ctx.fs * 0.45
    if freq_offset_hz is None:
        try:
            occ = estimate_band_occupancy_by_correlation("station_a")
        except Exception:
            occ = {}
        peaks_r = occ.get("residual_peaks") or []
        if not peaks_r:
            return {"error": "no residual peaks found; run correlation strip first",
                    "sources": []}
        f0 = peaks_r[0]["freq_offset_hz"]
        bw = peaks_r[0]["bandwidth_hz"]
    else:
        f0, bw = float(freq_offset_hz), float(bandwidth_hz or 60e3)
    lo, hi = _band_edges(f0, bw)
    # 每站多峰 → 候选射线
    per_st: dict[str, list] = {}
    for st in stations:
        try:
            scan = _u_scan_cached(st, f0, bw, on_residual=on_residual)
        except Exception:
            continue
        pos, axis = _station_geometry(st)
        for u, rel in scan["u_peaks"][:3]:
            for d in _u_to_world_rays(u, pos, axis):
                per_st.setdefault(st, []).append((pos, d))
    usable = [s for s in stations if s in per_st]
    if len(usable) < 3:
        return {"error": "insufficient stations with DOA peaks",
                "sources": [], "station_peaks": {s: len(per_st.get(s, [])) for s in stations}}
    # RANSAC 交会 + 空间聚类
    hits: list[np.ndarray] = []
    rng = np.random.default_rng(7)
    for _ in range(n_trials):
        trio = rng.choice(usable, 3, replace=False)
        rays = [per_st[s][int(rng.integers(len(per_st[s])))] for s in trio]
        hit = intersect_rays([r[0] for r in rays], [r[1] for r in rays])
        if hit and 0 < hit["position_m"][2] <= 300 and max(hit["ranges_m"]) < 2000:
            hits.append(np.array(hit["position_m"]))
    if not hits:
        return {"sources": [], "n_intersections": 0,
                "n_stations": len(usable), "reason": "no valid intersections"}
    H = np.array(hits)
    # 密度聚类：按位置近邻（40m）找密度峰
    clusters: list[dict] = []
    for p in H:
        near = [c for c in clusters if np.linalg.norm(p - c["mu"]) < 40]
        if near:
            c = near[0]
            c["pts"].append(p); c["mu"] = np.mean(c["pts"], axis=0)
        else:
            clusters.append({"mu": p, "pts": [p]})
    clusters.sort(key=lambda c: -len(c["pts"]))
    sources = []
    for c in clusters[:max_sources]:
        if len(c["pts"]) < 5:
            break                       # 低密度 = 噪声/磨损假交点
        mu = c["mu"]
        sources.append({
            "position_xy_m": [round(float(mu[0]), 1), round(float(mu[1]), 1)],
            "height_m": round(float(mu[2]), 1),
            "n_agreeing_triads": len(c["pts"]),
            "confidence": round(float(len(c["pts"]) / max(len(H), 1)), 2),
        })
    return {"band_center_hz": round(f0, 0), "n_stations": len(usable),
            "n_intersections": len(H), "n_peaks_per_station":
                {s: len(per_st.get(s, [])) for s in stations},
            "sources": sources,
            "note": ("sources = multi-station cone-intersection density peaks (>=40m apart, "
                     ">=5 agreeing triads). Each source is a spatially-separable co-channel "
                     "interferer or the target remnant; K = len(sources) is the spatial source "
                     "count evidence. Re-inspect with on_residual=true (default) — target is "
                     "already stripped.")}


TOOL_FUNCTIONS = {
    "analyze_spectrum_station": analyze_spectrum_station,
    "estimate_num_sources_station": estimate_num_sources_station,
    "estimate_doa_station": estimate_doa_station,
    "estimate_position": estimate_position,
    "detect_time_domain_station": detect_time_domain_station,
    "estimate_modulation_features_station": estimate_modulation_features_station,
    "list_radiator_catalog": list_radiator_catalog,
    "estimate_band_occupancy_by_correlation": estimate_band_occupancy_by_correlation,
    "analyze_time_frequency_station": analyze_time_frequency_station,
    "estimate_sources_by_spatial_clustering": estimate_sources_by_spatial_clustering,
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
                        "multipath, noise/interference-free). Fits a short FIR receive channel, "
                        "reports matched (target) vs unmatched residual energy vs noise floor, and "
                        "caches the target-stripped residual band IQ. If band_occupied_beyond_target "
                        "is true, the target band carries co-channel interferers — then re-run "
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
