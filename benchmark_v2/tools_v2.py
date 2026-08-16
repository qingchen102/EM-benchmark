"""Signal-analysis tools for the v2 benchmark (multi-antenna IQ, LLM Agents).

v2 工具集（与 v1 tools.py 隔离，供 evaluator_v2.py 使用）：
- analyze_spectrum:      多天线功率合成频谱 + 多峰检测（频偏估计）+ OBW99 带宽
- estimate_num_sources:  干扰源数量估计（MDL 信息论准则为主 + 显著谱峰旁证）
- estimate_doa:          MUSIC 空间谱 DOA 估计（复用 visualization.music_spectrum）
- detect_time_domain:    时域特征（PAPR、脉冲占空比）

所有工具返回结构化 JSON（测量值 + 文本洞察），不做分类结论 ——
分类/参数结论由 Agent 综合判断（填空题/选择题由 evaluator 打分）。
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import numpy as np

# 信号模型库位于 simulation/ 子目录：根目录运行本模块时需把它加入导入路径
_SIM_DIR = Path(__file__).resolve().parent / "simulation"
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from em_signal_simulator.baseband import measure_obw99


def _spectral_features(x):
    """频谱特征（v1 验证有效）：平坦度、分段频率漂移、峰均比。"""
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    eps = np.finfo(float).tiny
    geom = np.exp(np.mean(np.log(psd + eps)))
    arith = float(np.mean(psd)) + eps
    flatness = float(geom / arith)                       # 宽带/OFDM 高，成型调制低
    seg_size = n // 8
    if seg_size > 16:
        seg_peaks = []
        for i in range(8):
            seg = x[i * seg_size:(i + 1) * seg_size]
            s = np.abs(np.fft.fft(seg)) ** 2
            f = np.fft.fftfreq(seg_size, 1.0)
            seg_peaks.append(float(f[int(np.argmax(s))]))
        drift = float(np.std(seg_peaks))                 # 扫频/跳频高，静止调制低
    else:
        drift = 0.0
    peak_to_avg = float(10.0 * np.log10(max(psd.max(), eps) / arith))   # 单音高
    return flatness, drift, peak_to_avg


def _time_features(x):
    """时域特征：PAPR、占空比、幅度分布峰度。"""
    power = np.abs(x) ** 2
    mean_p = float(np.mean(power))
    papr = float(10.0 * np.log10(max(power.max(), 1e-12) / max(mean_p, 1e-12)))
    median = float(np.median(power))
    mad = float(np.median(np.abs(power - median)))
    thr = median + 3.0 * max(mad, 1e-12)
    duty = float(np.mean(power > thr))
    amp = np.abs(x)
    mu = float(np.mean(amp))
    var = float(np.mean((amp - mu) ** 2))
    kurt = float(np.mean((amp - mu) ** 4) / (var ** 2 + 1e-12)) if var > 0 else 0.0
    return papr, duty, kurt


def _feature_vector(x):
    """7 维特征向量（调制识别用）。"""
    flat, drift, p2a = _spectral_features(x)
    papr, duty, kurt = _time_features(x)
    return np.array([measure_obw99(x), flat, drift, p2a, kurt, papr, duty])


def _merge_peaks(peaks, freq, psd, min_gap=0.1, drop_db=20.0):
    """把间隔 < min_gap 的相邻峰合并为同一源（成型调制的频谱纹波）。

    成型调制（QPSK/OFDM 等）的频谱内部有多个近等幅纹波峰，直接当独立峰会导致：
    源数虚高、频偏选错纹波（低估）、per-peak 带宽只切到单个纹波（严重低估）。
    合并规则：频率间隔 < min_gap（0.1）视为同一源 —— 与真实源的分离约束
    （FREQ_MIN_SEP=0.15）有安全距离。

    返回 [{freq(组内功率加权中心), rel_db, bandwidth(-drop_db 轮廓跨度), num_peaks}]，
    按幅度降序。
    """
    if not peaks:
        return []
    peaks = sorted(peaks, key=lambda p: p[0])
    groups = [[peaks[0]]]
    for p in peaks[1:]:
        if p[0] - groups[-1][-1][0] < float(min_gap):
            groups[-1].append(p)
        else:
            groups.append([p])

    out = []
    for g in groups:
        n_pk = len(g)
        g_sorted = sorted(g, key=lambda p: -p[1])
        f_peak, db_peak = g_sorted[0]
        # 组内功率加权中心（幅度加权）作为频偏估计（纹波组 → 带中心）
        amps = [10.0 ** (db / 20.0) for _, db in g]
        fs = [f for f, _ in g]
        wsum = sum(a * f for a, f in zip(amps, fs))
        f_center = wsum / (sum(amps) + 1e-12)
        # 带宽：组最左/最右峰向外扩展到 -drop_db 的轮廓跨度
        n = len(freq)
        i_lo = int(np.argmin(np.abs(freq - g[0][0])))
        i_hi = int(np.argmin(np.abs(freq - g[-1][0])))
        thr_lo = float(psd[i_lo]) * 10.0 ** (-float(drop_db) / 10.0)
        thr_hi = float(psd[i_hi]) * 10.0 ** (-float(drop_db) / 10.0)
        while i_lo > 0 and float(psd[i_lo]) >= thr_lo:
            i_lo -= 1
        while i_hi < n - 1 and float(psd[i_hi]) >= thr_hi:
            i_hi += 1
        bw = max(float(freq[i_hi] - freq[i_lo]), 0.01)
        out.append({"freq": float(f_center), "rel_db": float(db_peak),
                    "bandwidth": float(bw), "num_peaks": int(n_pk)})
    out.sort(key=lambda d: -d["rel_db"])
    return out


_CANDIDATES = tuple(sorted(["BPSK", "QPSK", "16QAM", "64QAM", "GFSK", "OOK", "OFDM",
                            "FHSS", "LFM", "single_tone", "swept", "pulse",
                            "broadband", "nfm"]))
_TEMPLATE_FEATURES = {}


def _template_features(name, rng_seed=0):
    """在线生成候选模板并计算特征向量（缓存，每次会话只算一次）。"""
    key = (name, rng_seed)
    if key in _TEMPLATE_FEATURES:
        return _TEMPLATE_FEATURES[key]
    rng = np.random.default_rng(rng_seed)
    from em_signal_simulator.baseband import generate_baseband, MODULATIONS
    from em_signal_simulator.jamming import generate_interferer_waveform
    if name in MODULATIONS:
        x = generate_baseband(name, 1024, rng=rng)
    else:
        x = generate_interferer_waveform(name, 1024, rng=rng)
    fv = _feature_vector(x)
    _TEMPLATE_FEATURES[key] = fv
    return fv


def _peak_half_bw(psd, freq, f_p, drop_db=10.0, lo=0.01, hi=0.25):
    """峰 f_p 附近降到 -drop_db 的半带宽（自适应切片窗口，clamp 到 [lo, hi]）。"""
    n = len(freq)
    i_p = int(np.argmin(np.abs(freq - float(f_p))))
    thr = float(psd[i_p]) * 10.0 ** (-float(drop_db) / 10.0)
    i_lo = i_p
    while i_lo > 0 and float(psd[i_lo]) >= thr:
        i_lo -= 1
    i_hi = i_p
    while i_hi < n - 1 and float(psd[i_hi]) >= thr:
        i_hi += 1
    bw = float(freq[i_hi] - freq[i_lo])
    return min(max(bw / 2.0, lo), hi)


def _peak_slice_features(x, freq, psd, f_p, half_bw=0.04):
    """对频谱峰 f_p 附近的频带切片信号测特征（近似分离单个源）。

    混合信号（目标+干扰+噪声）的全局特征被目标主导；对明显分离的干扰
    （blocking/adjacent/大频偏），其峰附近的频带切片近似其本体，特征才有效。
    """
    n = len(x)
    mask = np.abs(freq - float(f_p)) <= float(half_bw)
    if int(np.sum(mask)) < 16:
        return None
    spec = np.fft.fftshift(np.fft.fft(x))
    x_bp = np.fft.ifft(np.fft.ifftshift(spec * mask))
    return _feature_vector(x_bp)


def estimate_modulation_features(sample_path: str) -> dict[str, Any]:
    """调制识别特征：全局特征 + 按谱峰分离的切片特征 + 与 14 种候选模板的特征距离。

    只返回测量值（含模板特征距离），不做匹配结论——由 Agent 自行判断。
    全局特征（频谱平坦度/漂移/峰均比等）对宽带/扫频/单音有判别力（v1 验证）；
    逐谱峰切片近似分离单个源（对频带分离的干扰有效，co_channel 时仅供参考）。
    """
    iq = _load_iq(sample_path)
    x = _combine_channels(iq)
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    flat, drift, p2a = _spectral_features(x)
    papr, duty, kurt = _time_features(x)

    # 模板特征矩阵 + min-max 归一化（用干净模板的分布）
    tpl = np.stack([_template_features(c) for c in _CANDIDATES])
    lo, hi = tpl.min(axis=0), tpl.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def _distances(fv):
        """返回与各模板的归一化欧氏距离（忠实数据，不排名）。"""
        fv_n = (fv - lo) / span
        tpl_n = (tpl - lo) / span
        dist = np.sqrt(((tpl_n - fv_n) ** 2).sum(axis=1))
        return [{"template": _CANDIDATES[i], "feature_distance": round(float(dist[i]), 2)}
                for i in range(len(_CANDIDATES))]

    global_dist = _distances(_feature_vector(x))
    peaks = _find_peaks(psd, freq, rel_thresh_db=-8.0, min_sep=1.0 / 64.0)
    per_peak = []
    for f_p, _ in peaks[:2]:           # 只对最强 2 个峰做切片（控制输出量）
        hbw = _peak_half_bw(psd, freq, f_p)
        fv_slice = _peak_slice_features(x, freq, psd, f_p, half_bw=hbw)
        if fv_slice is not None:
            per_peak.append({"peak_freq": round(float(f_p), 4),
                             "template_distances": _distances(fv_slice)})

    return {
        "spectral_flatness": round(float(flat), 3),
        "time_frequency_drift_std": round(float(drift), 4),
        "peak_to_average_ratio_db": round(float(p2a), 1),
        "amplitude_kurtosis": round(float(kurt), 2),
        "papr_db": round(float(papr), 1),
        "pulse_duty_cycle": round(float(duty), 3),
        "obw99_normalized": round(float(measure_obw99(x)), 3),
        "global_template_distances": global_dist,
        "per_peak_template_distances": per_peak,
    }


def _load_iq(sample_path) -> np.ndarray:
    data = np.asarray(np.load(sample_path))
    if data.ndim not in (1, 2):
        raise ValueError(f"IQ sample must be 1D or (num_antennas, num_samples), got {data.shape}")
    return data.astype(np.complex128, copy=False)


def _combine_channels(iq: np.ndarray) -> np.ndarray:
    """多天线功率合成（幅度 = 平均功率，相位 = 天线 0），用于频谱类测量。"""
    if iq.ndim == 1:
        return iq
    power = np.mean(np.abs(iq) ** 2, axis=0)
    ref = iq[0] / (np.abs(iq[0]) + 1e-12)
    return ref * np.sqrt(power)


def _find_peaks(psd, freq, rel_thresh_db=-12.0, min_sep=1.0 / 64.0):
    """在 PSD 中找显著峰（相对主峰阈值 + 最小间距去重）。

    返回 [(freq, rel_dB), ...]，按幅度降序。
    """
    psd = np.asarray(psd, dtype=float)
    freq = np.asarray(freq, dtype=float)
    if psd.max() <= 0:
        return []
    rel = psd / psd.max()
    thr = 10.0 ** (rel_thresh_db / 10.0)
    mask = rel > thr
    idx = np.where(mask)[0]
    # 局部最大（含边界）
    peaks_idx = []
    for i in idx:
        left = rel[i - 1] <= rel[i] if i > 0 else True
        right = rel[i + 1] <= rel[i] if i < len(rel) - 1 else True
        if left and right:
            peaks_idx.append(i)
    # 按幅度排序，再按最小间距去重（保留强的）
    peaks_idx.sort(key=lambda i: rel[i], reverse=True)
    chosen = []
    for i in peaks_idx:
        if all(abs(freq[i] - freq[j]) >= min_sep for j in chosen):
            chosen.append(i)
    chosen.sort(key=lambda i: freq[i])
    return [(float(freq[i]), float(10.0 * np.log10(rel[i] + 1e-12))) for i in chosen]


def analyze_spectrum(sample_path: str, target_modulation: str = "unknown") -> dict[str, Any]:
    """多天线频谱分析：合并纹波后的源峰检测 + 每源占用带宽 + 全局 OBW99。

    只返回结构化测量值，无结论性文本。
    - peaks_normalized: 合并后源峰（成型调制纹波已合并，峰数≈真实源数）
    - per_peak_bandwidth: 每源峰的能量轮廓带宽（-20dB 跨度）
    - peaks_excluding_center: |f|>0.05 的峰（干扰候选，目标在 0Hz 附近）
    """
    iq = _load_iq(sample_path)
    x = _combine_channels(iq)
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))

    raw_peaks = _find_peaks(psd, freq, rel_thresh_db=-14.0, min_sep=1.0 / 128.0)
    merged = _merge_peaks(raw_peaks[:24], freq, psd)   # 纹波合并
    merged = merged[:6]
    obw99 = measure_obw99(x)

    peaks_list = [{"freq": round(m["freq"], 4), "rel_db": round(m["rel_db"], 1),
                   "num_peaks": m["num_peaks"]} for m in merged]
    per_peak_bw = [{"freq": round(m["freq"], 4), "bandwidth": round(m["bandwidth"], 4)}
                   for m in merged[:4]]
    # 干扰候选：排除中心（目标）附近的峰
    excluding_center = [{"freq": round(m["freq"], 4), "rel_db": round(m["rel_db"], 1),
                         "bandwidth": round(m["bandwidth"], 4)}
                        for m in merged if abs(m["freq"]) > 0.05]

    return {
        "num_samples": int(n),
        "num_antennas": int(iq.shape[0]) if iq.ndim == 2 else 1,
        "peaks_normalized": peaks_list,
        "main_peak_freq_normalized": round(merged[0]["freq"], 4) if merged else None,
        "obw99_normalized": round(float(obw99), 4),
        "per_peak_bandwidth": per_peak_bw,
        "peaks_excluding_center": excluding_center,
    }


def _mdl_num_sources(evals, n_samples):
    """MDL 准则估计源数（空间协方差特征值，降序输入）。"""
    m = len(evals)
    n = n_samples
    best_k, best_v = 0, np.inf
    for k in range(0, m):  # k = 信号数（0..M-1）
        lam = evals[k:]
        nz = m - k
        if nz <= 0 or lam[0] <= 0:
            continue
        avg = float(np.mean(lam))
        geo = float(np.exp(np.mean(np.log(lam + 1e-30))))
        mdl = -n * nz * np.log(geo / (avg + 1e-30)) + 0.5 * k * (2 * m - k) * np.log(n)
        if mdl < best_v:
            best_v, best_k = mdl, k
    return best_k


def estimate_num_sources(sample_path: str, max_sources: int = 3) -> dict[str, Any]:
    """干扰源数量估计：MDL 信息论准则（空间协方差，主）+ 显著谱峰（旁证）。"""
    iq = _load_iq(sample_path)
    n = iq.shape[-1]

    # 1) 显著谱峰（仅作旁证：-6dB 相对主峰、最小间距 1/32）
    x = _combine_channels(iq)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    n_peak = min(len(_find_peaks(psd, freq, rel_thresh_db=-6.0, min_sep=1.0 / 32.0)),
                 max_sources)

    # 2) MDL（多天线空间协方差）
    n_mdl = None
    if iq.ndim == 2 and iq.shape[1] >= iq.shape[0]:
        r = (iq @ iq.conj().T) / iq.shape[1]
        evals = np.linalg.eigvalsh(r)[::-1]
        n_mdl = int(np.clip(_mdl_num_sources(evals, iq.shape[1]), 1, max_sources))

    if n_mdl is None:
        estimate = max(1, min(int(n_peak), max_sources))
        confidence = 0.55
    else:
        estimate = n_mdl                     # MDL 为主
        confidence = 0.85 if n_peak == n_mdl else 0.6

    return {
        "num_sources_estimate": int(estimate),
        "spectral_peak_count": int(n_peak),
        "eigenvalue_estimate": n_mdl,
        "confidence": round(float(confidence), 2),
    }


def estimate_doa(sample_path: str, num_sources: int = 1) -> dict[str, Any]:
    """MUSIC 空间谱 DOA 估计（4 元 ULA）。返回峰值 DOA 列表与谱摘要。"""
    iq = _load_iq(sample_path)
    if iq.ndim != 2 or iq.shape[0] < 2:
        return {"error": "DOA estimation requires multi-antenna IQ (num_antennas >= 2)"}
    m = iq.shape[0]
    ns = int(np.clip(num_sources, 1, m - 1))
    from em_signal_simulator.visualization import music_spectrum
    doas, spec = music_spectrum(iq, num_sources=ns)
    # 峰提取
    rel = spec / (spec.max() + 1e-300)
    mask = rel > 1e-3  # -30dB
    idx = np.where(mask)[0]
    peaks_idx = []
    for i in idx:
        left = rel[i - 1] <= rel[i] if i > 0 else True
        right = rel[i + 1] <= rel[i] if i < len(rel) - 1 else True
        if left and right:
            peaks_idx.append(i)
    peaks_idx.sort(key=lambda i: rel[i], reverse=True)
    chosen = []
    for i in peaks_idx:
        if all(abs(doas[i] - doas[j]) >= 12.0 for j in chosen):
            chosen.append(i)
        if len(chosen) >= ns:
            break
    chosen.sort(key=lambda i: doas[i])
    return {
        "doa_estimates_deg": [round(float(doas[i]), 1) for i in chosen],
        "num_sources_used": ns,
    }


def detect_time_domain(sample_path: str) -> dict[str, Any]:
    """时域特征：PAPR、脉冲占空比（区分 pulse 类干扰）。"""
    iq = _load_iq(sample_path)
    x = _combine_channels(iq)
    power = np.abs(x) ** 2
    mean_power = float(np.mean(power))
    papr_db = float(10.0 * np.log10(max(power.max(), 1e-12) / max(mean_power, 1e-12)))
    median = float(np.median(power))
    mad = float(np.median(np.abs(power - median)))
    thr = median + 3.0 * max(mad, 1e-12)
    active = power > thr
    duty = float(np.mean(active))
    # 脉冲计数（活动段）
    edges = int(np.sum(np.diff(active.astype(int)) == 1)) if len(active) > 1 else 0
    return {
        "papr_db": round(float(papr_db), 1),
        "pulse_duty_cycle": round(float(duty), 3),
        "pulse_edge_count": int(edges),
    }


TOOL_FUNCTIONS_V2 = {
    "analyze_spectrum": analyze_spectrum,
    "estimate_num_sources": estimate_num_sources,
    "estimate_doa": estimate_doa,
    "detect_time_domain": detect_time_domain,
    "estimate_modulation_features": estimate_modulation_features,
}

TOOL_SCHEMAS_V2 = [
    {
        "type": "function",
        "function": {
            "name": "analyze_spectrum",
            "description": "Multi-antenna spectrum analysis: peak detection (frequency-offset candidates) and 99% occupied bandwidth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "target_modulation": {"type": "string",
                                          "description": "Expected target modulation type."},
                },
                "required": ["sample_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_num_sources",
            "description": "Estimate the number of active sources (spectral peaks + spatial eigenvalue method).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "max_sources": {"type": "integer", "description": "Upper bound, default 3."},
                },
                "required": ["sample_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_doa",
            "description": "MUSIC spatial-spectrum DOA estimation for the ULA. Returns DOA peaks in degrees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "num_sources": {"type": "integer",
                                    "description": "Assumed number of sources for MUSIC noise-subspace dimension."},
                },
                "required": ["sample_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_time_domain",
            "description": "Time-domain features: PAPR and pulse duty cycle (for pulse-type interference).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                },
                "required": ["sample_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_modulation_features",
            "description": "Modulation-identification features: spectral flatness, time-frequency drift, peak-to-average ratio, amplitude kurtosis, PAPR, duty cycle, OBW99, and per-peak feature distances against 14 candidate modulation/jamming types. Returns measurements only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                },
                "required": ["sample_path"],
            },
        },
    },
]
