"""Signal-analysis tools for the v2 benchmark (multi-antenna IQ, LLM Agents).

v2 工具集（与 v1 tools.py 隔离，供 evaluator_v2.py 使用）：
- analyze_spectrum:      多天线功率合成频谱 + 多峰检测（频偏估计）+ OBW99 带宽
- estimate_num_sources:  源数估计：num_sources_estimate（MDL，默认建议）+ 三路证据（MDL/显著谱峰/纹波合并峰）+ 一致性，决策树修正
- estimate_doa:          MUSIC 空间谱 DOA 估计（复用 visualization.music_spectrum）；含跨阶稳定峰与 3 阶新峰
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


def _valley_split(group, freq, psd, thr_db, protect_radius=0.0):
    """按谱谷深度递归分裂峰组：相邻峰之间谷底比两者中较高者低 >= thr_db 视为异源。

    固定间距分组（min_gap）无法区分"同一源的浅纹波"与"两个近距源的深谷"
    （数据集允许两源频差 < 0.15、靠 DOA 分开）。真实异源之间频谱存在深谷，
    而单源成型纹波间谷浅 —— 以谷深为判据更物理。

    protect_radius: 目标保护半径（通常 = 目标 OBW99/2）。中点落在保护区内
    的峰对不分裂 —— 强宽带目标的内部谱零点（OFDM 子载波旁瓣/FHSS 跳频间隙）
    谷也很深，无保护时会把自己劈成多组强碎片（离线复现：干净样本被报出
    2 个幽灵干扰、单目标劈成 14 组）；目标中心/带宽是观测先验，带内本来
    就无需分裂。
    """
    g = sorted(group, key=lambda p: p[0])
    if len(g) < 2:
        return [g]
    for j in range(1, len(g)):
        a, b = g[j - 1], g[j]
        if abs((a[0] + b[0]) / 2.0) < float(protect_radius):
            continue                                    # 目标带内不分裂
        ia = int(np.argmin(np.abs(freq - a[0])))
        ib = int(np.argmin(np.abs(freq - b[0])))
        if ib - ia < 2:
            continue
        valley_db = float(10.0 * np.log10(float(np.min(psd[ia:ib + 1])) + 1e-300))
        if max(a[1], b[1]) - valley_db >= float(thr_db):
            return (_valley_split(g[:j], freq, psd, thr_db, protect_radius) +
                    _valley_split(g[j:], freq, psd, thr_db, protect_radius))
    return [g]


def _merge_peaks(peaks, freq, psd, min_gap=0.1, drop_db=20.0, split_valley_db=8.0,
                 protect_radius=0.0):
    """把间隔 < min_gap 的相邻峰合并为同一源（成型调制的频谱纹波），再按谱谷分裂。

    成型调制（QPSK/OFDM 等）的频谱内部有多个近等幅纹波峰，直接当独立峰会导致：
    源数虚高、频偏选错纹波（低估）、per-peak 带宽只切到单个纹波（严重低估）。
    两步：
    1. 频率间隔 < min_gap（0.1）先归并 —— 与真实源分离约束（FREQ_MIN_SEP=0.15）
       有安全距离；
    2. 组内谱谷分裂：相邻峰谷底比较高峰低 >= split_valley_db（8dB；阈值扫描
       7~14dB 权衡：越低覆盖越高但虚峰越多，8dB 时候选覆盖 0.39→0.64 而合并组
       计数失配与 10dB 持平）→ 拆成不同源。修复近距双源被并成一组导致候选丢失/
       中心拉偏的问题（离线诊断：未覆盖干扰中 94% 属此类偏斜）。

    返回 [{freq(组内功率加权中心), rel_db, bandwidth(-drop_db 轮廓跨度),
    num_peaks, span_idx(组最左/最右谱索引, 供 Hann 切片)}]，按幅度降序。
    """
    if not peaks:
        return []
    peaks = sorted(peaks, key=lambda p: p[0])
    raw_groups = [[peaks[0]]]
    for p in peaks[1:]:
        if p[0] - raw_groups[-1][-1][0] < float(min_gap):
            raw_groups[-1].append(p)
        else:
            raw_groups.append([p])

    groups = []
    for g in raw_groups:
        groups.extend(_valley_split(g, freq, psd, split_valley_db,
                                    protect_radius=protect_radius))

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
                    "bandwidth": float(bw), "num_peaks": int(n_pk),
                    "span_idx": (int(i_lo), int(i_hi))})
    out.sort(key=lambda d: -d["rel_db"])
    return out


# 调制候选集：默认收窄到数据集实际干扰池（真实调制 5 种 + 波形 5 种）。
# 可通过 estimate_modulation_features(candidates=...) 覆盖（换数据集/扩大任务时）。
_CANDIDATES = tuple(sorted(["BPSK", "QPSK", "GFSK", "LFM", "OFDM",
                            "single_tone", "swept", "pulse", "broadband", "nfm"]))
_TEMPLATE_FEATURES = {}

# 波形类候选按 Carson 带宽反推噪声调频参数的固定比例（模板等效化用）
_NFM_SPLIT_FDEV, _NFM_SPLIT_NB = 0.66, 0.17


def _waveform_kw_for_bandwidth(name, bw):
    """把波形类候选等效参数化到目标带宽 bw（与数据生成侧同一语义口径）。"""
    bw = float(np.clip(bw, 0.02, 0.95))
    if name == "single_tone":
        return {}
    if name == "swept":
        half = min(bw / 2.0, 0.45)
        return {"f_start": -half, "f_stop": half}
    if name == "pulse":
        return {"duty_cycle": 0.2}          # 典型脉冲占空比（数据生成范围 0.05~0.3）
    if name == "broadband":
        return {}                            # 全带弹幕，等效带宽不可调
    if name == "nfm":
        # Carson: bw ≈ 2(f_dev + noise_bw)
        return {"freq_deviation": _NFM_SPLIT_FDEV * bw / 2.0,
                "noise_bandwidth": min(_NFM_SPLIT_NB * bw / 2.0, 0.44)}
    return {}


def _template_features(name, bw, rng_seed=0):
    """在与切片实测带宽 bw 相同的带宽下生成候选模板并计算特征向量（缓存）。

    v2 修复：旧实现固定用各调制的默认带宽生成模板，而数据集中干扰带宽在
    0.03~0.59 随机采样 —— OBW 维度永远对不上，导致距离失真（64QAM 兜底）。
    现在先测切片 OBW99，再用同一带宽生成模板，特征距离才反映波形差异本身。
    """
    bw_q = round(float(np.clip(bw, 0.02, 0.95)) * 50.0) / 50.0   # 0.02 步长量化缓存
    key = (name, bw_q, rng_seed)
    if key in _TEMPLATE_FEATURES:
        return _TEMPLATE_FEATURES[key]
    rng = np.random.default_rng(rng_seed)
    from em_signal_simulator.baseband import generate_baseband, MODULATIONS
    from em_signal_simulator.jamming import generate_interferer_waveform
    if name in MODULATIONS:
        x = generate_baseband(name, 1024, rng=rng, bandwidth_normalized=bw_q)
    else:
        x = generate_interferer_waveform(name, 1024, rng=rng,
                                         **_waveform_kw_for_bandwidth(name, bw_q))
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


def _band_slice(x, freq, psd, i_lo, i_hi):
    """Hann 软掩膜频带切片 + CFO 解旋到基带（近似分离单个源）。

    混合信号全局特征被目标主导；对频带分离的干扰，其合并组（_merge_peaks 的
    span_idx）覆盖的频带切片近似其本体。相比旧砖墙矩形窗：
    - Hann 掩膜抑制频谱泄漏 ringing，切片更干净；
    - 切片后按组功率加权中心解旋到基带，与模板口径一致（模板均在零频生成）。
    返回 (x_baseband, bw_obw99)；切片无效（过窄/零功率）返回 (None, None)。
    """
    n = len(x)
    i_lo = max(int(i_lo) - 2, 0)
    i_hi = min(int(i_hi) + 2, n - 1)
    if i_hi - i_lo < 8:
        return None, None
    # 组内功率加权中心作为切片中心频点
    idx = np.arange(i_lo, i_hi + 1)
    w = psd[idx]
    f_c = float(np.sum(freq[idx] * w) / (np.sum(w) + 1e-30))
    mask = np.zeros(n)
    m = i_hi - i_lo + 1
    mask[i_lo:i_hi + 1] = np.hanning(m)
    spec = np.fft.fftshift(np.fft.fft(x))
    x_bp = np.fft.ifft(np.fft.ifftshift(spec * mask))
    t = np.arange(n)
    x_bb = x_bp * np.exp(-1j * 2.0 * np.pi * f_c * t)      # CFO 解旋到基带
    bw = measure_obw99(x_bb)
    if not np.isfinite(bw) or bw <= 0.002:
        return None, None
    return x_bb, bw


def _band_coherence(iq, freq, i_lo, i_hi):
    """频带空间相干性：带限切片的空间协方差主特征值与其余均值之比。

    物理判据（与 DOA 无关）：真实源（任意到达角）在各天线上的带限信号是
    秩 1 相干的，lambda1/mean(lambda_rest) 远大于 1；噪声起伏包各天线独立，
    比值接近 1。用于区分"弱真实源"与"噪声包" —— 两者的功率谱强度重叠
    （-6~-10dB），单靠幅度地板不可分。
    返回相干比；天线数 <2 时返回 inf（退化为无检验）。
    """
    if iq.ndim != 2 or iq.shape[0] < 2:
        return float("inf")
    n = iq.shape[-1]
    m_ = int(i_hi) - int(i_lo) + 1
    if m_ < 8:
        return float("inf")
    mask = np.zeros(n)
    mask[int(i_lo):int(i_hi) + 1] = np.hanning(m_)
    spec = np.fft.fftshift(np.fft.fft(iq, axis=-1), axes=-1)
    y = np.fft.ifft(np.fft.ifftshift(spec * mask, axes=-1), axis=-1)
    r_cov = (y @ y.conj().T) / y.shape[-1]
    evals = np.sort(np.linalg.eigvalsh(r_cov))[::-1]
    rest = float(np.mean(evals[1:])) + 1e-30
    return float(evals[0]) / rest


def _source_groups(raw_peaks, freq, psd, protect_radius=0.05,
                   floor_db=-10.0, top=6, drop_db=20.0):
    """候选源组公共管线：谱谷合并/分裂（含目标保护区）-> 强度地板 -> top-K。

    floor_db（默认 -10dB，相对全局主峰）：过滤噪声起伏包。低 SNR 样本上噪底
    局部极大可达 -6~-14dB，无地板时会以"候选源"身份污染列表（ds_run10 干净
    样本幽灵干扰的另一来源；r9 曾靠 [:24] 按频率截断碰巧抑制）。真实源组
    p25 ≈ -1.7dB，-10dB 地板仅损失 ~1pp 覆盖。
    """
    merged = _merge_peaks(raw_peaks, freq, psd, drop_db=drop_db,
                          protect_radius=protect_radius)
    if floor_db is not None:
        merged = [m for m in merged if m["rel_db"] >= float(floor_db)]
    merged.sort(key=lambda d: -d["rel_db"])
    return merged[:top] if top else merged


def estimate_modulation_features(sample_path: str, candidates: list | None = None,
                                 target_bandwidth_normalized: float | None = None) -> dict[str, Any]:
    """调制识别特征：全局特征 + 按合并源组切片的带宽自适应模板距离。

    只返回测量值（含模板特征距离），不做匹配结论——由 Agent 自行判断。

    v2 改进后的切片流程（每个非目标合并组）：
    1. 用 _merge_peaks 的合并组（-20dB 谱跨度 span_idx）定切片频带，避免把
       同一源的纹波峰切成两片；
    2. Hann 软掩膜切片（替代砖墙矩形窗，抑制频谱泄漏）；
    3. 按组功率加权中心 CFO 解旋到基带（与模板口径一致）；
    4. 测切片 OBW99，各候选在**同一带宽**下生成模板后比特征距离
       —— 旧版模板固定默认带宽，OBW 维度永远失配，是调制识别 0.11 的主因。

    切片优先选非目标带（|组中心| > 2×目标带 OBW99/2，无先验时 >0.05），
    最多 3 组；切片内模板距离已排除带宽混淆，直接比波形形状。
    co_channel（与目标频带重叠）干扰切片受目标污染，距离仅供参考。
    """
    iq = _load_iq(sample_path)
    x = _combine_channels(iq)
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    flat, drift, p2a = _spectral_features(x)
    papr, duty, kurt = _time_features(x)

    cand = tuple(sorted(set(candidates))) if candidates else _CANDIDATES

    def _distances(fv, bw):
        """fv 与候选模板（均在带宽 bw 下生成）的归一化欧氏距离，返回最近的 top-3。

        只保留 top-3：10 个候选全量输出会显著膨胀上下文（3 切片 × 10 距离），
        且长工具输出会诱导模型模仿性长推理；top-3 已足够支撑"最近模板 +
        前两差距 < 0.15 时取更简单者"的判定规则。
        """
        tpl = np.stack([_template_features(c, bw) for c in cand])
        lo, hi = tpl.min(axis=0), tpl.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        fv_n = (fv - lo) / span
        tpl_n = (tpl - lo) / span
        dist = np.sqrt(((tpl_n - fv_n) ** 2).sum(axis=1))
        rows = [{"template": cand[i],
                 "feature_distance": round(float(dist[i]), 2)} for i in range(len(cand))]
        rows.sort(key=lambda r: r["feature_distance"])
        return rows[:3]

    global_dist = _distances(_feature_vector(x), measure_obw99(x))

    # 按合并组切片（与 analyze_spectrum 同口径；谱谷分裂带目标保护区 + 强度地板，
    # 先验缺失时保守用 0.05）
    raw_peaks = _find_peaks(psd, freq, rel_thresh_db=-14.0, min_sep=1.0 / 128.0)
    protect = (float(target_bandwidth_normalized) / 2.0
               if target_bandwidth_normalized else 0.05)
    merged = _source_groups(raw_peaks, freq, psd, drop_db=20.0,
                            protect_radius=protect)
    obw99 = measure_obw99(x)
    target_half = max(obw99 / 2.0, 0.05)     # 目标带半径（无先验时的保守估计）
    # 非目标带优先（干扰更可能在目标带外），其次按强度，最多 3 组
    merged_sorted = sorted(merged, key=lambda m: (abs(m["freq"]) <= target_half, -m["rel_db"]))

    per_peak = []
    for m in merged_sorted:
        if len(per_peak) >= 3:
            break
        i_lo, i_hi = m["span_idx"]
        x_bb, bw_slice = _band_slice(x, freq, psd, i_lo, i_hi)
        if x_bb is None:
            continue
        per_peak.append({
            "peak_freq": round(float(m["freq"]), 4),
            "slice_obw99": round(float(bw_slice), 3),
            "in_target_band": bool(abs(m["freq"]) <= target_half),
            "template_distances": _distances(_feature_vector(x_bb), bw_slice),
        })

    return {
        "spectral_flatness": round(float(flat), 3),
        "time_frequency_drift_std": round(float(drift), 4),
        "peak_to_average_ratio_db": round(float(p2a), 1),
        "amplitude_kurtosis": round(float(kurt), 2),
        "papr_db": round(float(papr), 1),
        "pulse_duty_cycle": round(float(duty), 3),
        "obw99_normalized": round(float(obw99), 3),
        "candidates": list(cand),
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


def analyze_spectrum(sample_path: str, target_modulation: str = "unknown",
                     target_bandwidth_normalized: float | None = None) -> dict[str, Any]:
    """多天线频谱分析：合并纹波后的源峰检测 + 每源占用带宽 + 类别判定所需测量。

    只返回结构化测量值，无结论性文本。
    - sources_candidates: 每源一行 {freq, bandwidth, ratio_to_target_bw, power_ratio_approx_db}
      —— ratio 与功率近似由工具算好，模型直接读区间判定类别（co_channel/blocking/adjacent/none）
    - peaks_normalized / per_peak_bandwidth / peaks_excluding_center：兼容字段
    """
    iq = _load_iq(sample_path)
    x = _combine_channels(iq)
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))

    raw_peaks = _find_peaks(psd, freq, rel_thresh_db=-14.0, min_sep=1.0 / 128.0)
    obw99 = measure_obw99(x)
    tgt_bw = None if target_bandwidth_normalized is None else max(float(target_bandwidth_normalized), 1e-9)
    # 谱谷分裂的目标保护区 + 强度地板（公共管线）
    protect = (tgt_bw / 2.0) if tgt_bw else 0.05
    merged = _source_groups(raw_peaks, freq, psd, protect_radius=protect)

    # 每源一行的统一结构：freq / 带宽 / ratio（类别判定核心）/ 相对主峰功率近似
    # 功率参考 = 目标带内(|f|<=tgt_bw/2)的 PSD 积分功率，直接从频谱测量、
    # 不依赖分组/地板。积分（而非峰值）才与 GT 的总功率比语义一致 —— 宽带
    # 干扰的功率摊在宽频带上，峰值密度会系统性低估总功率。
    # 此前用全局主峰作参考：blocking 干扰自身恒测得 0dB、永远够不到规则的
    # >=10dB 线 —— 500 样本基线中 128 个 blocking 匹配对命中 0 的根因。
    zone = max(tgt_bw / 2.0, 0.02) if tgt_bw else 0.05
    iz = np.abs(freq) <= zone
    ref_power = float(np.sum(psd[iz])) if iz.any() else float(np.max(psd))
    ref_power = max(ref_power, 1e-30)

    sources_candidates = []
    for m in merged:
        ratio = abs(m["freq"]) / tgt_bw if tgt_bw else None
        i_lo, i_hi = m["span_idx"]
        band_power = float(np.sum(psd[max(i_lo, 0):min(i_hi, n - 1) + 1]))
        # 标定：-20dB 轮廓积分系统性低估总功率（300 样本审计中位 -4.3dB，
        # 轮廓裁掉裙边所致）。+4.5dB 使测量与 GT 总功率比语义对齐。
        power_rel_target = float(
            10.0 * np.log10(max(band_power, 1e-30) / ref_power)) + 4.5
        # 类别边界提示：ratio/功率落在判定阈值附近时标记，提醒模型该候选的
        # 类别对测量噪声敏感（全量诊断：σf=0.02 下规则命中率上限仅 0.89，
        # 混淆集中在 0.5 / 2.0 / 10dB 边界）。边界半宽随目标带宽缩放
        # （σ_ratio ≈ σf / tgt_bw）。
        near_boundary = False
        if ratio is not None:
            dz = max(0.08, 1.5 * 0.02 / tgt_bw)
            near_boundary = (abs(ratio - 0.5) < dz or abs(ratio - 2.0) < 4 * dz
                             or abs(power_rel_target - 10.0) < 3.0)
        sources_candidates.append({
            "freq": round(m["freq"], 4),
            "bandwidth": round(m["bandwidth"], 4),
            "ratio_to_target_bw": round(float(ratio), 3) if ratio is not None else None,
            "power_ratio_approx_db": round(power_rel_target, 1),   # 相对目标功率
            "near_category_boundary": bool(near_boundary),
        })

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
        "target_bandwidth_normalized": round(float(tgt_bw), 4) if tgt_bw else None,
        "sources_candidates": sources_candidates,
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


def _music_cross_order_peaks(iq):
    """MUSIC 2/3 阶空间谱峰提取（estimate_doa 与源数决策树共用）。

    返回 (stable_peaks_deg, peaks_new_at_order3_deg)：
    - stable: 2 阶与 3 阶都出现（容差 5°）的角度 —— 真实源跨阶稳定；
    - new3:   仅 3 阶出现的角度 —— 可能是第 3 源或过分辨伪峰。
    """
    from em_signal_simulator.visualization import music_spectrum
    d2, s2 = music_spectrum(iq, num_sources=2)
    d3, s3 = music_spectrum(iq, num_sources=3)
    p2 = _extract_doa_peaks(d2, s2, max_n=3)
    p3 = _extract_doa_peaks(d3, s3, max_n=3)
    stable = [a for a in p2 if any(abs(a - b) <= 5.0 for b in p3)]
    new3 = [b for b in p3 if not any(abs(a - b) <= 5.0 for a in p2)]
    return stable, new3


def estimate_num_sources(sample_path: str, max_sources: int = 3,
                         target_bandwidth_normalized: float | None = None) -> dict[str, Any]:
    """干扰源数量：MDL 为默认建议，工具端预应用决策树给出 final_suggestion。

    target_bandwidth_normalized: 目标 OBW99 先验（评估器自动注入）。用于谱谷
    分裂的目标保护区 —— 防止强宽带目标被自己的内部谱零点劈成多组碎片，
    污染 merged_peak_count 与候选列表（ds_run10 干净样本幽灵干扰的根因）。

    - num_sources_estimate: MDL 原始值（多天线空间协方差特征值）。强干扰
      （blocking，特征值 λ2/λ1 大）下可能低估；低 SNR 弱源/宽带纹波下可能高估。
    - final_suggestion: 在 MDL 上预应用决策树修正后的建议值（Agent 的首选依据）：
      1) 高估修正: merged_peak_count < mdl_estimate 且跨阶稳定峰数 < merged
         （宽带纹波把噪声特征值顶成信号）→ 用 merged_peak_count；
      2) 低估修正: merged_peak_count > mdl_estimate 且 3 阶新峰非空
         （强干扰压低 MDL）→ 用 merged_peak_count。
      决策树需要跨工具证据（MUSIC 跨阶峰），由工具代做可避免模型逐条验证条件时
      陷入元推理循环（ds_run8 中 8/50 样本因此未产出 JSON）。
    - mdl_estimate / spectral_peak_count / merged_peak_count / consistent /
      confidence: 支撑证据与一致性，供 Agent 覆盖 final_suggestion 时参考。
    """
    iq = _load_iq(sample_path)
    n = iq.shape[-1]

    # 1) 显著谱峰（-6dB 相对主峰、最小间距 1/32，多源敏感）
    x = _combine_channels(iq)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    n_peak = min(len(_find_peaks(psd, freq, rel_thresh_db=-6.0, min_sep=1.0 / 32.0)),
                 max_sources)

    # 2) 纹波合并峰计数（与 analyze_spectrum 同口径，含目标保护区）
    raw_all = _find_peaks(psd, freq, rel_thresh_db=-14.0, min_sep=1.0 / 128.0)
    protect = (float(target_bandwidth_normalized) / 2.0
               if target_bandwidth_normalized else 0.05)
    n_merged = min(len(_source_groups(raw_all, freq, psd, protect_radius=protect,
                                      top=None)), max_sources)

    # 3) MDL（多天线空间协方差）
    n_mdl = None
    if iq.ndim == 2 and iq.shape[1] >= iq.shape[0]:
        r = (iq @ iq.conj().T) / iq.shape[1]
        evals = np.linalg.eigvalsh(r)[::-1]
        n_mdl = int(np.clip(_mdl_num_sources(evals, iq.shape[1]), 1, max_sources))

    # 默认建议：MDL 为主（离线验证 37/50，最可靠单通道）；无 MDL 时用合并峰
    if n_mdl is None:
        estimate = max(1, min(int(n_merged), max_sources))
    else:
        estimate = n_mdl

    # 4) 决策树预计算（与评估 prompt 原两分支规则一致）：需要 MUSIC 跨阶证据
    stable, new3 = [], []
    if iq.ndim == 2 and iq.shape[0] >= 3:
        try:
            stable, new3 = _music_cross_order_peaks(iq)
        except Exception:
            stable, new3 = [], []

    suggestion = int(estimate)
    applied = None
    # 可信组判定（用于低估修正）：强度可信(rel_db>=-6dB) 或 空间相干可信。
    # 噪声包与弱真实源在功率谱上重叠（-6~-10dB），但真实源的带限切片在阵列上
    # 秩 1 相干（全量校准：真实组相干比 p10=7.2，噪声包 p75=3.6），以
    # lambda1/mean(rest) >= 4 为第二判据可同时保住弱源召回并剔除噪声包
    # （ds_run11 幽灵干扰 / 漏报弱源的共同根因）。
    groups_all = _source_groups(raw_all, freq, psd, protect_radius=protect, top=None)
    n_credible = 0
    for g_ in groups_all:
        strong = g_["rel_db"] >= -6.0
        coh = (_band_coherence(iq, freq, g_["span_idx"][0], g_["span_idx"][1])
               if not strong else float("inf"))
        if strong or coh >= 4.0:
            n_credible += 1
            if n_credible >= max_sources:
                break
    if n_mdl is not None:
        if n_merged < n_mdl and len(stable) < n_merged:
            suggestion, applied = int(n_merged), "overcount_corrected"
        elif n_credible > n_mdl:
            # 可信组（强或空间相干）多于 MDL 判定 -> 低估。不再要求 MUSIC 3 阶
            # 新峰佐证：显著度过滤会同时砍掉弱真源的角度证据；相干检验已提供
            # 独立于角度的确证。
            suggestion, applied = int(n_credible), "undercount_corrected"

    consistent = (n_mdl is not None and n_mdl == n_peak and n_peak == n_merged)
    if consistent:
        confidence = 0.9
    elif n_mdl is not None and n_mdl == n_merged:   # MDL 与合并峰一致（谱峰虚高→宽带纹波）
        confidence = 0.75
    elif n_mdl is None or n_peak == n_merged:
        confidence = 0.65
    else:
        confidence = 0.5                            # 完全分歧，按决策树裁决

    return {
        "num_sources_estimate": int(estimate),      # MDL 原始值
        "final_suggestion": int(suggestion),        # 决策树修正后的建议值（首选）
        "decision_tree_applied": applied,           # None / overcount_corrected / undercount_corrected
        "mdl_estimate": n_mdl,
        "spectral_peak_count": int(n_peak),
        "merged_peak_count": int(n_merged),
        "stable_peaks_deg": [round(float(a), 1) for a in stable],
        "peaks_new_at_order3_deg": [round(float(a), 1) for a in new3],
        "consistent": bool(consistent),
        "confidence": round(float(confidence), 2),
    }


def _extract_doa_peaks(doas, spec, max_n, prominence_db=25.0):
    """从 MUSIC 空间谱提取显著峰。

    两道过滤：
    - -30dB 相对主峰 + 最小间距 12°（原有）；
    - 显著度：只保留与全局最高峰差 <= prominence_db 的峰。真源的空间谱峰远高
      于噪声过分辨产生的浅峰 —— 无此过滤时，单源场景的 3 阶谱也会凑出 2 个
      噪声角，喂给源数决策树的 undercount 分支造成误修正（ds_run11 干净样本
      幽灵干扰根因：MDL 正确报 1，被 merged 虚高 + 假新峰联合覆盖成 3）。
    返回角度列表（升序）。
    """
    spec = np.asarray(spec, dtype=float)
    doas = np.asarray(doas, dtype=float)
    mx = float(spec.max()) if spec.size else 0.0
    if mx <= 0:
        return []
    rel = spec / mx
    mask = rel > 1e-3  # -30dB
    idx = np.where(mask)[0]
    peaks_idx = []
    for i in idx:
        left = rel[i - 1] <= rel[i] if i > 0 else True
        right = rel[i + 1] <= rel[i] if i < len(rel) - 1 else True
        if left and right:
            peaks_idx.append(i)
    # 按幅度排序 -> 最小间距去重 -> 显著度地板
    peaks_idx.sort(key=lambda i: rel[i], reverse=True)
    chosen = []
    for i in peaks_idx:
        if all(abs(doas[i] - doas[j]) >= 12.0 for j in chosen):
            chosen.append(i)
        if len(chosen) >= max_n:
            break
    floor = 10.0 ** (-float(prominence_db) / 10.0)
    chosen = [i for i in chosen if rel[i] >= floor]
    chosen.sort(key=lambda i: doas[i])
    return [float(doas[i]) for i in chosen]


def estimate_doa(sample_path: str, num_sources: int = 1) -> dict[str, Any]:
    """MUSIC 空间谱 DOA 估计（4 元 ULA）。

    返回请求阶数的峰值 DOA（doa_estimates_deg）；同时固定计算 num_sources=2 与 3
    两阶空间谱，给出跨阶测量（忠实数据，供源数裁决用）：
    - stable_peaks_deg: 在 2 阶与 3 阶都出现（容差 5°）的角度 —— 真实源跨阶稳定；
    - peaks_new_at_order3_deg: 仅在 3 阶出现、2 阶没有的角度 —— MUSIC 过分辨时
      高阶常出现伪峰，该字段用于提示"可能存在第 3 个源"（需与谱证据交叉验证）。
    """
    iq = _load_iq(sample_path)
    if iq.ndim != 2 or iq.shape[0] < 2:
        return {"error": "DOA estimation requires multi-antenna IQ (num_antennas >= 2)"}
    m = iq.shape[0]
    ns = int(np.clip(num_sources, 1, m - 1))
    from em_signal_simulator.visualization import music_spectrum
    doas, spec = music_spectrum(iq, num_sources=ns)
    peaks = _extract_doa_peaks(doas, spec, max_n=ns)

    stable, new3 = [], []
    if m >= 3:
        try:
            stable, new3 = _music_cross_order_peaks(iq)
        except Exception:
            stable, new3 = [], []

    return {
        "doa_estimates_deg": [round(float(a), 1) for a in peaks],
        "num_sources_used": ns,
        "stable_peaks_deg": [round(float(a), 1) for a in stable],
        "peaks_new_at_order3_deg": [round(float(a), 1) for a in new3],
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
    # 削顶比例：|x| 进入峰值 90% 邻域的样本占比。LNA 饱和（blocking 的物理
    # 本质，Rapp 压缩）会把波形顶部压平 —— 全量审计：blocking 样本中位 0.027，
    # 干净/无阻塞多源样本 0.008/0.011，是 blocking 的独立物理证据。
    amp = np.abs(x)
    clipping = float(np.mean(amp >= 0.9 * amp.max())) if len(amp) else 0.0
    return {
        "papr_db": round(float(papr_db), 1),
        "pulse_duty_cycle": round(float(duty), 3),
        "pulse_edge_count": int(edges),
        "clipping_fraction": round(clipping, 4),
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
            "description": "Multi-antenna spectrum analysis: merged source peaks with per-source bandwidth, ratio to target bandwidth, and approximate power ratio (for interference category rules).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "target_modulation": {"type": "string",
                                          "description": "Expected target modulation type."},
                    "target_bandwidth_normalized": {"type": "number",
                                                    "description": "Target occupied bandwidth (OBW99, normalized), from the receiver prior."},
                },
                "required": ["sample_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_num_sources",
            "description": ("Estimate the number of active sources (total, INCLUDING the target): "
                            "final_suggestion is the preferred count (MDL after an automatic "
                            "over/under-count correction using spectral and cross-order MUSIC "
                            "evidence; decision_tree_applied tells whether it was corrected); "
                            "num_sources_estimate / mdl_estimate / spectral_peak_count / "
                            "merged_peak_count are supporting evidence. Do not re-derive the "
                            "correction yourself."),
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
            "description": ("MUSIC spatial-spectrum DOA estimation for the ULA. Returns doa_estimates_deg "
                            "for the requested num_sources, plus stable_peaks_deg (angles present in BOTH "
                            "the order-2 and order-3 runs, within 5 deg — real sources are stable across "
                            "MUSIC order) and peaks_new_at_order3_deg (angles seen only at order 3 — "
                            "possible extra source or over-resolution artifact)."),
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
            "description": ("Modulation-identification features: spectral flatness, time-frequency drift, "
                            "peak-to-average ratio, amplitude kurtosis, PAPR, duty cycle, OBW99, and "
                            "per-source feature distances against candidate modulation/jamming types. "
                            "Each non-target source is isolated by a Hann-windowed spectral slice "
                            "(de-rotated to baseband), and every candidate template is generated at the "
                            "SAME bandwidth as the slice before distance comparison, so distances reflect "
                            "waveform shape rather than bandwidth mismatch. Returns measurements only."),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_path": {"type": "string"},
                    "candidates": {"type": "array", "items": {"type": "string"},
                                   "description": "Optional candidate list override; default is the dataset interferer pool."},
                },
                "required": ["sample_path"],
            },
        },
    },
]
