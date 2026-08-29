"""空间维度候选生成（MVDR 频率×角度联合分析）—— v6 候选层实验实现。

背景：analyze_spectrum 用 _combine_channels 把 4 路天线平均功率合成单路再
做一维谱峰检测，空间信息在第一步即丢失。而数据集生成约束
（_correct_separation: Δfreq≥0.15 或 ΔDOA≥15°）保证了：频谱上熔在一起的源，
在角度上必然可分。本模块在 (freq, angle) 二维上做 MVDR/Capon 波束扫描，
从谱脊中提取候选源，供与旧路径离线 A/B 对比（diag_spatial_ab.py）。

约定（与工具链既有实现保持一致）：
- 导向矢量 a(θ)_k = exp(j·2π·(d/λ)·sinθ·k)，d/λ = 0.5（同 visualization.music_spectrum）；
- 候选字段与 analyze_spectrum 同 schema，新增 angle_deg（只增不改）；
- 功率语义：候选在自己角度的 MVDR 波束内做 -10dB 轮廓积分，参考功率为
  0° 波束（目标恒在 0°）在目标带内的积分。标定常数 power_cal_db 由离线
  A/B 从 GT 重推（P2），默认 0（未标定）。

本文件不修改 tools_v2.py 的任何既有函数（P1 约定）；验收通过后再以
use_spatial 开关接入。
"""
from __future__ import annotations

import numpy as np

import tools_v2

DOA_RANGE = (-60.0, 60.0)
SEG_LEN = 256


def _steering(angles_deg, m):
    """ULA d=λ/2 导向矢量矩阵 (n_ang, M)，与 music_spectrum 同约定。"""
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    idx = np.arange(m)
    return np.exp(1j * 2.0 * np.pi * 0.5
                  * np.sin(theta[:, None]) * idx[None, :])


def _segment_covariance(iq, seg_len=SEG_LEN):
    """分段 STFT + 每频点空间协方差 R(f)。

    返回 (freqs (nf,), R (nf, M, M))。hann 窗，50% 重叠，多段平均。
    """
    m, n = iq.shape
    if n < seg_len:
        seg_len = n
    hop = max(seg_len // 2, 1)
    win = np.hanning(seg_len)
    starts = list(range(0, n - seg_len + 1, hop)) or [0]
    segs = np.stack([iq[:, s:s + seg_len] * win for s in starts])  # (S, M, L)
    X = np.fft.fftshift(np.fft.fft(segs, axis=-1), axes=-1)        # (S, M, nf)
    freqs = np.fft.fftshift(np.fft.fftfreq(seg_len, d=1.0))
    Xf = np.swapaxes(X, 0, 1)                                      # (M, S, nf)
    # R(f) = mean_s x_f x_f^H
    R = np.einsum("msf,nsf->fmn", Xf, Xf.conj()) / len(starts)
    return freqs, R


def _full_width(psd, i_p, drop_db=10.0, lo=0.01, hi=0.7):
    """峰 i_p 处 -drop_db 轮廓的完整宽度（含 skirt，与 GT 总功率语义配套）。"""
    n = len(psd)
    thr = float(psd[i_p]) * 10.0 ** (-float(drop_db) / 10.0)
    i_lo = i_p
    while i_lo > 0 and psd[i_lo] >= thr:
        i_lo -= 1
    i_hi = i_p
    while i_hi < n - 1 and psd[i_hi] >= thr:
        i_hi += 1
    return i_lo, i_hi


def spatial_candidates(sample_path: str, target_modulation: str = "unknown",
                       target_bandwidth_normalized: float | None = None,
                       angle_step_deg: float = 2.0, diag_load: float = 0.05,
                       rel_floor_db: float = -12.0, min_sep: float = 1.0 / 64.0,
                       max_candidates: int = 6,
                       power_cal_db: float = 0.0) -> dict:
    """MVDR 频率×角度联合候选生成。

    返回与 analyze_spectrum 兼容的结构（sources_candidates 同 schema，
    每行多一个 angle_deg），外加 angles_kept 等调试字段。
    """
    iq = tools_v2._load_iq(sample_path)
    if iq.ndim != 2 or iq.shape[0] < 2:
        return {"sources_candidates": [], "error": "requires >=2 antennas"}
    m = iq.shape[0]
    tgt_bw = (None if target_bandwidth_normalized is None
              else max(float(target_bandwidth_normalized), 1e-9))

    freqs, R = _segment_covariance(iq)
    # 对角加载：低 SNR / 段数少时稳定求逆
    load = diag_load * np.trace(R, axis1=1, axis2=2).real / m
    Rinv = np.linalg.inv(R + load[:, None, None] * np.eye(m)[None])

    # 角度网格 + Capon 伪谱 P(f,θ) = 1 / (a^H Rinv a)
    angles = np.arange(DOA_RANGE[0], DOA_RANGE[1] + 1e-9, angle_step_deg)
    A = _steering(angles, m)                                   # (A_, M)
    V = np.einsum("fij,aj->fai", Rinv, A)                      # (nf, A_, M)
    denom = np.einsum("fai,ai->fa", V, A.conj()).real
    P = 1.0 / np.clip(denom, 1e-12, None)                      # (nf, A_)

    # 角度选择：max_f P(f,θ)（窄带源只占少数频点，max 比 mean 灵敏），
    # 相对最强角 -15dB、最小角距 10°；0°（目标）强制保留作参考波束
    prof = 10.0 * np.log10(P.max(axis=0) + 1e-30)
    prof_rel = prof - prof.max()
    idx = np.where(prof_rel > -15.0)[0]
    peaks = []
    for i in idx:
        left = prof[i - 1] <= prof[i] if i > 0 else True
        right = prof[i + 1] <= prof[i] if i < len(prof) - 1 else True
        if left and right:
            peaks.append(i)
    peaks.sort(key=lambda i: prof[i], reverse=True)
    kept: list[int] = []
    for i in peaks:
        if all(abs(angles[i] - angles[j]) >= 10.0 for j in kept):
            kept.append(i)
    kept.sort(key=lambda i: prof[i], reverse=True)
    kept = kept[:max_candidates]
    i_center = int(np.argmin(np.abs(angles)))   # 0° 目标参考波束
    if i_center not in kept:
        kept.append(i_center)

    # 参考功率：0° 波束在目标带内的积分
    zone = max(tgt_bw / 2.0, 0.02) if tgt_bw else 0.05
    iz = np.abs(freqs) <= zone
    ref = float(P[iz, i_center].sum()) if iz.any() else float(P[:, i_center].max())
    ref = max(ref, 1e-30)

    # 每波束内按"连续超门限区域"分组提取：宽带源整块一个候选（防止纹波
    # 被逐个当成峰）。频点取 -10dB 轮廓中点（对平台内 argmax 漂移稳健），
    # 同波束内频差 < FREQ_MIN_SEP 的候选必为同一源（生成约束保证同角度
    # 源对 Δfreq≥0.15），按功率贪心合并。
    gmax = float(P.max())
    gthr = gmax * 10.0 ** (rel_floor_db / 10.0)
    raw = []
    for k in kept:
        beam = P[:, k]
        above = beam > gthr
        i = 0
        nb = len(beam)
        runs = []
        while i < nb:
            if not above[i]:
                i += 1
                continue
            j = i
            while j + 1 < nb and above[j + 1]:
                j += 1
            if j - i + 1 >= 2:
                runs.append((i, j))
            i = j + 1
        per_beam = []
        for (i_lo, i_hi) in runs:
            i_p = i_lo + int(np.argmax(beam[i_lo:i_hi + 1]))
            c_lo, c_hi = _full_width(beam, i_p)
            f_center = float(0.5 * (freqs[c_lo] + freqs[c_hi]))
            band_power = float(beam[c_lo:c_hi + 1].sum())
            bw = float(freqs[c_hi] - freqs[c_lo])
            per_beam.append({"f": f_center, "bw": bw, "pw": band_power,
                             "i_lo": c_lo, "i_hi": c_hi})
        per_beam.sort(key=lambda c: c["pw"], reverse=True)
        merged_beam = []
        for c in per_beam:
            if any(abs(c["f"] - d["f"]) < 0.15 for d in merged_beam):
                continue
            merged_beam.append(c)
        for c in merged_beam:
            f_p, bw, band_power = c["f"], c["bw"], c["pw"]
            ratio = abs(f_p) / tgt_bw if tgt_bw else None
            power_rel = 10.0 * np.log10(max(band_power, 1e-30) / ref) + power_cal_db
            near_boundary = False
            if ratio is not None:
                dz = max(0.08, 1.5 * 0.02 / tgt_bw)
                near_boundary = (abs(ratio - 0.5) < dz or abs(ratio - 2.0) < 4 * dz
                                 or abs(power_rel - 10.0) < 3.0)
            raw.append({
                "freq": round(f_p, 4),
                "bandwidth": round(bw, 4),
                "ratio_to_target_bw": round(float(ratio), 3) if ratio is not None else None,
                "power_ratio_approx_db": round(float(power_rel), 1),
                "near_category_boundary": bool(near_boundary),
                "angle_deg": round(float(angles[k]), 1),
                "_pk": band_power,
            })

    # 跨波束去重：同源泄漏（频差 <0.02 且角差 <5°）保留强者
    raw.sort(key=lambda c: c["_pk"], reverse=True)
    cands = []
    for c in raw:
        if any(abs(c["freq"] - d["freq"]) < 0.02
               and abs(c["angle_deg"] - d["angle_deg"]) < 5.0 for d in cands):
            continue
        cands.append(c)
    cands = cands[:max_candidates]
    for c in cands:
        c.pop("_pk")

    return {
        "num_samples": int(iq.shape[1]),
        "num_antennas": int(m),
        "target_bandwidth_normalized": round(float(tgt_bw), 4) if tgt_bw else None,
        "sources_candidates": cands,
        "angles_kept_deg": [round(float(angles[k]), 1) for k in kept],
        "method": "mvdr_freq_angle",
    }
