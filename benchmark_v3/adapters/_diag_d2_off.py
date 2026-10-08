# -*- coding: utf-8 -*-
# D2 cond=0.473 的"每源载频"杠杆：把**可实现的无真值载频估计器**接到提取链路上，量 cond。
#
# 机理（rad_cls/data_gen16.py）：
#   * simulate_residual() 第 1 步只给 f_off = U(±20kHz)，第 4 步 Hann 带限**以 DC 为中心**
#     （edge = bw*fs/2，用 abs(freq)）→ 训练样本里源永远**贴在 DC 附近**。
#   * simulate_beamformed_train()（2026-10-02 新增的训练终段）全程**不再搬移频率**：
#     叠兄弟源 → 接收滤波 → 全源 LS 提取 → 可选的以主源为中心的砖墙/软边切片。
#   * 所以 CNN 期望的输入 = **载频归零后**的带限波形。
#   工具路径 (capture_tools.py) 的 bfw 落盘却是 `w *= exp(-1j*2π*cand_c*n/fs)`，
#   `cand_c = 0.5*(band_hz[0]+band_hz[1])`，而 band_hz 来自**旧宽带判据**的 ~858kHz
#   共道带 → 共道多源下带中心对谁都不是自己的载频。
#
# 本脚本对同一批 55 个认领对，比较：
#   a0/a1 工具 A 配 cand_c（基线 0.473）/ 真值频偏（上限 0.509）
#   a2/a3 工具 A 配**无真值**估计器：门控窄带自测 / 谱重心
#   b0/b1 GT A 配 cand_c / 真值频偏（已知上限 0.600）→ 分离"列不全"与"载频错"两个因素
#   b2/b3 GT A 配两个无真值估计器
# 目的：判定"修每源载频估计"单独能把 cond 推到多少、以及它与"导向矩阵列不全"是否可叠加。
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rad_cls"))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402
import oracle16                     # noqa: E402
import _diag_d2_extract as dx       # noqa: E402
from data_gen16 import CLS_IDX      # noqa: E402

D2 = dx.D2
MATCH_M = dx.MATCH_M
OUT = os.path.join(HERE, "_diag_d2_off.json")

VARIANTS = ["a0_base_nm", "a1_true_nm", "a2_gated_nm", "a3_cent_nm", "a4_bf_nm",
            "a5_cs_nm", "a6_cs4_nm", "a7_base_mk", "a8_true_mk", "a9_shift0_mk",
            "a10_shift0_nm", "a11_cs4cm_mk", "a12_cs4cm_nm",
            "b1_true_nm", "b7_base_mk", "b8_true_mk", "b9_shift0_mk", "b11_cs4cm_mk",
            "a13_db_cent_mk", "a14_db_zero_mk", "a15_db_true_mk",
            "b13_db_cent_mk", "b15_db_true_mk", "a16_db50_cent_mk"]


def centroid_hz(w, fs, half_hz=400e3):
    """提取波形的谱重心（-20dB 门内），无真值。"""
    n = w.shape[0]
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    P = np.abs(np.fft.fftshift(np.fft.fft(w))) ** 2
    m = (f >= -half_hz) & (f <= half_hz)
    Pf, ff = P[m], f[m]
    if Pf.size == 0 or Pf.max() <= 0:
        return None
    sel = Pf >= Pf.max() * 0.01
    if not sel.any():
        return None
    return float(np.sum(ff[sel] * Pf[sel]) / np.sum(Pf[sel]))


def gated_self_hz(w, fs):
    """拿工具自己的新峰判据（局部底 + 相干门控）在**提取波形上**找载频残差。

    单通道输入下 _band_coherence 恒为 1（无空间维），这里只用它当稳健的频率定位器。
    """
    wb = np.asarray(w, np.complex128)[None, :]
    try:
        bands = ct._residual_bands(wb, fs, fs * 0.45)
    except Exception:
        return None
    if not bands:
        return None
    return float(bands[0]["freq_offset_hz"])

def bf_peak_hz(resid, fs, fc, rec, pos, win_hz=450e3, smooth_hz=20e3):
    """方向感知载频估计（免真值）：把残差投到该候选方向的导向矢量上，
    取波束输出功率在频域最大的位置，作为该源的载频残差。

    共道多源的载频互不相同、来向也不同；单一候选的载频 = 该来向上
    波束形成器输出功率最大处的频率，故按方向分离可绕开“带中心”。
    """
    n = resid.shape[1]
    a = dx.steering(rec, [pos], True, fc)[:, 0]
    y = np.abs(np.conj(a) @ np.fft.fft(resid, axis=1)) ** 2
    f = np.fft.fftfreq(n, 1 / fs)
    k = max(1, int(round(smooth_hz / (fs / n))))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    ys = np.convolve(np.r_[y[-k:], y, y[:k]], ker, mode="same")[k:-k]
    m = np.abs(f) <= win_hz
    if not m.any():
        return None
    fm, ym = f[m], np.asarray(ys, dtype=float)[m]
    return float(fm[int(np.argmax(ym))])
def _support_center(c, f, fs, n, win_hz, smooth_hz, frac):
    """在 |f|<=win_hz 内取“方向相干度”支撑段的中心（两端中值）。"""
    k = max(1, int(round(smooth_hz / (fs / n))))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    cs = np.convolve(np.r_[c[-k:], c, c[:k]], ker, mode="same")[k:-k]
    m = np.abs(f) <= win_hz
    if not m.any():
        return None
    fm, cm = f[m], np.asarray(cs, dtype=float)[m]
    thr = frac * float(cm.max())
    idx = np.nonzero(cm >= thr)[0]
    if idx.size == 0:
        return None
    return float(0.5 * (fm[int(idx[0])] + fm[int(idx[-1])]))


def coh_sup_hz(resid, fs, fc, rec, pos, win_hz=450e3, smooth_hz=15e3, frac=0.6,
               common_mode=False):
    """方向相干度支撑中心（单站，免真值）。

    `c(f) = |a^H X(f)|^2 / (mean_m |X_m(f)|^2)` 是“该频点的能量有多少来自候选
    方向”的无量纲量（0..m）。共道多源方向不同 → 各自频段上 c 高、对方频段上
    低；取高相干段的中心作为载频，比“功率 argmax”稳（源谱是平顶 Hann 带限）。

    common_mode=True 时先减掉阵元共模分量——残差里存在跨阵元完全相同的直流
    分量，它让 c(0) 顶到度量上限（≈m），把 frac*max 支撑段锁死在 0 附近。
    """
    n = resid.shape[1]
    a = dx.steering(rec, [pos], True, fc)[:, 0]
    a = a / (np.linalg.norm(a) + 1e-12)
    X = np.fft.fft(resid, axis=1)
    if common_mode:
        X = X - X.mean(axis=0, keepdims=True)
    num = np.abs(np.conj(a) @ X) ** 2
    den = np.sum(np.abs(X) ** 2, axis=0) / resid.shape[0]
    c = np.fft.fftshift(num / (den + 1e-30))
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    return _support_center(c, f, fs, n, win_hz, smooth_hz, frac)


def coh_sup_ms(obs, pos, cache, win_hz=450e3, smooth_hz=15e3, frac=0.6,
               common_mode=False):
    """四站非相干叠加的方向相干度支撑中心（免真值）。

    跳站相位不可用（各站独立多径），但“该频点是否来自候选方向”无量纲可加；
    四站叠加等效把角度分辨力提高一代，用于把共道多源载频分开。
    """
    fc = obs["receivers"][0]["center_frequency_hz"]
    fs = obs["receivers"][0]["sample_rate_hz"]
    tot, f = None, None
    for rr in obs["receivers"]:
        key = (rr["receiver_id"], bool(common_mode))
        if key not in cache:
            try:
                r = ct._station_iq_raw(rr["receiver_id"], on_residual=True).astype(np.complex64)
            except Exception:
                cache[key] = None
                continue
            Xc = np.fft.fft(r, axis=1).astype(np.complex64)
            if common_mode:
                Xc = Xc - Xc.mean(axis=0, keepdims=True)
            den = (np.sum(np.abs(Xc) ** 2, axis=0) / r.shape[0]).astype(np.float32)
            cache[key] = (Xc, den, np.fft.fftshift(np.fft.fftfreq(r.shape[1], 1 / fs)))
        ent = cache[key]
        if ent is None:
            continue
        Xc, den, f = ent
        a = dx.steering(rr, [pos], True, fc)[:, 0]
        a = (a / (np.linalg.norm(a) + 1e-12)).astype(np.complex64)
        c = np.fft.fftshift(np.abs(np.conj(a) @ Xc) ** 2 / (den + 1e-30))
        tot = c if tot is None else tot + c
    if tot is None or f is None:
        return None
    return _support_center(tot, f, fs, len(f), win_hz, smooth_hz, frac)


def dir_band_ms(obs, pos, cache, drop_db=6.0, win_hz=1800e3, smooth_hz=20e3,
                pad_frac=0.0):
    """候选方向上的功率谱 → 以峰为起点按 drop_db 阈值向两侧连续扩张，定出该源自己的频带。

    与 coh_sup_ms 的区别：这里用的是**功率**(|a^H X|^2)而不是归一化相干度。源谱是平顶
    (带宽 130~600kHz)，峰位置在平顶上没意义，**边缘**才有意义——所以用阈值找边缘，
    而不是 argmax。返回 (lo, hi, 功率重心)。pad_frac>0 时把带向两侧各放宽这么多倍宽。
    """
    fc = obs["receivers"][0]["center_frequency_hz"]
    fs = obs["receivers"][0]["sample_rate_hz"]
    tot, f = None, None
    for rr in obs["receivers"]:
        key = (rr["receiver_id"], False)
        if key not in cache:
            try:
                r = ct._station_iq_raw(rr["receiver_id"], on_residual=True).astype(np.complex64)
            except Exception:
                cache[key] = None
                continue
            Xc = np.fft.fft(r, axis=1).astype(np.complex64)
            den = (np.sum(np.abs(Xc) ** 2, axis=0) / r.shape[0]).astype(np.float32)
            cache[key] = (Xc, den, np.fft.fftshift(np.fft.fftfreq(r.shape[1], 1 / fs)))
        ent = cache[key]
        if ent is None:
            continue
        Xc, den, f = ent
        a = dx.steering(rr, [pos], True, fc)[:, 0]
        a = (a / (np.linalg.norm(a) + 1e-12)).astype(np.complex64)
        P = np.fft.fftshift(np.abs(np.conj(a) @ Xc) ** 2)
        tot = P if tot is None else tot + P
    if tot is None or f is None:
        return None
    n = len(f)
    k = max(1, int(round(smooth_hz / (fs / n))))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    tot = np.convolve(np.r_[tot[-k:], tot, tot[:k]], ker, mode="same")[k:-k]
    m = np.abs(f) <= win_hz
    fm, pm = f[m], np.asarray(tot, float)[m]
    ipk = int(np.argmax(pm))
    thr = pm[ipk] * 10 ** (-drop_db / 10)
    lo = ipk
    while lo > 0 and pm[lo - 1] >= thr:
        lo -= 1
    hi = ipk
    while hi < len(pm) - 1 and pm[hi + 1] >= thr:
        hi += 1
    flo, fhi = float(fm[lo]), float(fm[hi])
    if pad_frac > 0:
        wpad = pad_frac * (fhi - flo)
        flo, fhi = flo - wpad, fhi + wpad
    seg = pm[lo:hi + 1]
    cent = float(np.sum(fm[lo:hi + 1] * seg) / (np.sum(seg) + 1e-30))
    return flo, fhi, cent


def _bb(spec):
    """dir_band_ms 的 (lo, hi, cent) → 掩蔽用 (lo, hi)。"""
    return (spec[0], spec[1]) if spec else None

def main(n_samples=50):
    model, _ = oracle16.load_model()
    idx = json.loads(Path(D2, "index.json").read_text(encoding="utf-8"))
    acc = {v: [0, 0] for v in VARIANTS}
    err = {v: [] for v in VARIANTS}
    n_pair = 0
    freq_rows = []

    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = os.path.join(D2, sid, "capture")
        obs = json.loads(Path(cap, "public", "observations.json").read_text(encoding="utf-8"))
        private = Path(cap) / "private"
        ns = obs["receivers"][0]["shape"][1]
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ct.set_context(ct.ToolContext(
            capture_dir=os.path.join(cap, "public"),
            cache_dir=os.path.join(D2, "_agent_cache", sid),
            allowed_stations=allowed,
            fs=obs["receivers"][0]["sample_rate_hz"],
            fc=obs["receivers"][0]["center_frequency_hz"],
            target_bw_normalized=0.25, aoi_center=aoi,
            target_reference=rca._build_target_reference(private, ns)))
        fs = obs["receivers"][0]["sample_rate_hz"]
        fc = obs["receivers"][0]["center_frequency_hz"]
        rec = next(r for r in obs["receivers"] if r["receiver_id"] == "station_a")
        truth = json.loads((private / "d1_truth.json").read_text(encoding="utf-8"))
        full = json.loads((private / "truth.json").read_text(encoding="utf-8"))
        vg = dx.vis_gts(truth, full, allowed)
        vg_a = dx.vis_gts(truth, full, allowed, only_station="station_a")
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cand = clu.get("candidates") or []
        if not cand:
            continue
        resid = ct._station_iq_raw("station_a", on_residual=True).astype(np.complex128)
        scache = {}

        def st_corr(p1, p2):
            cs = []
            for rr in obs["receivers"]:
                A1 = dx.steering(rr, [p1], True, fc)
                A2 = dx.steering(rr, [p2], True, fc)
                cs.append(abs(np.vdot(A1[:, 0], A2[:, 0])) / 8)
            return float(np.mean(cs))

        sets = []
        for k, c in enumerate(cand):
            blo, bhi = c["band_hz"]
            part = []
            for m, cm in enumerate(cand):
                if m == k:
                    continue
                mlo, mhi = cm["band_hz"]
                if not (mhi < blo or mlo > bhi) and st_corr(c["position_xy_m"], cm["position_xy_m"]) < 0.8:
                    part.append(m)
            part.sort(key=lambda m: cand[m].get("perp_rms_m", 999.0))
            sets.append(sorted(set([k] + part[:1])))

        pair = []
        for k, sp in enumerate(cand):
            p2 = np.asarray(sp["position_xy_m"])
            for i_g, (g, _n, _ps) in enumerate(vg):
                d = float(np.linalg.norm(p2 - np.asarray(g["position_m"][:2])))
                if d < MATCH_M:
                    pair.append((d, k, i_g))
        pair.sort()
        used_k, used_g = set(), set()
        for _d, k, i_g in pair:
            if k in used_k or i_g in used_g:
                continue
            used_k.add(k)
            used_g.add(i_g)
            n_pair += 1
            g = vg[i_g][0]
            gt_i = CLS_IDX[g["kind"]]
            ms = sets[k]
            lo_g = min(cand[m]["band_hz"][0] for m in ms)
            hi_g = max(cand[m]["band_hz"][1] for m in ms)
            cand_c = 0.5 * (cand[k]["band_hz"][0] + cand[k]["band_hz"][1])
            poses_cand = [cand[m]["position_xy_m"] for m in ms]
            row_k = ms.index(k)
            poses_gt = [g["position_m"]] + [gg["position_m"] for (gg, _n, _ps) in vg_a if gg is not g]
            f0 = float(g["freq_offset_hz"])

            w_base = dx.extract(resid, fs, fc, rec, poses_cand, True, lo_g, hi_g, True, cand_c, row_k)
            if w_base is None:
                continue
            gs = gated_self_hz(w_base, fs)
            off_gated = (cand_c + gs) if gs is not None else None
            cg = centroid_hz(w_base, fs)
            off_cent = (cand_c + cg) if cg is not None else None
            off_bf = bf_peak_hz(resid, fs, fc, rec, cand[k]["position_xy_m"])
            px = cand[k]["position_xy_m"]
            off_cs = coh_sup_hz(resid, fs, fc, rec, px)
            off_cs4 = coh_sup_ms(obs, px, scache)
            off_cs4cm = coh_sup_ms(obs, px, scache, common_mode=True)
            db_cand = dir_band_ms(obs, px, scache)
            db_gt = dir_band_ms(obs, g["position_m"], scache)
            db50 = dir_band_ms(obs, px, scache, pad_frac=0.5)
            bc = db_cand[2] if db_cand else None
            bg = db_gt[2] if db_gt else None
            b50 = db50[2] if db50 else None

            plans = [
                ("a0_base_nm", poses_cand, True, row_k, cand_c, False),
                ("a1_true_nm", poses_cand, True, row_k, f0, False),
                ("a2_gated_nm", poses_cand, True, row_k, off_gated, False),
                ("a3_cent_nm", poses_cand, True, row_k, off_cent, False),
                ("a4_bf_nm", poses_cand, True, row_k, off_bf, False),
                ("a5_cs_nm", poses_cand, True, row_k, off_cs, False),
                ("a6_cs4_nm", poses_cand, True, row_k, off_cs4, False),
                ("a7_base_mk", poses_cand, True, row_k, cand_c, True),
                ("a8_true_mk", poses_cand, True, row_k, f0, True),
                ("a9_shift0_mk", poses_cand, True, row_k, 0.0, True),
                ("a10_shift0_nm", poses_cand, True, row_k, 0.0, False),
                ("a11_cs4cm_mk", poses_cand, True, row_k, off_cs4cm, True),
                ("a12_cs4cm_nm", poses_cand, True, row_k, off_cs4cm, False),
                ("b1_true_nm", poses_gt, False, 0, f0, False),
                ("b7_base_mk", poses_gt, False, 0, cand_c, True),
                ("b8_true_mk", poses_gt, False, 0, f0, True),
                ("b9_shift0_mk", poses_gt, False, 0, 0.0, True),
                ("b11_cs4cm_mk", poses_gt, False, 0, off_cs4cm, True),
                ("a13_db_cent_mk", poses_cand, True, row_k, bc, True, _bb(db_cand)),
                ("a14_db_zero_mk", poses_cand, True, row_k, 0.0, True, _bb(db_cand)),
                ("a15_db_true_mk", poses_cand, True, row_k, f0, True, _bb(db_cand)),
                ("b13_db_cent_mk", poses_gt, False, 0, bg, True, _bb(db_gt)),
                ("b15_db_true_mk", poses_gt, False, 0, f0, True, _bb(db_gt)),
                ("a16_db50_cent_mk", poses_cand, True, row_k, b50, True, _bb(db50)),
            ]
            for spec in plans:
                name, poses, horz, row, shift, mask = spec[:6]
                band = spec[6] if len(spec) > 6 else None
                if shift is None:
                    continue
                blo, bhi = band if band else (lo_g, hi_g)
                w = dx.extract(resid, fs, fc, rec, poses, horz, blo, bhi, mask, shift, row)
                if w is None:
                    continue
                pr = int(oracle16._pred(model, [w])[0])
                acc[name][1] += 1
                acc[name][0] += int(pr == gt_i)
                err[name].append(abs(float(shift) - f0))
            freq_rows.append({"sid": sid, "gt": g["kind"], "f0": f0, "cand_c": cand_c,
                              "lo": lo_g, "hi": hi_g, "bf": off_bf, "cs": off_cs,
                              "cs4": off_cs4, "cs4cm": off_cs4cm,
                              "gated": off_gated, "cent": off_cent,
                              "db_lo": db_cand[0] if db_cand else None,
                              "db_hi": db_cand[1] if db_cand else None,
                              "db_c": bc, "bg": bg, "b50": b50})
        print(f"  {sid} 认领 {len(used_k)}  累计 {n_pair}")

    print("=" * 78)
    print(f"认领对 n = {n_pair}")
    print(f"{'变量':22s} {'cond':>10s} {'载频估计误差中位(kHz)':>22s}")
    for v in VARIANTS:
        c, n = acc[v]
        e = np.median(err[v]) / 1e3 if err[v] else float("nan")
        print(f"  {v:20s} {c:4d}/{n:<4d} = {c / max(n, 1):.3f}   {e:18.1f}")
    json.dump({"n": n_pair, "acc": acc,
               "err_med_khz": {v: (float(np.median(err[v])) / 1e3 if err[v] else None)
                               for v in VARIANTS},
               "freq_rows": freq_rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
