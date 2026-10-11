# -*- coding: utf-8 -*-
# B4-b：cond=0.473 的损失分解 —— 工具的"提取后波形"到底坏在哪一步？
#
# 已知：
#   * CNN 自测 gate_clean = 0.860
#   * GT 位置全源 LS 提取（gate_gt_slice）= 0.738   ← 提取正确性上限
#   * 工具自己的聚类+bfw（gate_tool_view cond）= 0.473
#   * 判对/判错的几何特征（perp/triads/defl/可见站数）**无区分度** → 不是定位问题
#
# 与 gate_gt_slice 逐行对比，工具路径有三处口径差异（见 oracle16.py:90-186）：
#   (a) 导向矩阵 A 的**列不完整**：只放 {k} ∪ top-1 伙伴（按 perp），
#       GT 路径放 station_a 全部可见真源 → 未建模源原样漏进提取波形。
#   (b) 频带**brick-wall 掩蔽**到组频带（~858kHz）；GT 路径 `X = resid` 全带不切。
#       GT 路径自己的实验：二次带限曾把 GT-slice 从 0.71 打到 0.11
#       （训练分布通带外恒有噪声，通带外为 0 的切片 CNN 从未见过）。
#   (c) 变频用**测量到的 band_hz 中心** `cand_c`；GT 路径用**真值** `freq_offset_hz`。
#       共道场景下同一 858kHz 带里挤 2~3 个源，带中心对谁都不是自己的载频。
#   (d) 导向矢量：工具**水平化**（z 置 0）；GT 路径用 3D 单位向量。
#
# 本脚本对同一批认领对，逐项打开/关闭这些差异，量出各自的 pp 贡献。
import json
import os
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from collections import Counter
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rad_cls"))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402
import oracle16                     # noqa: E402
from data_gen16 import CLS_IDX      # noqa: E402

ROOT = os.path.dirname(HERE)
D2 = os.path.join(ROOT, "dataset_d2")
OUT = os.path.join(HERE, "_diag_d2_extract.json")
MATCH_M = 50.0
C = 299792458.0
VARIANTS = ["v0_file", "v1_nomask", "v2_trueoff", "v3_gtA3d", "v4_gtAhor",
            "v5_solo", "v6_mask_trueoff", "v7_partner_trueoff", "v8_selfpeak"]


def vis_gts(truth, full, allowed, only_station=None):
    vis = np.zeros(len(truth["interferers"]), int)
    per_st = [{} for _ in truth["interferers"]]
    for x in full["stations"]:
        if x["receiver_id"] not in allowed:
            continue
        nz = 10 * np.log10(x["receiver"]["expected_output_noise_w"]) + 30
        for i_g in range(len(truth["interferers"])):
            eid = "interferer_%d" % (i_g + 1)
            src_x = next((y for y in x["sources"] if y["emitter_id"] == eid), None)
            if src_x is None:
                continue
            pr = 10 * np.log10(float(np.mean(src_x["mean_port_power_w"])) + 1e-30) + 30
            per_st[i_g][x["receiver_id"]] = pr - nz
            if pr - nz >= 3.0:
                vis[i_g] += 1
    out = [(g, int(vis[i_g]), per_st[i_g]) for i_g, g in enumerate(truth["interferers"])
           if vis[i_g] >= 1 and g["kind"] in CLS_IDX]
    if only_station is not None:
        out = [(g, n, ps) for (g, n, ps) in out if ps.get(only_station, -999) >= 3.0]
    return out


def steering(rec, poses, horizontal, fc):
    off = np.asarray(rec["element_offsets_world_m"], float)
    rp3 = np.asarray(rec["receiver_position_m"], float)
    A = np.zeros((8, len(poses)), complex)
    for k, sp in enumerate(poses):
        sp3 = np.asarray(sp[:3], float) if len(sp) >= 3 else np.array([sp[0], sp[1], rp3[2]])
        d = sp3 - rp3
        if horizontal:
            d = np.array([d[0], d[1], 0.0])
        u = d / (np.linalg.norm(d) + 1e-12)
        A[:, k] = np.exp(2j * np.pi * fc / C * (off @ u))
    return A


def extract(resid, fs, fc, rec, poses, horizontal, lo, hi, mask_on, shift_hz, row=0):
    n = resid.shape[1]
    if mask_on:
        f = np.fft.fftfreq(n, 1 / fs)
        m = (f >= lo) & (f <= hi)
        X = np.fft.ifft(np.fft.fft(resid, axis=1) * m, axis=1) if m.any() else resid
    else:
        X = resid
    A = steering(rec, poses, horizontal, fc)
    if not np.isfinite(np.linalg.cond(A)) or np.linalg.cond(A) > 1e6:
        return None
    S = np.linalg.pinv(A) @ X
    w = S[row].astype(np.complex128)
    w = w * np.exp(-1j * 2 * np.pi * shift_hz * np.arange(n) / fs)
    w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)
    return w.astype(np.complex64)
def self_peak_hz(w, fs, half_hz=400e3):
    """从已提取波形自己的频谱里量峰位置（无真值可用，纯自测）。"""
    n = w.shape[0]
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    P = np.abs(np.fft.fftshift(np.fft.fft(w)))
    m = (f >= -half_hz) & (f <= half_hz)
    return float(f[m][int(np.argmax(P[m]))])


def main(n_samples=50):
    model, _ = oracle16.load_model()
    idx = json.loads(open(os.path.join(D2, "index.json"), encoding="utf-8").read())
    acc = {v: [0, 0] for v in VARIANTS}
    v0_mismatch = 0
    conf_best = Counter()
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = os.path.join(D2, sid, "capture")
        obs = json.loads(open(os.path.join(cap, "public", "observations.json"), encoding="utf-8").read())
        private = Path(cap) / "private"
        ns = obs["receivers"][0]["shape"][1]
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ct.set_context(ct.ToolContext(capture_dir=os.path.join(cap, "public"),
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
        vg = vis_gts(truth, full, allowed)
        vg_a = vis_gts(truth, full, allowed, only_station="station_a")
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cand = clu.get("candidates") or []
        if not cand:
            continue
        resid = ct._station_iq_raw("station_a", on_residual=True).astype(np.complex128)
        # 复刻 extract_sets（自己 + 频带重叠且导向去相关的 perp 最低伙伴）
        def st_corr(p1, p2):
            cs = []
            for rr in obs["receivers"]:
                A1 = steering(rr, [p1], True, fc)
                A2 = steering(rr, [p2], True, fc)
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
        # 认领
        pair = []
        for k, sp in enumerate(cand):
            p2 = np.asarray(sp["position_xy_m"])
            for i_g, (g, _n, _ps) in enumerate(vg):
                d = float(np.linalg.norm(p2 - np.asarray(g["position_m"][:2])))
                if d < MATCH_M:
                    pair.append((d, k, i_g))
        pair.sort()
        used_k, used_g = set(), set()
        for d, k, i_g in pair:
            if k in used_k or i_g in used_g:
                continue
            used_k.add(k)
            used_g.add(i_g)
            g = vg[i_g][0]
            ms = sets[k]
            blo, bhi = cand[k]["band_hz"]
            lo_g = min(cand[m]["band_hz"][0] for m in ms)
            hi_g = max(cand[m]["band_hz"][1] for m in ms)
            cand_c = 0.5 * (blo + bhi)
            poses_cand = [cand[m]["position_xy_m"] for m in ms]
            row_k = ms.index(k)
            # v3/v4：GT 路径 —— 目标源置第一列，其余为 station_a 可见的其他真源
            poses_gt = [g["position_m"]] + [gg["position_m"] for (gg, _n, _ps) in vg_a
                                             if gg is not g]
            f0 = float(g["freq_offset_hz"])
            variants = {}
            variants["v1_nomask"] = extract(resid, fs, fc, rec, poses_cand, True, lo_g, hi_g, False, cand_c, row_k)
            variants["v2_trueoff"] = extract(resid, fs, fc, rec, poses_cand, True, lo_g, hi_g, False, f0, row_k)
            variants["v3_gtA3d"] = extract(resid, fs, fc, rec, poses_gt, False, lo_g, hi_g, False, f0, 0)
            variants["v4_gtAhor"] = extract(resid, fs, fc, rec, poses_gt, True, lo_g, hi_g, False, f0, 0)
            variants["v5_solo"] = extract(resid, fs, fc, rec, [cand[k]["position_xy_m"]], True, lo_g, hi_g, False, f0, 0)
            variants["v6_mask_trueoff"] = extract(resid, fs, fc, rec, poses_cand, True, lo_g, hi_g, True, f0, row_k)
            variants["v7_partner_trueoff"] = (extract(resid, fs, fc, rec, poses_cand, True, lo_g, hi_g, False, f0, row_k)
                                              if len(ms) > 1 else None)
            # v8：完全不用真值 —— 先按 cand_c 提取，再量提取波形自己的谱峰，按该峰二次变频
            w_pre = extract(resid, fs, fc, rec, poses_cand, True, lo_g, hi_g, False, cand_c, row_k)
            if w_pre is not None:
                pk = self_peak_hz(w_pre, fs)
                t = np.arange(w_pre.shape[0])
                w8 = w_pre.astype(np.complex128) * np.exp(-1j * 2 * np.pi * pk * t / fs)
                variants["v8_selfpeak"] = (w8 / (np.sqrt(np.mean(np.abs(w8) ** 2)) + 1e-12)).astype(np.complex64)
            else:
                variants["v8_selfpeak"] = None
            fpath = Path(D2) / "_agent_cache" / sid / ("bfw_station_a_%d.npy" % k)
            v0 = None
            if fpath.exists():
                v0 = np.load(fpath).astype(np.complex64)
            # v0 用工具自己的路径重算一次，验证复刻保真
            v0_re = extract(resid, fs, fc, rec, poses_cand, True, lo_g, hi_g, True, cand_c, row_k)
            if v0 is not None and v0_re is not None:
                r = float(np.abs(np.vdot(v0 / (np.linalg.norm(v0) + 1e-30),
                                         v0_re / (np.linalg.norm(v0_re) + 1e-30))))
                if r < 0.999:
                    v0_mismatch += 1
            variants["v0_file"] = v0
            gt_i = CLS_IDX[g["kind"]]
            for name, w in variants.items():
                if w is None:
                    continue
                pr = int(oracle16._pred(model, [w])[0])
                acc[name][1] += 1
                acc[name][0] += int(pr == gt_i)
                if name == "v3_gtA3d":
                    conf_best[(g["kind"], pr)] += 1
        print(f"  {sid} 认领 {len(used_k)}")
    json.dump({k: v for k, v in acc.items()}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("=" * 78)
    print(f"v0 复刻保真检查：{v0_mismatch} 个样本的 bfw 与本脚本重算不一致（应为 0）")
    print("变量                                cond       n")
    for v in VARIANTS:
        c, n = acc[v]
        print(f"  {v:20s} {c:4d}/{n:<4d} = {c / max(n, 1):.3f}")
    print("--- v3_gtA3d (GT 上限) 混淆 ---")
    for (gk, pi), n in conf_best.most_common(12):
        print(f"  {n:3d}  {gk:16s} → {pi}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
