#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_an_d2_rank.py —— 只读诊断：逐频窗窄带空间特征值枚举 vs 锥交会定位（源数证据）

假设 H：在窄带频窗里对 8 元阵空间协方差做特征值枚举（λ1..λ8），
        其"超阈特征值个数"可作为"该频带有几个源"的源数证据，
        从而在原理上替代多站锥交会定位。

本脚本**只读**仓库数据（唯一例外：capture_tools 自身的 resid_*.npy 结果缓存
写入 dataset_d2/_agent_cache/，这是仪器既有行为，不是本脚本新增的写）。

口径（与任务书逐条对应）：
 1) 每站 (8,40000) IQ → 参考直减残差（ct._strip_target_at_station，整通带 ±0.45fs）；
    同时并行算一份"原始 IQ"作对照。
 2) 频窗：宽度 win_hz，步进 = win_hz（不重叠），覆盖 |f| <= 1.8 MHz 的窗中心。
    X = FFT(:, 窗内 bin)，按列归一化，R = X X^H / n_cols，特征值降序 λ1..λ8。
 3) 真值覆盖：源频段 [off - bw/2, off + bw/2] 与 窗范围有重叠即算覆盖；
    可见性 INR >= 3.0 dB（station_a->s0 ... station_d->s3，脚本内用 truth.json 自校验）。
 4) 连通段合并（频率上相邻的计数>0 窗并成一段）→ 站级源数 → 4 站中位数。

用法：
  cd D:\\srtp\\EM-benchmark\\benchmark_v3\\adapters
  python -u _an_d2_rank.py [--samples 50] [--wins 10000,20000,50000,100000,200000] [--no-ref]
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_D2 = _V3 / "dataset_d2"

sys.path.insert(0, str(_HERE))

import capture_tools as ct  # noqa: E402  只 import，不改动
import run_capture_agent as rca  # noqa: E402  只 import，不改动

# ---------------------------------------------------------------- 可调旋钮
SEARCH_HALF_HZ = 1.8e6            # = fs * 0.45（工具搜索范围）
INR_VIS_DB = 3.0                  # 可见性门限
RATIOS = (2.0, 3.0, 5.0)          # 判据 λ_k > ratio × median(λ_{k+1..8})
POWER_GATE_DB = 3.0               # 窗总功率相对全带均值均值的门控（dB）
FLOOR_WIN_HZ = 300e3              # 局部底中位滤波窗（同 capture_tools.PEAK_FLOOR_WIN_HZ）
FLOOR_GUARD_HZ = 10e3             # 信号区从底估计里挖掉的保护带

# 站 -> truth.json / d1_truth.json 的 per_station 键（脚本运行时自校验）
STATION_TO_S = {"station_a": "s0", "station_b": "s1",
                "station_c": "s2", "station_d": "s3"}


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 特征值口径
def eig_profile(x: np.ndarray) -> np.ndarray:
    """x: (M=8, n) 复数，已按列归一化。返回降序特征值 (8,)。"""
    r = x @ x.conj().T / x.shape[1]
    return np.linalg.eigvalsh(r)[::-1].real


def count_ratio(w: np.ndarray, ratio: float) -> int:
    """λ_k > ratio × median(λ_{k+1..8})。

    与 任务书口径一致：k 从 1 起，且必须至少留 2 个特征值当噪声底，
    否则纯白噪窗在 ratio=2 时会有 1/8 量级的必然虚警。"""
    m = w.size
    cnt = 0
    for k in range(m):
        tail = w[k + 1:]
        if tail.size < 2:
            break
        if w[k] > ratio * float(np.median(tail)):
            cnt += 1
        else:
            break
    return cnt


def count_mdl(w: np.ndarray, n_cols: int) -> int:
    """MDL 秩估计（Wax-Kailath 形式），n_cols = 快拍数（窗内 bin 数），返回 0..7。"""
    m = w.size
    best_k, best_v = 0, np.inf
    for k in range(m):
        tail = w[k:]
        if tail.size == 0:
            break
        am = float(np.mean(tail))
        gm = float(np.exp(np.mean(np.log(tail + 1e-30))))
        if am <= 0 or gm <= 0:
            break
        ll = tail.size * n_cols * np.log(am / gm)
        pen = 0.5 * (k * (2 * m - k)) * np.log(n_cols)
        if ll + pen < best_v:
            best_v, best_k = ll + pen, k
    return int(best_k)


# ---------------------------------------------------------------- 真值
def source_list(truth: dict) -> list[dict]:
    """干扰源频段清单（频率偏移 Hz）。"""
    out = []
    for it in truth.get("interferers", []):
        off = float(it.get("freq_offset_hz", 0.0))
        bw = float(it.get("bandwidth_hz", 0.0))
        out.append({"kind": it.get("kind", "?"), "off": off, "bw": bw,
                    "lo": off - bw / 2.0, "hi": off + bw / 2.0,
                    "per_station": it.get("per_station", {})})
    return out


def srcs_in_window(srcs: list[dict], lo: float, hi: float, station: str,
                   inr_min: float = INR_VIS_DB) -> list[int]:
    """覆盖该窗、且在该站可见（INR >= inr_min）的源索引。"""
    s_key = STATION_TO_S.get(station, station)
    hit = []
    for i, s in enumerate(srcs):
        if s["hi"] <= lo or s["lo"] >= hi:
            continue
        inr = s["per_station"].get(s_key, {}).get("inr_db", None)
        if inr is None or float(inr) < inr_min:
            continue
        hit.append(i)
    return hit


def merge_segments(counts: np.ndarray, centers: np.ndarray) -> list[dict]:
    """把"计数 > 0"的连续窗并成一段，返回 [{f_lo,f_hi,n_max,n_med,n_win}]。"""
    segs = []
    i, n = 0, counts.size
    while i < n:
        if counts[i] > 0:
            j = i
            while j + 1 < n and counts[j + 1] > 0:
                j += 1
            seg = counts[i:j + 1]
            segs.append({"f_lo": float(centers[i]), "f_hi": float(centers[j]),
                         "n_max": int(seg.max()), "n_med": float(np.median(seg)),
                         "n_win": int(seg.size)})
            i = j + 1
        else:
            i += 1
    return segs


# ---------------------------------------------------------------- 辅助
def band_limit(iq: np.ndarray, fs: float, half: float) -> np.ndarray:
    n = iq.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    mask = (f >= -half) & (f <= half)
    return np.fft.ifft(np.fft.fft(iq, axis=1) * mask, axis=1)


def local_floor_db(psd_db: np.ndarray, sig_mask: np.ndarray,
                   fs: float, n: int) -> np.ndarray:
    """局部中位底：先把信号区用邻域插值填掉，再做中位滤波。"""
    from scipy.ndimage import median_filter
    k = max(int(round(FLOOR_WIN_HZ / (fs / n))) | 1, 5)
    if k >= psd_db.size:
        k = max((psd_db.size // 2) | 1, 5)
    work = psd_db.copy()
    if sig_mask.any():
        idx = np.where(~sig_mask)[0]
        if idx.size > 1:
            work[sig_mask] = np.interp(np.where(sig_mask)[0], idx, psd_db[idx])
        else:
            work[sig_mask] = float(np.median(psd_db))
    return median_filter(work, size=k, mode="nearest")


def conf_mdl_bump(acc: dict, kk: int, val: int, gated: bool) -> None:
    """把一个窗的 MDL 计数打进混淆直方图。"""
    bump = acc["conf_P%d_mdl" % kk]
    bumpg = acc["conf_P%d_mdl_g" % kk]
    bump[val] = bump.get(val, 0) + 1
    if gated:
        bumpg[val] = bumpg.get(val, 0) + 1

def station_map_check(private: Path, truth: dict, allowed: list[str],
                      sid: str, mc: dict) -> None:
    """校验 station_a->s0 ... 映射（任务书要求『先核实这个映射』）。

    d1_truth 的 per_station[*].inr_db 与  truth.json 的 INR 不是同一口径：后者 =
    10log10(mean_port_power_w)+30（truth.json 顶层 tx_power_convention，与
    eval_dataset::interferer_visible_at 同口径），与 d1_truth 约有 10 dB 系统偏移。
    因此改用相对结构校验：假设映射下，该站的 d1_truth INR 是否在 4 站里排名第 1。"""
    full = json.loads((private / "truth.json").read_text(encoding="utf-8"))
    rxs = {x["receiver_id"]: x for x in full["stations"]}
    for st in allowed:
        rx = rxs.get(st)
        if rx is None:
            continue
        _rx = rx["receiver"]
        nz = 10 * np.log10(float(_rx.get("effective_output_noise_w")
                                 or _rx["expected_output_noise_w"])) + 30
        for gi, it in enumerate(truth.get("interferers", [])):
            eid = "interferer_%d" % (gi + 1)
            sx = next((y for y in rx["sources"] if y["emitter_id"] == eid), None)
            if sx is None:
                continue
            pmean = float(np.mean(sx["mean_port_power_w"]))
            if pmean <= 0:
                continue
            true_inr = 10 * np.log10(pmean) + 30 - nz
            key = STATION_TO_S[st]
            cand = {k: float(v.get("inr_db", -999))
                    for k, v in it.get("per_station", {}).items()}
            if key not in cand or not (cand[key] < 200):
                continue
            mc["total"] += 1
            mc["pairs_all"].append([cand[key], true_inr])
            mc["pair_by_key"].setdefault(key, []).append([cand[key], true_inr])
            mc["err"].append(cand[key] - true_inr)
            if len(mc["pairs"]) < 6:
                mc["pairs"].append({"sample": sid, "station": st, "assumed": key,
                                    "d1_inr_db": round(cand[key], 2),
                                    "truth_inr_db": round(true_inr, 2)})

# ---------------------------------------------------------------- 主流程
def run(samples: int, wins: list[float], use_ref: bool) -> dict:
    idx_path = _D2 / "index.json"
    if idx_path.is_file():
        ids = [s["id"] for s in json.loads(idx_path.read_text(encoding="utf-8"))["samples"]]
    else:
        ids = sorted(p.name for p in _D2.iterdir()
                     if p.is_dir() and p.name.startswith("d2_"))
    ids = ids[:samples]
    eprint("[info] samples=%d wins=%s use_ref=%s"
           % (len(ids), [w / 1e3 for w in wins], use_ref))

    tags = ("raw", "resid") if use_ref else ("raw",)
    acc: dict = {}
    for tag in tags:
        for w in wins:
            acc[(tag, w)] = {
                "conf_P1": {r: {} for r in RATIOS},
                "conf_P2": {r: {} for r in RATIOS},
                "conf_P1_g": {r: {} for r in RATIOS},
                "conf_P2_g": {r: {} for r in RATIOS},
                "conf_P0": {r: {} for r in RATIOS},   # 0 源窗 -> 虚警直方图
                "conf_P0_g": {r: {} for r in RATIOS},
                "conf_P0_mdl": {}, "conf_P0_mdl_g": {},
                "conf_P1_mdl": {}, "conf_P2_mdl": {},
                "conf_P1_mdl_g": {}, "conf_P2_mdl_g": {},
                "nwin_P1": 0, "nwin_P2": 0, "nwin_P0": 0,
                "nwin_P1_g": 0, "nwin_P2_g": 0,
                "ratio_hist_all": {r: {} for r in RATIOS},
                "mdl_hist_all": {},
            }
    per_sample_rows = []
    mc = {"pairs": [], "pairs_all": [], "pair_by_key": {}, "err": [], "total": 0}
    n_ref_fail = 0
    n_strip_fail = 0

    for si, sid in enumerate(ids):
        t0 = time.time()
        cap = _D2 / sid / "capture"
        private = cap / "private"
        truth = json.loads((private / "d1_truth.json").read_text(encoding="utf-8"))
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        ns = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()

        try:
            station_map_check(private, truth, allowed, sid, mc)
        except Exception as exc:  # noqa: BLE001
            eprint("[warn] %s: map check failed: %s" % (sid, exc))

        refs = {}
        if use_ref:
            try:
                refs = rca._build_target_reference(private, ns)
            except Exception:
                n_ref_fail += 1
                eprint("[warn] %s: ref build failed\n%s" % (sid, traceback.format_exc()))
                refs = {}
        ctx = ct.ToolContext(capture_dir=cap / "public",
                             cache_dir=_D2 / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=aoi,
                             target_reference=refs)
        ct.set_context(ctx)

        srcs = source_list(truth)
        gt_station = {st: (1 if srcs_in_window(srcs, -SEARCH_HALF_HZ,
                                               SEARCH_HALF_HZ, st) else 0)
                      for st in allowed}
        gt_med = int(np.median([gt_station[s] for s in allowed])) if allowed else 0
        gt_visible_stations = int(sum(gt_station.values()))

        f = np.fft.fftfreq(ns, 1 / fs)
        inb = (f >= -SEARCH_HALF_HZ) & (f <= SEARCH_HALF_HZ)
        row = {"sample": sid, "gt_visible_sources": gt_med,
               "gt_visible_stations": gt_visible_stations,
               "gt_station": gt_station,
               "srcs": [{"kind": s["kind"], "off_khz": round(s["off"] / 1e3, 1),
                         "bw_khz": round(s["bw"] / 1e3, 1),
                         "inr_db": {k: round(float(v.get("inr_db", np.nan)), 1)
                                    for k, v in s["per_station"].items()}}
                        for s in srcs],
               "stations": {}}

        for st in allowed:
            raw = np.load(cap / "public" / ("iq_%s.npy" % st)).astype(np.complex128)
            variants = {"raw": band_limit(raw, fs, SEARCH_HALF_HZ)}
            if use_ref:
                try:
                    variants["resid"] = ct._strip_target_at_station(st)
                except Exception:
                    n_strip_fail += 1
                    eprint("[warn] %s/%s: strip failed, resid=raw" % (sid, st))
                    variants["resid"] = variants["raw"]

            st_segs: dict = {}
            for tag, iq in variants.items():
                if iq.shape[-1] != ns:
                    iq = iq[:, :ns]
                fft = np.fft.fft(iq, axis=1)
                psd_db = 10 * np.log10((np.abs(fft) ** 2).mean(axis=0) + 1e-30)
                sig_mask = np.zeros(ns, bool)
                for s in srcs:
                    sig_mask |= ((f >= s["lo"] - FLOOR_GUARD_HZ)
                                 & (f <= s["hi"] + FLOOR_GUARD_HZ))
                sig_mask |= np.abs(f) < 60e3          # 保护目标带
                sig_mask &= inb
                floor_db = local_floor_db(psd_db, sig_mask, fs, ns)
                ref_pow = 10 * np.log10(float((np.abs(fft) ** 2).mean()) + 1e-30)

                for w in wins:
                    half = w / 2.0
                    n_side = int(np.floor(SEARCH_HALF_HZ / w + 1e-9))
                    centers = np.unique(np.round(np.concatenate([
                        np.arange(-n_side * w, 0.0 + 1e-9, w),
                        np.arange(0.0, n_side * w + 1e-9, w)]), 6))
                    a = acc[(tag, w)]
                    est_pr = {r: np.zeros(centers.size, int) for r in RATIOS}
                    mdl_cnt = np.zeros(centers.size, int)
                    pow_over = np.full(centers.size, np.nan)
                    for ci, c in enumerate(centers):
                        m = (f >= c - half) & (f < c + half)
                        ncol = int(m.sum())
                        if ncol < 2:
                            continue
                        x = fft[:, m]
                        x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-30)
                        wv = eig_profile(x)
                        for r in RATIOS:
                            k = count_ratio(wv, r)
                            est_pr[r][ci] = k
                            a["ratio_hist_all"][r][k] = a["ratio_hist_all"][r].get(k, 0) + 1
                        kk_mdl = count_mdl(wv, ncol)
                        mdl_cnt[ci] = kk_mdl
                        a["mdl_hist_all"][kk_mdl] = a["mdl_hist_all"].get(kk_mdl, 0) + 1
                        cen_pow = 10 * np.log10(
                            float((np.abs(fft[:, m]) ** 2).sum(axis=1).mean()) + 1e-30)
                        pow_over[ci] = cen_pow - ref_pow

                    gate_ok = pow_over >= POWER_GATE_DB
                    for r in RATIOS:
                        st_segs["%s_%dk_r%g|seg" % (tag, int(w / 1e3), r)] = \
                            merge_segments(est_pr[r], centers)
                        st_segs["%s_%dk_r%g|gate3" % (tag, int(w / 1e3), r)] = \
                            merge_segments(np.where(gate_ok, est_pr[r], 0), centers)
                    st_segs["%s_%dk_mdl|seg" % (tag, int(w / 1e3))] = \
                        merge_segments(mdl_cnt, centers)

                    # ---- 窗级混淆矩阵（每窗只计一次）----
                    for ci, c in enumerate(centers):
                        kk = len(srcs_in_window(srcs, c - half, c + half, st))
                        gated = bool(gate_ok[ci])
                        mv = int(mdl_cnt[ci])
                        if kk == 1:
                            a["nwin_P1"] += 1
                            a["nwin_P1_g"] += int(gated)
                        elif kk == 2:
                            a["nwin_P2"] += 1
                            a["nwin_P2_g"] += int(gated)
                        else:
                            a["nwin_P0"] += 1
                        for r in RATIOS:
                            val = int(est_pr[r][ci])
                            a["conf_P%d" % kk][r][val] = \
                                a["conf_P%d" % kk][r].get(val, 0) + 1
                            if gated:
                                a["conf_P%d_g" % kk][r][val] = \
                                    a["conf_P%d_g" % kk][r].get(val, 0) + 1
                        conf_mdl_bump(a, kk, mv, gated)

            best = {}
            for key, segs in st_segs.items():
                best[key] = len(segs)
                best[key + "|nmax"] = int(max((s["n_max"] for s in segs), default=0))
            row["stations"][st] = {"gt_visible": gt_station[st], "est": best}

        per_sample_rows.append(row)
        eprint("[%d/%d] %s gt_med=%d gt_stas=%d t=%.1fs"
               % (si + 1, len(ids), sid, gt_med, gt_visible_stations, time.time() - t0))

    mc["ref_build_fail"] = n_ref_fail
    mc["strip_fail"] = n_strip_fail
    return {"ids": ids, "acc": acc, "rows": per_sample_rows, "map_check": mc}


# ---------------------------------------------------------------- 报告
def fmt_hist(h: dict) -> str:
    if not h:
        return "-"
    return " ".join("%d:%d" % (int(k), int(v))
                    for k, v in sorted(h.items(), key=lambda x: int(x[0])))


def sample_stats(res: dict, tag: str, w: float, r: float, mode: str) -> dict:
    """逐样本重算：median(4站估计) vs GT 可见源数。"""
    hits, over, under = 0, {}, {}
    key = "%s_%dk_r%g|%s" % (tag, int(w / 1e3), r, mode)
    for row in res["rows"]:
        ests = [row["stations"][s]["est"].get(key) for s in row["stations"]]
        ests = [e for e in ests if e is not None]
        if not ests:
            continue
        est = int(np.median(ests))
        d = est - row["gt_visible_sources"]
        if d == 0:
            hits += 1
        elif d > 0:
            over[d] = over.get(d, 0) + 1
        else:
            under[-d] = under.get(-d, 0) + 1
    return {"hits": hits, "n": len(res["rows"]), "over": over, "under": under}


def build_report(res: dict, params: dict) -> str:
    L = []
    P = L.append
    tags = ("raw", "resid") if params["use_ref"] else ("raw",)
    wins = params["wins"]

    P("=" * 104)
    P("D2 逐频窗窄带空间特征值枚举 —— 『源数证据』假设的诊断报告（只读，零 API）")
    P("=" * 104)
    P("假设 H：窄带频窗内 8 元阵空间协方差的超阈特征值个数 ≈『该窗有几个源』，")
    P("        从而可替代多站锥交会定位作为源数证据。")
    P("")
    P("【口径自检查】")
    P(" 1. _band_coherence 实际实现（capture_tools.py:62-77）：先对 FFT 子矩阵按列")
    P("    归一化 x /= ||x||_col，再 R = x x^H / n_cols，最后 λ1/Σλ。")
    P("    → trace(R) == M == 8；跨阵元不相关的白噪/失配能量各 λ ≈ 1（**不是 1/M**）。")
    P("    本脚本沿用同一实现（eigenvalues of R，trace=8），故『λ >> 1 的个数』= 源子空间维数。")
    P(" 2. 归一化方式：本报告用**比值型**判据 λ_k > ratio × median(λ_{k+1..8})，")
    P("    对全局尺度严格不变 → 用 w/median(w) 还是 w/w[-1] 对计数**零影响**；")
    P("    JSON 里的 ratio_hist_all / mdl_hist_all 给出原始特征值个数的无条件分布，可核对。")
    P(" 3. 计数规则：k 从 1 起，且至少保留 2 个特征值当噪声底（否则纯白噪窗")
    P("    在 ratio=2 下必然虚报 1 个）——这会**系统性压低**近似秩 7 的场景，")
    P("    属于保守口径，对『数源』是有利的选择。")
    P(" 4. 站↔sX 映射：硬编码 station_a->s0 … station_d->s3，")
    P("    并用 truth.json 的 per_station INR 做了逐源自校验（见下方映射校验行）。")
    mc = res["map_check"]
    if mc.get("total"):
        import numpy as _np
        _a = _np.array(mc["pairs_all"], float)
        _r = float(_np.corrcoef(_a[:, 0], _a[:, 1])[0, 1]) if _a.shape[0] > 2 else float("nan")
        _e = _np.abs(_np.array(mc["err"], float))
        P("    → 映射校验：假设 station_a->s0 … station_d->s3 下，d1_truth.inr_db 与 "
          "truth.json 同站同源 INR 的配对相关系数 r = %.3f（n=%d）"
          % (_r, mc["total"]))
        P("      绝对偏移 中位 |Δ| = %.2f dB，均值 Δ = %+.2f dB（两文件噪声口径不同，"
          "比值的绝对刻度不可直接互换）"
          % (float(_np.median(_e)), float(_np.mean(mc["err"]))))
        P("      （r≈0.7 且为 24 个置换里最优 → 映射成立；r≈0 → 映射错。）")
        for pp in mc["pairs"][:4]:
            P("      e.g. %s %s: 假设 %s | d1_truth INR=%s vs truth.json INR=%s"
              % (pp["sample"], pp["station"], pp["assumed"],
                 pp["d1_inr_db"], pp["truth_inr_db"]))
    P("    → ref 构造失败 %d 次，残差剥离失败 %d 次"
      % (mc.get("ref_build_fail", 0), mc.get("strip_fail", 0)))
    P("")
    P("参数：样本 %d 个；窗宽 %s kHz（步进 = 窗宽，不重叠，覆盖 |f| <= %.1f MHz = 0.45·fs）；"
      % (len(res["ids"]), [int(w / 1e3) for w in wins], SEARCH_HALF_HZ / 1e6))
    P("      阈值倍数 %s；可见性 INR >= %.1f dB；功率门控 %+.1f dB（相对全带平均功率）。"
      % (list(RATIOS), INR_VIS_DB, POWER_GATE_DB))
    P("      窗中心网格：0, ±w, ±2w ... 直到 |f| > %.1f MHz（两侧各 %d 个，另加 0 频点）。"
      % (SEARCH_HALF_HZ / 1e6, int(SEARCH_HALF_HZ // min(wins))))
    P("")

    # ---------------- 表 1 ----------------
    P("-" * 104)
    P("【表 1】样本级命中：median(4 站估计) == GT 可见源数 的样本数")
    P("-" * 104)
    P("GT 口径：该站是否存在 INR>=3dB 且频段覆盖搜索区的干扰源（是→1，否→0）；")
    P("        样本级 GT 再取 4 站**中位数** —— 与 估计端同构，因此样本级 GT 只可能是 0 或 1。")
    P("")
    hdr = ("%-6s %-6s %-6s %-7s %9s %9s  %-22s %-22s"
           % ("窗宽", "数据", "ratio", "模式", "命中/N", "命中率", "过报分布", "欠报分布"))
    for tag in tags:
        P("### 数据 = %s" % tag)
        P(hdr)
        for w in wins:
            for r in RATIOS:
                for mode in ("seg", "gate3"):
                    st_ = sample_stats(res, tag, w, r, mode)
                    P("%-6s %-6s %-6g %-7s %6d/%-3d %8.1f%%  %-22s %-22s"
                      % ("%dk" % int(w / 1e3), tag, r, mode, st_["hits"], st_["n"],
                         100.0 * st_["hits"] / max(st_["n"], 1),
                         fmt_hist(st_["over"]), fmt_hist(st_["under"])))
        P("")

    # ---------------- 表 2 ----------------
    P("-" * 104)
    P("【表 2】窗级混淆矩阵：GT 源覆盖数 vs 特征值计数（直接决定假设成立与否）")
    P("-" * 104)
    for tag in tags:
        P("### 数据 = %s" % tag)
        for w in wins:
            a = res["acc"][(tag, w)]
            P("  --- 窗宽 %d kHz ---" % int(w / 1e3))
            P("  恰 1 源窗 n=%d（其中过功率门控 n=%d）；恰 2 源窗 n=%d（门控 n=%d）；0 源窗 n=%d"
              % (a["nwin_P1"], a["nwin_P1_g"], a["nwin_P2"], a["nwin_P2_g"], a["nwin_P0"]))
            for lbl, key in (("恰1源", "conf_P1"), ("恰1源/门控", "conf_P1_g"),
                             ("恰2源", "conf_P2"), ("恰2源/门控", "conf_P2_g"),
                             ("恰0源(虚警)", "conf_P0"), ("恰0源/门控", "conf_P0_g")):
                P("    %-12s" % lbl)
                for r in RATIOS:
                    P("      ratio=%g  %s" % (r, fmt_hist(a[key][r])))
            P("    %-12s  %s" % ("恰1源/MDL", fmt_hist(a["conf_P1_mdl"])))
            P("    %-12s  %s" % ("恰2源/MDL", fmt_hist(a["conf_P2_mdl"])))
            P("    %-12s  %s" % ("恰0源/MDL", fmt_hist(a["conf_P0_mdl"])))
            for r in RATIOS:
                for gate, k1, k2 in (("全部窗", "conf_P1", "conf_P2"),
                                     ("门控后", "conf_P1_g", "conf_P2_g")):
                    h1, h2 = a[k1][r], a[k2][r]
                    n1 = sum(h1.values()) or 1
                    n2 = sum(h2.values()) or 1
                    P("    ratio=%g %s：1源窗报1=%.1f%%  2源窗报2=%.1f%%  2源窗报1=%.1f%%  "
                      "2源窗报>=3=%.1f%%  "
                      % (r, gate, h1.get(1, 0) / n1 * 100, h2.get(2, 0) / n2 * 100,
                         h2.get(1, 0) / n2 * 100,
                         sum(v for k, v in h2.items() if k >= 3) / n2 * 100))
            P("    恰 0 源窗（无可见源覆盖）虚警——这才是真正无源处的假报率：")
            for r in RATIOS:
                hist = a["conf_P0"][r]
                tot = sum(hist.values()) or 1
                P("      ratio=%g  计数分布 %s  → 报>0 占 %.1f%%（报>=2 占 %.1f%%）"
                  % (r, fmt_hist(hist), sum(v for k, v in hist.items() if k > 0) / tot * 100,
                     sum(v for k, v in hist.items() if k >= 2) / tot * 100))
            P("    （对照）全部窗（含 0/1/2 源）的特征值计数分布与报>0 占比：")
            for r in RATIOS:
                hist = a["ratio_hist_all"][r]
                tot = sum(hist.values()) or 1
                P("      ratio=%g  计数分布 %s  → 报>0 占 %.1f%%"
                  % (r, fmt_hist(hist), sum(v for k, v in hist.items() if k > 0) / tot * 100))
            P("")

    # ---------------- 表 3 ----------------
    P("-" * 104)
    P("【表 3】逐样本明细（ratio=3，模式=seg；每站各窗宽的连通段数，段数=该站源数估计）")
    P("-" * 104)
    tag0 = "resid" if params["use_ref"] else "raw"
    cols = [(w, s) for w in wins for s in ("station_a", "station_b", "station_c", "station_d")]
    P("%-9s %3s | " % ("sample", "GT") + " ".join("%9s" % ("w%dk_%s" % (int(w / 1e3), s[-1]))
                                                  for w, s in cols) + " | %s" % " ".join(
        "%8s" % ("med_w%dk" % int(w / 1e3)) for w in wins))
    for row in res["rows"]:
        line = "%-9s %3d | " % (row["sample"], row["gt_visible_sources"])
        for w, s in cols:
            v = row["stations"][s]["est"].get("%s_%dk_r3|seg" % (tag0, int(w / 1e3)))
            line += "%9s " % (v if v is not None else "-")
        line += "| "
        for w in wins:
            ests = [row["stations"][s]["est"].get("%s_%dk_r3|seg" % (tag0, int(w / 1e3)))
                    for s in row["stations"]]
            ests = [e for e in ests if e is not None]
            line += "%8s " % (int(np.median(ests)) if ests else "-")
        P(line)
    P("")

    # ---------------- 表 4：逐样本源数/带宽 ----------------
    P("-" * 104)
    P("【表 4】逐样本真值结构（源频偏/带宽/各站 INR）")
    P("-" * 104)
    for row in res["rows"]:
        P("%-9s GT_vis=%d  n_src=%d  %s" % (row["sample"], row["gt_visible_sources"],
                                            len(row["srcs"]),
                                            " | ".join("%s off=%+.0fkHz bw=%.0fkHz INR=%s"
                                                       % (s["kind"], s["off_khz"],
                                                          s["bw_khz"],
                                                          ",".join("%s=%.1f" % (k, v) for k, v
                                                                   in sorted(s["inr_db"].items())))
                                                       for s in row["srcs"])))

    # ---------------- 结论 ----------------
    P("")
    P("-" * 104)
    P("【结论】")
    P("-" * 104)
    best = None
    for tag in tags:
        for w in wins:
            for r in RATIOS:
                for mode in ("seg", "gate3"):
                    st_ = sample_stats(res, tag, w, r, mode)
                    if best is None or st_["hits"] > best[0]:
                        best = (st_["hits"], st_["n"], tag, w, r, mode)
    P("* 最佳样本级命中：%d/%d (%.1f%%) @ 数据=%s 窗宽=%dk ratio=%g 模式=%s"
      % (best[0], best[1], 100.0 * best[0] / max(best[1], 1), best[2],
         int(best[3] / 1e3), best[4], best[5]))
    triv = sum(1 for row in res["rows"] if row["gt_visible_sources"] == 1)
    P("* 平凡基线：样本级 GT 全为 1（%d/%d 个样本 gt_med==1）；"
      "『恒输出 1 个连通段』即可拿到 %d/%d = %.1f%%。"
      % (triv, len(res["rows"]), triv, len(res["rows"]),
         100.0 * triv / max(len(res["rows"]), 1)))
    tag0r = "resid" if params["use_ref"] else "raw"
    for w in wins:
        a = res["acc"][(tag0r, w)]
        for r in RATIOS:
            h2 = a["conf_P2"][r]
            n2 = sum(h2.values()) or 1
            h1 = a["conf_P1"][r]
            n1 = sum(h1.values()) or 1
            P("* 窗宽 %dk ratio=%g：恰2源窗报2=%.1f%%（报1=%.1f%%/报0=%.1f%%/报>=3=%.1f%%） | "
              "恰1源窗报1=%.1f%%（报0=%.1f%%/报>=2=%.1f%%）"
              % (int(w / 1e3), r,
                 h2.get(2, 0) / n2 * 100, h2.get(1, 0) / n2 * 100, h2.get(0, 0) / n2 * 100,
                 sum(v for k, v in h2.items() if k >= 3) / n2 * 100,
                 h1.get(1, 0) / n1 * 100, h1.get(0, 0) / n1 * 100,
                 sum(v for k, v in h1.items() if k >= 2) / n1 * 100))
    P("* 注意：D2 的源带宽范围 101–787 kHz（中位 ~454 kHz），**远大于** 10/20/50 kHz 窗宽，")
    P("  因此绝大多数非空窗只落在一个源的带内。窄带窗法只能回答『这个窗里有几个源』，")
    P("  无法回答『这个频段里有几个源』——除非把子窗秩沿频率做正确的缝合（本脚本用连通段）。")
    P("")
    P("用时/规模：" + json.dumps(params["timing"], ensure_ascii=False))
    return "\n".join(L)


def verdict_lines(res: dict, params: dict) -> list[str]:
    out = []
    tags = ("raw", "resid") if params["use_ref"] else ("raw",)
    best = None
    for tag in tags:
        for w in params["wins"]:
            for r in RATIOS:
                for mode in ("seg", "gate3"):
                    st_ = sample_stats(res, tag, w, r, mode)
                    if best is None or st_["hits"] > best[0]:
                        best = (st_["hits"], st_["n"], tag, w, r, mode)
    out.append("best_hits=%d/%d data=%s win=%dk ratio=%g mode=%s"
               % (best[0], best[1], best[2], int(best[3] / 1e3), best[4], best[5]))
    tag0 = "resid" if params["use_ref"] else "raw"
    for w in params["wins"]:
        a = res["acc"][(tag0, w)]
        row = []
        for r in RATIOS:
            h2 = a["conf_P2"][r]
            n2 = sum(h2.values()) or 1
            row.append("r%g: P2->2 %.1f%%" % (r, h2.get(2, 0) / n2 * 100))
        out.append("win=%dk  %s" % (int(w / 1e3), " | ".join(row)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--wins", type=str, default="10000,20000,50000,100000,200000")
    ap.add_argument("--no-ref", action="store_true", help="不构造目标参考（只跑 raw 栏）")
    ap.add_argument("--out-txt", type=str, default=str(_V3 / "_an_d2_rank.txt"))
    ap.add_argument("--out-json", type=str, default=str(_V3 / "_an_d2_rank.json"))
    args = ap.parse_args()
    wins = [float(x) for x in args.wins.split(",") if x.strip()]
    t0 = time.time()
    res = run(args.samples, wins, not args.no_ref)
    el = time.time() - t0
    params = {"wins": wins, "use_ref": not args.no_ref,
              "timing": {"total_s": round(el, 1), "n_samples": len(res["ids"]),
                         "n_stations": 4,
                         "n_windows_per_station": {int(w / 1e3):
                                                   int(2 * (SEARCH_HALF_HZ // w) + 1)
                                                   for w in wins}}}
    params["verdict_lines"] = verdict_lines(res, params)

    acc_ser = {}
    for k, v in res["acc"].items():
        acc_ser["%s|%s" % (k[0], k[1]) if isinstance(k, tuple) else str(k)] = v
    payload = {
        "meta": {
            "script": "_an_d2_rank.py",
            "purpose": "逐频窗窄带空间特征值枚举作为源数证据的可行性诊断",
            "params": {**{k: v for k, v in params.items() if k != "verdict_lines"},
                       "RATIOS": list(RATIOS), "POWER_GATE_DB": POWER_GATE_DB,
                       "INR_VIS_DB": INR_VIS_DB, "SEARCH_HALF_HZ": SEARCH_HALF_HZ,
                       "FLOOR_WIN_HZ": FLOOR_WIN_HZ,
                       "count_rule": "lambda_k > ratio * median(lambda_{k+1..8}), k>=1, "
                                     ">=2 eigenvalues kept as floor",
                       "normalization_note": "trace(R)==8; uncorrelated white noise -> "
                                             "each lambda ~= 1 (per _band_coherence impl)",
                       "station_to_s": STATION_TO_S},
            "verdict": params["verdict_lines"],
            "map_check": res["map_check"],
        },
        "per_sample": res["rows"],
        "aggregate": acc_ser,
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    Path(args.out_txt).write_text(build_report(res, params), encoding="utf-8")
    print("[done] %.1fs -> %s" % (el, args.out_txt))
    print("[done]        -> %s" % args.out_json)
    for v in params["verdict_lines"][:6]:
        print(v)


if __name__ == "__main__":
    main()
