"""只读实验：给每个源"单独定边"的三条判据对照（用户 m12150 提案：谱阶跃 + 跨阵元相干度）。

背景（本轮实测）
  · 现行判据 `capture_tools._residual_bands`：把 over(局部底 +8 dB) 与 ±3 kHz 空间相干(≥0.6)
    在 **10 kHz 固定网格**上打点，再把相邻点合并成带 → 带只有 ~6 kHz 宽；
    网格还可能整条跳过 1 bin 宽的纯音（`d1_00026`：真源 over=60 dB / coh=0.90 却返回空）。
  · 旧宽带判据 `_residual_peaks_legacy`：全局中位底 + ≥5 bin 连段 → 干净样本虚警（已废）。

本脚本在同一份残差（同一 station、同一 `_strip_target_at_station`）上对比：
  A 现行           : `capture_tools._residual_bands`（原样调用，作为基准）
  B 相干支撑连通域 : 同样的 over/coh 判据，但**去掉 10 kHz 网格与最小长度**，
                     并允许跨"相干仍然很高"的谷把碎片并回同一个源
  C 谱阶跃边       : 平滑谱一阶差分找强阶跃 → 边处相干门控 → 用"外侧相干"区分
                     **真边界**（一侧是背景）与**源内结构**（两侧都相干）；再配对成带

指标（逐 GT 源，station 可指定）：
  · 覆盖 = 某返回带中心距 GT 载频 ≤40 kHz（注册口径 `eval_dataset.FREQ_TOL`）
           或与 GT 频带重叠 ≥50%
  · 多余带 = 未覆盖任何 GT 的带
  · 带宽：返回带宽度中位 vs GT 带宽中位
  · 边准：GT 带宽 ≥20 kHz 的源，是否有接受边落在真边 ±60 kHz 内（报告中位 |Δ边|）
  · 干净样本：返回带数（即虚警带数）

用法：
  python -X utf8 diag\\_sweep_step_edges.py <dataset_root> [--n 100] [--station station_a]
  python -X utf8 diag\\_sweep_step_edges.py <dataset_root> --cases d2_00001:station_c
"""
from __future__ import annotations

import json as _json
import pathlib as _pl
import shutil
import sys
import tempfile

import numpy as np

_HERE = _pl.Path(__file__).resolve().parent
_V3 = _HERE.parent
for _p in (str(_V3), str(_V3 / "simulation"), str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

FREQ_TOL_HZ = 40e3

# ---- 判据参数（本轮实验用；与 capture_tools 现有常量同源）----
GRID_HZ = 5e3            # 我自己的扫描网格（比现行的 10 kHz 细一半）
SMOOTH_HZ = 50e3         # 阶跃检测的平滑尺度（参数，见提案补丁②）
EDGE_STEP_MIN = 0.35     # dB/kHz，接受为"强阶跃"的下限
EDGE_COH_MIN = 0.80      # 边处相干门限（真源边 0.98~1.0、地毯边 0.19~0.25）
EDGE_OUTSIDE_HZ = 40e3   # 判断"真边界 vs 源内结构"要看多远的外侧
EDGE_NMS_HZ = 25e3       # C 法：同一条真边上的多个局部极大做非极大抑制的频距
EDGE_INSIDE_FRAC_MIN = 0.35   # C 法：带内"源样"（over≥阈 且 相干≥阈）占比下限
EDGE_INSIDE_COH_MIN = 0.60    # C 法：带内平均相干下限
MERGE_GAP_HZ = 90e3      # B 法：允许跨谷合并的最大间隙
MERGE_COH_MIN = 0.60     # B 法：间隙内相干中位仍需 ≥ 此值


def _prep(resid: np.ndarray, fs: float, rx_half: float):
    """返回 (频率轴, over 曲线, 平滑谱, 全 FFT)。频率轴已按升序排列（正频段 + 负频段）。"""
    from scipy.ndimage import median_filter, uniform_filter1d
    n = resid.shape[1]
    f = np.fft.fftfreq(n, 1 / fs)
    inb = np.where((f >= -rx_half) & (f <= rx_half))[0]
    order = inb[np.argsort(f[inb])]
    X = np.fft.fft(resid, axis=-1)
    psd = (np.abs(X) ** 2).mean(axis=0)[order]
    db = 10 * np.log10(psd + 1e-30)
    fs_bin = fs / n
    k_floor = max(int(round(ct.PEAK_FLOOR_WIN_HZ / fs_bin)) | 1, 5)
    over = db - median_filter(db, size=k_floor, mode="nearest")
    sm = uniform_filter1d(db, size=max(int(round(SMOOTH_HZ / fs_bin)), 1), mode="nearest")
    return f[order], over, sm, X


def _coh_curve(X: np.ndarray, f_full: np.ndarray, centers: np.ndarray, win_hz: float) -> np.ndarray:
    """一次 FFT 复用：逐中心点算 ±win 内 λ1/Σλ（与 ct._band_coherence 同定义）。"""
    out = np.full(centers.size, np.nan)
    for i, c in enumerate(centers):
        m = np.abs(f_full - c) <= win_hz
        if int(m.sum()) < 2:
            continue
        x = X[:, m]
        x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-30)
        r = x @ x.conj().T / x.shape[1]
        w = np.linalg.eigvalsh(r)[::-1].real
        out[i] = float(w[0] / (w.sum() + 1e-30))
    return out


def _band(f, over, i0: int, i1: int) -> dict:
    seg = over[i0:i1 + 1]
    k = i0 + int(np.argmax(seg))
    lo = float(f[i0]) - ct.PEAK_COH_WIN_HZ
    hi = float(f[i1]) + ct.PEAK_COH_WIN_HZ
    return {"freq_offset_hz": round(float(f[k]), 0),
            "over_floor_db": round(float(seg.max()), 1),
            "bandwidth_hz": round(hi - lo, 0),
            "band_hz": [round(lo, 0), round(hi, 0)]}


def bands_support(f, over, coh) -> list[dict]:
    """B：相干支撑连通域（无网格、无最小长度；允许跨"仍相干"的谷合并）。"""
    sup = (over >= ct.PEAK_OVER_DB) & (coh >= ct.PEAK_COH_MIN)
    idx = np.where(sup)[0]
    if idx.size == 0:
        return []
    raw: list[list[int]] = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        i = int(i)
        if f[i] - f[raw[-1][1]] <= GRID_HZ * 1.5:
            raw[-1][1] = i
        else:
            raw.append([i, i])
    merged = [list(raw[0])]
    for a, b in raw[1:]:
        pa, pb = merged[-1]
        gap = f[a] - f[pb]
        gap_coh = float(np.nanmedian(coh[pb:a + 1])) if a > pb else 1.0
        if gap <= MERGE_GAP_HZ and gap_coh >= MERGE_COH_MIN:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [_band(f, over, a, b) for a, b in merged]


def bands_steps(f, over, coh, sm) -> tuple[list[dict], list[dict]]:
    """C：谱阶跃 + 相干（真边界 vs 源内结构）→ 配对成带。返回 (带, 接受边明细)。"""
    if f.size < 7:
        return [], []
    d = np.gradient(sm, f) * 1e3                                     # dB/kHz（dB/Hz × 1000）
    ad = np.abs(d)
    cand = ((ad[1:-1] >= ad[:-2]) & (ad[1:-1] >= ad[2:]) & (ad[1:-1] >= EDGE_STEP_MIN))
    edges: list[dict] = []
    for i in (np.where(cand)[0] + 1):
        i = int(i)
        c = coh[i]
        if not np.isfinite(c) or c < EDGE_COH_MIN:
            continue
        jl = int(np.argmin(np.abs(f - (f[i] - EDGE_OUTSIDE_HZ))))
        jr = int(np.argmin(np.abs(f - (f[i] + EDGE_OUTSIDE_HZ))))
        cl, cr = coh[jl], coh[jr]
        outside_ok = (not np.isfinite(cl)) or (not np.isfinite(cr)) or (min(cl, cr) < EDGE_COH_MIN)
        if not outside_ok:                                          # 两侧都相干 = 源内结构
            continue
        edges.append({"f_hz": float(f[i]), "step_db_per_khz": float(abs(d[i])),
                      "coh": float(c), "coh_left": float(cl) if np.isfinite(cl) else None,
                      "coh_right": float(cr) if np.isfinite(cr) else None})
    # 非极大抑制：同一条真边上会连着好几个局部极大（实测 top12 里有 4 个都属于 DECT 下沿）
    edges.sort(key=lambda e: -e["step_db_per_khz"])
    kept: list[dict] = []
    for e in edges:
        if all(abs(e["f_hz"] - k["f_hz"]) > EDGE_NMS_HZ for k in kept):
            kept.append(e)
    edges = sorted(kept, key=lambda e: e["f_hz"])
    bands: list[dict] = []
    for e1, e2 in zip(edges, edges[1:]):
        i0 = int(np.argmin(np.abs(f - e1["f_hz"])))
        i1 = int(np.argmin(np.abs(f - e2["f_hz"])))
        if i1 - i0 < 2:
            continue
        seg = slice(i0, i1 + 1)
        src = (over[seg] >= ct.PEAK_OVER_DB) & (coh[seg] >= ct.PEAK_COH_MIN)
        # 带内"源样"占比而不是中位：跳频/突发源内部本来有大片谷（蓝牙实测占比 ~0.5）
        if float(np.mean(src)) < EDGE_INSIDE_FRAC_MIN:
            continue
        if float(np.nanmean(coh[seg])) < EDGE_INSIDE_COH_MIN:
            continue
        b = _band(f, over, i0, i1)
        b["edge_step_db_per_khz"] = round(max(e1["step_db_per_khz"], e2["step_db_per_khz"]), 2)
        bands.append(b)
    return bands, edges


def cover_stats(bands: list[dict], gts: list[dict]) -> dict:
    hit_idx: set[int] = set()
    cov = 0
    for g in gts:
        f0 = float(g["freq_offset_hz"])
        bw = float(g.get("bandwidth_hz") or 0.0)
        lo, hi = f0 - bw / 2, f0 + bw / 2
        for i, b in enumerate(bands):
            blo, bhi = b["band_hz"]
            ok = abs(b["freq_offset_hz"] - f0) <= FREQ_TOL_HZ
            if not ok and bw > 0:
                ok = max(0.0, min(hi, bhi) - max(lo, blo)) / bw >= 0.5
            if ok:
                hit_idx.add(i)
                cov += 1
                break
    extras = len(bands) - len(hit_idx)
    w_band = float(np.median([b["bandwidth_hz"] for b in bands])) if bands else 0.0
    w_gt = float(np.median([float(g.get("bandwidth_hz") or 0.0) for g in gts])) if gts else 0.0
    return {"cov": cov, "extras": extras, "n_band": len(bands),
            "w_band": w_band, "w_gt": w_gt}


def edge_stats(edges: list[dict], gts: list[dict]) -> dict:
    """GT 带宽 ≥20 kHz 的源：接受边是否落在真边 ±60 kHz 内。"""
    truth_edges = []
    for g in gts:
        bw = float(g.get("bandwidth_hz") or 0.0)
        if bw < 20e3:
            continue
        f0 = float(g["freq_offset_hz"])
        truth_edges += [f0 - bw / 2, f0 + bw / 2]
    if not truth_edges:
        return {"n_gt_edge": 0, "matched": 0, "med_abs_err_hz": None}
    errs = []
    for t in truth_edges:
        if not edges:
            break
        d = min(abs(e["f_hz"] - t) for e in edges)
        if d <= 60e3:
            errs.append(d)
    return {"n_gt_edge": len(truth_edges), "matched": len(errs),
            "med_abs_err_hz": float(np.median(errs)) if errs else None}


def residual_of(root: _pl.Path, sid: str, station: str):
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = _json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    tr = _json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    refs = rca._build_target_reference(priv, n)
    tmp = _pl.Path(tempfile.mkdtemp(prefix=f"steps_{sid}_"))
    ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                         allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                         fs=float(obs["receivers"][0]["sample_rate_hz"]),
                         fc=float(obs["receivers"][0]["center_frequency_hz"]),
                         target_bw_normalized=0.25, aoi_center=[0, 0],
                         target_reference=refs)
    ct.set_context(ctx)
    resid = np.asarray(ct._strip_target_at_station(station), np.complex128)
    return ctx, resid, tr, obs, tmp


def one(root: _pl.Path, sid: str, station: str) -> dict:
    try:
        ctx, resid, tr, _obs, tmp = residual_of(root, sid, station)
    except Exception as exc:                      # 该站无目标参考（0 径）等
        return {"sid": sid, "skip": f"{type(exc).__name__}: {exc}"}
    try:
        fs, rx_half = ctx.fs, ctx.fs * 0.45
        gts = list(tr.get("interferers") or [])
        f, over, sm, X = _prep(resid, fs, rx_half)
        f_full = np.fft.fftfreq(resid.shape[1], 1 / fs)
        coh = _coh_curve(X, f_full, f, ct.PEAK_COH_WIN_HZ)
        a = ct._residual_bands(resid, fs, rx_half)
        b = bands_support(f, over, coh)
        c, edges = bands_steps(f, over, coh, sm)
        out = {"sid": sid, "n_gt": len(gts)}
        for tag, bands in (("A", a), ("B", b), ("C", c)):
            st = cover_stats(bands, gts)
            st["edges"] = edge_stats(edges, gts) if tag == "C" else None
            out[tag] = st
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def accum(rows: list[dict]) -> None:
    skipped = [r for r in rows if r.get("skip")]
    rows = [r for r in rows if not r.get("skip")]
    clean = [r for r in rows if r["n_gt"] == 0]
    dirty = [r for r in rows if r["n_gt"] > 0]
    print(f"\n== 干净样本 {len(clean)} / 有干扰样本 {len(dirty)} / 跳过 {len(skipped)} ==")
    print(f"{'判据':<22}{'虚警带':>8}{'覆盖':>10}{'覆盖%':>8}{'多余带':>8}"
          f"{'带中位宽_kHz':>14}{'GT中位宽_kHz':>15}")
    for tag, name in (("A", "A 现行（10k 网格）"), ("B", "B 相干支撑连通域"), ("C", "C 谱阶跃边")):
        fa = sum(r[tag]["n_band"] for r in clean)
        cov = sum(r[tag]["cov"] for r in dirty)
        gt_n = sum(r["n_gt"] for r in dirty)
        ex = sum(r[tag]["extras"] for r in dirty)
        wb = [r[tag]["w_band"] for r in dirty if r[tag]["w_band"]]
        wg = [r[tag]["w_gt"] for r in dirty if r[tag]["w_gt"]]
        print(f"{name:<22}{fa:>8}{cov:>6}/{gt_n:<4}{100*cov/max(gt_n,1):>7.1f}%{ex:>8}"
              f"{(np.median(wb)/1e3 if wb else 0):>14.1f}{(np.median(wg)/1e3 if wg else 0):>15.1f}")
    eg = [r["C"]["edges"] for r in dirty if r["C"]["edges"] and r["C"]["edges"]["n_gt_edge"]]
    if eg:
        ne = sum(e["n_gt_edge"] for e in eg)
        me = sum(e["matched"] for e in eg)
        errs = [e["med_abs_err_hz"] for e in eg if e["med_abs_err_hz"] is not None]
        print(f"\nC 法边的定位（GT 带宽 ≥20 kHz 的源）：真边 {ne} 条、被接受边命中 {me} 条"
              f"（{100*me/max(ne,1):.1f}%），命中边中位 |Δ边| = "
              f"{(np.median(errs)/1e3 if errs else float('nan')):.1f} kHz")


def cases(root: _pl.Path, spec: str) -> int:
    sid, _, station = spec.partition(":")
    station = station or "station_a"
    ctx, resid, tr, _obs, tmp = residual_of(root, sid, station)
    try:
        fs, rx_half = ctx.fs, ctx.fs * 0.45
        gts = list(tr.get("interferers") or [])
        f, over, sm, X = _prep(resid, fs, rx_half)
        f_full = np.fft.fftfreq(resid.shape[1], 1 / fs)
        coh = _coh_curve(X, f_full, f, ct.PEAK_COH_WIN_HZ)
        c, edges = bands_steps(f, over, coh, sm)
        print(f"== {root.name}/{sid} @ {station} ==")
        for g in gts:
            f0 = float(g["freq_offset_hz"])
            bw = float(g.get("bandwidth_hz") or 0.0)
            print(f"  GT {g['kind']:<14} f={f0/1e3:9.1f}k  bw={bw/1e3:8.1f}k  真边 "
                  f"{(f0-bw/2)/1e3:9.1f}k / {(f0+bw/2)/1e3:9.1f}k")
        dd = np.gradient(sm, f) * 1e3
        ad = np.abs(dd)
        lp = np.where((ad[1:-1] >= ad[:-2]) & (ad[1:-1] >= ad[2:]))[0] + 1
        top = sorted(lp.tolist(), key=lambda i: -ad[i])[:12]
        print(f"  -- |d| 局部极大 top12（max|d|={ad.max():.3f} dB/kHz，平滑 {SMOOTH_HZ/1e3:.0f}k，"
              f"接受阈 {EDGE_STEP_MIN} dB/kHz + 边处相干≥{EDGE_COH_MIN} + 外侧相干<{EDGE_COH_MIN}）--")
        for i in top:
            jl = int(np.argmin(np.abs(f - (f[i] - EDGE_OUTSIDE_HZ))))
            jr = int(np.argmin(np.abs(f - (f[i] + EDGE_OUTSIDE_HZ))))
            print(f"     {f[i]/1e3:9.1f}k  |d|={ad[i]:5.2f}  coh={coh[i]:.3f}"
                  f"  out=({coh[jl]:.3f},{coh[jr]:.3f})")
        print(f"  -- C 法接受的 {len(edges)} 条边（阶跃 dB/kHz，边处相干，外侧相干）--")
        for e in sorted(edges, key=lambda x: -x["step_db_per_khz"]):
            tag = []
            for g in gts:
                f0 = float(g["freq_offset_hz"])
                bw = float(g.get("bandwidth_hz") or 0.0)
                for name, t in (("下沿", f0 - bw / 2), ("上沿", f0 + bw / 2)):
                    if abs(e["f_hz"] - t) <= 60e3:
                        tag.append(f"{g['kind']}{name}")
                if abs(e["f_hz"] - f0) <= 20e3:
                    tag.append(f"{g['kind']}载频")
            print(f"     {e['f_hz']/1e3:9.1f}k  step={e['step_db_per_khz']:5.2f}  "
                  f"coh={e['coh']:.3f}  out=({e['coh_left']},{e['coh_right']})"
                  + ("   <= " + "/".join(sorted(set(tag))) if tag else ""))
        print(f"  -- C 法返回的带 --")
        for b in c:
            print(f"     {b}")
        print(f"  -- A 法（现行）返回的带 --")
        for b in ct._residual_bands(resid, fs, rx_half):
            print(f"     {b}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=", 1)[1] for a in argv if a.startswith("--") and "=" in a}
    if not args:
        print(__doc__)
        return 2
    root = _pl.Path(args[0]).resolve()
    if "--cases" in opts:
        return cases(root, opts["--cases"])
    n_max = int(opts.get("--n", 0))
    station = opts.get("--station", "station_a")
    idx = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    samples = idx["samples"][:n_max] if n_max else idx["samples"]
    rows = []
    for s in samples:
        r = one(root, s["id"], station)
        rows.append(r)
        if r.get("skip"):
            print(f"  {r['sid']} 跳过：{r['skip']}", flush=True)
            continue
        print(f"  {r['sid']} GT={r['n_gt']}  A:cov={r['A']['cov']}/ex={r['A']['extras']}"
              f"/w={r['A']['w_band']/1e3:.1f}k  B:cov={r['B']['cov']}/ex={r['B']['extras']}"
              f"/w={r['B']['w_band']/1e3:.1f}k  C:cov={r['C']['cov']}/ex={r['C']['extras']}"
              f"/w={r['C']['w_band']/1e3:.1f}k", flush=True)
    print(f"\n== {root.name} @ {station}（n={len(rows)}）==")
    accum(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
