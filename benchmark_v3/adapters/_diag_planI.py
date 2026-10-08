"""只读：最终候选方案 I —— 旧判据的带结构 + 每带两个判别量（零 token）。

方案 I：保留旧判据（D2 基线所依赖的宽带结构不动），给每个带加：
   * coherence_peak = 带内 over 最大处 ±10k 的 λ1/Σλ
   * coherence_inband = 通带内（|f|<=810k）扫描的 λ1/Σλ 最大值（±5k 窄窗）
   * keep = coherence_inband >= 0.6  ← 只踢"通带内根本没有点源"的带
     （D1 的 |f|>810k 失配能量带会被踢；D2 的真源带必然保留）

本脚本量它：D1 干净虚警 / D1 真源覆盖 / D2 真源覆盖 / D2 假带。

用法：python _diag_planI.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

D1 = _V3 / "dataset_d1"
D2 = _V3 / "dataset_d2"
RX_FRAC = 0.45 * 0.9              # 生成器"名义通带" ±810k


def plan_i(resid, fs, cv_win=5e3, cv_thr=0.6):
    rx_half = 0.45 * fs
    nom = RX_FRAC * fs
    peaks = ct._residual_peaks_legacy(resid, fs, rx_half)
    n = resid.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    inb = np.where((f >= -nom) & (f <= nom))[0]
    step = max(int(round(20e3 / (fs / n))), 1)
    best = (-1.0, 0.0)
    for ci in range(0, inb.size, step):
        fc = float(f[inb[ci]])
        c = ct._band_coherence(resid, fs, fc, cv_win)
        if np.isfinite(c) and c > best[0]:
            best = (c, fc)
    out = []
    for p in peaks:
        cp = ct._band_coherence(resid, fs, p["freq_offset_hz"], 10e3)
        out.append({**p, "coh_peak": round(float(cp), 3),
                    "coh_inband": round(best[0], 3), "coh_f": best[1],
                    "keep": bool(best[0] >= cv_thr)})
    return out


def evaluate(root: Path, sids, station, gate=True):
    real = fake = missed = clean_fa = 0
    detail = []
    for sid in sids:
        cap = root / sid / "capture" / "public"
        priv = root / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        if root is D2:
            nz = tr["noise"]["noise_floor_dbm"]
            ach = tr.get("achieved") or {}
            if not any((ach.get(f"interferer_{j+1}", {}).get(f"s{i}", {})
                        .get("pr_dbm", -999) - nz) >= 3.0
                       for j in range(len(ifw)) for i in range(len(allowed))):
                continue
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                     # noqa: BLE001
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=root / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        st = station if station in allowed else allowed[0]
        try:
            ct.estimate_band_occupancy_by_correlation(st)
            resid = np.load(ctx.cache_dir / f"resid_{st}.npy")
        except Exception:                                     # noqa: BLE001
            continue
        bands = plan_i(resid, fs)
        if gate:
            bands = [b for b in bands if b["keep"]]
        if not ifw:
            if bands:
                clean_fa += 1
                detail.append((sid, [(round(b["freq_offset_hz"] / 1e3),
                                      round(b["coh_inband"], 2)) for b in bands]))
            continue
        matched = set()
        for b in bands:
            ctr = 0.5 * (b["band_hz"][0] + b["band_hz"][1])
            hit = False
            for gi, g in enumerate(ifw):
                if abs(ctr - g["freq_offset_hz"]) <= 250e3:
                    matched.add(gi)
                    hit = True
            if hit:
                real += 1
            else:
                fake += 1
        missed += len(ifw) - len(matched)
    return real, fake, missed, clean_fa, detail


def main() -> None:
    idx1 = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    idx2 = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    s1 = [s["id"] for s in idx1["samples"][:12]]
    s2 = [s["id"] for s in idx2["samples"][:12]]
    print(f"{'方案':<34}{'D1虚警':>8}{'D1真带':>8}{'D1假带':>8}"
          f"{'D2真带':>8}{'D2假带':>8}{'D2漏源':>8}")
    for label, gate in (("旧判据（无门控）", False), ("方案I 通带内相干门控", True)):
        r1, f1, m1, c1, d1 = evaluate(D1, s1, "station_a", gate)
        r2, f2, m2, c2, d2 = evaluate(D2, s2, "station_b", gate)
        print(f"{label:<34}{c1:>8}{r1:>8}{f1:>8}{r2:>8}{f2:>8}{m2:>8}")
        if gate:
            print(f"    （D1 干净样本残留虚警明细：{d1}）")


if __name__ == "__main__":
    main()
