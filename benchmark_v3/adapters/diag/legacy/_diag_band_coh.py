"""只读：方案 H —— 旧判据的宽带，改为"带内任意处存在相干点源"即保留（零 token）。

动机（前序实测）：
  * D1 假峰 = 通带边缘 |f|>810k 的**失配能量**，跨阵元不相干（coh~0.2），
    agent 只要不看它们就 0 虚警；
  * D2 假峰 = RT **多径**（相干！coh~0.8），相干判据救不了 D2；
  * 但 D2 的真源带宽本身 300~800 kHz，所以"带内存在相干点源"这条判据对 D2 真源
    天然成立，同时把 D1 那些纯失配能量的带踢掉。

判据：对旧判据的每个 hot 段，在段内以 20 kHz 步长扫 ±20k 相干度，取 max。
输出：D1 干净虚警样本数 / 真源覆盖，D2 真源覆盖 / 假带数。

用法：python _diag_band_coh.py
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


def bands_h(resid, fs, rx_half, coh_thr=0.6, win=20e3, thr_db=8.0):
    """旧判据的 hot 段 + 段内最大相干度；返回带清单（含被相干门控保留/剔除标记）。"""
    n = resid.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    inb = np.where((f >= -rx_half) & (f <= rx_half))[0]
    psd = (np.abs(np.fft.fft(resid, axis=-1)) ** 2).mean(axis=0)
    db = 10 * np.log10(psd[inb] + 1e-30)
    floor = float(np.median(db))
    over = db - floor
    hot = over >= thr_db
    out = []
    k = 0
    while k < len(hot):
        if hot[k]:
            k2 = k
            while k2 + 1 < len(hot) and hot[k2 + 1]:
                k2 += 1
            if k2 - k + 1 >= 5:
                seg = over[k:k2 + 1]
                kc = k + int(np.argmax(seg))
                lo = float(f[inb[k]])
                hi = float(f[inb[k2]])
                best_c, best_cf = -1.0, float(f[inb[kc]])
                step = max(int(round(20e3 / (fs / n))), 1)
                for ci in range(k, k2 + 1, step):
                    fc = float(f[inb[ci]])
                    c = ct._band_coherence(resid, fs, fc, win)
                    if np.isfinite(c) and c > best_c:
                        best_c, best_cf = c, fc
                out.append({"peak": float(f[inb[kc]]), "over": float(seg.max()),
                            "band": (lo, hi), "coh_max": best_c,
                            "coh_f": best_cf,
                            "keep": bool(best_c >= coh_thr)})
            k = k2 + 1
        else:
            k += 1
    out.sort(key=lambda p: -p["over"])
    return out[:6]


def evaluate(root: Path, sids, station, use_gate: bool):
    real = fake = missed = clean_fa = 0
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
        bands = bands_h(resid, fs, 0.45 * fs)
        if use_gate:
            bands = [b for b in bands if b["keep"]]
        if not ifw:
            if bands:
                clean_fa += 1
            continue
        matched = set()
        for b in bands:
            ctr = 0.5 * (b["band"][0] + b["band"][1])
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
    return real, fake, missed, clean_fa


def main() -> None:
    idx1 = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    idx2 = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    s1 = [s["id"] for s in idx1["samples"][:12]]
    s2 = [s["id"] for s in idx2["samples"][:12]]
    print(f"{'配置':<32}{'D1虚警':>8}{'D1真带':>8}{'D1假带':>8}"
          f"{'D2真带':>8}{'D2假带':>8}{'D2漏源':>8}")
    for label, gate in (("旧判据（无门控）", False), ("旧判据+带内相干门控", True)):
        r1, f1, m1, c1 = evaluate(D1, s1, "station_a", gate)
        r2, f2, m2, c2 = evaluate(D2, s2, "station_b", gate)
        print(f"{label:<32}{c1:>8}{r1:>8}{f1:>8}{r2:>8}{f2:>8}{m2:>8}")


if __name__ == "__main__":
    main()
