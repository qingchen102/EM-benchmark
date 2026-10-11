"""临时只读：D2 全量估算——解除残差带限后，真源 u 峰的可召回率增益。"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct  # noqa: E402

D2 = _V3 / "dataset_d2"
TOL = 0.05


def main(limit: int, mode: str) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    import run_capture_agent as rca
    tot = hit_band = hit_full = hit_slice = 0
    per_band_limit = {}
    for s in idx["samples"][:limit]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        try:
            ref = rca._build_target_reference(priv, n)
        except Exception:                                   # noqa: BLE001
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25,
                             aoi_center=[0, 0], target_reference=ref)
        ct.set_context(ctx)
        truth = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = truth["noise"]["noise_floor_dbm"]
        ach = truth.get("achieved") or {}
        fq = np.fft.fftfreq(n, 1 / fs)
        mband = np.abs(fq) <= 500e3
        mfull = np.abs(fq) <= 0.45 * fs
        for st in allowed:
            sk = f"s{allowed.index(st)}"
            iq = ct._station_iq_raw(st)
            if ref.get(st) is None:
                continue                                    # 该站目标 0 径，无参考
            rf = np.asarray(ref[st], complex)
            if rf.ndim == 1:
                rf = rf[None, :]
            r_band = np.fft.ifft(np.fft.fft(iq, axis=1) * mband, axis=1) \
                - np.fft.ifft(np.fft.fft(rf, axis=1) * mband, axis=1)
            r_full = np.fft.ifft(np.fft.fft(iq, axis=1) * mfull, axis=1) - rf
            peaks = {}
            for tag, r in (("band", r_band), ("full", r_full)):
                orig = ct._station_iq_raw
                ct._station_iq_raw = lambda x, on_residual=False, _o=orig, _r=r: (
                    _r if on_residual else _o(x, on_residual=False))
                try:
                    sc = ct._u_scan(st, 0.0, 0.9 * fs, on_residual=True)
                finally:
                    ct._station_iq_raw = orig
                peaks[tag] = [u for u, _ in sc["u_peaks"]]
            pos, axis = ct._station_geometry(st)
            for j, g in enumerate(truth.get("interferers", [])):
                if ach.get(f"interferer_{j+1}", {}).get(sk, {}).get("pr_dbm", -999) - nz < 3.0:
                    continue                                   # 该站不可见，不计
                p3 = np.asarray(g["position_m"], float)[:3]
                d = p3 - pos
                u_t = float(axis @ d) / (np.linalg.norm(d) + 1e-12)
                # C：整通带残差 + 该源 own band 切片（分带测向上限）
                orig = ct._station_iq_raw
                ct._station_iq_raw = lambda x, on_residual=False, _o=orig, _r=r_full: (
                    _r if on_residual else _o(x, on_residual=False))
                try:
                    sc_c = ct._u_scan(st, float(g["freq_offset_hz"]),
                                      float(g["bandwidth_hz"]), on_residual=True)
                finally:
                    ct._station_iq_raw = orig
                u_c = [u for u, _ in sc_c["u_peaks"]]
                hb = min((abs(u_t - u) for u in peaks["band"]), default=9) < TOL
                hf = min((abs(u_t - u) for u in peaks["full"]), default=9) < TOL
                hc = min((abs(u_t - u) for u in u_c), default=9) < TOL
                tot += 1
                hit_band += int(hb)
                hit_full += int(hf)
                hit_slice += int(hc)
                key = ("band" if hb else "miss") + "->" + ("full" if hf else "miss") \
                    + "->" + ("slice" if hc else "noslice")
                per_band_limit[key] = per_band_limit.get(key, 0) + 1
    print(f"\n[{mode}] 可见干扰×站 = {tot}")
    print(f"  A 现行(目标带残差+宽带扫描)   : {hit_band}/{tot} = {hit_band/max(tot,1):.0%}")
    print(f"  B 整通带残差 + 宽带扫描       : {hit_full}/{tot} = {hit_full/max(tot,1):.0%}")
    print(f"  C 整通带残差 + 源自身带切片   : {hit_slice}/{tot} = {hit_slice/max(tot,1):.0%}")
    for k in sorted(per_band_limit):
        print(f"    {k:24s} {per_band_limit[k]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, "D2")
