"""临时只读 A/B：残差带限(±target_bw/2) vs 整通带(±0.45fs) 对源可辨识性的影响。"""
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


def run(sid: str) -> None:
    priv = D2 / sid / "capture" / "private"
    cap = D2 / sid / "capture" / "public"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    fc = float(obs["receivers"][0]["center_frequency_hz"])
    n = int(obs["receivers"][0]["shape"][1])
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    import run_capture_agent as rca
    ref = rca._build_target_reference(priv, n)
    ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                         allowed_stations=allowed, fs=fs, fc=fc,
                         target_bw_normalized=0.25, aoi_center=aoi, target_reference=ref)
    ct.set_context(ctx)
    truth = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    scen = json.loads((priv / "scenario.json").read_text(encoding="utf-8"))
    tgt_pos = np.asarray([e for e in scen["emitters"] if e["id"] == "target"][0]
                         ["position_m"], float)[:3]
    srcs = [("target", tgt_pos)]
    for j, g in enumerate(truth.get("interferers", [])):
        srcs.append((f"ifw{j+1}/{g['kind']}", np.asarray(g["position_m"], float)[:3]))
    f = np.fft.fftfreq(n, 1 / fs)

    print(f"\n===== {sid} =====")
    for st in allowed:
        iq = ct._station_iq_raw(st)
        rf = np.asarray(ref.get(st), complex)
        if rf.ndim == 1:
            rf = rf[None, :]
        # 目标剥离质量
        a = iq[0] - iq[0].mean()
        b = rf[0] - rf[0].mean()
        corr = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
        # 整通带残差
        mfull = np.abs(f) <= 0.45 * fs
        resid_full = np.fft.ifft(np.fft.fft(iq, axis=1) * mfull, axis=1) - rf
        # 目标带残差（现行口径）
        mband = np.abs(f) <= 500e3
        resid_band = np.fft.ifft(np.fft.fft(iq, axis=1) * mband, axis=1) \
            - np.fft.ifft(np.fft.fft(rf, axis=1) * mband, axis=1)
        pos, axis = ct._station_geometry(st)
        ups_band = [u for u, _ in ct._u_scan(st, 0.0, 2 * 0.45 * fs * 0.98,
                                             on_residual=True)["u_peaks"]] \
            if (D2 / "_agent_cache" / sid / f"resid_{st}.npy").exists() else []
        # 用整通带残差重跑 u 扫描
        orig = ct._station_iq_raw
        ct._station_iq_raw = lambda s, on_residual=False, _o=orig, _r=resid_full: (
            _r if on_residual and s == st else _o(s, on_residual=on_residual))
        try:
            scan_full = ct._u_scan(st, 0.0, 2 * 0.45 * fs * 0.98, on_residual=True)
        finally:
            ct._station_iq_raw = orig
        ups_full = [u for u, _ in scan_full["u_peaks"]]

        def rec(u, ups):
            return min((abs(u - uu) for uu in ups), default=9.0)
        line = []
        for name, p3 in srcs:
            d = p3 - pos
            u_t = float(axis @ d) / (np.linalg.norm(d) + 1e-12)
            line.append(f"{name[:14]} u={u_t:+.3f} Δband={rec(u_t, ups_band):.3f} Δfull={rec(u_t, ups_full):.3f}")
        print(f" {st:10s} corr(iq,ref)={corr:.3f}  现行峰={[round(u,2) for u in ups_band]}")
        print(f"            整通带峰={[round(u,2) for u in ups_full]} rank={scan_full['rank']}")
        for L in line:
            print("            ", L)


if __name__ == "__main__":
    for sid in (sys.argv[1:] or ["d2_00008"]):
        run(sid)
