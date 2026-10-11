"""临时只读：整通带残差下跑完整空间聚类，看 d2_00008 的真源能否交出。"""
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
    cap = D2 / sid / "capture" / "public"
    priv = D2 / sid / "capture" / "private"
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
    truth_pos = [np.asarray([e for e in scen["emitters"] if e["id"] == "target"][0]
                            ["position_m"], float)[:3]]
    truth_pos += [np.asarray(g["position_m"], float)[:3]
                  for g in truth.get("interferers", [])]

    # 预生成整通带残差并按站缓存到内存
    fq = np.fft.fftfreq(n, 1 / fs)
    mfull = np.abs(fq) <= 0.45 * fs
    resid_full = {}
    for st in allowed:
        if ref.get(st) is None:
            continue
        rf = np.asarray(ref[st], complex)
        if rf.ndim == 1:
            rf = rf[None, :]
        resid_full[st] = np.fft.ifft(np.fft.fft(ct._station_iq_raw(st), axis=1) * mfull,
                                     axis=1) - rf
    orig = ct._station_iq_raw
    ct._station_iq_raw = lambda s, on_residual=False: (
        resid_full[s] if (on_residual and s in resid_full) else orig(s, on_residual=on_residual))

    print(f"\n===== {sid}  (整通带残差) =====")
    for st in allowed:
        if st not in resid_full:
            print(f"  {st:10s} 无目标参考，跳过"); continue
        sc = ct._u_scan(st, 0.0, 0.9 * fs, on_residual=True)
        print(f"  {st:10s} rank={sc['rank']} peaks={sc['u_peaks']}")
    for tag, kw in (("现行(目标带残差)", {}), ("整通带残差", {})):
        ct._npy = None
        if tag.startswith("现行"):
            ct._station_iq_raw = orig          # 恢复：读盘上 band-limited 残差
        res = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        print(f"  -- {tag}: n_int={res.get('n_intersections')} bands={res.get('bands')}")
        for s in res.get("sources", []):
            p = np.array([s["position_xy_m"][0], s["position_xy_m"][1]])
            dd = sorted(float(np.linalg.norm(p - t[:2])) for t in truth_pos)
            print(f"       xy={s['position_xy_m']} triads={s['n_agreeing_triads']} "
                  f"→ 距最近真源 {dd[0]:.0f} m")
        if res.get("error"):
            print("       error:", res["error"])
        if tag.startswith("现行"):
            ct._station_iq_raw = lambda s, on_residual=False: (
                resid_full[s] if (on_residual and s in resid_full)
                else orig(s, on_residual=on_residual))


if __name__ == "__main__":
    for sid in (sys.argv[1:] or ["d2_00008"]):
        run(sid)
