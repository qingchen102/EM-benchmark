"""临时只读：分带测向 vs 宽带扫描 —— 真源 u 是否成为 top-1 峰。

比"峰表里是否出现"更严格的口径：MUSIC 的**最强峰**是否落在真源 u 的容差内。
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

import capture_tools as ct  # noqa: E402

D2 = _V3 / "dataset_d2"
TOL = 0.05


def run(n_samples: int) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    import run_capture_agent as rca
    tot = t1_wide = t1_band = t1_band_tight = 0
    any_wide = any_band = 0
    for s in idx["samples"][:n_samples]:
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
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=ref)
        ct.set_context(ctx)
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        for st in allowed:
            sk = f"s{allowed.index(st)}"
            pos, axis = ct._station_geometry(st)
            try:
                wide = ct._u_scan_cached(st, 0.0, 0.9 * fs, on_residual=True)
            except Exception:
                continue
            w1 = wide["u_peaks"][0][0] if wide["u_peaks"] else None
            wups = [u for u, _ in wide["u_peaks"]]
            for j, g in enumerate(tr.get("interferers", [])):
                if ach.get(f"interferer_{j+1}", {}).get(sk, {}).get("pr_dbm", -999) - nz < 3.0:
                    continue
                p = np.asarray(g["position_m"], float)[:3]
                d = p - pos
                u = float((d / np.linalg.norm(d)) @ axis)
                # 源自身频带（GT 带宽，±50% 余量）切片测向
                bw = float(g["bandwidth_hz"])
                try:
                    sc = ct._u_scan_cached(st, float(g["freq_offset_hz"]), bw, on_residual=True)
                except Exception:
                    continue
                b1 = sc["u_peaks"][0][0] if sc["u_peaks"] else None
                bups = [uu for uu, _ in sc["u_peaks"]]
                tot += 1
                t1_wide += int(w1 is not None and abs(w1 - u) < TOL)
                t1_band += int(b1 is not None and abs(b1 - u) < TOL)
                any_wide += int(min((abs(u - x) for x in wups), default=9) < TOL)
                any_band += int(min((abs(u - x) for x in bups), default=9) < TOL)
    print(f"\n== 可见干扰×站 = {tot}（前 {n_samples} 样本）==")
    print(f"  宽带扫描：真源 u 出现在峰表任意位置 {any_wide}/{tot} = {any_wide/max(tot,1):.0%}")
    print(f"           真源 u 是**最强峰**        {t1_wide}/{tot} = {t1_wide/max(tot,1):.0%}")
    print(f"  分带测向：真源 u 出现在峰表任意位置 {any_band}/{tot} = {any_band/max(tot,1):.0%}")
    print(f"           真源 u 是**最强峰**        {t1_band}/{tot} = {t1_band/max(tot,1):.0%}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
