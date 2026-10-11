"""只读：剥目标后残差 PSD 的绝对剖面（10 kHz 细粒度），判断 ±810k 的 47dB 台面是
目标残留、还是别的结构化能量（零 token）。

同时给出：
  - 原始 PSD 与残差 PSD 的绝对 dB（同一 1e-30 参考，可直接比）
  - 残差在目标带内 (-500~500k) 的能量占比
用法：python _diag_resid_abs.py d1_00001
"""
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

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

D1 = _V3 / "dataset_d1"


def main(sid: str) -> None:
    cap = D1 / sid / "capture" / "public"
    priv = D1 / sid / "capture" / "private"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    try:
        refs = rca._build_target_reference(priv, n)
    except Exception:                                        # noqa: BLE001
        refs = {}
    ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                         allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                         fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                         target_bw_normalized=0.25, aoi_center=[0, 0],
                         target_reference=refs)
    ct.set_context(ctx)
    occ = ct.estimate_band_occupancy_by_correlation("station_a")
    print(json.dumps(occ, indent=1, ensure_ascii=False)[:1200])
    iq = ct._station_iq_raw("station_a")
    resid = np.load(ctx.cache_dir / "resid_station_a.npy")
    print("iq shape", iq.shape, "resid shape", resid.shape,
          "resid/target ref keys:", list(refs.keys()))
    if refs:
        for k, v in refs.items():
            print("  ref", k, np.asarray(v).shape, "mean|.|=", float(np.mean(np.abs(v))))
    print("mean|iq|^2 = %.3e   mean|resid|^2 = %.3e   ratio = %.3f"
          % (float(np.mean(np.abs(iq) ** 2)), float(np.mean(np.abs(resid) ** 2)),
             float(np.mean(np.abs(resid) ** 2)) / float(np.mean(np.abs(iq) ** 2))))
    # 参考与 iq 的时域相关（station_a）
    if refs and "station_a" in refs:
        r = np.asarray(refs["station_a"])
        for el in range(min(3, iq.shape[0])):
            a = iq[el]
            b = r[el] if r.ndim > 1 else r
            L = min(len(a), len(b))
            c = abs(np.vdot(b[:L], a[:L])) / (np.linalg.norm(a[:L]) * np.linalg.norm(b[:L]) + 1e-30)
            print("  corr el%d = %.4f" % (el, float(c)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d1_00001")
