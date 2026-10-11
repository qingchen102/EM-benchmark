"""只读探针：D1 的 `estimate_position` 到底还出不出位置（零 token）。

背景：10-08 修完虚警后，agent 的 detection=1.0 / freq=1.0 / doa=0.75，但
S1/S2 的 pos=0.0 —— 因为答案里 `position_m` 恒为 [0,0]。09-27 的旧运行同口径
是 0.61（24/24 条都有非零位置）。本脚本直接调工具，判定是"工具坏了"还是
"agent 拿到结果没写进去"。

用法：python _diag_d1_position.py [n]
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


def main(n_samples: int) -> None:
    idx = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    n_ok = n_tot = 0
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                   # noqa: BLE001
            refs = {}
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D1 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        if not ifw:
            continue
        g = ifw[0]
        f0 = float(g.get("freq_offset_hz", 0))
        bw = float(g.get("bandwidth_hz", 60e3)) or 60e3
        print(f"\n=== {sid}  GT {g.get('kind')} f={f0/1e3:.0f}k bw={bw/1e3:.0f}k"
              f" 真位置={np.round(np.asarray(g.get('position_m'), float)[:2], 0)} ===")
        for label, kw in (("定带(默认)", {}),
                          ("on_residual", {"on_residual": True}),
                          ("整通带", {"freq_offset_hz": f0, "bandwidth_hz": max(bw, 400e3)})):
            try:
                r = ct.estimate_position(freq_offset_hz=f0, bandwidth_hz=bw, **kw)
            except Exception as exc:                        # noqa: BLE001
                print(f"  [{label}] 异常 {exc}")
                continue
            if isinstance(r, dict) and r.get("error"):
                print(f"  [{label}] error = {r['error']}")
                continue
            hyps = (r or {}).get("hypotheses") or []
            n_tri = (r or {}).get("n_triads") or (r or {}).get("n_valid_triads")
            print(f"  [{label}] 假设 {len(hyps)} 条  n_triads={n_tri}  keys={sorted((r or {}).keys())}")
            for h in hyps[:3]:
                print(f"      {np.round(np.asarray(h.get('position_xy_m'), float), 0)}"
                      f"  agree={h.get('n_agreeing_triads')} rms={h.get('perp_rms_m')}")
        n_tot += 1
        n_ok += 1

    print(f"\n处理样本 {n_ok}/{n_tot}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
