# -*- coding: utf-8 -*-
# 一次性探针：直接验证 _dir_coh_support_center 是否生效（不写任何缓存）。
import json
import os
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rad_cls"))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402

print("capture_tools from:", ct.__file__)
D2 = os.path.join(os.path.dirname(HERE), "dataset_d2")
idx = json.loads(Path(D2, "index.json").read_text(encoding="utf-8"))
for s in idx["samples"][:4]:
    sid = s["id"]
    cap = os.path.join(D2, sid, "capture")
    obs = json.loads(Path(cap, "public", "observations.json").read_text(encoding="utf-8"))
    private = Path(cap) / "private"
    ns = obs["receivers"][0]["shape"][1]
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    ct.set_context(ct.ToolContext(
        capture_dir=os.path.join(cap, "public"),
        cache_dir=os.path.join(D2, "_agent_cache", sid),
        allowed_stations=allowed,
        fs=obs["receivers"][0]["sample_rate_hz"],
        fc=obs["receivers"][0]["center_frequency_hz"],
        target_bw_normalized=0.25, aoi_center=aoi,
        target_reference=rca._build_target_reference(private, ns)))
    ct.estimate_band_occupancy_by_correlation("station_a")
    clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
    cands = clu.get("candidates") or []
    print(f"{sid}  candidates={len(cands)}")
    for k, c in enumerate(cands):
        px = c["position_xy_m"]
        try:
            e = ct._dir_coh_support_center(px, sorted(allowed))
        except Exception as exc:
            e = f"EXC {type(exc).__name__}: {exc}"
        bc = 0.5 * (c["band_hz"][0] + c["band_hz"][1])
        es = e if isinstance(e, str) else (None if e is None else f"{e / 1e3:8.1f}")
        print(f"   k={k} band_center={bc / 1e3:8.1f}kHz  coh={es}  "
              f"band=({c['band_hz'][0] / 1e3:.0f},{c['band_hz'][1] / 1e3:.0f})kHz")
