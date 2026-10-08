# -*- coding: utf-8 -*-
# 一次性探针 2：确认真实工具链路里 bfw 落盘用的变频量（carriers_hz）是否生效。
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rad_cls"))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402

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
    fs = obs["receivers"][0]["sample_rate_hz"]
    ct.set_context(ct.ToolContext(
        capture_dir=os.path.join(cap, "public"),
        cache_dir=os.path.join(D2, "_agent_cache", sid),
        allowed_stations=allowed,
        fs=fs, fc=obs["receivers"][0]["center_frequency_hz"],
        target_bw_normalized=0.25, aoi_center=aoi,
        target_reference=rca._build_target_reference(private, ns)))
    clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
    out = ct.extract_source_waveforms()
    cands = clu.get("candidates") or []
    print(f"{sid}  candidates={len(cands)}  carriers_hz={out.get('carriers_hz')}")
    for k, c in enumerate(cands):
        bc = 0.5 * (c["band_hz"][0] + c["band_hz"][1])
        print(f"   k={k} band_center={bc / 1e3:8.1f}kHz band="
              f"({c['band_hz'][0] / 1e3:.0f},{c['band_hz'][1] / 1e3:.0f})kHz")
    for k in range(len(cands)):
        p = Path(D2, "_agent_cache", sid, f"bfw_station_a_{k}.npy")
        if p.exists():
            w = np.load(p)
            P = np.abs(np.fft.fftshift(np.fft.fft(w))) ** 2
            f = np.fft.fftshift(np.fft.fftfreq(w.shape[0], 1 / fs))
            m = np.abs(f) <= 450e3
            print(f"   bfw[{k}] peak@ {f[m][np.argmax(P[m])] / 1e3:8.1f}kHz  "
                  f"n={w.shape[0]}  mtime={p.stat().st_mtime:.0f}")
