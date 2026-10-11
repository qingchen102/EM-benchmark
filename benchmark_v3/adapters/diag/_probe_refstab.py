# -*- coding: utf-8 -*-
"""探针：为什么 gate 的 GT 切片会「target reference not available for station 'station_a'」。

这 7 个样本（d2_00006/15/19/33/35/43/45）在 gate 冷缓存重建路径上被跳过、分母从 80 掉到 71，
而同一份数据热缓存路径分母是 80 —— 同一个 checkpoint 两套口径，属于测量卫生问题。
本脚本只读：列出目标链路文件与径数、truth.json 的 target path_count、实际 refs 的键、
以及 gate 私有缓存 / agent 缓存里有没有该样本的残差。

用法：python -u _probe_refstab.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # adapters（迁移后 HERE 下沉一级）
sys.path.insert(0, str(HERE / "rad_cls"))

import run_capture_agent as rca  # noqa: E402

ROOT = HERE.parent / "dataset_d2"
SIDS = ["d2_00000", "d2_00006", "d2_00015", "d2_00019", "d2_00033",
        "d2_00035", "d2_00043", "d2_00045"]

for sid in SIDS:
    cap = ROOT / sid / "capture"
    priv = cap / "private"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    ns = obs["receivers"][0]["shape"][1]
    truth = json.loads((priv / "truth.json").read_text(encoding="utf-8"))
    # truth.json 里每站每源的 path_count / geometry
    pc = {}
    for st in truth.get("stations", []):
        if st["receiver_id"] != "station_a":
            continue
        for s in st.get("sources", []):
            if s.get("emitter_id") == "target":
                pc = {"path_count": s.get("path_count"),
                      "geometry": s.get("geometry")}
    files = {}
    for f in sorted(priv.glob("channel_target_*.npz")):
        with np.load(f, allow_pickle=False) as d:
            files[f.name] = int(d["delays_s"].size)
    try:
        refs = rca._build_target_reference(priv, ns)
        keys = sorted(refs.keys())
    except Exception as exc:                                    # noqa: BLE001
        keys = f"EXC {type(exc).__name__}: {str(exc)[:70]}"
    gate_resid = (ROOT / "_gate_cache" / sid / "resid_station_a.npy").exists()
    agent_resid = (ROOT / "_agent_cache" / sid / "resid_station_a.npy").exists()
    print(f"{sid}: target_station_a={pc}")
    print(f"    files={files}")
    print(f"    refs_keys={keys}  gate_resid={gate_resid}  agent_resid={agent_resid}")
