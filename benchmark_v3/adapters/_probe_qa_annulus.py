# -*- coding: utf-8 -*-
# 一次性探针：QA 约束 1（每个干扰 >=3 站 LoS 可见）的通过率
# 对"干扰相对目标的距离环"有多敏感？
#
# 背景：D3/D4 的监测网固定环绕 AOI 中心（stations_center），目标刻意放楼后
# （target_nlos），而干扰从"距目标 250–800m"的环里抽 → 天生继承目标遮蔽，
# QA 约束 1 实测一次通过率只有 34%（D1/D2 是 96%/100%）。
#
# 本脚本 monkeypatch Placer.interferer 的环参数，量出：
#   通过率 / 失败样本在 80 次重抽中达到的最好可见站数
# 从而判断"是放宽阈值"还是"放宽布点环"更有效。
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as G  # noqa: E402

occ = G.SceneOccupancy.load(G._PLATFORM / "priors" / "hangzhou_geometry" / "occupancy.npz")
placer = G.Placer(occ)
_orig_vis = G._n_visible_stations
_orig_ifw = G.Placer.interferer
BEST = {"max": -1}
CFG = {"mode": "ring", "lo": 250, "hi": 800}


def patched_vis(src, stations, occ_):
    v = _orig_vis(src, stations, occ_)
    if v > BEST["max"]:
        BEST["max"] = v
    return v


def patched_ifw(self, rng, tgt, d_low=400, d_high=900):
    if CFG["mode"] == "free":
        p = self.occ.free_point(rng, z_low=10, z_high=50)
        if p is None:
            return None
        x, y = self._clamp(p[0], p[1])
        if not self.occ.is_free(x, y) or G.dist((x, y), tgt[:2]) < 120:
            return None
        return (x, y, p[2])
    return _orig_ifw(self, rng, tgt, d_low=CFG["lo"], d_high=CFG["hi"])


G._n_visible_stations = patched_vis
G.Placer.interferer = patched_ifw

CONFIGS = [
    ("ring 250-800 (现状)", "ring", 250, 800),
    ("ring 400-900", "ring", 400, 900),
    ("ring 250-1500", "ring", 250, 1500),
    ("ring 400-1500", "ring", 400, 1500),
    ("ring 600-1500", "ring", 600, 1500),
    ("ring 600-2500", "ring", 600, 2500),
    ("自由布点(任意自由点)", "free", 0, 0),
]

DIFFS = [(sys.argv[1], int(sys.argv[2]))] if len(sys.argv) > 2 else [("d3", 61000), ("d4", 72000)]

for diff, seed0 in DIFFS:
    for name, mode, lo, hi in CONFIGS:
        CFG.update(mode=mode, lo=lo, hi=hi)
        npass = 0
        hist = Counter()
        for i in range(50):
            BEST["max"] = -1
            try:
                G.build_sample(i, seed0 + i, placer, diff)
                npass += 1
            except RuntimeError:
                hist[BEST["max"]] += 1
        print(f"  {diff} seed0={seed0}  {name:22s} 一次通过 {npass:2d}/50 "
              f"({npass * 2:3d}%)   失败样本最好可见站数 {dict(sorted(hist.items()))}")
        sys.stdout.flush()
