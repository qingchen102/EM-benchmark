# -*- coding: utf-8 -*-
# 一次性探针：量化 QA 约束 1 在 D3/D4 下的失败率与"可达上限"。
# 背景：d3 重生成卡在样本 26（6 轮 × 6 次内部重抽全败）。
# 假设：D3/D4 用 stations_center(固定环布) + target_nlos(目标在楼后)，
#       干扰却从"距目标 250–800m"里抽 → 天生继承目标的 NLoS 遮蔽，
#       "≥3 站 LoS" 几乎不可达（代码注释假设的 21% 是 D1/D2 几何下的数）。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as G  # noqa: E402

occ = G.SceneOccupancy.load(G._PLATFORM / "priors" / "hangzhou_geometry" / "occupancy.npz")
placer = G.Placer(occ)
_orig_vis = G._n_visible_stations
BEST = {"max": -1}


def patched_vis(src, stations, occ_):
    v = _orig_vis(src, stations, occ_)
    if v > BEST["max"]:
        BEST["max"] = v
    return v


G._n_visible_stations = patched_vis

PAIRS = [(sys.argv[1::2][k], int(sys.argv[2::2][k])) for k in range(len(sys.argv[1::2]))]
for diff, seed0 in (PAIRS or [("d3", 61000), ("d4", 72000)]):
    npass = 0
    fails = []
    best_hist = {}
    for i in range(50):
        BEST["max"] = -1
        try:
            G.build_sample(i, seed0 + i, placer, diff)
            npass += 1
        except RuntimeError as e:
            fails.append((i, str(e)[:58], BEST["max"]))
            best_hist[BEST["max"]] = best_hist.get(BEST["max"], 0) + 1
    print(f"=== {diff}  seed0={seed0} ===")
    print(f"  一次通过 {npass}/50   失败 {len(fails)}/50  ({len(fails) / 50 * 100:.0f}%)")
    print(f"  失败样本在 80 次重抽中达到的**最好** LoS 可见站数分布: "
          f"{dict(sorted(best_hist.items()))}")
    for i, msg, best in fails[:15]:
        print(f"    idx={i:2d}  best_vis={best}  {msg}")
