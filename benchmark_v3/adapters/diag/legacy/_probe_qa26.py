# -*- coding: utf-8 -*-
# 一次性探针：查 d3_00026 的 QA 约束 1 为何 36 个 seed 全败。
# 假设：occ.interferer() 恒返回 None → 位置从不更新 → 80 次空转后抛错。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as G  # noqa: E402

occ = G.SceneOccupancy.load(G._PLATFORM / "priors" / "hangzhou_geometry" / "occupancy.npz")
placer = G.Placer(occ)

S = {"calls": 0, "none": 0, "ok": 0}
V = {"n": 0, "hist": {}}
_orig_if = G.Placer.interferer
_orig_vis = G._n_visible_stations


def patched_if(self, rng, tgt, d_low=400, d_high=900):
    S["calls"] += 1
    r = _orig_if(self, rng, tgt, d_low, d_high)
    if r is None:
        S["none"] += 1
    else:
        S["ok"] += 1
    return r


def patched_vis(src, stations, occ_):
    v = _orig_vis(src, stations, occ_)
    V["n"] += 1
    V["hist"][v] = V["hist"].get(v, 0) + 1
    return v


G.Placer.interferer = patched_if
G._n_visible_stations = patched_vis

for seed in (61031, 62031, 63031, 64031, 65031, 66031):
    S.update(calls=0, none=0, ok=0)
    V.update(n=0, hist={})
    try:
        spec = G.build_sample(26, seed, placer, "d3")
        print(f"seed={seed} OK  out={spec.get('out_name')}")
    except RuntimeError as e:
        print(f"seed={seed} FAIL {str(e)[:70]}")
    print(f"    interferer() calls={S['calls']:3d}  None={S['none']:3d}  ok={S['ok']:3d}"
          f"  |  可见站检查 {V['n']:3d} 次  LoS可见站数直方图={dict(sorted(V['hist'].items()))}")
