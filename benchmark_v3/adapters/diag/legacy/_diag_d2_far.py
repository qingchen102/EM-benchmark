# -*- coding: utf-8 -*-
# B2：把 gate_tool_view 的"假簇"分解成可归因的三类。
#
# gate 口径（oracle16.py:348-352）：
#   recall = 认领真源数 / 任一站可见(INR>=3dB)的真源数
#   far    = 1 - 认领真源数 / 工具输出簇总数
# D2 现状：146 簇 / 认领 55 / 可见真源 98 → recall 0.561, far 0.623
#
# 本脚本把"未认领的 91 个簇"拆成：
#   A. 幽灵簇    = 该样本 0 个可见真源，工具却出了簇（本该报空）
#   B. 过分割簇  = 与某个可见真源 ≤50m 都没到，但被去重规则挤掉的同源重复簇
#   C. 远距幻影  = 与任何可见真源都 >50m（真·无中生有）
# 并对比"命中簇"与"未命中簇"的四个可观测特征，看能否线性分开（B3 定阈依据）。
#
# 只读，不写数据集；只读/复用 _agent_cache。
import json
import os
from pathlib import Path
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rad_cls"))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402
from em_signal_simulator_v3.emitters import EMITTERS  # noqa: E402

ROOT = os.path.dirname(HERE)
D2 = os.path.join(ROOT, "dataset_d2")
MATCH_M = 50.0
NEAR_M = 50.0


def visible_gts(truth, full, allowed):
    """与 oracle16.gate_tool_view 完全同口径：任一 allowed 站 INR>=3dB 即计入。"""
    vis_count = np.zeros(len(truth["interferers"]), int)
    for x in full["stations"]:
        if x["receiver_id"] not in allowed:
            continue
        nz = 10 * np.log10(x["receiver"]["expected_output_noise_w"]) + 30
        for i_g in range(len(truth["interferers"])):
            eid = "interferer_%d" % (i_g + 1)
            src_x = next((y for y in x["sources"] if y["emitter_id"] == eid), None)
            if src_x is None:
                continue
            pr = 10 * np.log10(float(np.mean(src_x["mean_port_power_w"])) + 1e-30) + 30
            if pr - nz >= 3.0:
                vis_count[i_g] += 1
    return [g for i_g, g in enumerate(truth["interferers"])
            if vis_count[i_g] >= 1 and g["kind"] in set(EMITTERS)]


def main(n_samples=50):
    idx = json.loads(open(os.path.join(D2, "index.json"), encoding="utf-8").read())
    rows = []
    claimed_feats, far_feats, over_feats = [], [], []
    phantom_samples = 0
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = os.path.join(D2, sid, "capture")
        obs = json.loads(open(os.path.join(cap, "public", "observations.json"), encoding="utf-8").read())
        private = Path(cap) / "private"
        ns = obs["receivers"][0]["shape"][1]
        refs = rca._build_target_reference(private, ns)
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ctx = ct.ToolContext(capture_dir=os.path.join(cap, "public"),
                             cache_dir=os.path.join(D2, "_agent_cache", sid),
                             allowed_stations=allowed,
                             fs=obs["receivers"][0]["sample_rate_hz"],
                             fc=obs["receivers"][0]["center_frequency_hz"],
                             target_bw_normalized=0.25, aoi_center=aoi,
                             target_reference=refs)
        ct.set_context(ctx)
        truth = json.loads(open(os.path.join(private, "d1_truth.json"), encoding="utf-8").read())
        full = json.loads(open(os.path.join(private, "truth.json"), encoding="utf-8").read())
        gts = visible_gts(truth, full, allowed)

        try:
            clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        except Exception as exc:
            rows.append((sid, len(gts), -1, 0, 0, 0, "EXC:" + str(exc)[:30]))
            continue
        cand = clu.get("candidates") or []

        # 贪心最近认领（与 gate 同规则）
        pair = []
        for k, sp in enumerate(cand):
            p2 = np.asarray(sp["position_xy_m"])
            for i_g, g in enumerate(gts):
                d = float(np.linalg.norm(p2 - np.asarray(g["position_m"][:2])))
                if d < MATCH_M:
                    pair.append((d, k, i_g))
        pair.sort()
        used_k, used_g = set(), set()
        for d, k, i_g in pair:
            if k in used_k or i_g in used_g:
                continue
            used_k.add(k)
            used_g.add(i_g)
        n_claim = len(used_k)

        # 逐簇归类
        n_near_dup = n_phantom = n_remote = 0
        for k, sp in enumerate(cand):
            p2 = np.asarray(sp["position_xy_m"])
            dmin = min((float(np.linalg.norm(p2 - np.asarray(g["position_m"][:2]))) for g in gts),
                       default=1e9)
            f = (sp["deflation_db"], sp["perp_rms_m"], sp["max_du"],
                 sp["n_explained_stations"], sp["n_agreeing_triads"])
            if k in used_k:
                claimed_feats.append(f)
            elif not gts:
                n_phantom += 1
                far_feats.append(f)
            elif dmin < NEAR_M:
                n_near_dup += 1
                over_feats.append(f)
            else:
                n_remote += 1
                far_feats.append(f)
        if not gts and cand:
            phantom_samples += 1
        rows.append((sid, len(gts), len(cand), n_claim, n_near_dup, n_phantom + n_remote, ""))

    # ---- 汇总 ----
    good = [r for r in rows if r[2] >= 0]
    n_clusters = sum(r[2] for r in good)
    n_claim = sum(r[3] for r in good)
    n_dup = sum(r[4] for r in good)
    n_far = sum(r[5] for r in good)
    print("=" * 78)
    print(f"样本 {len(good)}  工具输出簇 {n_clusters}  认领 {n_claim}  → far={1 - n_claim / max(n_clusters, 1):.3f}")
    print(f"  可见真源总数      = {sum(r[1] for r in good)}")
    print("  --- 未认领簇分解 ---")
    print(f"  B 过分割（<50m 同源重复） = {n_dup:3d}  ({n_dup / max(n_clusters, 1) * 100:.1f}% 总簇, "
          f"{n_dup / max(n_clusters - n_claim, 1) * 100:.1f}% 未认领)")
    print(f"  A+C 真幻影（>50m 或无真源） = {n_far:3d}  ({n_far / max(n_clusters, 1) * 100:.1f}% 总簇, "
          f"{n_far / max(n_clusters - n_claim, 1) * 100:.1f}% 未认领)")
    print(f"  其中 0 可见真源却有簇的样本 = {phantom_samples}/{len(good)}")

    def stat(name, feats, i):
        if not feats:
            print(f"    {name:16s} n=0")
            return
        v = np.array([f[i] for f in feats], float)
        print(f"    {name:16s} n={len(v):3d}  中位={np.median(v):8.2f}  "
              f"P10={np.percentile(v, 10):8.2f}  P90={np.percentile(v, 90):8.2f}")

    for i, fname in enumerate(("deflation_db", "perp_rms_m", "max_du",
                               "n_explained_stations", "n_agreeing_triads")):
        print(f"  --- {fname} ---")
        stat("命中(认领)", claimed_feats, i)
        stat("过分割", over_feats, i)
        stat("幻影(A+C)", far_feats, i)

    print("=" * 78)
    print("逐样本：sid  可见真源  输出簇  认领  过分割  幻影")
    for r in rows:
        print(f"  {r[0]}  gt={r[1]:2d}  cand={r[2]:2d}  claim={r[3]:2d}  dup={r[4]:2d}  far={r[5]:2d}  {r[6]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
