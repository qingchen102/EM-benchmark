# -*- coding: utf-8 -*-
# B3：D2 候选门控阈值扫描。
#
# 依据 B2（_diag_d2_far.py）：146 个候选里 89 个是"纯幻影"（>50m 无真源）、
# 只有 2 个是过分割。可用特征只有两个：
#   perp_rms_m  : 命中中位 3.9 vs 幻影中位 21.2
#   triads      : 命中中位  21 vs 幻影中位   8
# 本脚本先落盘每个候选的特征+到最近真源的距离（_diag_d2_cands.json），
# 再离线扫阈值，输出 (recall, far, n_clusters) 取舍表。
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
from em_signal_simulator_v3.emitters import EMITTERS  # noqa: E402

ROOT = os.path.dirname(HERE)
D2 = os.path.join(ROOT, "dataset_d2")
CACHE_JSON = os.path.join(HERE, "_diag_d2_cands.json")
MATCH_M = 50.0


def visible_gts(truth, full, allowed):
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


def collect(n_samples=50):
    idx = json.loads(open(os.path.join(D2, "index.json"), encoding="utf-8").read())
    recs = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = os.path.join(D2, sid, "capture")
        obs = json.loads(open(os.path.join(cap, "public", "observations.json"), encoding="utf-8").read())
        private = Path(cap) / "private"
        ns = obs["receivers"][0]["shape"][1]
        refs = rca._build_target_reference(private, ns)
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ct.set_context(ct.ToolContext(capture_dir=os.path.join(cap, "public"),
                                      cache_dir=os.path.join(D2, "_agent_cache", sid),
                                      allowed_stations=allowed,
                                      fs=obs["receivers"][0]["sample_rate_hz"],
                                      fc=obs["receivers"][0]["center_frequency_hz"],
                                      target_bw_normalized=0.25, aoi_center=aoi,
                                      target_reference=refs))
        truth = json.loads((private / "d1_truth.json").read_text(encoding="utf-8"))
        full = json.loads((private / "truth.json").read_text(encoding="utf-8"))
        gts = visible_gts(truth, full, allowed)
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cand = clu.get("candidates") or []
        src_keys = {tuple(round(float(x), 3) for x in sp["position_xy_m"]) for sp in (clu.get("sources") or [])}
        gpos = [np.asarray(g["position_m"][:2], float) for g in gts]
        cs = []
        for sp in cand:
            p = np.asarray(sp["position_xy_m"], float)
            ds = [float(np.linalg.norm(p - g)) for g in gpos]
            cs.append({"perp": float(sp["perp_rms_m"]), "triads": int(sp["n_agreeing_triads"]),
                       "defl": float(sp["deflation_db"]), "nexp": int(sp["n_explained_stations"]),
                       "dmin": (min(ds) if ds else 1e9),
                       "gi": (int(np.argmin(ds)) if ds else -1),
                       "insrc": tuple(round(float(x), 3) for x in sp["position_xy_m"]) in src_keys})
        recs.append({"sid": sid, "gts": len(gts),
                     "n_usable": int(clu["support_threshold"]["n_usable_stations"]),
                     "cands": cs})
        print(f"  {sid}  gt={len(gts)} cand={len(cand)} src={sum(1 for c in cs if c['insrc'])}")
    json.dump(recs, open(CACHE_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    return recs


def evaluate(recs, rule):
    rn = rd = nclu = 0
    for r in recs:
        rd += r["gts"]
        keep = [c for c in r["cands"] if rule(c, r)]
        nclu += len(keep)
        pair = sorted((c["dmin"], k, c["gi"]) for k, c in enumerate(keep) if c["dmin"] < MATCH_M)
        used_k, used_g = set(), set()
        for d, k, gi in pair:
            if k in used_k or gi in used_g:
                continue
            used_k.add(k)
            used_g.add(gi)
        rn += len(used_k)
    return rn / max(rd, 1), 1 - rn / max(nclu, 1), nclu, rn, rd


def main():
    recs = json.load(open(CACHE_JSON, encoding="utf-8")) if os.path.exists(CACHE_JSON) else collect()
    if not isinstance(recs[0]["cands"], list):
        recs = collect()
    print("=" * 86)

    def show(tag, rule):
        rc, far, nclu, rn, rd = evaluate(recs, rule)
        mark = "PASS" if (far < 0.5 and rc >= 0.5) else "    "
        print(f"  {mark} {tag:34s} recall={rc:.3f} ({rn:3d}/{rd})  far={far:.3f}  簇={nclu:3d}")

    show("基线：全用 candidates", lambda c, r: True)
    show("工具自带 sources 门控", lambda c, r: c["insrc"])
    print("  --- perp_rms_m <= T ---")
    for T in (5, 8, 10, 12, 15, 20, 25, 30, 40):
        show(f"perp<={T}", lambda c, r, T=T: c["perp"] <= T)
    print("  --- triads >= T ---")
    for T in (3, 5, 8, 10, 12, 15, 20, 30):
        show(f"triads>={T}", lambda c, r, T=T: c["triads"] >= T)
    print("  --- 组合 perp<=A 且 triads>=B ---")
    for A in (8, 12, 15, 20, 25, 30, 40):
        for B in (5, 8, 10, 15, 20):
            show(f"perp<={A} & triads>={B}", lambda c, r, A=A, B=B: c["perp"] <= A and c["triads"] >= B)
    print("  --- 每样本只留 perp 最小的前 K 个 ---")
    for K in (1, 2, 3):
        show(f"top-K by perp, K={K}", lambda c, r, K=K:
             c in sorted(r["cands"], key=lambda z: z["perp"])[:K])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
