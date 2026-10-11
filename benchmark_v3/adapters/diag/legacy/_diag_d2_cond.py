# -*- coding: utf-8 -*-
# B4-a：D2 tool-view gate 卡在 cond=0.473 —— 到底是哪类被认错？
#
# oracle16.py:362 的通过判定是 `recall >= 0.50 and cond >= 0.50`；
# 假簇率 FAR 只是打印项（不在判定里）。所以 D2 的阻塞点是**条件分类准确率**：
# 认领成功的 55 个候选里，工具落盘的 bfw_station_a_<k>.npy（波束形成后波形）
# 喂给 16 类 CNN，只有 26 个判对。
#
# 本脚本按 gate_tool_view 同口径复算认领对，并导出每一对的
# (GT 类 → 预测类) 混淆，找出系统性错配。
import json
import os
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from collections import Counter
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rad_cls"))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402
import oracle16                     # noqa: E402
from data_gen16 import CLS_IDX      # noqa: E402
from em_signal_simulator_v3.emitters import EMITTERS  # noqa: E402

ROOT = os.path.dirname(HERE)
D2 = os.path.join(ROOT, "dataset_d2")
OUT = os.path.join(HERE, "_diag_d2_cond.json")
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
    return [(g, int(vis_count[i_g])) for i_g, g in enumerate(truth["interferers"])
            if vis_count[i_g] >= 1 and g["kind"] in CLS_IDX]


def main(n_samples=50):
    model, _classes = oracle16.load_model()
    idx = json.loads(open(os.path.join(D2, "index.json"), encoding="utf-8").read())
    rows = []
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
        vg = visible_gts(truth, full, allowed)
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cand = clu.get("candidates") or []
        ct.extract_source_waveforms()          # 落盘 bfw_station_a_<k>.npy
        pair = []
        for k, sp in enumerate(cand):
            p2 = np.asarray(sp["position_xy_m"])
            for i_g, (g, _n) in enumerate(vg):
                d = float(np.linalg.norm(p2 - np.asarray(g["position_m"][:2])))
                if d < MATCH_M:
                    pair.append((d, k, i_g))
        pair.sort()
        used_k, used_g = set(), set()
        n_ok = n_tot = 0
        for d, k, i_g in pair:
            if k in used_k or i_g in used_g:
                continue
            used_k.add(k)
            used_g.add(i_g)
            wf = Path(D2) / "_agent_cache" / sid / ("bfw_station_a_%d.npy" % k)
            if not wf.exists():
                rows.append({"sid": sid, "k": k, "gt": vg[i_g][0]["kind"], "pred": None,
                             "d": d, "vis": vg[i_g][1], "perp": cand[k]["perp_rms_m"],
                             "triads": cand[k]["n_agreeing_triads"],
                             "defl": cand[k]["deflation_db"]})
                continue
            wk = np.load(wf).astype(np.complex64)
            pr = int(oracle16._pred(model, [wk])[0])
            gt = vg[i_g][0]["kind"]
            ok = int(pr == CLS_IDX[gt])
            n_ok += ok
            n_tot += 1
            rows.append({"sid": sid, "k": k, "gt": gt, "pred": int(pr),
                         "pred_name": (list(CLS_IDX)[pr] if pr < len(CLS_IDX) else str(pr)),
                         "ok": ok, "d": d, "vis": vg[i_g][1],
                         "perp": cand[k]["perp_rms_m"], "triads": cand[k]["n_agreeing_triads"],
                         "defl": cand[k]["deflation_db"]})
        print(f"  {sid}  认领 {len(used_k)}  cond {n_ok}/{n_tot}")
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

    scored = [r for r in rows if r.get("pred") is not None]
    print("=" * 78)
    print(f"认领对有 bfw 的 {len(scored)}，无 bfw 的 {len(rows) - len(scored)}")
    print(f"cond = {sum(r['ok'] for r in scored)}/{len(scored)}"
          f" = {sum(r['ok'] for r in scored) / max(len(scored), 1):.3f}")
    conf = Counter((r["gt"], r["pred_name"]) for r in scored if not r["ok"])
    print("--- 错配（GT 类 → 预测类）计数 ---")
    for (g, p), n in conf.most_common(20):
        print(f"  {n:3d}  {g:18s} → {p}")
    print("--- 按 GT 类 ---")
    by = {}
    for r in scored:
        o, t = by.get(r["gt"], (0, 0))
        by[r["gt"]] = (o + r["ok"], t + 1)
    for g, (o, t) in sorted(by.items()):
        print(f"  {g:18s} {o:2d}/{t:2d}")
    print("--- 判错的对，其几何特征 vs 判对 ---")
    for name, key in (("perp_rms_m", "perp"), ("triads", "triads"), ("defl", "defl"), ("visible_stations", "vis")):
        okf = [r[key] for r in scored if r["ok"]]
        baf = [r[key] for r in scored if not r["ok"]]
        f = lambda v: f"中位={np.median(v):7.2f} P10={np.percentile(v, 10):7.2f} P90={np.percentile(v, 90):7.2f}" if v else "n=0"
        print(f"  {name:17s} 判对(n={len(okf):2d}) {f(okf)}   判错(n={len(baf):2d}) {f(baf)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
