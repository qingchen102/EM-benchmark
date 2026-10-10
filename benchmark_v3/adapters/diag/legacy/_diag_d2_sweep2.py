# -*- coding: utf-8 -*-
# B4-c：把"候选过滤规则"的搜索变成离线可复算 —— 一次收集，任意规则秒级重算。
#
# 动机：`_diag_d2_sweep.py` 只能算 recall/far（不需要分类），
# 而 gate 真正卡的是 **cond**（认领对的条件分类准确率）。cond 依赖"哪些候选被认领"
# 与"每个候选的 CNN 预测"，两者都可离线重算——前提是把**每个候选**的预测
# 都算出来存下来（而不是只算被认领的那些）。
#
# 用法：
#   python _diag_d2_sweep2.py collect 50     # 跑工具链路，落 _diag_d2_sweep2.json
#   python _diag_d2_sweep2.py sweep          # 离线扫规则，秒级出 (recall, far, cond)
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
import oracle16                     # noqa: E402
from data_gen16 import CLS_IDX      # noqa: E402

ROOT = os.path.dirname(HERE)
D2 = os.path.join(ROOT, "dataset_d2")
OUT = os.path.join(HERE, "_diag_d2_sweep2.json")
MATCH_M = 50.0


def vis_gts(truth, full, allowed):
    vis = np.zeros(len(truth["interferers"]), int)
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
                vis[i_g] += 1
    return [(g, int(vis[i_g])) for i_g, g in enumerate(truth["interferers"])
            if vis[i_g] >= 1 and g["kind"] in CLS_IDX]


def collect(n_samples=50):
    model, _ = oracle16.load_model()
    idx = json.loads(open(os.path.join(D2, "index.json"), encoding="utf-8").read())
    data = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = os.path.join(D2, sid, "capture")
        obs = json.loads(open(os.path.join(cap, "public", "observations.json"), encoding="utf-8").read())
        private = Path(cap) / "private"
        ns = obs["receivers"][0]["shape"][1]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ct.set_context(ct.ToolContext(capture_dir=os.path.join(cap, "public"),
                                      cache_dir=os.path.join(D2, "_agent_cache", sid),
                                      allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                                      fs=obs["receivers"][0]["sample_rate_hz"],
                                      fc=obs["receivers"][0]["center_frequency_hz"],
                                      target_bw_normalized=0.25, aoi_center=aoi,
                                      target_reference=rca._build_target_reference(private, ns)))
        truth = json.loads((private / "d1_truth.json").read_text(encoding="utf-8"))
        full = json.loads((private / "truth.json").read_text(encoding="utf-8"))
        vg = vis_gts(truth, full, [r["receiver_id"] for r in obs["receivers"]])
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cand = clu.get("candidates") or []
        ct.extract_source_waveforms()
        rows = []
        for k, sp in enumerate(cand):
            wf = Path(D2) / "_agent_cache" / sid / ("bfw_station_a_%d.npy" % k)
            pred = None
            if wf.exists():
                pred = int(oracle16._pred(model, [np.load(wf).astype(np.complex64)])[0])
            rows.append({"k": k, "pos": list(map(float, sp["position_xy_m"])),
                         "perp": float(sp.get("perp_rms_m", 1e9)),
                         "triads": int(sp.get("n_agreeing_triads", 0)),
                         "defl": float(sp.get("deflation_db", 0.0)),
                         "nexp": int(sp.get("n_explained_stations", 0)),
                         "maxdu": float(sp.get("max_du", 9.9)),
                         "pred": pred})
        data.append({"sid": sid,
                     "cands": rows,
                     "gts": [{"pos": list(map(float, g["position_m"][:2])), "kind": g["kind"],
                              "cls": CLS_IDX[g["kind"]], "vis": int(n)} for g, n in vg]})
        print(f"  {sid} cand {len(rows)} gt {len(vg)}")
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"已落 {OUT}")


def _claim(cands, gts):
    """gate_tool_view 同口径：候选×真源距离<50m 的距离排序 + 双向贪心。"""
    pair = []
    for ci, c in enumerate(cands):
        p = np.asarray(c["pos"])
        for gi, g in enumerate(gts):
            d = float(np.linalg.norm(p - np.asarray(g["pos"])))
            if d < MATCH_M:
                pair.append((d, ci, gi))
    pair.sort()
    used_c, used_g = set(), set()
    out = []
    for d, ci, gi in pair:
        if ci in used_c or gi in used_g:
            continue
        used_c.add(ci)
        used_g.add(gi)
        out.append((ci, gi))
    return out


def evaluate(data, rule):
    tot_c = rec_n = 0
    gt_tot = sum(len(s["gts"]) for s in data)
    cond_c = cond_t = 0
    for s in data:
        cands = [c for c in s["cands"] if rule(c)]
        tot_c += len(cands)
        for ci, gi in _claim(cands, s["gts"]):
            rec_n += 1
            if cands[ci]["pred"] is not None:
                cond_t += 1
                cond_c += int(cands[ci]["pred"] == s["gts"][gi]["cls"])
    recall = rec_n / max(gt_tot, 1)
    far = 1.0 - rec_n / max(tot_c, 1)
    cond = cond_c / max(cond_t, 1)
    return recall, far, cond, tot_c


RULES = [
    ("基线（全 candidates）", lambda c: True),
    ("工具自带 sources 门控(nexp>=n_usable-1 & maxdu<=0.3)", lambda c: c["nexp"] >= 3 and c["maxdu"] <= 0.3),
    ("perp<=5", lambda c: c["perp"] <= 5),
    ("perp<=8", lambda c: c["perp"] <= 8),
    ("perp<=10", lambda c: c["perp"] <= 10),
    ("perp<=12", lambda c: c["perp"] <= 12),
    ("perp<=15", lambda c: c["perp"] <= 15),
    ("perp<=20", lambda c: c["perp"] <= 20),
    ("perp<=12 & triads>=5", lambda c: c["perp"] <= 12 and c["triads"] >= 5),
    ("perp<=15 & triads>=5", lambda c: c["perp"] <= 15 and c["triads"] >= 5),
    ("perp<=20 & triads>=5", lambda c: c["perp"] <= 20 and c["triads"] >= 5),
    ("perp<=12 & triads>=10", lambda c: c["perp"] <= 12 and c["triads"] >= 10),
    ("perp<=15 & triads>=10", lambda c: c["perp"] <= 15 and c["triads"] >= 10),
    ("perp<=20 & defl>=40", lambda c: c["perp"] <= 20 and c["defl"] >= 40),
    ("perp<=15 & defl>=45", lambda c: c["perp"] <= 15 and c["defl"] >= 45),
]


def sweep():
    data = json.loads(open(OUT, encoding="utf-8").read())
    print(f"{'规则':44s} {'recall':>8s} {'far':>7s} {'cond':>7s} {'簇数':>5s}")
    for name, rule in RULES:
        r, f, c, n = evaluate(data, rule)
        gate = "TOOLVIEW PASS" if (r >= 0.50 and c >= 0.50) else ""
        print(f"  {name:42s} {r:8.3f} {f:7.3f} {c:7.3f} {n:5d}  {gate}")
    # 每样本 perp 最小 top-K
    for K in (1, 2, 3):
        def rule(c, K=K):
            return c["_rank"] < K
        for s in data:
            order = sorted(range(len(s["cands"])), key=lambda i: s["cands"][i]["perp"])
            for rank, i in enumerate(order):
                s["cands"][i]["_rank"] = rank
        r, f, c, n = evaluate(data, rule)
        gate = "TOOLVIEW PASS" if (r >= 0.50 and c >= 0.50) else ""
        print(f"  {'每样本 perp 最小 top-%d' % K:42s} {r:8.3f} {f:7.3f} {c:7.3f} {n:5d}  {gate}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "collect":
        collect(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        sweep()
