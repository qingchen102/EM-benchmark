"""临时只读：把 agent 分数变化分解成「配对口径」与「阈值放宽」两部分。

对同一批 50 样本答案（answers_S2_agent_full）算 2×2：
  配对 = {旧:贪心频偏 | 新:位置优先+频偏兜底}
  阈值 = {注册口径: pos 50m / freq 40kHz | 放宽: pos 300m / freq 200kHz}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parent.parent
D2 = V3 / "dataset_d2"
ANS = D2 / "answers_S2_agent_full"


def gt_visible(g, ach, nz):
    eid = None
    return any((ach or {}).get(g, {}).get(f"s{i}", {}).get("pr_dbm", -999) - nz >= 3.0
               for i in range(4))


def match_old(pred, gt_list, gate_mhz=0.45):
    """旧口径：贪心频偏最近（含门槛）。"""
    pairs, used = [], set()
    order = sorted(range(len(pred)), key=lambda i: abs(pred[i].get("freq_offset_hz", 0)))
    for i in order:
        best, bd = None, 1e9
        for j, g in enumerate(gt_list):
            if j in used:
                continue
            d = abs(pred[i].get("freq_offset_hz", 0) - g["freq_offset_hz"]) / 2e6
            if d <= gate_mhz and d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            pairs.append((pred[i], gt_list[best]))
    return pairs


def match_new(pred, gt_list, pos_gate=50.0, freq_gate_mhz=0.45):
    """新口径：位置优先（双侧有位置且 ≤50m）→ 频偏兜底。"""
    pairs, used_p, used_g = [], set(), set()
    cands = []
    for i, p in enumerate(pred):
        if "position_m" not in p:
            continue
        for j, g in enumerate(gt_list):
            d = float(np.hypot(p["position_m"][0] - g["position_m"][0],
                               p["position_m"][1] - g["position_m"][1]))
            if d <= pos_gate:
                cands.append((d, i, j))
    for d, i, j in sorted(cands):
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((pred[i], gt_list[j]))
    rest = [i for i in range(len(pred)) if i not in used_p]
    rem = [j for j in range(len(gt_list)) if j not in used_g]
    for i in sorted(rest, key=lambda k: abs(pred[k].get("freq_offset_hz", 0))):
        best, bd = None, 1e9
        for j in rem:
            if j in used_g:
                continue
            d = abs(pred[i].get("freq_offset_hz", 0) - gt_list[j]["freq_offset_hz"]) / 2e6
            if d <= freq_gate_mhz and d < bd:
                best, bd = j, d
        if best is not None:
            used_g.add(best)
            pairs.append((pred[i], gt_list[best]))
    return pairs


def main() -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    preds, gts = {}, {}
    for s in idx["samples"]:
        sid = s["id"]
        f = ANS / f"{sid}.json"
        if not f.exists():
            continue
        preds[sid] = json.loads(f.read_text(encoding="utf-8"))
        priv = D2 / sid / "capture" / "private"
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        vis = []
        for j, g in enumerate(tr.get("interferers", [])):
            eid = f"interferer_{j+1}"
            if gt_visible(eid, ach, nz):
                vis.append(g)
        gts[sid] = vis
    print(f"样本 {len(preds)}")
    rows = []
    for tag_m, mf in (("旧配对(频偏)", match_old), ("新配对(位置优先)", match_new)):
        for tag_t, ptol, ftol in (("注册口径 50m/40k", 50.0, 0.02),
                                  ("放宽口径 300m/200k", 300.0, 200e3 / 2e6)):
            ph = ph_n = fh = fh_n = kh = kh_n = 0
            for sid, ans in preds.items():
                gt = gts[sid]
                pr = ans.get("interferers") or []
                for p, g in mf(pr, gt):
                    pe = float(np.hypot(p["position_m"][0] - g["position_m"][0],
                                        p["position_m"][1] - g["position_m"][1])) if "position_m" in p else 1e9
                    df = abs(p.get("freq_offset_hz", 0) - g["freq_offset_hz"]) / 2e6
                    ph += int(pe <= ptol); ph_n += 1
                    fh += int(df <= ftol); fh_n += 1
                    kh += int(str(p.get("kind", "")).strip() == g["kind"]); kh_n += 1
            rows.append((tag_m, tag_t, ph, ph_n, fh, fh_n, kh, kh_n))
    print(f"\n{'配对':<16}{'阈值':<20}{'pos_hits':>16}{'freq_hits':>16}{'kind_hits':>16}")
    for r in rows:
        tag_m, tag_t, ph, ph_n, fh, fh_n, kh, kh_n = r
        print(f"{tag_m:<16}{tag_t:<20}"
              f"{ph}/{ph_n} = {100*ph/max(ph_n,1):4.1f}%"
              f"{fh}/{fh_n} = {100*fh/max(fh_n,1):4.1f}%"
              f"{kh}/{kh_n} = {100*kh/max(kh_n,1):4.1f}%")


if __name__ == "__main__":
    main()
