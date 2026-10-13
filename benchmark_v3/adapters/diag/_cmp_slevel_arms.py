#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同一批样本上比较"真跑的观测档"（S0 单站 / S1 三站 / S2 四站）。

背景：历史上 D2–D4 只真跑过 S2，报告里的 S0/S1 块是同一份答案按可见性重打分
（见 diag/_audit_s_level_runs.py）。本脚本用于 2026-10-13 补跑的 S0/S1 独立作答：
对每个难度档，取**所有臂都作答了的同一批样本**，各自调用官方 `eval_dataset.evaluate`
（这样候选池化/意图/机理口径与注册报告完全一致），再并列打印。

用法：
    python -X utf8 diag/_cmp_slevel_arms.py                 # d2,d3,d4 全部可用臂
    python -X utf8 diag/_cmp_slevel_arms.py d2 d3           # 只看某些档
    python -X utf8 diag/_cmp_slevel_arms.py d2 --limit 25    # 只取前 25 个共同样本

注意：临时目录 benchmark_v3/_tmp_slevel_cmp/ 会在结束时删除。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ADP = HERE.parents[1]                 # benchmark_v3/adapters
V3 = ADP.parent                      # benchmark_v3
sys.path.insert(0, str(ADP))          # eval_dataset / capture_tools 都在这层

import eval_dataset as ed  # noqa: E402
CMP = V3 / "_tmp_slevel_cmp"

ARMS = ["S0", "S1", "S2"]            # 真跑档 → 目录 answers_<S>_agent_quota


def arm_dir(deck: str, s: str) -> Path:
    return V3 / f"dataset_{deck}" / f"answers_{s}_agent_quota"


def sample_ids(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("d*_*.json")):
        if p.name.startswith("_"):
            continue
        out.append(p.stem)
    return out


def gt_counts(root: Path, ids: list[str], stations: list[str]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for sid in ids:
        truth = ed.load_truth(root, sid)
        k = len(ed.gt_at_slevel(truth, stations))
        dist[k] = dist.get(k, 0) + 1
    return dict(sorted(dist.items()))


def score_one(deck: str, s: str, ids: list[str]) -> dict | None:
    root = V3 / f"dataset_{deck}"
    src = arm_dir(deck, s)
    if not src.is_dir():
        return None
    tmp = CMP / f"{deck}_{s}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    have = set(sample_ids(src))
    for sid in ids:
        if sid in have:
            shutil.copy2(src / f"{sid}.json", tmp / f"{sid}.json")
    log = src / "_run_log.json"
    if log.exists():
        shutil.copy2(log, tmp / "_run_log.json")
    rep = ed.evaluate(str(root), str(tmp))
    blk = rep.get(s)
    if not blk:
        return None
    return {"n": blk.get("n"), "rep": blk}


def fmt(v) -> str:
    if v is None:
        return "   -  "
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return f"{v}"


def main() -> None:
    raw = [a for a in sys.argv[1:] if not a.startswith("--")]
    decks = [d for a in raw for d in a.replace(",", " ").split()] or ["d2", "d3", "d4"]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    CMP.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("同一批样本上的观测档对比（真跑 S0 / S1 / S2，官方 eval_dataset 口径）")
    print("=" * 96)
    for deck in decks:
        root = V3 / f"dataset_{deck}"
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        subsets = idx["observation_subsets"]
        ids_per = {s: set(sample_ids(arm_dir(deck, s))) for s in ARMS}
        avail = [s for s in ARMS if ids_per[s]]
        if not avail:
            print(f"\n[{deck}] 没有任何 answers_*_agent_quota 目录")
            continue
        common = set.intersection(*[ids_per[s] for s in avail])
        ids = sorted(common)
        if limit:
            ids = ids[:limit]
        print(f"\n[{deck}] 可用臂 = {', '.join(avail)}｜共同已答样本 = {len(ids)}"
              f"（各臂已答 {', '.join(f'{s}:{len(ids_per[s])}' for s in ARMS)}）")
        if not ids:
            print("  尚无共同样本（作业还在跑）")
            continue
        # 共同样本上的常数基线（GT 源数众数占比）
        for s in avail:
            dist = gt_counts(root, ids, subsets[s])
            tot = sum(dist.values()) or 1
            top = max(dist.items(), key=lambda kv: kv[1])
            print(f"  · 基线 {s}（{len(subsets[s])} 站）：GT 源数分布 {dist}"
                  f"  众数占比 {top[1] / tot:.3f}")
        hdr = (f"  {'臂':<4}{'n':>4}{'检测计数':>10}{'±1':>8}{'体制':>8}{'载频':>8}"
               f"{'频率AE':>9}{'空间':>8}{'空间AE':>9}{'意图':>8}{'机理':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        rows = []
        for s in avail:
            got = score_one(deck, s, ids)
            rows.append((s, got))
            if got is None:
                print(f"  {s:<4} 评估失败")
                continue
            r = got["rep"]
            spatial = r.get("doa_hits") if s == "S0" else r.get("pos_hits")
            sae = r.get("doa_aes") if s == "S0" else r.get("pos_errs")
            print(f"  {s:<4}{got['n']:>4}{fmt(r.get('detection_hit')):>10}"
                  f"{fmt(r.get('detection_within1')):>8}{fmt(r.get('kind_hits')):>8}"
                  f"{fmt(r.get('freq_hits')):>8}{fmt(r.get('freq_aes')):>9}"
                  f"{fmt(spatial):>8}{fmt(sae):>9}"
                  f"{fmt(r.get('intent_accuracy')):>8}{fmt(r.get('mech_accuracy')):>8}")
        parts = []
        for s, got in rows:
            if got is None:
                continue
            mp = (got["rep"].get("mech_pairs") or 0) + (got["rep"].get("intent_pairs") or 0)
            parts.append(f"{s}:{mp}")
        print("  （意图/机理配对分母：" + ", ".join(parts) + "）")

    shutil.rmtree(CMP, ignore_errors=True)
    print("\n[done] 临时目录已清理")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
