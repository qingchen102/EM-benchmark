"""从 _diag_d2_cluster_ab.json 里算：仪器自己的 K（聚类 sources 数）与 GT 源数的精确匹配率，
OLD（加噪前数据）与 NEW（加噪后数据）各一行；并列出 K≠GT 的样本。只读。"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]   # benchmark_v3
SRC = ROOT / "_diag_d2_cluster_ab.json"


def main() -> int:
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    recs = [dict(rec, sid=sid, tag=tag) for sid, per in raw.items()
            for tag, rec in per.items()]
    print(f"记录 {len(recs)} 条")
    for tag in ("OLD", "NEW"):
        rows = [r for r in recs if r.get("tag") == tag and "n_sources" in r]
        if not rows:
            continue
        hit = [r for r in rows if r["n_sources"] == r["gt_n"]]
        s_gt = sum(r["gt_n"] for r in rows)
        s_k = sum(r["n_sources"] for r in rows)
        s_c = sum(r.get("n_candidates") or 0 for r in rows)
        over = [r for r in rows if r["n_sources"] > r["gt_n"]]
        under = [r for r in rows if r["n_sources"] < r["gt_n"]]
        print(f"\n[{tag}] n={len(rows)}  K==GT 的样本 {len(hit)}/{len(rows)} = {len(hit)/len(rows):.3f}"
              f"   ΣK={s_k} vs ΣGT={s_gt}（K/GT={s_k/s_gt:.2f}）  Σcand={s_c}")
        print(f"   过报 {len(over)} 个  欠报 {len(under)} 个")
        print(f"   过报样本: {[r['sid'] for r in over][:18]}")
    # 聚类候选板（含未过门）与 GT
    rows = [r for r in recs if r.get("tag") == "NEW" and r.get("n_candidates") is not None]
    exact_cand = sum(1 for r in rows if r["n_candidates"] == r["gt_n"])
    print(f"\n[NEW] 候选板 len(candidates) == GT 的样本 {exact_cand}/{len(rows)}；"
          f"候选板中位数 {sorted(r['n_candidates'] for r in rows)[len(rows)//2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
