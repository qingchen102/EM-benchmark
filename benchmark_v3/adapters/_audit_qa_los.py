# -*- coding: utf-8 -*-
# 按 QA 约束 1 的真实口径（几何 LoS 可见）重算 D2/D3/D4 的干扰可见站数。
#
# QA 代码原文（gen_dataset.py，91caa6e 引入）：
#   QA 约束 1：每个干扰 >=3 站 LoS 可见（INR 终检由 achieved 承担）。
# 此前 _audit_datasets.py 用的是 achieved INR>=3dB，口径不同，必须分开统计。
# 只读，不写任何数据集。
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_truth(sample_dir):
    p = os.path.join(sample_dir, "capture", "private", "d1_truth.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump_keys(ds, n=1):
    d = os.path.join(ROOT, ds)
    if not os.path.isdir(d):
        return
    names = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    for nm in names[:n]:
        t = load_truth(os.path.join(d, nm))
        if not t:
            continue
        print(f"  [{ds}/{nm}] 干扰 per_station 键示例:")
        for j, it in enumerate(t.get("interferers", [])):
            per = it.get("per_station") or {}
            for st, v in list(per.items())[:1]:
                print(f"     interferer[{j}] kind={it.get('kind')} station={st} 字段={sorted((v or {}).keys())}")
        tg = t.get("target", {})
        per = tg.get("per_station") or {}
        for st, v in list(per.items())[:1]:
            print(f"     target station={st} 字段={sorted((v or {}).keys())}")


def count(ds, key):
    """key: 'los' 用 los_blocked；'inr' 用 inr_db>=3"""
    d = os.path.join(ROOT, ds)
    if not os.path.isdir(d):
        print(f"{ds}: 不存在")
        return
    dist = Counter()
    per_sample_min = []
    n_samp = n_if = 0
    for nm in sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))):
        t = load_truth(os.path.join(d, nm))
        if not t:
            continue
        n_samp += 1
        mins = []
        for it in t.get("interferers", []):
            n_if += 1
            per = it.get("per_station") or {}
            if key == "los":
                vis = sum(1 for v in per.values() if not (v or {}).get("los_blocked", False))
            else:
                vis = sum(1 for v in per.values() if (v or {}).get("inr_db", -99) >= 3.0)
            dist[vis] += 1
            mins.append(vis)
        if mins:
            per_sample_min.append(min(mins))
    if not n_if:
        print(f"{ds} ({key}): 无干扰")
        return
    ok = sum(c for v, c in dist.items() if v >= 3)
    fail = sum(c for v, c in dist.items() if v < 3)
    print(f"{ds} [{key}]  样本={n_samp} 干扰={n_if}")
    print(f"   可见站数分布 {dict(sorted(dist.items()))}")
    print(f"   >=3 站（满足 QA 约束 1）: {ok}/{n_if} = {ok / n_if * 100:.1f}%   "
          f"违反: {fail}/{n_if} = {fail / n_if * 100:.1f}%")
    print(f"   每样本最少可见站数分布 {dict(sorted(Counter(per_sample_min).items()))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== per_station 字段 ===")
    for ds in ("dataset_d1", "dataset_d2", "dataset_d3", "dataset_d4"):
        dump_keys(ds)
    print()
    for key, title in (("los", "几何 LoS 可见（= QA 约束 1 的口径）"), ("inr", "achieved INR>=3dB（旧审计口径）")):
        print(f"########## {title} ##########")
        for ds in ("dataset_d1", "dataset_d2", "dataset_d3", "dataset_d4"):
            count(ds, key)
        print()
