# -*- coding: utf-8 -*-
# 各难度 × 各站数子集（S0/S1/S2）下，答案键里"真正可检"的干扰数分布。
# 直接复用 eval_dataset.py 的口径（interferer_visible_at / gt_at_slevel），
# 不自己造判据。目的：量化"检测"任务在每一档是否退化成"大多数样本报 0"。
# 只读，不写任何数据集。
import json
import os
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_dataset import gt_at_slevel  # noqa: E402

SUBSETS = ("S0", "S1", "S2")


def station_lists():
    """从任一样本的 station_subsets 读真实定义，避免硬编码。"""
    for ds in ("dataset_d1", "dataset_d2", "dataset_d3", "dataset_d4"):
        d = os.path.join(ROOT, ds)
        if not os.path.isdir(d):
            continue
        for nm in sorted(os.listdir(d)):
            p = os.path.join(d, nm, "capture", "private", "d1_truth.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    t = json.load(f)
                ss = t.get("station_subsets")
                if ss:
                    return ds, nm, ss
    return None, None, None


def main():
    ds0, nm0, ss = station_lists()
    print(f"station_subsets 取自 {ds0}/{nm0}: {json.dumps(ss, ensure_ascii=False)}")
    for level in SUBSETS:
        if ss and level in ss:
            print(f"   {level} = {ss[level]}")
    print()

    for ds in ("dataset_d1", "dataset_d2", "dataset_d3", "dataset_d4"):
        d = os.path.join(ROOT, ds)
        if not os.path.isdir(d):
            continue
        print(f"########## {ds} ##########")
        for level in SUBSETS:
            stations = (ss or {}).get(level)
            if not stations:
                continue
            hist = Counter()
            n_ifw_total = 0
            n_samp = 0
            for nm in sorted(os.listdir(d)):
                p = os.path.join(d, nm, "capture", "private", "d1_truth.json")
                if not os.path.exists(p):
                    continue
                with open(p, encoding="utf-8") as f:
                    t = json.load(f)
                n_samp += 1
                gts = gt_at_slevel(t, stations)
                hist[len(gts)] += 1
                n_ifw_total += len(gts)
            if not n_samp:
                continue
            pct0 = hist.get(0, 0) / n_samp * 100
            print(f"  [{level}] 站={stations} 样本={n_samp} 键内干扰总数={n_ifw_total} "
                  f"均值={n_ifw_total / n_samp:.2f}")
            print(f"       每样本可检干扰数分布 {dict(sorted(hist.items()))}   "
                  f"'报 0 即正确'的样本占比 {pct0:.1f}%")
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
