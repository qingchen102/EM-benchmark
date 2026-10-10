"""用 10-05/10-07 的五次 D2 小样本诊断跑（公开层之前）量化「同一代码、同一数据」的
运行间方差——这是判断本轮 D2 detection 0.86→0.48 是「prompt 伪影」还是「本来就不稳」
的对照材料。

只读。用法：python -u _check_d2_runsvar.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import glob
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PRE = HERE.parent / "_pre_noise_20261010" / "dataset_d2"
LIVE = HERE.parent / "dataset_d2"

runs = sorted(glob.glob(str(PRE / "answers_agent_check*")))
sids = ["d2_00000", "d2_00001", "d2_00002", "d2_00003", "d2_00004"]

gt = {}
for sid in sids:
    t = json.loads((LIVE / sid / "capture" / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    gt[sid] = len(t["interferers"])

print("五次小样本诊断跑（均为公开层之前的代码）：")
print(f"{'run':<28s}" + "".join(f"{s[-5:]:>8s}" for s in sids))
for d in runs:
    row = []
    for sid in sids:
        p = os.path.join(d, sid + ".json")
        if os.path.exists(p):
            o = json.loads(Path(p).read_text(encoding="utf-8"))
            row.append(f"{o.get('num_interferers'):>8d}")
        else:
            row.append(f"{'-':>8s}")
    print(f"{os.path.basename(d):<28s}" + "".join(row))
print(f"{'GT(S2 可见)':<28s}" + "".join(f"{gt[s]:>8d}" for s in sids))
print()

n_run_per_sample = len(runs)
unstable = []
for sid in sids:
    vals = []
    for d in runs:
        p = os.path.join(d, sid + ".json")
        if os.path.exists(p):
            vals.append(json.loads(Path(p).read_text(encoding="utf-8")).get("num_interferers"))
    det = sum(1 for v in vals if v == gt[sid])
    print(f"  {sid}: 五次计数={vals}  detection_hit={det}/{len(vals)} = {det/len(vals):.2f}")
    if len(set(vals)) > 1:
        unstable.append(sid)
print()
print(f"计数不一致的样本: {len(unstable)}/{len(sids)} → {unstable}")
print("说明：这些跑是小样本诊断（10-05/10-07，公开层之前），每样本 3~5 次重复；")
print("     若同一个样本在同一代码下就能在 0 与 2 之间跳，则 50 样本单次跑的 detection")
print("     本身带有数十个百分点量级的不确定度。")
