"""D2 detection 的「运行间方差」对照：把 10-05~10-07（公开层之前）的多次 D2 答案与
本轮 layer1 答案排在一起，看同一批样本在**未改 prompt** 的时代是否就已经摇摆。

只读。用法：python -u _check_d2_variance.py
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
FLIPPED = ["d2_00001", "d2_00005", "d2_00008", "d2_00009", "d2_00019",
           "d2_00020", "d2_00021", "d2_00022", "d2_00024", "d2_00027", "d2_00028"]

# GT（S2 可见）源数
truth_counts = {}
for sid in FLIPPED:
    t = json.loads((LIVE / sid / "capture" / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    truth_counts[sid] = len(t["interferers"])
print("GT(S2 可见) 源数:", {k: v for k, v in truth_counts.items()})
print()

dirs = sorted(glob.glob(str(PRE / "answers_agent_check*"))) + [
    str(PRE / "answers_S2_agent_full"), str(LIVE / "answers_S2_agent_layer1")]
print(f"{'答案目录':<46s} {'文件':>4s}  " + "  ".join(f"{s[-5:]:>5s}" for s in FLIPPED))
for d in dirs:
    row = []
    n = 0
    for sid in FLIPPED:
        p = os.path.join(d, sid + ".json")
        if os.path.exists(p):
            n += 1
            o = json.loads(Path(p).read_text(encoding="utf-8"))
            row.append(f"{o.get('num_interferers'):>5d}")
        else:
            row.append(f"{'-':>5s}")
    tag = os.path.relpath(d, str(HERE.parent))
    print(f"{tag:<46s} {n:>4d}  " + "  ".join(row))

print()
print("历史 check* 目录覆盖的样本（若与上面不同）：")
for d in sorted(glob.glob(str(PRE / "answers_agent_check*"))):
    ids = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, "*.json"))
                 if not f.endswith("_run_log.json"))
    print(" ", os.path.basename(d), ids)
