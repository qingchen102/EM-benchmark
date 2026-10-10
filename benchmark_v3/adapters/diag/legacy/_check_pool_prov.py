# 一次性核对：新候选池 index.json 的 seed 是否等于真值里的实际 seed（provenance 修复验证）
import json
from pathlib import Path

for tier in ("d3", "d4"):
    root = Path(__file__).resolve().parent.parent / "_regen_d34" / f"pool_{tier}"
    idx = json.load(open(root / "index.json", encoding="utf-8"))
    print(f"--- {tier}  index 条数={len(idx['samples'])}  顶层键={list(idx.keys())}")
    samples = idx["samples"]
    for s in samples[:3] + samples[-2:]:
        sid = s["id"]
        t = json.load(open(root / sid / "capture" / "private" / "d1_truth.json", encoding="utf-8"))
        flag = "OK" if s["seed"] == t["seed"] else "MISMATCH"
        print(f"   {sid}  index.seed={s['seed']}  truth.seed={t['seed']}  {flag}")
    mism = 0
    for s in samples:
        t = json.load(open(root / s["id"] / "capture" / "private" / "d1_truth.json", encoding="utf-8"))
        if s["seed"] != t["seed"]:
            mism += 1
    print(f"   全局 mismatches = {mism}/{len(samples)}")
    # 槽位是否连续
    ids = [s["id"] for s in samples]
    expect = [f"{tier}_{i:05d}" for i in range(len(samples))]
    print(f"   槽位连续 = {ids == expect}")
