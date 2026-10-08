# -*- coding: utf-8 -*-
# 核对 D3/D4 的 station_subsets 定义与 index.json schema（重生成 vs 旧）。
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PAIRS = [
    (os.path.join(ROOT, "dataset_d3"), "d3 旧"),
    (os.path.join(ROOT, "_regen_d34", "dataset_d3"), "d3 新"),
    (os.path.join(ROOT, "dataset_d4"), "d4 旧"),
    (os.path.join(ROOT, "_regen_d34", "dataset_d4"), "d4 新"),
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for d, lab in PAIRS:
        ip = os.path.join(d, "index.json")
        if not os.path.exists(ip):
            print(f"{lab}: 无 index.json")
            continue
        with open(ip, encoding="utf-8") as f:
            idx = json.load(f)
        sid = idx["samples"][0]["id"]
        with open(os.path.join(d, sid, "capture", "private", "d1_truth.json"),
                  encoding="utf-8") as f:
            t = json.load(f)
        print(f"--- {lab}  ({sid}) ---")
        print(f"  index 顶层键 : {sorted(idx.keys())}")
        print(f"  samples[0]   : {json.dumps(idx['samples'][0], ensure_ascii=False)}")
        print(f"  首样本数     : {len(idx['samples'])}")
        print(f"  subsets      : {json.dumps(t.get('station_subsets'), ensure_ascii=False)}")
        print()
