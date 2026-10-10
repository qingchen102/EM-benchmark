"""一次性：打印四档 index.json 的 envelope 文案 + 采样统计（供现状报告核对文案与数据是否一致）。"""
import json
from pathlib import Path

for d in ("d1", "d2", "d3", "d4"):
    p = Path(f"../dataset_{d}/index.json")
    j = json.loads(p.read_text(encoding="utf-8"))
    e = j["envelope"]
    n = len(j["samples"])
    per = {}
    for s in j["samples"]:
        k = len(s.get("interferers", []))
        per[k] = per.get(k, 0) + 1
    print(f"[{d}] n={n}  每样本干扰数 {dict(sorted(per.items()))}")
    print(f"     interferer: {e.get('interferer')}")
    print(f"     target    : {e.get('target')}")
    print(f"     noise     : {e.get('noise')}   propagation: {e.get('propagation')}")
    print()
