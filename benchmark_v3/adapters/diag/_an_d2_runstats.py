"""统计两次 D2 运行：原始输出里用负哨兵(-1)的样本、被 _sanitize 落成 0 的样本、报数分布、
平均耗时/工具数。只读。"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]   # benchmark_v3
D2 = ROOT / "dataset_d2"
NEG = re.compile(r'"num_interferers"\s*:\s*-')


def load(name: str) -> list[dict]:
    p = D2 / name / "_run_log.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def main() -> int:
    for name in ("answers_S2_agent_layer1", "answers_S2_agent_layer1b"):
        d = load(name)
        if not d:
            print(f"[{name}] 无日志")
            continue
        neg = [e["sample_id"] for e in d if NEG.search(e.get("raw_output") or "")]
        zero = [e["sample_id"] for e in d if (e.get("prediction") or {}).get("num_interferers") == 0]
        dist = {k: sum(1 for e in d if (e.get("prediction") or {}).get("num_interferers") == k)
                for k in range(0, 8)}
        print(f"[{name}] 条目 {len(d)}")
        print(f"  原始输出用负哨兵(-1) 的样本: {len(neg)}  {neg}")
        print(f"  prediction 落成 0 的样本:     {len(zero)}  {zero}")
        print(f"  报数分布: {dist}")
        print(f"  平均耗时 {st.mean(e['elapsed_s'] for e in d):.0f}s   平均工具数 "
              f"{st.mean(e['tool_calls'] for e in d):.1f}   累计 {sum(e['elapsed_s'] for e in d) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
