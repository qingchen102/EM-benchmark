"""统计各档 agent 运行里"禁用哨兵 -1"（= 拒答）的比例。

背景：`run_capture_agent.py` 的 SYSTEM_PROMPT_CAPTURE 明写
"NEVER use a negative num_interferers (e.g. -1) or any sentinel"，
而 `_sanitize()` 会把任何负数清成 0 —— 于是一次拒答在评分里等价于
"我判定没有干扰"。本脚本只读 `_run_log.json` 的 raw_output（模型原话）
统计该行为，用于把 detection_hit 拆成"拒答"与"真的报了 0"两部分。

用法：python -u _an_abstain.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = pathlib.Path(__file__).resolve().parent.parent.parent          # benchmark_v3
PRE = BASE / "_pre_bw_20261009"

# (标签, 数据集目录, 答案目录, 观测档)
RUNS = [
    ("D1 x S0", BASE / "dataset_d1", BASE / "dataset_d1" / "answers_S0_agent_layer1", "S0"),
    ("D2 x S2 带宽修前", PRE / "dataset_d2", PRE / "dataset_d2" / "answers_S2_agent_layer1", "S2"),
    ("D2 x S2 带宽修后", BASE / "dataset_d2", BASE / "dataset_d2" / "answers_S2_agent_bwfix", "S2"),
    ("D3 x S2 带宽修前", PRE / "dataset_d3", PRE / "dataset_d3" / "answers_S2_agent_layer1", "S2"),
    ("D4 x S2 带宽修前", PRE / "dataset_d4", PRE / "dataset_d4" / "answers_S2_agent_layer1", "S2"),
]

SENT = re.compile(r'"num_interferers"\s*:\s*(-?\d+)')


def load_records(ans: pathlib.Path) -> list[dict]:
    p = ans / "_run_log.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    recs = d if isinstance(d, list) else d.get("runs", [])
    return recs if isinstance(recs, list) else []


def registered_detection(ds: pathlib.Path, level: str) -> float | None:
    p = ds / f"eval_report_{level}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    blk = d.get(level)
    return float(blk["detection_hit"]) if isinstance(blk, dict) and "detection_hit" in blk else None


def main() -> int:
    print(f"{'run':<18} {'records':>7} {'sentinel(-1)':>13} {'等于0':>7} {'真0?':>7} {'detection':>10}")
    print("-" * 70)
    for label, ds, ans, lv in RUNS:
        recs = load_records(ans)
        if not recs:
            print(f"{label:<18} {'-':>7}  (无 _run_log.json)")
            continue
        sent = 0
        zero = 0
        for r in recs:
            ro = r.get("raw_output") or ""
            m = SENT.search(ro)
            if m and int(m.group(1)) < 0:
                sent += 1
            if (r.get("prediction") or {}).get("num_interferers") == 0:
                zero += 1
        det = registered_detection(ds, lv)
        det_s = f"{det:.3f}" if det is not None else "-"
        # 注：eval_report 覆盖全部 50 样本，而 _run_log 可能只有续跑那几轮
        print(f"{label:<18} {len(recs):>7} {sent:>13} {zero:>7} {zero - sent:>7} {det_s:>10}")
    print("\n注：sentinel(-1) = 模型写了 prompt 明令禁止的负哨兵；`_sanitize` 把它变成 0，")
    print("    评分上等价于'判定无干扰'；真0 = 清理出的 0 减去哨兵数（可能是真的报 0）。")
    print("    `_run_log.json` 只保存本轮（续跑目录可能只含末轮记录），故 records 可能 < 50。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
