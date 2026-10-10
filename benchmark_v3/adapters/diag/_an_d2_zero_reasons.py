"""读答案目录里的 _run_log.json，把「报 0」样本的 Agent 自述理由与工具调用序列打出来。

用途：判断 Agent 为什么在自己仪器给出 K>=1 的情况下报 0（证据不足？分类失败？
提取失败？还是它自己选择保守）。只读。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = (HERE / ".." / "dataset_d2").resolve()


def main() -> int:
    runs = sys.argv[1:] or ["answers_S2_agent_layer1"]
    for run in runs:
        d = ROOT / run
        log = d / "_run_log.json"
        if not log.exists():
            print(f"[{run}] 无 _run_log.json")
            continue
        entries = json.loads(log.read_text(encoding="utf-8"))
        zeros = []
        for e in entries:
            pred = e.get("prediction") or {}
            if int(pred.get("num_interferers", -1)) == 0:
                zeros.append(e)
        print("=" * 92)
        print(f"[{run}] 条目 {len(entries)}  报 0 的 {len(zeros)} 个")
        for e in zeros:
            sid = e["sample_id"]
            truth = json.loads((ROOT / sid / "capture" / "private" / "d1_truth.json")
                               .read_text(encoding="utf-8"))
            gt = len(truth["interferers"])
            raw = (e.get("raw_output") or "").strip()
            names = [n for r in (e.get("round_log") or []) for n in (r.get("names") or [])]
            uniq = []
            for n in names:
                if n not in uniq:
                    uniq.append(n)
            print("-" * 92)
            print(f"{sid}  GT={gt}  工具序列({len(names)}): {uniq}")
            print("  自述尾部：" + raw[-700:].replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
