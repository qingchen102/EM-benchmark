"""盘点：历史上到底"真跑过"哪些观测档（S0/S1/S2），还是只跑一档再用三套口径重打分？

事实源只有两个（不猜）：
  1) 每档 dataset 下的 `answers_*` 目录名 → 名字里的 S 就是当时 `--slevel` 的值；
  2) 所有 `_api_*.log` 首行（包装脚本回显的 python 命令行）里的 `--slevel` / `--answers`。

结论（2026-10-13）：全树 21 个 answers 目录中，D1 只有 `answers_S0_*`，D2–D4 只有 `answers_S2_*`，
**没有任何 `answers_S1_*`**；D1 的工具集也证明它是真单站跑（无 spatial_clustering / extract_source_waveforms）。
⇒ D2–D4 报告里的 S0/S1 块是同一份 S2 答案按少站可见性**重打分**，不是"少给几个站"的实测。

用法：python -X utf8 diag/_audit_s_level_runs.py
"""
from __future__ import annotations

import glob
import io
import os
import re
import sys
from pathlib import Path

ADP = Path(__file__).resolve().parents[1]      # benchmark_v3/adapters
V3 = ADP.parent                                # benchmark_v3
ROOT = V3.parent                               # 仓库根


def main():
    print("== 全树的 answers_* 目录（名字里的 S 即当时的 --slevel） ==")
    hits = []
    for dp, dn, _fn in os.walk(V3):
        if "_archive" in dp:
            continue
        for x in dn:
            if x.startswith("answers_"):
                hits.append(str(Path(dp) / x))
    for h in sorted(hits):
        print("  " + h.replace(str(ROOT) + os.sep, "").replace(os.sep, "/"))
    lv = {}
    for h in hits:
        m = re.match(r"answers_(S\d)_", os.path.basename(h))
        if m:
            lv[m.group(1)] = lv.get(m.group(1), 0) + 1
    print(f"  合计 {len(hits)} 个；按档统计 {lv}")

    print("\n== 所有 _api_*.log 的运行命令行 ==")
    logs = sorted(glob.glob(str(V3 / "_api_*.log")))
    logs += sorted(glob.glob(str(ADP / "_api_*.log")))
    for f in logs:
        first = io.open(f, encoding="utf-8", errors="replace").readline().strip()
        m1 = re.search(r"--slevel\s+(\S+)", first)
        m2 = re.search(r"--answers\s+(\S+)", first)
        m3 = re.search(r"--max-samples\s+(\S+)", first)
        print(f"  {os.path.basename(f):40s} slevel={m1.group(1) if m1 else '?':4s} "
              f"n={m3.group(1) if m3 else '-':4s} answers={os.path.basename(m2.group(1)) if m2 else '-'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
