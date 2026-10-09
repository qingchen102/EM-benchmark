"""一次性检查：run_summary.json 的 hashes 是否与盘上文件一致（重点是 observations.json）。

只读。背景：`regen_noise.py:refresh_hashes` 会把 run root 下每个文件都摘要进
run_summary.json；若在那之后再改写 observations.json（受害者层/合作先验投影），
记录的摘要就会失效。本脚本用平台自己的 `cityrf.export.digest` 对全部文件逐一对账。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import gen_dataset  # noqa: F401,E402  —— 先导入它把平台 src 插进 sys.path
from cityrf.export import digest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TIERS = ["d1", "d2", "d3", "d4"]


def main() -> int:
    total = stale = missing = 0
    stale_files: dict[str, int] = {}
    for tier in TIERS:
        for sd in sorted((ROOT / f"dataset_{tier}").glob(f"{tier}_*")):
            sp = sd / "capture" / "run_summary.json"
            if not sp.is_file():
                continue
            summ = json.loads(sp.read_text(encoding="utf-8"))
            for rel, want in (summ.get("hashes") or {}).items():
                p = sd / "capture" / rel
                total += 1
                if not p.is_file():
                    missing += 1
                    continue
                if digest(p) != want:
                    stale += 1
                    stale_files[rel] = stale_files.get(rel, 0) + 1
    print(f"对账摘要 {total} 项：失效 {stale}  文件缺失 {missing}")
    for rel, n in sorted(stale_files.items(), key=lambda kv: -kv[1]):
        print(f"  失效 {n:>3} 个样本: {rel}")
    return 1 if (stale or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
