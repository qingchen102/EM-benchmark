"""对照：同一次评测器下，重算"点状重跑各臂"的答案目录，逐档并排比。

用法::

    python -X utf8 diag/_cmp_answer_arms.py                       # 自动发现各档答案目录
    python -X utf8 diag/_cmp_answer_arms.py d2                    # 只看一档
    python -X utf8 diag/_cmp_answer_arms.py d2 --arms a=..,b=..   # 指定臂

只读答案文件 + 真值，调用 `eval_dataset.evaluate` 重算，不写任何报告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # adapters/
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eval_dataset as ed  # noqa: E402

V3 = HERE.parents[1]                          # benchmark_v3/
AXES = ("detection_hit", "detection_within1", "detect_abs_err", "kind_hits",
        "freq_hits", "pos_hits", "doa_hits", "intent_accuracy",
        "mech_accuracy", "spoof_fa_rate")


def discover(difficulty: str) -> list[Path]:
    out = []
    for p in sorted((V3 / f"dataset_{difficulty}").glob("answers_*")):
        if p.is_dir():
            out.append(p)
    return out


def answers_summary(d: Path) -> tuple[int, dict[int, int], dict[int, int]]:
    """(样本数, 报数分布, 机理非空条数分布)。"""
    n = 0
    counts: dict[int, int] = {}
    mech: dict[int, int] = {}
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        obj = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or "interferers" not in obj:
            continue
        n += 1
        items = obj["interferers"] or []
        counts[len(items)] = counts.get(len(items), 0) + 1
        k = sum(1 for it in items if it.get("mechanism"))
        mech[k] = mech.get(k, 0) + 1
    return n, counts, mech


def main() -> int:
    ap = argparse.ArgumentParser(description="重算并并排比各答案臂（只读）")
    ap.add_argument("difficulty", nargs="?", default=None)
    ap.add_argument("--arms", default=None, help="路径对，逗号分隔，如 a=../dataset_d2/answers_x,b=...")
    args = ap.parse_args()

    diffs = [args.difficulty] if args.difficulty else ["d1", "d2", "d3", "d4"]
    for d in diffs:
        root = V3 / f"dataset_{d}"
        if not root.is_dir():
            continue
        arms = [Path(x.split("=", 1)[1]) for x in args.arms.split(",")] if args.arms \
            else discover(d)
        arms = [a if a.is_absolute() else (HERE / a).resolve() for a in arms]
        print(f"\n{'='*100}\n=== dataset_{d}  臂 = {len(arms)} ===")
        table: dict[str, dict[str, float]] = {}
        for a in arms:
            n, counts, mech = answers_summary(a)
            rep = ed.evaluate(root, a)
            lvl = "S0" if d == "d1" else "S2"
            row = {k: rep[lvl].get(k) for k in AXES}
            table[a.name] = row
            print(f"\n  [{a.name}]  样本 {n}  报数分布 {dict(sorted(counts.items()))}  "
                  f"机理非空条数 {dict(sorted(mech.items()))}")
            line = "   ".join(f"{k}={'-' if row[k] is None else round(row[k], 4)}" for k in AXES[:6])
            print("   " + line)
            line = "   ".join(f"{k}={'-' if row[k] is None else round(row[k], 4)}" for k in AXES[6:])
            print("   " + line)
        names = list(table)
        if len(names) > 1:
            print(f"\n  --- 差值（{names[-1]} − {names[0]}）---")
            for k in AXES:
                a, b = table[names[0]].get(k), table[names[-1]].get(k)
                if a is None or b is None:
                    continue
                print(f"    {k:<20} {a:>7.4f} → {b:>7.4f}  ({b - a:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
