"""检测计数轴的"常数基线"体检（零 API，只读）。

为什么需要它：注册口径 `detection_hit` 要求报数与真值**完全相等**。如果某档真值的
可见源数高度集中（例如 48/50 个样本都恰好 2 个），那么"永远答多数类"就能拿到很高分，
该轴在这些档上几乎没有判别力——所有跨臂/跨模型比较都必须先减掉这个常数基线。

用法：
    python -X utf8 diag/_count_axis_baseline.py            # 四档全跑
    python -X utf8 diag/_count_axis_baseline.py d2 d4      # 指定档

输出：每档 × 每个 S 级子集（index.json 的 observation_subsets）×
  · 真值可见源数分布、常数多数类基线、计数熵
  · 该档所有 answers_* 臂的 detection_hit，以及"相对常数基线"的差值
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ADAPTERS = HERE.parent
sys.path.insert(0, str(_ADAPTERS))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eval_dataset as ed  # noqa: E402

_V3 = _ADAPTERS.parent
DEFAULTS = ("d1", "d2", "d3", "d4")


def gt_counts(root: Path, sids: list[str], stations: list[str]) -> list[int]:
    out = []
    for sid in sids:
        truth = ed.load_truth(root, sid)
        out.append(len(ed.gt_at_slevel(truth, stations)))
    return out


def entropy(counter: collections.Counter, n: int) -> float:
    h = 0.0
    for v in counter.values():
        p = v / n
        if p:
            h -= p * math.log2(p)
    return h


def arm_detection(root: Path, sids: list[str], stations: list[str],
                  ans_dir: Path) -> tuple[float | None, int]:
    hit = tot = 0
    for sid in sids:
        p = ans_dir / f"{sid}.json"
        if not p.exists():
            continue
        ans = json.loads(p.read_text(encoding="utf-8"))
        truth = ed.load_truth(root, sid)
        gt = ed.gt_at_slevel(truth, stations)
        n_pre = len(ans.get("interferers") or []) if "interferers" in ans \
            else int(ans.get("num_interferers", 0) or 0)
        hit += int(n_pre == len(gt))
        tot += 1
    return ((hit / tot) if tot else None), tot


def main(argv: list[str]) -> int:
    decks = [a for a in argv if not a.startswith("-")] or list(DEFAULTS)
    out: list[str] = []
    for deck in decks:
        root = _V3 / f"dataset_{deck}"
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        sids = [s["id"] for s in idx["samples"]]
        subsets = idx["observation_subsets"]
        arms = sorted(p for p in root.glob("answers_*") if p.is_dir())
        out.append(f"\n{'=' * 84}")
        out.append(f"dataset_{deck}   样本 {len(sids)}   S 级子集 {list(subsets)}   答案臂 {len(arms)}")
        out.append(f"{'=' * 84}")
        for slevel, stations in subsets.items():
            counts = gt_counts(root, sids, stations)
            cnt = collections.Counter(counts)
            n = len(counts)
            maj = max(cnt.values()) / n
            out.append(f"\n[{slevel}] 用站 {stations}   真值可见源数分布 "
                       f"{dict(sorted(cnt.items()))}   均值 {sum(counts) / n:.2f}")
            out.append(f"  常数多数类基线（永远答 {max(cnt, key=cnt.get)}） = {maj:.3f}"
                       f"   计数熵 = {entropy(cnt, n):.2f} bit（理论最大 {math.log2(len(cnt)):.2f}）")
            for a in arms:
                d, n_arm = arm_detection(root, sids, stations, a)
                if d is None:
                    continue
                partial = "（跑中/不完整）" if n_arm < n else ""
                mark = "  ← 低于常数基线" if (d < maj and not partial) else ""
                out.append(f"    {a.name:<42} detection_hit = {d:.3f} (n={n_arm})"
                           f"   相对常数基线 {d - maj:+.3f}{mark}{partial}")
    text = "\n".join(out) + "\n"
    print(text, end="")
    dst = _V3 / "evidence" / "count_axis_baseline.txt"
    dst.write_text(text, encoding="utf-8")
    print(f"[已写入] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
