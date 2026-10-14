"""新增场景与主档的并排对照：读各档 canonical 报告（零 API、只读）。

用法：
    python -X utf8 diag/_cmp_scene_decks.py                # 默认 d2 d3 d4 d5 d6 d7 d8 @ S2
    python -X utf8 diag/_cmp_scene_decks.py d5 d6 --slevel S2
"""
from __future__ import annotations

import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_V3 = _HERE.parents[1]           # benchmark_v3

TIER = {"d1": "T1", "d2": "T2", "d3": "T3", "d4": "T4",
        "d5": "T3", "d6": "T2", "d7": "T4", "d8": "T2"}
NAME = {"d1": "物流航线", "d2": "居民区共存", "d3": "城市峡谷", "d4": "对抗演习",
        "d5": "卡11 建筑工地", "d6": "卡7 电力线", "d7": "卡3 蜂群", "d8": "卡10 滨水"}
COLS = [("detection_hit", "检测计数"), ("detection_within1", "±1"),
        ("kind_hits", "体制"), ("freq_hits", "载频"),
        ("pos_hits", "定位"), ("doa_hits", "DOA"),
        ("intent_accuracy", "意图"), ("mech_accuracy", "机理")]


def main(argv: list[str]) -> int:
    decks, slevel = ["d2", "d3", "d4", "d5", "d6", "d7", "d8"], "S2"
    for i, a in enumerate(argv[1:]):
        if a == "--slevel":
            slevel = argv[i + 2]
    plain = [a for a in argv[1:]
             if not a.startswith("--") and a not in ("S0", "S1", "S2")]
    if plain:
        decks = " ".join(plain).replace(",", " ").split()
    hdr = f"{'档':<4}{'场景':<14}{'难度':<5}{'n':>4}"
    for _k, cn in COLS:
        hdr += f"{cn:>9}"
    print(hdr)
    print("-" * len(hdr))
    for dk in decks:
        p = _V3 / f"dataset_{dk}" / f"eval_report_{slevel}.json"
        if not p.is_file():
            print(f"{dk:<4}{NAME.get(dk, ''):<14}{TIER.get(dk, '?'):<5}{'—':>4}   （无 {slevel} 报告）")
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        blk = rep.get(slevel) or {}
        if not blk:
            print(f"{dk:<4}{NAME.get(dk, ''):<14}{TIER.get(dk, '?'):<5}{'—':>4}   （报告无 {slevel} 段）")
            continue
        row = f"{dk:<4}{NAME.get(dk, ''):<14}{TIER.get(dk, '?'):<5}{blk.get('n', 0):>4}"
        for k, _cn in COLS:
            v = blk.get(k)
            row += f"{'—':>9}" if v is None else f"{v:>9.3f}"
        print(row)
        if blk.get("excluded_zero_visible_n"):
            print(f"       （{slevel} 剔除零可见样本 {blk['excluded_zero_visible_n']} / 共见 "
                  f"{blk.get('n_total_seen', '?')}）")
    print("\n注：d5–d8 的难度等级为预设值（跑分后调整）；定位/DOA 两列互斥（S0 出 DOA，S1/S2 出定位）。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv))
