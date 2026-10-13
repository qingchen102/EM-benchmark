"""审计：各档在每个观测档下的「可见干扰源数」分布（零 API、只读）。

为什么需要它
------------
S0 只允许用 station_a。若某个样本在 station_a 一个可见源都没有，那么该样本在
S0 口径下的真值源数就是 0 —— 此时「报 0」既正确又不需要任何能力（它还会把
detection_hit 抬到常数基线之上）。这与 D1 那种**设计上就没有干扰**的样本
（D1 一半样本是真·干净样本，报 0 才是题目要考的事）是两回事。

判据与评测口径一致：`eval_dataset.gt_at_slevel(truth, subset_stations)`，
即「子集内任一站的 INR / RT achieved pr 达标即算可见」。
（`index.json` 的 `observation_subsets`：S0=[station_a] / S1=[a,b,c] / S2=[a,b,c,d]）

用法
----
    python -X utf8 diag/_probe_s0_visibility.py [d1,d2,d3,d4] [--out <path>]

默认四档、输出到 `benchmark_v3/evidence/_s0_visibility.txt`。
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ADP = HERE.parent                          # benchmark_v3/adapters
V3 = ADP.parent                            # benchmark_v3
sys.path.insert(0, str(ADP))

import eval_dataset as ed                  # noqa: E402


def survey(root: pathlib.Path) -> dict:
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    subsets = idx["observation_subsets"]
    levels = sorted(subsets)
    counts = {lv: collections.Counter() for lv in levels}
    n = 0
    for s in idx["samples"]:
        truth = ed.load_truth(root, s["id"])
        if truth is None:
            continue
        n += 1
        for lv in levels:
            counts[lv][len(ed.gt_at_slevel(truth, subsets[lv]))] += 1
    return {"n": n, "subsets": subsets, "counts": counts}


def _fmt(counter: collections.Counter) -> str:
    return "  ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="?", default="d1,d2,d3,d4")
    ap.add_argument("--out", default=str(V3 / "evidence" / "_s0_visibility.txt"))
    args = ap.parse_args(argv)

    decks = [d.strip() for d in args.decks.replace(",", " ").split() if d.strip()]
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("各观测档「可见干扰源数」分布（判据 = 子集内任一站可见；与 eval_dataset 同源）")
    emit(f"{'档':<4s} {'样本':>4s}  " + "  ".join(f"{lv:>22s}" for lv in ("S0", "S1", "S2")))
    for d in decks:
        root = V3 / f"dataset_{d}"
        if not (root / "index.json").is_file():
            emit(f"{d:<4s}   （跳过：缺 index.json）")
            continue
        r = survey(root)
        cells = []
        for lv in ("S0", "S1", "S2"):
            c = r["counts"].get(lv)
            cells.append("—" if c is None else _fmt(c))
        emit(f"{d:<4s} {r['n']:>4d}  " + "  ".join(f"{x:>22s}" for x in cells))
        c0 = r["counts"].get("S0")
        if c0:
            zero = c0.get(0, 0)
            emit(f"     └ S0 零可见样本 {zero}/{r['n']} = {zero / max(r['n'], 1):.1%}"
                 f"（子集 {r['subsets'].get('S0')}）")
    emit()
    emit("读法：D1 的 0 可见样本 = 设计上的「真·无干扰」样本，必须留在分母里；")
    emit("      D2/D3/D4 的 0 可见样本 = 源存在但 station_a 看不到，S0 轴上不度量能力。")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
