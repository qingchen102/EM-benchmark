"""逐样本比较两个 D2 答案目录（detection 命中、报源数、GT 可见源数）。

用法：
    python _cmp_d2_runs.py <answers_dir_A> <answers_dir_B> [--root ../dataset_d2] [--slevel S2]

只读；A/B 两目录可有不同样本数（B 跑动中也能看）。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # adapters（迁移后 HERE 下沉一级）
import eval_dataset as ed          # noqa: E402

SL = {"S0": ["station_a"], "S1": ["station_a", "station_b", "station_c"],
      "S2": ["station_a", "station_b", "station_c", "station_d"]}


def load(root: Path, d: Path, stations: list[str]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        ans = json.loads(p.read_text(encoding="utf-8"))
        truth = json.loads((root / p.stem / "capture" / "private" / "d1_truth.json")
                           .read_text(encoding="utf-8"))
        gt = ed.gt_at_slevel(truth, stations)
        out[p.stem] = (int(ans.get("num_interferers", -1)), len(gt))
    return out


def hits(d: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return sum(1 for v in d.values() if v[0] == v[1]), len(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--root", default="../dataset_d2")
    ap.add_argument("--slevel", default="S2", choices=list(SL))
    args = ap.parse_args()
    root = (HERE / args.root).resolve()
    A, B = load(root, (HERE / args.a).resolve(), SL[args.slevel]), load(root, (HERE / args.b).resolve(), SL[args.slevel])
    ha, na = hits(A)
    hb, nb = hits(B)
    print(f"A {args.a}: {ha}/{na} = {ha / max(na, 1):.3f}")
    print(f"B {args.b}: {hb}/{nb} = {hb / max(nb, 1):.3f}")
    print()
    print("sid        GT  A   B   标记")
    same = diff = 0
    miss_b = 0
    for k in sorted(set(A) | set(B)):
        va, vb = A.get(k), B.get(k)
        ga = va[1] if va else (vb[1] if vb else -1)
        ca = va[0] if va else None
        cb = vb[0] if vb else None
        if ca is None or cb is None:
            mark = "B缺" if cb is None else "A缺"
            miss_b += cb is None
        elif ca == cb:
            mark = "="
            same += 1
        else:
            mark = "DIFF"
            diff += 1
        ha_ = va is not None and va[0] == va[1]
        hb_ = vb is not None and vb[0] == vb[1]
        tag = ""
        if va is not None and vb is not None and ha_ != hb_:
            tag = ("  A对B错" if ha_ else "  A错B对")
        print(f"{k}  {ga}   {ca if ca is not None else '-'}   {cb if cb is not None else '-'}   {mark}{tag}")
    print(f"\n两边都有的样本中：报数相同 {same} / 不同 {diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
