"""clean 臂的**按体制**分解：19 类各自在"独立种子纯波形"上的命中率。

为什么：gate 的 clean 分只有总体 0.877。要回答"D2 的体制识别为什么弱"，
必须先分清两类原因：
  (a) 分类器自己对某类就弱（clean 上就低）→ 训练/标签问题；
  (b) 分类器能做到，但过了解析式切片链后被毁（clean 高、GT 切片/tool-view 低）。
这个探针直接复用 oracle16.gate_clean 的抽样口径（seed_base=700000、n_per_class），
把混淆矩阵摊成每类命中率与"最常被误判成谁"。

用法：python -X utf8 _probe_clean_perclass.py [n_per_class]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # adapters（capture_tools 等）

import numpy as np  # noqa: E402

import oracle16 as o16  # noqa: E402


def main(n_per_class: int = 60) -> None:
    model, classes = o16.load_model()
    acc, conf = o16.gate_clean(model, n_per_class=n_per_class)
    tot = conf.sum()
    print(f"\n== clean 臂按体制分解（n_per_class={n_per_class}, 合计 {int(tot)}, 总 {acc:.3f}）==")
    print(f"{'体制':16s} {'n':>4s} {'命中':>5s} {'命中率':>7s}   最常误判（次数）")
    rows = []
    for ci, name in enumerate(classes):
        n = int(conf[ci].sum())
        hit = int(conf[ci, ci])
        others = [(int(conf[ci, cj]), classes[cj]) for cj in range(len(classes)) if cj != ci]
        others.sort(reverse=True)
        top = ", ".join(f"{k}({v})" for v, k in others[:3] if v > 0)
        rows.append((name, n, hit, hit / n if n else 0.0, top))
    for name, n, hit, r, top in sorted(rows, key=lambda x: x[3]):
        print(f"{name:16s} {n:>4d} {hit:>5d} {r:>7.1%}   {top}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
