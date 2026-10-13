"""S0/S1/S2 三个观测档为什么给出一模一样的分数？（零 API、只读）

结论先行：报告里的 S0/S1/S2 **不是三次跑**，而是同一次作答（`--slevel S2`，Agent 看得到 4 个站）
用三套"可见性口径"各评一遍（见 `run_capture_agent.py:609 allowed = list(subsets[slevel])`
与 `eval_dataset.py:350` 的 `for slevel, stations in subsets.items()`）。
可见性规则 `interferer_visible_at` = 子集内**任一站** INR / RT achieved pr 达标即算可见
⇒ 只有当某干扰"只被 station_d 看见"时，S1 与 S2 的 GT 分母才会不同。

本脚本把这个量数出来（当前四档都是 0，所以 S1 == S2 逐位相同）。

用法：python -X utf8 diag/_probe_s_levels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ADP = Path(__file__).resolve().parents[1]        # benchmark_v3/adapters
V3 = ADP.parent                                  # benchmark_v3
sys.path.insert(0, str(ADP))

import eval_dataset as ed  # noqa: E402


def main():
    for deck in ("d1", "d2", "d3", "d4"):
        root = V3 / f"dataset_{deck}"
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        subs = idx["observation_subsets"]
        n_diff = 0
        only_d = 0
        tot = {k: 0 for k in subs}
        mix = {}
        for s in idx["samples"]:
            sid = s["id"]
            truth = ed.load_truth(root, sid)
            counts = {k: len(ed.gt_at_slevel(truth, v)) for k, v in subs.items()}
            for k in subs:
                tot[k] += counts[k]
            if counts.get("S1") != counts.get("S2"):
                n_diff += 1
            only_d += max(0, counts.get("S2", 0) - counts.get("S1", 0))
            key = (counts.get("S0"), counts.get("S1"), counts.get("S2"))
            mix[key] = mix.get(key, 0) + 1
        print(f"[{deck}] samples={len(idx['samples'])}  subsets={subs}")
        print(f"    可见 GT 总数 S0/S1/S2 = {tot['S0']} / {tot['S1']} / {tot['S2']}"
              f"   仅 d 可见的干扰数 = {only_d}   n_gt(S1)!=n_gt(S2) 的样本 = {n_diff}")
        top = sorted(mix.items(), key=lambda kv: -kv[1])[:4]
        print("    (n0,n1,n2) 分布 top: " + "  ".join(f"{k}×{v}" for k, v in top))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
