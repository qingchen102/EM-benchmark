"""只读：d2_00000 的真值/几何/QA 状态（解释冒烟失败的样本为何失败，零 token）。

用法：python _diag_d2_00000_truth.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402

ROOT = _V3 / "dataset_d2"


def main(sid: str = "d2_00000") -> None:
    cap = ROOT / sid / "capture" / "public"
    priv = ROOT / sid / "capture" / "private"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in tr.items()
                      if k in ("sample_id", "difficulty", "interferers", "noise",
                               "answerable", "qa")},
                     ensure_ascii=False, indent=1)[:2500])
    print("\n--- achieved (逐站 pr_dbm / los) ---")
    print(json.dumps(tr.get("achieved"), ensure_ascii=False, indent=1)[:1500])
    print("\n--- 站坐标 ---")
    for r in obs["receivers"]:
        print("  ", r["receiver_id"], [round(v, 1) for v in r["receiver_position_m"]])
    print("  目标(GT)  ", [round(v, 1) for v in tr["target"]["position_m"]])
    for j, g in enumerate(tr.get("interferers", [])):
        print(f"  干扰{j+1} {g['kind']:>12s}",
              [round(v, 1) for v in g["position_m"][:2]],
              "f=", round(g["freq_offset_hz"] / 1e3, 1), "k")
    # 目标-干扰 与 干扰-干扰 的 u 分离（阵列可分辨性预检）
    print("\n--- |Δu| 预检（阵轴 +y 假定，仅作诊断）---")
    def u_of(p, o):
        d = np.asarray(p[:2], float) - np.asarray(o[:2], float)
        return float(d[1] / (np.linalg.norm(d) + 1e-12))
    for r in obs["receivers"]:
        o = r["receiver_position_m"]
        us = [(f"ifw{j+1}", u_of(g["position_m"], o))
              for j, g in enumerate(tr.get("interferers", []))]
        ut = ("target", u_of(tr["target"]["position_m"], o))
        print(f"  {r['receiver_id']}: {ut} " + " ".join(f"{n}={v:.3f}" for n, v in us))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d2_00000")
