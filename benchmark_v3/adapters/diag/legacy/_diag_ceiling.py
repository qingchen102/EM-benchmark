"""临时只读：D2 召回天花板分解 —— 可见性硬上限 vs 峰检测层损失。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "dataset_d2"
RAY = 2.0 / 8.0


def main() -> None:
    idx = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
    c: Counter = Counter()
    n = 0
    pair_tot = pair_sub = 0
    for s in idx["samples"][:50]:
        t = json.loads((ROOT / s["id"] / "capture" / "private" / "d1_truth.json")
                       .read_text(encoding="utf-8"))
        nz = t["noise"]["noise_floor_dbm"]
        ach = t.get("achieved") or {}
        ifw = t.get("interferers", [])
        for j in range(len(ifw)):
            eid = f"interferer_{j+1}"
            k = sum(1 for i in range(4)
                    if (ach.get(eid, {}).get(f"s{i}", {}).get("pr_dbm", -999) - nz) >= 3.0)
            c[k] += 1
            n += 1
        for i in range(len(ifw)):
            for j in range(i + 1, len(ifw)):
                pair_tot += 1
                for sk in ("s0", "s1", "s2", "s3"):
                    a = ifw[i]["per_station"].get(sk)
                    b = ifw[j]["per_station"].get(sk)
                    if not a or not b:
                        continue
                    du = abs(abs(np.sin(np.deg2rad(a["doa_deg"])))
                             - abs(np.sin(np.deg2rad(b["doa_deg"]))))
                    if du < RAY:
                        pair_sub += 1
                        break
    ge3 = sum(v for k, v in c.items() if k >= 3)
    ge2 = sum(v for k, v in c.items() if k >= 2)
    print(f"D2 前50样本 干扰总数 = {n}")
    print(f"  每干扰 可见站数(INR>=3dB) 分布: {dict(sorted(c.items()))}")
    print(f"  >=3 站可见 = 任何算法的硬上限: {ge3}/{n} = {100*ge3/n:.0f}%")
    print(f"  >=2 站可见 = 硬上限: {ge2}/{n} = {100*ge2/n:.0f}%")
    print(f"  干扰对中存在某站 Δu<2/M 的对: {pair_sub}/{pair_tot} = "
          f"{100*pair_sub/max(pair_tot,1):.0f}%")
    print(f"  当前 gate 召回 29.2% / 可见性硬上限 {100*ge3/n:.0f}% "
          f"=> 峰检测层已损失 {100*(1-29.2/(100*ge3/n)):.0f}%")


if __name__ == "__main__":
    main()
