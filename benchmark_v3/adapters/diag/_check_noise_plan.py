"""零 API：核对"按站随机配方"的分配是否均衡、是否覆盖到位。

检查四件事（对着 `noise_profiles.py` 的菜单与权重）：
  1. 每档配方**总占比**是否等于权重（分层是否真的按权重落）；
  2. **站间均衡**：四个站位置各自看到的配方分布是否一致（防止 station_a 系统性偏干净，
     因为 S0 永远只用 station_a —— 那会让 S0 整体偏简单）；
  3. 每档"四站全热噪"的样本数（档内对照组够不够）；
  4. 打印每档前 8 个样本的分配表供人眼过一遍。

用法：python _check_noise_plan.py [--n 50] [--stations 4]
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import argparse
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))  # adapters（迁移后 HERE 下沉一级）

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import noise_profiles as nprof  # noqa: E402

STATIONS = ["station_a", "station_b", "station_c", "station_d"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="每档样本数")
    ap.add_argument("--stations", type=int, default=4)
    args = ap.parse_args()

    stations = STATIONS[: args.stations]
    ok = True
    for diff in ("d1", "d2", "d3", "d4"):
        plan_fn = getattr(nprof, "plan_sample", None)
        if plan_fn is None:
            raise SystemExit("noise_profiles.py 里还没有 plan_sample()")
        rows = [plan_fn(diff, stations, i) for i in range(args.n)]

        total = collections.Counter()
        per_station = [collections.Counter() for _ in stations]
        all_thermal = 0
        for r in rows:
            archs = [r[sid] for sid in stations]
            total.update(archs)
            for k, a in enumerate(archs):
                per_station[k][a] += 1
            if set(archs) == {"thermal"}:
                all_thermal += 1

        n_slots = args.n * len(stations)
        print(f"\n=== {diff}  菜单 {nprof.MENUS[diff]}  样本 {args.n} × 站 {len(stations)} = {n_slots} 槽 ===")
        for name, w_slots in nprof.MENUS[diff]:
            w = w_slots / nprof.B_SLOTS   # 菜单权重是"10 个槽里占几个"，占比 = w_slots/B_SLOTS
            cnt = total.get(name, 0)
            share = cnt / n_slots
            flag = "OK" if abs(share - w) <= 0.02 else "**偏差**"
            print(f"  {name:<12s} 目标 {w:5.2f}  实际 {share:5.2f} ({cnt:3d} 槽)  {flag}")
            if abs(share - w) > 0.02:
                ok = False
        print(f"  四站全热噪的样本数: {all_thermal}/{args.n}")

        print("  站间均衡（每站各配方槽数，应完全一致）:")
        keys = [n for n, _ in nprof.MENUS[diff]]
        for k, sid in enumerate(stations):
            line = "  ".join(f"{key}:{per_station[k].get(key, 0):3d}({per_station[k].get(key, 0)/args.n:.2f})" for key in keys)
            print(f"    {sid:<10s} {line}")
        for key in keys:
            vals = [per_station[k].get(key, 0) for k in range(len(stations))]
            if max(vals) - min(vals) > 1:
                print(f"    **站间不均衡**: {key} -> {vals}")
                ok = False

        print("  前 8 个样本的分配:")
        for i, r in enumerate(rows[:8]):
            print(f"    样本 {i:2d}: " + "  ".join(f"{sid.split('_')[1]}={r[sid]}" for sid in stations))

    print("\n[结论]", "全部均衡" if ok else "**存在偏差，见上**")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
