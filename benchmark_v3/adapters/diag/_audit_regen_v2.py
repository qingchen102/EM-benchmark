"""新生成数据集（机理/源数配额 v2）的只读验收入口（零 API）。

用法：
    python -X utf8 diag\\_audit_regen_v2.py <root> [d1,d2,d3,d4]
    # root 例：../_regen_v2   → 每档位于 <root>/dataset_dX

检查四项：
  1. 结构：index.json 与磁盘样本目录是否一致；每样本是否齐 4 站 IQ +
     private/d1_truth.json + public/observations.json；样本号是否连续；
  2. 配额：机理（用 mechanism.derive 从真值物理量复算）/ S2 可见源数 / 意图
     三项分布 + QA4 判定（D1 豁免，口径见 docs/机理配额与重生成方案.md §3）；
  3. 公开层：cooperative_sources（含 4 站链路遥测）与 interference_events 条数；
  4. 0-path：channel_target_station_*.npz 的 delays_s 是否为空（目标在该站不可见）
     —— 与 oracle16._target_ref_state 同判据。

exit code：0 = 全部通过；1 = 有 FAIL 项（明细打印在上方）。
"""
from __future__ import annotations

import json
import pathlib as _pl
import sys
from collections import Counter

import numpy as np

_HERE = _pl.Path(__file__).resolve().parent
_ADAPTERS = _HERE.parent
sys.path.insert(0, str(_ADAPTERS))
sys.path.insert(0, str(_ADAPTERS.parent / "simulation"))

import dataset_plan as dp                                        # noqa: E402
from em_signal_simulator_v3.emitters import INTENT               # noqa: E402

STATIONS = ["station_a", "station_b", "station_c", "station_d"]
D1_EXEMPT = ("d1",)


def _link_state(cap: _pl.Path, st: str) -> str:
    p = cap / "private" / f"channel_target_{st}.npz"
    if not p.is_file():
        return "missing"
    try:
        return "ok" if np.asarray(np.load(p)["delays_s"]).size > 0 else "invisible"
    except Exception:                                            # noqa: BLE001
        return "missing"


def audit_one(root: _pl.Path, diff: str) -> bool:
    ok = True
    print(f"\n=== dataset_{diff}  ({root}) ===")
    idx_p = root / "index.json"
    if not idx_p.is_file():
        print("  FAIL 缺 index.json")
        return False
    idx = json.loads(idx_p.read_text(encoding="utf-8"))
    sids = [s["id"] for s in idx["samples"]]
    dirs = sorted(p.name for p in root.glob(f"{diff}_*") if p.is_dir())
    print(f"  index 样本 {len(sids)} | 磁盘目录 {len(dirs)} | "
          f"子集 {idx.get('observation_subsets')}")
    if sids != dirs:
        ok = False
        print(f"  FAIL index 与磁盘目录不一致：只在一侧 {sorted(set(sids) ^ set(dirs))[:6]}")
    want_names = [f"{diff}_{i:05d}" for i in range(len(sids))]
    if sids and sids != want_names:
        ok = False
        print(f"  FAIL 样本号不连续（应为 {want_names[0]}..{want_names[-1]}）")

    truths, iq_sizes, link_cnt = [], Counter(), Counter()
    n_ev, n_coop = Counter(), Counter()
    for sid in sids:
        cap = root / sid / "capture"
        t_p = cap / "private" / "d1_truth.json"
        o_p = cap / "public" / "observations.json"
        if not t_p.is_file() or not o_p.is_file():
            ok = False
            print(f"  FAIL {sid} 缺 private/d1_truth.json 或 public/observations.json")
            continue
        truths.append(json.loads(t_p.read_text(encoding="utf-8")))
        for st in STATIONS:
            f = cap / "public" / f"iq_station_{st[-1]}.npy"
            iq_sizes[f"{st}"] += f.stat().st_size if f.is_file() else -1
            link_cnt[(st, _link_state(cap, st))] += 1
        obs = json.loads(o_p.read_text(encoding="utf-8"))
        cs = obs.get("cooperative_sources") or []
        n_coop["有" if (cs and cs[0].get("link_quality_per_station")) else "无"] += 1
        n_ev[len(obs.get("interference_events") or [])] += 1

    print("  IQ 单站字节：" + "  ".join(f"{st[-1]}={iq_sizes[st]}" for st in STATIONS))
    print("  公开层 cooperative_sources：" + "  ".join(f"{k} {v}" for k, v in n_coop.items())
          + "   |  事件日志条数分布 " + "  ".join(f"{k}:{v}" for k, v in sorted(n_ev.items())))
    if n_coop.get("无"):
        ok = False
        print(f"  FAIL {n_coop['无']} 个样本缺 cooperative_sources.link_quality_per_station")
    if diff in ("d3", "d4") and not [k for k in n_ev if k > 0]:
        ok = False
        print("  FAIL d3/d4 没有任何 interference_events")
    if diff in ("d1", "d2") and any(k > 0 for k in n_ev):
        print("  WARN d1/d2 出现 interference_events（按设计应为 0）")

    for st in STATIONS:
        bad = link_cnt[(st, "missing")] + link_cnt[(st, "invisible")]
        print(f"  0-path {st:10s} ok {link_cnt[(st, 'ok')]:3d}  "
              f"invisible {link_cnt[(st, 'invisible')]:3d}  missing {link_cnt[(st, 'missing')]:3d}")
        if st == "station_a" and bad > len(sids) * 0.25:
            print(f"  WARN station_a 不可用 {bad}/{len(sids)}（>25%）")

    qa_ok, lines = dp.audit_truths(truths, INTENT, diff)
    for ln in lines:
        print("  " + ln)
    if diff in D1_EXEMPT:
        print("  QA4：D1 豁免（0/1 源、legacy tone/noise 不计意图轴）")
    elif not qa_ok:
        ok = False
    return ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    root = _pl.Path(argv[0]).resolve()
    diffs = (argv[1].split(",") if len(argv) > 1 else ["d1", "d2", "d3", "d4"])
    all_ok = True
    for d in diffs:
        r = root / f"dataset_{d}"
        if not r.is_dir():
            print(f"\n=== dataset_{d} 缺目录：{r}")
            all_ok = False
            continue
        all_ok = audit_one(r, d) and all_ok
    print(f"\n{'=' * 60}\n总判定：{'全部通过' if all_ok else '存在 FAIL（见上）'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
