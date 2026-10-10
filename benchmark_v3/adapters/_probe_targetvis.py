"""只读探针（零 token）：逐档统计「目标在哪些站根本没有链路」。

判据与 `oracle16._target_ref_state` 一致：
  missing   —— capture/private/channel_target_{st}.npz 不存在；
  invisible —— 文件在但 delays_s.size == 0（该站对目标 0 径）；
  ok        —— 有路径。

后果：目标参考缺失的站会让 `estimate_band_occupancy_by_correlation(st)` 直接返回
{"error": "target reference not available for station ..."} —— 该站的**残差工具整条不可用**。
S0（只有 station_a）的样本若 station_a 不可用则完全无法作答；S1/S2 可退到别的站。

用法：python _probe_targetvis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_V3 = Path(__file__).resolve().parent.parent
STATIONS = ["station_a", "station_b", "station_c", "station_d"]


def state(cap: Path, st: str) -> str:
    p = cap / "private" / f"channel_target_{st}.npz"
    if not p.is_file():
        return "missing"
    try:
        d = np.load(p)
        return "ok" if np.asarray(d["delays_s"]).size > 0 else "invisible"
    except Exception:                                             # noqa: BLE001
        return "missing"


def main() -> None:
    for d in ("d1", "d2", "d3", "d4"):
        root = _V3 / f"dataset_{d}"
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        sids = [s["id"] for s in idx["samples"]]
        cnt = Counter()
        bad_a = []
        bad_all = []
        for sid in sids:
            cap = root / sid / "capture"
            states = {st: state(cap, st) for st in STATIONS}
            for st, v in states.items():
                cnt[(st, v)] += 1
            if states["station_a"] != "ok":
                bad_a.append((sid, states["station_a"]))
            if all(v != "ok" for v in states.values()):
                bad_all.append((sid, dict(states)))
        print(f"\n=== {d}: {len(sids)} 样本 ===")
        for st in STATIONS:
            print(f"   {st:10s} ok {cnt[(st,'ok')]:3d}  invisible {cnt[(st,'invisible')]:3d}"
                  f"  missing {cnt[(st,'missing')]:3d}")
        print(f"   station_a 不可用: {len(bad_a)}/{len(sids)}  {[s for s, _ in bad_a][:12]}")
        print(f"   四站全不可用:    {len(bad_all)}/{len(sids)}")


if __name__ == "__main__":
    main()
