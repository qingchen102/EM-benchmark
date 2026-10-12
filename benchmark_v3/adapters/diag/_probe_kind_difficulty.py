"""按档统计"源侧难度"：体制构成、功率/带宽/可见性、同频带共占。

为什么：四档的体制识别分数无法直接横向比——分母（认领/可见的 GT 切片）的体制构成、
源功率与带宽、以及同频伙伴共占情况都不同。这个只读探针把"源侧"这些量摊开，
再与 gate 的按类命中率并列读，才能判断"D2 为什么弱"。

用法：python -X utf8 diag/_probe_kind_difficulty.py [root] [d1,d2,d3,d4]
      默认 root = benchmark_v3（用当前配额数据），可传 ../_pre_quota_20261012 做旧档对照。
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # benchmark_v3/adapters/diag
_V3 = HERE.parents[1]                            # benchmark_v3

DEAD = {"Zigbee", "WiFi_burst"}                  # gate 里恒为 0 的两类


def load(root: Path, deck: str):
    d = root / f"dataset_{deck}"
    idx_p = d / "index.json"
    if not idx_p.is_file():
        return None
    idx = json.loads(idx_p.read_text(encoding="utf-8"))
    rows = []
    for s in idx["samples"]:
        tp = d / s["id"] / "capture" / "private" / "d1_truth.json"
        if not tp.is_file():
            continue
        rows.append(json.loads(tp.read_text(encoding="utf-8")))
    return rows


def overlap_hz(a0, ab, b0, bb):
    lo = max(a0 - ab / 2, b0 - bb / 2)
    hi = min(a0 + ab / 2, b0 + bb / 2)
    return max(0.0, hi - lo)


def report(root: Path, deck: str):
    truths = load(root, deck)
    if not truths:
        print(f"[{deck}] 无数据，跳过")
        return
    per_kind = {}
    n_src = 0
    n_samp = len(truths)
    per_sample_n = []
    ovl_fracs = []          # 每个源：被同频伙伴覆盖的带宽占自身带宽比例
    inr_all = []
    for t in truths:
        ifws = t.get("interferers") or []
        per_sample_n.append(len(ifws))
        for i, a in enumerate(ifws):
            n_src += 1
            k = a["kind"]
            ps = a.get("per_station") or {}
            inrs = [v.get("inr_db") for v in ps.values() if isinstance(v.get("inr_db"), (int, float))]
            vis = sum(1 for v in inrs if v >= 3.0)
            rec = per_kind.setdefault(k, {"n": 0, "bw": [], "pw": [], "inr": [], "vis": []})
            rec["n"] += 1
            rec["bw"].append(float(a.get("bandwidth_hz") or 0.0))
            rec["pw"].append(float(a.get("power_dbm") or 0.0))
            rec["inr"].append(max(inrs) if inrs else 0.0)
            rec["vis"].append(vis)
            inr_all.append(max(inrs) if inrs else 0.0)
            # 与同样本其它源的频带共占
            covered = 0.0
            for j, b in enumerate(ifws):
                if i == j:
                    continue
                covered = max(covered, overlap_hz(a["freq_offset_hz"], a["bandwidth_hz"],
                                                  b["freq_offset_hz"], b["bandwidth_hz"]))
            bw = float(a.get("bandwidth_hz") or 0.0)
            ovl_fracs.append(covered / bw if bw > 0 else 0.0)

    print(f"\n===== {deck}（{root.name}）  样本 {n_samp}  源 {n_src}  "
          f"每样本源数 中位 {st.median(per_sample_n):.0f} 均值 {st.mean(per_sample_n):.2f}  =====")
    print(f"{'体制':16s} {'n':>4s} {'占比':>7s} {'带宽中位(kHz)':>13s} {'发射功率(dBm)':>13s} "
          f"{'最强INR(dB)':>12s} {'可见站数':>9s} {'死类':>5s}")
    for k, r in sorted(per_kind.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{k:16s} {r['n']:>4d} {r['n'] / n_src:>7.1%} {st.median(r['bw']) / 1e3:>13.1f} "
              f"{st.mean(r['pw']):>13.2f} {st.median(r['inr']):>12.1f} "
              f"{st.mean(r['vis']):>9.2f} {'★' if k in DEAD else '':>5s}")
    print(f"  ── 窄带源占比（带宽中位 < 300 kHz 的类，按源数加权）："
          f"{sum(r['n'] for k, r in per_kind.items() if st.median(r['bw']) < 300e3) / n_src:.1%}")
    print(f"  ── 死类（Zigbee/WiFi_burst）源占比：{sum(r['n'] for k, r in per_kind.items() if k in DEAD) / n_src:.1%}")
    print(f"  ── 同频共占：有同频伙伴（覆盖>0）的源占比 "
          f"{sum(1 for v in ovl_fracs if v > 0) / len(ovl_fracs):.1%}；"
          f"覆盖>50% 的源占比 {sum(1 for v in ovl_fracs if v > 0.5) / len(ovl_fracs):.1%}；"
          f"覆盖比例中位 {st.median(ovl_fracs):.1%}")
    print(f"  ── 最强 INR 中位 {st.median(inr_all):.1f} dB（{len(inr_all)} 个源·站取最大）")


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else _V3
    if not root.is_absolute():
        here = (Path.cwd() / root)
        root = here.resolve() if here.is_dir() else (_V3 / root).resolve()
    decks = (argv[2] if len(argv) > 2 else "d1,d2,d3,d4").split(",")
    print(f"[root] {root}")
    for d in decks:
        report(root, d.strip())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv)
