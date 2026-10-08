# -*- coding: utf-8 -*-
# _regen_d34 验收：完整性 + RT 实测可见站数 + 答案键尺寸（官方口径）+ 干扰种类分布。
# 只读，不写任何数据集。旧 dataset_d3/d4 作为对照一起打印。
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_dataset import gt_at_slevel  # noqa: E402

SUBSETS = ("S0", "S1", "S2")
def _targets():
    """默认对照表；命令行可覆盖：python _audit_regen.py <label>=<path> ..."""
    if len(sys.argv) > 1:
        out = []
        for a in sys.argv[1:]:
            lab, _, p = a.partition("=")
            out.append((os.path.abspath(p or lab),
                        lab if p else os.path.basename((p or lab).rstrip("/\\"))))
        return out
    return [
        (os.path.join(ROOT, "_regen_d34", "pool_d3"), "D3 候选池(补QA+换seed0)"),
        (os.path.join(ROOT, "_regen_d34", "pool_d4"), "D4 候选池(补QA+换seed0)"),
        (os.path.join(ROOT, "_regen_d34", "dataset_d3"), "D3 重生成(旧 bump 机制)"),
        (os.path.join(ROOT, "_regen_d34", "dataset_d4"), "D4 重生成(旧 bump 机制)"),
        (os.path.join(ROOT, "dataset_d3"), "D3 旧(QA前) 对照"),
        (os.path.join(ROOT, "dataset_d4"), "D4 旧(QA前) 对照"),
    ]

TARGETS = _targets()


def audit(ds_path, label):
    print("=" * 78)
    print(f"### {label}")
    print(f"    {ds_path}")
    if not os.path.isdir(ds_path):
        print("    （不存在）")
        return
    ip = os.path.join(ds_path, "index.json")
    sids = []
    if os.path.exists(ip):
        with open(ip, encoding="utf-8") as f:
            sids = [s["id"] for s in json.load(f)["samples"]]
    dirs = sorted(d for d in os.listdir(ds_path)
                  if os.path.isdir(os.path.join(ds_path, d)))
    print(f"    index 样本 {len(sids)}   目录 {len(dirs)}")

    incomplete = []
    vis_hist = Counter()
    kinds = Counter()
    per_sub = {k: [] for k in SUBSETS}
    n_no_ifw = 0
    n_ok = 0
    for sid in dirs:
        priv = os.path.join(ds_path, sid, "capture", "private")
        publ = os.path.join(ds_path, sid, "capture", "public")
        iqs = ([f for f in os.listdir(publ) if f.startswith("iq_station_")]
               if os.path.isdir(publ) else [])
        tp = os.path.join(priv, "d1_truth.json")
        if len(iqs) != 4 or not os.path.exists(tp):
            incomplete.append(f"{sid}(iq={len(iqs)},truth={os.path.exists(tp)})")
            continue
        n_ok += 1
        with open(tp, encoding="utf-8") as f:
            t = json.load(f)
        if not t.get("interferers"):
            n_no_ifw += 1
        for g in t["interferers"]:
            kinds[g["kind"]] += 1
        ach = t.get("achieved") or {}
        nf = (t.get("noise") or {}).get("noise_floor_dbm")
        for i_g, _g in enumerate(t["interferers"]):
            eid = "interferer_%d" % (i_g + 1)
            nvis = 0
            for _sk, a in (ach.get(eid) or {}).items():
                try:
                    if a["pr_dbm"] - nf >= 3.0:
                        nvis += 1
                except (TypeError, KeyError):
                    pass
            vis_hist[nvis] += 1
        ss = t.get("station_subsets") or {}
        for k in SUBSETS:
            if ss.get(k):
                per_sub[k].append(len(gt_at_slevel(t, ss[k])))
    print(f"    完整样本 {n_ok}   不完整 {len(incomplete)} {incomplete[:6]}")
    print(f"    无干扰样本 {n_no_ifw}")
    tot = sum(kinds.values())
    print(f"    干扰总数 {tot}   种类 {dict(kinds.most_common())}")
    print(f"    RT 实测可见站数分布（全部干扰, >=3dB）{dict(sorted(vis_hist.items()))}")
    full = vis_hist.get(3, 0) + vis_hist.get(4, 0)
    if tot:
        print(f"    ≥3 站可见占比 {full / tot:.1%}   <3 站（白占仿真时间）{1 - full / tot:.1%}")
    for k in SUBSETS:
        v = per_sub[k]
        if not v:
            continue
        print(f"    [{k}] 键内干扰均值 {sum(v) / len(v):.2f}  "
              f"分布 {dict(sorted(Counter(v).items()))}  "
              f"'报0即正确' {sum(1 for x in v if x == 0) / len(v):.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p, lab in TARGETS:
        audit(p, lab)
