#!/usr/bin/env bash
# 对比：新生成 _probe_d4/d4_00000（当前代码, seed0=31000）vs 现存 dataset_d4/d4_00000
set -u
BASE=/mnt/d/srtp/EM-benchmark/benchmark_v3
PY="$BASE/城市低空电磁信号仿真平台/.venv-rt/bin/python"
"$PY" - "$BASE/_probe_d4/d4_00000" "$BASE/dataset_d4/d4_00000" <<'EOF'
import json, sys, os
import numpy as np
new, old = sys.argv[1], sys.argv[2]

def load(root):
    p = os.path.join(root, "capture/private/truth.json")
    d1 = os.path.join(root, "capture/private/d1_truth.json")
    return json.load(open(p, encoding="utf-8")), json.load(open(d1, encoding="utf-8"))

tn, dn = load(new)
to, do = load(old)

print("== truth.json 顶层键 ==")
print("  新:", sorted(tn.keys()))
print("  旧:", sorted(to.keys()))

def show(t, tag):
    print(f"-- {tag} --")
    tgt = t.get("target", {})
    for k in ("position_m", "position_xy_m", "kind", "class", "tx_power_dbm", "velocity_mps"):
        if k in tgt:
            v = tgt[k]
            print(f"   target.{k} = {np.round(v,3) if isinstance(v,list) else v}")
    ifs = t.get("interferers", [])
    print(f"   干扰数 = {len(ifs)}")
    for j, it in enumerate(ifs):
        per = it.get("per_station") or {}
        vis = sum(1 for st, d in per.items() if (d or {}).get("inr_db", -99) >= 3.0)
        print(f"     [{j}] kind={it.get('kind'):16s} pos={np.round(it.get('position_m',[0,0,0]),1)} "
              f"fc={it.get('center_hz', it.get('freq_hz','?'))} 可见站={vis}/4")
    for key in ("interference_events", "envelope", "noise"):
        if key in t:
            print(f"   {key} = {json.dumps(t[key], ensure_ascii=False)[:300]}")

show(tn, "新生成")
show(to, "现存 d4_00000")

print("\n== d1_truth.json（agent 读的那份）键 ==")
print("  新:", sorted(dn.keys()))
print("  旧:", sorted(do.keys()))
for tag, d in (("新", dn), ("旧", do)):
    tg = d.get("target", {})
    ifs = d.get("interferers", [])
    print(f"  {tag}: target keys={sorted(tg.keys())}")
    for j, it in enumerate(ifs):
        per = it.get("per_station") or {}
        n = sum(1 for st, x in per.items() if (x or {}).get("inr_db", -99) >= 3.0)
        print(f"     [{j}] {it.get('kind'):16s} 可见站={n}/4")
EOF
