#!/usr/bin/env bash
# 复现性探测：当前 gen_dataset.py 能否复现 d4_00000（seed0=31000）？
set -u
BASE=/mnt/d/srtp/EM-benchmark/benchmark_v3
PY="$BASE/城市低空电磁信号仿真平台/.venv-rt/bin/python"
cd "$BASE/adapters" || exit 1
OUT="$BASE/_probe_d4"
rm -rf "$OUT"
echo "== 复现探测：d4 seed0=31000 count=1  $(date +%H:%M:%S) =="
time "$PY" gen_dataset.py --difficulty d4 --count 1 --seed0 31000 --out "$OUT" 2>&1 | tail -25
echo "== 结束 $(date +%H:%M:%S) =="
NEW="$OUT/d4_00000"
OLD="$BASE/dataset_d4/d4_00000"
if [ ! -d "$NEW" ]; then echo "!! 未生成，复现失败"; exit 0; fi
echo "== 与现存 d4_00000 对比 =="
"$PY" - "$NEW" "$OLD" <<'EOF'
import json, sys, numpy as np
new, old = sys.argv[1], sys.argv[2]
def load(root):
    t = json.load(open(f"{root}/capture/private/d4_truth.json", encoding="utf-8"))
    return t
try:
    tn, to = load(new), load(old)
except Exception as e:
    print("  读真值失败:", e); raise SystemExit
def tgt(t):
    a = t.get("achieved", {}); return t["target"]
for shade in (None,):
    pass
# 目标位置
try:
    pn = tn["target"]["position_m"]; po = to["target"]["position_m"]
    print(f"  目标位置  新 {np.round(pn,1)}   旧 {np.round(po,1)}")
except Exception as e:
    print("  目标位置键缺失:", e)
kn = [i["kind"] for i in tn.get("interferers", [])]
ko = [i["kind"] for i in to.get("interferers", [])]
print(f"  干扰体制  新 {kn}")
print(f"            旧 {ko}")
# 逐站可见性
def vis(t, tag):
    out = []
    for j, it in enumerate(t.get("interferers", [])):
        n = sum(1 for st, d in (it.get("per_station") or {}).items()
                if (d or {}).get("inr_db", -99) >= 3.0)
        out.append(f"{it['kind']}:{n}/4")
    print(f"  {tag} 可见站数 {out}")
vis(tn, "新"); vis(to, "旧")
# 目标逐站 SNR
def snr(t, tag):
    a = t.get("achieved") or {}
    print(f"  {tag} achieved 键: {sorted(a.keys())[:6]}")
snr(tn, "新"); snr(to, "旧")
EOF
