#!/usr/bin/env bash
# 对比目标几何/种子：新生成 vs 现存 d4_00000
set -u
BASE=/mnt/d/srtp/EM-benchmark/benchmark_v3
PY="$BASE/城市低空电磁信号仿真平台/.venv-rt/bin/python"
"$PY" - "$BASE/_probe_d4/d4_00000" "$BASE/dataset_d4/d4_00000" <<'EOF'
import json, sys, os
import numpy as np

def L(root):
    return json.load(open(os.path.join(root, "capture/private/d1_truth.json"), encoding="utf-8"))

for tag, root in (("新", sys.argv[1]), ("旧", sys.argv[2])):
    t = L(root)
    tg = t["target"]
    print(f"== {tag} ==")
    print(f"  seed            = {t.get('seed')}")
    print(f"  目标位置         = {np.round(tg['position_m'], 1)}")
    print(f"  目标功率         = {tg['power_dbm']:.2f} dBm")
    print(f"  目标频偏         = {tg['freq_offset_hz']}")
    print(f"  目标 per_station = {json.dumps(tg['per_station'], ensure_ascii=False)[:600]}")
    print(f"  干扰数           = {len(t['interferers'])}")
    for j, it in enumerate(t["interferers"]):
        print(f"    [{j}] {it['kind']:12s} pos={np.round(it['position_m'],1)} "
              f"pwr={it['power_dbm']:.2f}dBm 频偏={it.get('freq_offset_hz')} "
              f"带宽={it.get('bandwidth_hz')}")
    print(f"  envelope        = {t.get('envelope')}")
    print(f"  events          = {json.dumps(t.get('interference_events'), ensure_ascii=False)[:300]}")
    print(f"  scene_type      = {t.get('scene_type')}  difficulty={t.get('difficulty')}")
EOF
