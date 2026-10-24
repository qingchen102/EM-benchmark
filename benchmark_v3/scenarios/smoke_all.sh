#!/usr/bin/env bash
# 冒烟：每个场景只生成 2 个接受样本，验证链路是否通（不写入正式 dataset_d*）。
# 用法：bash benchmark_v3/scenarios/smoke_all.sh [scene ...]
# 不带参数时跑 d1–d12；输出到 benchmark_v3/_regen_scene_smoke/（已 gitignore）。
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
V3="$(cd -- "$HERE/.." && pwd)"
OUT_ROOT="$V3/_regen_scene_smoke"

if [ "$#" -gt 0 ]; then
  SCENES=("$@")
else
  SCENES=(d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12)
fi

for d in "${SCENES[@]}"; do
  echo "########## smoke $d"
  bash "$HERE/run_dataset.sh" "$d" 2 31000 6 "$OUT_ROOT/dataset_$d"
done

echo "SMOKE_DONE $(date '+%F %T')"
