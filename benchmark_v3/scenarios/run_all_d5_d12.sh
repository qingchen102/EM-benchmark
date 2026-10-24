#!/usr/bin/env bash
# 一次生成 d5–d12（十二张场景卡中的后八张）。每档 50 个接受样本。
# 用法：bash benchmark_v3/scenarios/run_all_d5_d12.sh
# 输出：benchmark_v3/dataset_d5 … dataset_d12（已被 .gitignore 排除）
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
V3="$(cd -- "$HERE/.." && pwd)"

# 场景 -> seed0。与已入库数据集使用的种子保持一致，便于复现。
declare -A SEED=(
  [d5]=51000
  [d6]=52000
  [d7]=53000
  [d8]=54000
  [d9]=55000
  [d10]=56000
  [d11]=57000
  [d12]=58000
)

for d in d5 d6 d7 d8 d9 d10 d11 d12; do
  pool=50
  case "$d" in
    d7) pool=200 ;;          # 同频约束强，首次 50 候选只接受 41 条
    d9|d10|d11|d12) pool=120 ;;  # 含布点/QA 拒绝，留足候选
  esac
  echo "########## $d  seed0=${SEED[$d]}  pool=$pool"
  bash "$HERE/run_dataset.sh" "$d" 50 "${SEED[$d]}" "$pool" "$V3/dataset_$d"
done

echo "ALL_DONE $(date '+%F %T')"
