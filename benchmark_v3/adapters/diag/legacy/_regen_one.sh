#!/usr/bin/env bash
# 重生成 D3/D4（补 QA 三约束 + 换 seed0，见用户 2026-10-08 决定）。
# 产出到 _regen_d34/dataset_<tier>/，**不动现存 dataset_d3 / dataset_d4**（它们是唯一副本、且未入库）。
# 用 --resume 断点续跑；若某样本 6 次 QA 重抽全败，就 bump seed0 重试，
# 已完成样本由 --resume 跳过、不被重做（仅该样本自身用到 seed，故 bump 不影响已完成的）。
set -u
TIER="${1:?用法: _regen_one.sh <d3|d4> <seed0>}"
SEED0="${2:?}"
BASE=/mnt/d/srtp/EM-benchmark/benchmark_v3
PY="$BASE/城市低空电磁信号仿真平台/.venv-rt/bin/python"
OUT="$BASE/_regen_d34/dataset_$TIER"
LOG="$BASE/_regen_d34/_regen_$TIER.log"
mkdir -p "$BASE/_regen_d34"
cd "$BASE/adapters" || exit 1

echo "=== $TIER 重生成开始 $(date +'%F %T')  初始 seed0=$SEED0 ===" | tee -a "$LOG"
# 每轮快则几十秒；某轮若在样本 i 处 6 次 QA 全败，bump 后已完成样本被 --resume 跳过、
# 只有 i 及之后的样本换新 seed。60 轮足够把 50/50 凑满（几何一次通过率实测 34%）。
for bump in $(seq 0 59); do
  s=$((SEED0 + bump))
  echo "--- 第 $((bump + 1)) 轮  seed0=$s  $(date +'%T') ---" | tee -a "$LOG"
  "$PY" -u gen_dataset.py --difficulty "$TIER" --count 50 --seed0 "$s" \
        --out "$OUT" --resume >>"$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== $TIER 全部完成 $(date +'%T') ===" | tee -a "$LOG"
    break
  fi
  echo "!! 第 $((bump + 1)) 轮退出码 $rc，bump seed0 重试（已完成样本会跳过）" | tee -a "$LOG"
done
echo "--- $TIER 磁盘上已生成样本数（应为 50）---" | tee -a "$LOG"
N=$(ls "$OUT"/${TIER}_*/capture/private/d1_truth.json 2>/dev/null | wc -l)
echo "$N" | tee -a "$LOG"
if [ "$N" -ne 50 ]; then
  echo "!! $TIER 只生成 $N/50，未完成（最后一轮退出码 $rc）" | tee -a "$LOG"
  exit 1
fi
echo "=== $TIER 收工：50/50 ===" | tee -a "$LOG"
exit 0
