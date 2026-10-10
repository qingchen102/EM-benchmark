#!/usr/bin/env bash
# 一次性候选池重生成（用户 2026-10-08 m01643 方案）：
#   「一个 seed 多生成一些样本，再按顺序筛出 --want 条满足的」——
#   不再用 _regen_one.sh 那种 bump seed0 的外层循环。
# 产出到 _regen_d34/pool_<tier>/，**不动现存 dataset_d3 / dataset_d4**。
# --qa-skip：单候选连续 6 次 QA/RT 失败就跳过该候选（用下一个 seed0 候选），不中断整跑。
# 输出槽位按已接受数连续命名，index 的 seed 记真值里的实际 seed。
set -u
TIER="${1:?用法: _regen_pool.sh <d3|d4> <seed0> [count] [want]}"
SEED0="${2:?}"
COUNT="${3:-150}"
WANT="${4:-50}"
BASE=/mnt/d/srtp/EM-benchmark/benchmark_v3
PY="$BASE/城市低空电磁信号仿真平台/.venv-rt/bin/python"
OUT="$BASE/_regen_d34/pool_$TIER"
LOG="$BASE/_regen_d34/_regen_pool_$TIER.log"
mkdir -p "$BASE/_regen_d34"
cd "$BASE/adapters" || exit 1

echo "=== $TIER 候选池重生成开始 $(date +'%F %T')  seed0=$SEED0  count=$COUNT  want=$WANT -> $OUT ===" | tee -a "$LOG"
"$PY" -u gen_dataset.py --difficulty "$TIER" --count "$COUNT" --want "$WANT" \
      --qa-skip --seed0 "$SEED0" --out "$OUT" >>"$LOG" 2>&1
rc=$?
echo "--- $TIER 磁盘上已生成样本数（应为 $WANT）---" | tee -a "$LOG"
N=$(ls "$OUT"/${TIER}_*/capture/private/d1_truth.json 2>/dev/null | wc -l)
echo "$N" | tee -a "$LOG"
if [ "$N" -ne "$WANT" ]; then
  echo "!! $TIER 只生成 $N/$WANT，未完成（退出码 $rc）" | tee -a "$LOG"
  exit 1
fi
echo "=== $TIER 收工：$WANT/$WANT（候选池消耗见日志末尾） ===" | tee -a "$LOG"
exit 0
