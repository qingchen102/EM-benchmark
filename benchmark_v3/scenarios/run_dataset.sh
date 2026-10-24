#!/usr/bin/env bash
# 生成单个场景的数据集。所有“按场景跑”的入口都放在本目录，避免散落在 adapters/ 里。
#
# 用法（在 WSL/Linux 下，需先有平台 .venv-rt）：
#   bash benchmark_v3/scenarios/run_dataset.sh <scene> [want] [seed0] [pool] [out]
#
#   scene   d1..d12
#   want    最终要接受的样本数，默认 50
#   seed0   起始随机种子；各场景的约定值见 README
#   pool    候选池大小，默认等于 want；d7 这类约束强的场景需要更大的池
#   out     输出目录，默认 benchmark_v3/dataset_<scene>
#
# 可选环境变量：
#   PY       指定 Python（默认平台 .venv-rt/bin/python）
#   DURATION 窗长秒数（默认沿用生成器配置；长窗冒烟可设 1.0）
#
# 示例：
#   bash benchmark_v3/scenarios/run_dataset.sh d5 50 51000
#   bash benchmark_v3/scenarios/run_dataset.sh d7 50 53000 200
set -euo pipefail

if [ "$#" -lt 1 ]; then
  sed -n '2,20p' "$0"
  exit 2
fi

SCENE="$1"
WANT="${2:-50}"
SEED0="${3:-31000}"
POOL="${4:-$WANT}"
OUT="${5:-}"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
V3="$(cd -- "$HERE/.." && pwd)"
PLATFORM="$V3/城市低空电磁信号仿真平台"
PY="${PY:-$PLATFORM/.venv-rt/bin/python}"

if [ -z "$OUT" ]; then
  OUT="$V3/dataset_$SCENE"
fi
case "$OUT" in
  /*) ;;
  *) OUT="$V3/$OUT" ;;
esac

if [ ! -x "$PY" ]; then
  echo "找不到可执行的 Python：$PY" >&2
  echo "请先准备平台 .venv-rt，或用 PY=/path/to/python 指定。" >&2
  exit 3
fi

mkdir -p "$OUT"
cd "$V3/adapters"

EXTRA=()
if [ -n "${DURATION:-}" ]; then
  EXTRA+=(--duration "$DURATION")
fi

echo "=== scene=$SCENE want=$WANT pool=$POOL seed0=$SEED0 out=$OUT"
exec "$PY" -u gen_dataset.py \
  --difficulty "$SCENE" \
  --count "$POOL" \
  --want "$WANT" \
  --qa-skip \
  --seed0 "$SEED0" \
  --out "$OUT" \
  "${EXTRA[@]}"
