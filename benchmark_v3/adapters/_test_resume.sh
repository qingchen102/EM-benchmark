#!/usr/bin/env bash
# 验证 gen_dataset.py 的 --resume：应跳过已存在的 d4_00000 且完全不跑 RT。
set -u
BASE=/mnt/d/srtp/EM-benchmark/benchmark_v3
PY="$BASE/城市低空电磁信号仿真平台/.venv-rt/bin/python"
cd "$BASE/adapters" || exit 1
echo "--- 语法检查 ---"
"$PY" -m py_compile gen_dataset.py && echo "语法 OK"
echo "--- --resume 测试（应打印'已存在，跳过'，耗时几秒）---"
time "$PY" gen_dataset.py --difficulty d4 --count 1 --seed0 31000 \
     --out "$BASE/_probe_d4" --resume 2>&1 | tail -6
echo "--- 生成的 index.json ---"
cat "$BASE/_probe_d4/index.json"
