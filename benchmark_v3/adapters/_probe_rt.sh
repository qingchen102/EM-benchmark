#!/usr/bin/env bash
# 探测 RT 生成链路（一次性验证脚本）
set -u
PY="/mnt/d/srtp/EM-benchmark/benchmark_v3/城市低空电磁信号仿真平台/.venv-rt/bin/python"
echo "== python: $PY"
ls -l "$PY" 2>&1 | head -2
"$PY" -V 2>&1
echo "== import 探测 =="
"$PY" - <<'EOF' 2>&1
import sys, importlib
for m in ("numpy", "mitsuba", "sionna", "sionna.rt"):
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:12s} {type(e).__name__}: {e}")
EOF
echo "== 平台源码可导入？ =="
cd /mnt/d/srtp/EM-benchmark/benchmark_v3/adapters || exit 1
"$PY" - <<'EOF' 2>&1
import sys
sys.path.insert(0, "/mnt/d/srtp/EM-benchmark/benchmark_v3/城市低空电磁信号仿真平台/src")
try:
    from cityrf import runner
    print("  OK   cityrf.runner")
except Exception as e:
    print(f"  FAIL cityrf.runner {type(e).__name__}: {e}")
EOF
