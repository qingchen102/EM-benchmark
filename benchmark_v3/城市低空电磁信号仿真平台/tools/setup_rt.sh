#!/usr/bin/env bash
set -euo pipefail
# Run from project directory inside WSL; isolated, no system package changes.
cd -- "$(dirname -- "$0")/.."
runtime_path="${CITYRF_RT_ENV:-$PWD/.venv-rt}"
python3 -m venv "$runtime_path"
"$runtime_path/bin/python" -m pip install -e '.[rt,web,test]'
printf 'Ready: %s/bin/python\n' "$runtime_path"
