#!/usr/bin/env bash
# ============================================================================
# 城市低空电磁信号仿真平台 —— 跨机器启动脚本
#
# 把整个文件夹拷到另一台电脑后，只要有 Python 3.12+，直接跑这个脚本就行：
# 它会自己建虚拟环境、装依赖、挑端口、起服务、开浏览器。
#
#   ./start.sh                 首次运行自动安装 Web 依赖并启动
#   ./start.sh --rt            连 Sionna RT 一起装（体积大、耗时长，但城市多径
#                              与「电磁态势图」需要它）
#   ./start.sh --port 9000     换端口（默认 8765）
#   ./start.sh --no-browser    只起服务，不开浏览器
#   ./start.sh --recreate      删掉已有虚拟环境重装
#   ./start.sh --env DIR       使用指定目录下的虚拟环境
#   ./start.sh --help          显示这段说明
#
# 想脱离本脚本手动跑，等价命令是：
#   PYTHONPATH="$PWD/src" ./.venv/bin/python -m webui.server --port 8765
# ============================================================================

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT=8765
WANT_RT=0
RECREATE=0
OPEN_BROWSER=1
ENV_DIR=""

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//' -e '/^=\{4,\}$/d'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --rt)          WANT_RT=1 ;;
    --no-browser)  OPEN_BROWSER=0 ;;
    --recreate)    RECREATE=1 ;;
    --port)        PORT="${2:?--port 需要一个端口号}"; shift ;;
    --port=*)      PORT="${1#*=}" ;;
    --env)         ENV_DIR="${2:?--env 需要一个目录}"; shift ;;
    --env=*)       ENV_DIR="${1#*=}" ;;
    -h|--help)     usage; exit 0 ;;
    *) printf '未知参数：%s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$PORT" in
  ''|*[!0-9]*) printf '端口必须是数字，收到：%s\n' "$PORT" >&2; exit 2 ;;
esac

say() { printf '\033[1;34m==\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 找 Python

PY_OK='import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'

pick_base_python() {
  local candidate
  for candidate in python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c "$PY_OK" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------- 虚拟环境

RUNTIME=""
if [ -n "$ENV_DIR" ]; then
  RUNTIME="$ENV_DIR"
elif [ -x "$ROOT/.venv-rt/bin/python" ]; then
  RUNTIME="$ROOT/.venv-rt"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  RUNTIME="$ROOT/.venv"
fi

if [ "$RECREATE" = 1 ]; then
  for dir in "$ROOT/.venv-rt" "$ROOT/.venv"; do
    if [ -d "$dir" ]; then say "删除已有虚拟环境 $dir"; rm -rf -- "$dir"; fi
  done
  [ -z "$ENV_DIR" ] && RUNTIME=""
fi

if [ -z "$RUNTIME" ]; then
  BASE_PYTHON="$(pick_base_python)" || die "找不到 Python 3.12 或更高版本。请先安装 Python 3.12+（Debian/Ubuntu 还需要 python3-venv 包），然后重跑本脚本。"
  if [ "$WANT_RT" = 1 ]; then TARGET="$ROOT/.venv-rt"; else TARGET="$ROOT/.venv"; fi
  say "创建虚拟环境 $TARGET（基于 $(command -v "$BASE_PYTHON")）"
  "$BASE_PYTHON" -m venv "$TARGET" || die "创建虚拟环境失败。Linux 上通常是缺 python3-venv：sudo apt install python3-venv"
  RUNTIME="$TARGET"
fi

PY="$RUNTIME/bin/python"
[ -x "$PY" ] || die "$RUNTIME 里没有可执行的 bin/python，请删掉该目录后重跑，或用 --env 指定别的环境。"

has_module() {
  "$PY" -c "import importlib.util,sys; raise SystemExit(0 if importlib.util.find_spec('$1') else 1)" >/dev/null 2>&1
}

# ---------------------------------------------------------------- 装依赖

EXTRAS="web,test"
[ "$WANT_RT" = 1 ] && EXTRAS="rt,web,test"

if ! has_module cityrf || ! has_module fastapi; then
  say "安装依赖 .[$EXTRAS]（首次运行需要联网，可能要几分钟）"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -e ".[$EXTRAS]" || die "依赖安装失败。请检查网络，或先手动执行：$PY -m pip install -e \".[$EXTRAS]\""
elif [ "$WANT_RT" = 1 ] && ! has_module sionna.rt; then
  say "补装 Sionna RT（.[rt]）"
  "$PY" -m pip install -e ".[rt]" || die "Sionna RT 安装失败。请检查网络后重试。"
fi

# ---------------------------------------------------------------- 端口

port_in_use() {
  "$PY" - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(0)          # 被占用
finally:
    sock.close()
raise SystemExit(1)
PY
}

BASE_URL="http://127.0.0.1:$PORT/"

open_browser() {
  [ "$OPEN_BROWSER" = 1 ] || return 0
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then open "$url" >/dev/null 2>&1 &
  elif command -v wslview >/dev/null 2>&1; then wslview "$url" >/dev/null 2>&1 &
  elif command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$url" >/dev/null 2>&1 &
  else warn "没有可用的浏览器打开命令，请手动访问 $url"
  fi
}

# 端口状态：0 = 本平台已在此端口应答（直接复用）；2 = 被别的程序占用；1 = 空闲
probe_port() {
  local port="$1" body rc=0
  if command -v curl >/dev/null 2>&1; then
    body="$(curl -fsS --max-time 2 "http://127.0.0.1:${port}/api/bootstrap" 2>/dev/null || true)"
    case "$body" in
      *available_backends*) return 0 ;;
    esac
  fi
  # WSL 里 127.0.0.1 是本发行版自己的回环，看不见 Windows 侧的占用；而 Windows 侧一旦
  # 被别的程序占住（比如另一个项目起了静态服务器），localhost 转发就会失效、网页根本
  # 打不开——所以必须再问一次 Windows，否则会“启动成功但浏览器打开是别人的页面”。
  if command -v powershell.exe >/dev/null 2>&1; then
    timeout 10 powershell.exe -NoProfile -Command "try{ \$null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:${port}/api/bootstrap'; exit 0 }catch{ if(\$_.Exception.Response){ exit 2 } else { exit 1 } }" </dev/null >/dev/null 2>&1 || rc=$?
    if [ "$rc" = 0 ]; then return 0; fi
    if [ "$rc" = 2 ]; then return 2; fi
  fi
  if port_in_use "$port"; then return 2; fi
  return 1
}

PORT_STATE=0
probe_port "$PORT" || PORT_STATE=$?

if [ "$PORT_STATE" = 0 ]; then
  say "平台已经在运行：$BASE_URL"
  open_browser "$BASE_URL"
  exit 0
fi

if [ "$PORT_STATE" = 2 ]; then
  warn "端口 $PORT 已被别的程序占用（可能是另一个项目或另一个终端），自动往后找。"
  FOUND=0
  for candidate in $(seq $((PORT + 1)) $((PORT + 20))); do
    STATE=0
    probe_port "$candidate" || STATE=$?
    if [ "$STATE" = 0 ]; then
      PORT="$candidate"; BASE_URL="http://127.0.0.1:$PORT/"
      say "端口 $PORT 上已经跑着本平台，直接复用"
      open_browser "$BASE_URL"
      exit 0
    fi
    if [ "$STATE" = 1 ]; then PORT="$candidate"; FOUND=1; break; fi
  done
  [ "$FOUND" = 1 ] || die "从原端口起连续 20 个端口都被占用，请用 --port 指定别的端口。"
  BASE_URL="http://127.0.0.1:$PORT/"
  say "改用端口 $PORT"
fi

# ---------------------------------------------------------------- 起服务

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8

if has_module sionna.rt; then
  say "Sionna RT：可用（城市多径与电磁态势图都能算）"
else
  say "Sionna RT：未安装（只有自由空间与抽头后端；需要城市多径时跑 ./start.sh --rt）"
fi
say "Python：$PY"
say "打开浏览器访问：$BASE_URL"
say "停止服务：在本终端按 Ctrl+C"
echo

if [ "$OPEN_BROWSER" = 1 ]; then
  ( sleep 3; open_browser "$BASE_URL" ) &
fi

exec "$PY" -m webui.server --port "$PORT"
