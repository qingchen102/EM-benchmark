"""零成本验收：事件日志段落「按需注入」后，两种情形都必须与历史版本逐字相同。

- 无事件（D1/D2）→ 应等于公开层之前的 prompt（commit 52e306d = 1bfdc24 的父提交）
- 有事件（D3/D4）→ 应等于公开层提交 1bfdc24 的 prompt

做法：把两个历史版本的 run_capture_agent.py 取到临时目录，用 importlib 各自加载，
直接比较模块级 SYSTEM_PROMPT_CAPTURE。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REL = "benchmark_v3/adapters/run_capture_agent.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def show(rev: str, dest: Path) -> None:
    txt = subprocess.run(["git", "show", f"{rev}:{REL}"], cwd=REPO,
                         capture_output=True, check=True).stdout.decode("utf-8")
    dest.write_text(txt, encoding="utf-8")


tmp = Path(tempfile.mkdtemp(prefix="promptvar_"))
show("52e306d", tmp / "pre_public_layer.py")   # 公开层之前
show("1bfdc24", tmp / "public_layer.py")       # 公开层提交（含段落）

pre = load("pre_prompt_mod", tmp / "pre_public_layer.py")
pub = load("pub_prompt_mod", tmp / "public_layer.py")
cur = load("cur_prompt_mod", HERE / "run_capture_agent.py")

no_ev = cur.system_prompt_for([])
with_ev = cur.system_prompt_for([{"event_id": "E1", "source_id": "interferer_1"}])

print(f"pre(52e306d) prompt 长度 = {len(pre.SYSTEM_PROMPT_CAPTURE)}")
print(f"pub(1bfdc24) prompt 长度 = {len(pub.SYSTEM_PROMPT_CAPTURE)}")
print(f"cur 无事件长度 = {len(no_ev)}   有事件长度 = {len(with_ev)}")
print()
print("① 无事件 == 公开层之前 :", no_ev == pre.SYSTEM_PROMPT_CAPTURE)
print("② 有事件 == 公开层提交 :", with_ev == pub.SYSTEM_PROMPT_CAPTURE)
print("③ 有事件比无事件多长度 :", len(with_ev) - len(no_ev), "(= 段落字节数)")

if no_ev != pre.SYSTEM_PROMPT_CAPTURE:
    a, b = no_ev, pre.SYSTEM_PROMPT_CAPTURE
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    print("\n①差异位置", i)
    print("  cur:", repr(a[max(0, i - 60):i + 60]))
    print("  pre:", repr(b[max(0, i - 60):i + 60]))
if with_ev != pub.SYSTEM_PROMPT_CAPTURE:
    a, b = with_ev, pub.SYSTEM_PROMPT_CAPTURE
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    print("\n②差异位置", i)
    print("  cur:", repr(a[max(0, i - 60):i + 60]))
    print("  pub:", repr(b[max(0, i - 60):i + 60]))

# 顺带核对：默认（模块级 ev2.SYSTEM_PROMPT）应是无事件版
print("\n④ 模块默认 ev2.SYSTEM_PROMPT == 无事件版 :", cur.ev2.SYSTEM_PROMPT == no_ev)
print(f"\n临时目录：{tmp}")
