"""零 API 自检：机理轴对照开关的逐字等价性。

断言四件事：
1. 当前文件里的 SYSTEM_PROMPT_CAPTURE 与 git HEAD 的**逐字节相同**（题面没被改坏）；
2. prompt_without_mech_axis(SYSTEM_PROMPT_CAPTURE) 与 git 5cd795a^ 的题面**逐字节相同**
   （对照臂的题面 = 机理段落入之前的那一份）；
3. 默认路径（DSH_NO_MECH_AXIS 未设）返回的 prompt 仍含机理四段；
4. system_prompt_for([]) == SYSTEM_PROMPT_CAPTURE（无事件时不注入事件段落）。

用法：python -X utf8 diag/_check_prompt_mech.py
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ADAPTERS = _HERE.parent
sys.path.insert(0, str(_ADAPTERS))

import run_capture_agent as rca  # noqa: E402

ROOT = _ADAPTERS.parents[1]
REL = "benchmark_v3/adapters/run_capture_agent.py"
OLD_REV = "5cd795a^"

FAILED: list[str] = []


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def prompt_at(rev: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:{REL}"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8")
    for node in ast.parse(out).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "SYSTEM_PROMPT_CAPTURE" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("SYSTEM_PROMPT_CAPTURE not found in " + rev)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def main() -> int:
    cur = rca.SYSTEM_PROMPT_CAPTURE
    old = prompt_at(OLD_REV)
    head = prompt_at("HEAD")

    print(f"当前  len={len(cur):>6} md5={md5(cur)}")
    print(f"HEAD  len={len(head):>6} md5={md5(head)}")
    print(f"旧版  len={len(old):>6} md5={md5(old)}  ({OLD_REV})")
    print()

    check("题面未被改动（当前 == git HEAD）", cur == head,
          f"len {len(cur)} vs {len(head)}")

    stripped = rca.prompt_without_mech_axis(cur)
    check(f"剥掉机理后 == {OLD_REV} 题面（逐字节）", stripped == old,
          f"len {len(stripped)} vs {len(old)}；md5 {md5(stripped)} vs {md5(old)}")
    if stripped != old:
        diff = list(difflib.unified_diff(
            old.splitlines(keepends=True), stripped.splitlines(keepends=True),
            fromfile=OLD_REV, tofile="stripped", n=2))
        print("".join(diff[:80]))

    seg_ok = all(new in cur for new, _ in rca._MECH_AXIS_SEGMENTS)
    check("机理四段都还在默认题面里", seg_ok,
          f"{len(rca._MECH_AXIS_SEGMENTS)} 段")

    check("默认（无环境变量）不剥机理", rca._NO_MECH_AXIS is False)
    check("system_prompt_for([]) == SYSTEM_PROMPT_CAPTURE",
          rca.system_prompt_for([]) == cur)

    saved = rca._NO_MECH_AXIS
    try:
        rca._NO_MECH_AXIS = True
        check("DSH_NO_MECH_AXIS=1 时 system_prompt_for([]) == 旧题面",
              rca.system_prompt_for([]) == old)
    finally:
        rca._NO_MECH_AXIS = saved

    print()
    if FAILED:
        print(f"=== 未通过：{len(FAILED)} 项 -> {FAILED} ===")
        return 1
    print("=== 全部通过：对照臂题面与 5cd795a^ 逐字节一致 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
