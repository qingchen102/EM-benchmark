"""一次性工具：扫描 JSON 文件里"上一个成员行末缺逗号"的位置。

背景：用 edit 往 JSON 对象末尾插字段时，容易忘记给前一行补逗号；
json.load 只报**第一个**错误，迭代修很慢。本脚本一次列出全部可疑点。

用法：python _check_json_commas.py <file.json>
"""
from __future__ import annotations

import json
import re
import sys

KEY = re.compile(r'^"')


def main() -> int:
    path = sys.argv[1]
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    prev = None
    prevno = 0
    bad: list[tuple[int, int, str, str]] = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s:
            continue
        if prev is not None and KEY.match(s):
            if not prev.endswith((",", "[", "{")):
                bad.append((prevno, i, prev[-70:], s[:70]))
        prev = s
        prevno = i
    print(f"candidates: {len(bad)}")
    for a, b, pa, pb in bad:
        print(f"  L{a} -> L{b}\n     prev: ...{pa}\n     next: {pb}")
    try:
        json.load(open(path, encoding="utf-8"))
        print("JSON OK")
    except Exception as exc:  # noqa: BLE001
        print("JSON ERR:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
