"""探针：百炼上这批模型能否被强制成"无推理"（enable_thinking=False）。

背景（2026-10-11）：`deepseek-v3.2` 从百炼下架（404 Model not exist），
`siliconflow/deepseek-v3.2` 需要另开市场订阅（400 Aliyun market app does not exist）。
本探针回答"换哪个考官还能保持无推理口径"——评测口径要求考官**不出思维链**，
所以必须验证 enable_thinking=False 真的生效。

判据（每题十几个 token，成本可忽略）：
    开开关后 reasoning_content 为空 + completion_tokens≈1 + 直接给答案 -> 无推理模式生效
    不带开关时 reasoning_content 有内容 -> 说明该模型默认开思考，必须靠开关关掉

用法（在仓库根）：python -X utf8 benchmark_v3/adapters/diag/_probe_model_think.py
key 只从 gitignore 的 run_capture_agent.ps1 读取，输出绝不回显 key。
"""
from __future__ import annotations

import re
import sys
import time
import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]          # 仓库根（本文件在 benchmark_v3/adapters/diag/）
# 可选参数：base_url [模型1,模型2,…]（默认百炼直连 + 三个候选）
BASE = sys.argv[1] if len(sys.argv) > 1 else "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELS = (sys.argv[2].split(",") if len(sys.argv) > 2
          else ["deepseek-v4.1-flash", "deepseek-v4-flash-0731", "deepseek-v4-pro"])
PROMPT = "1+1=? 只回答数字，不要别的。"


def _key() -> str:
    ps1 = (ROOT / "run_capture_agent.ps1").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"sk-[A-Za-z0-9._\-]+", ps1)
    if not m:
        raise SystemExit("run_capture_agent.ps1 里找不到 key（该文件被 gitignore，属本地文件）")
    return m.group(0)


def probe(cli: OpenAI, model: str, nothink: bool) -> None:
    kw = dict(model=model, messages=[{"role": "user", "content": PROMPT}],
              max_tokens=64, temperature=0.0)
    if nothink:
        kw["extra_body"] = {"enable_thinking": False}
    tag = "nothink" if nothink else "plain  "
    t0 = time.time()
    try:
        r = cli.chat.completions.create(**kw)
    except Exception as e:                                    # noqa: BLE001
        print(f"{model:24s} {tag} ERR  {str(e).replace(chr(10), ' ')[:150]}")
        return
    ch = r.choices[0].message
    reason = getattr(ch, "reasoning_content", None)
    txt = (ch.content or "").strip().replace("\n", " ")
    print(f"{model:24s} {tag} OK   reasoning_content={'有' if reason else '无'}"
          f"({len(reason or '')}字)  out_tok={r.usage.completion_tokens}"
          f"  {time.time() - t0:.1f}s  text={txt[:60]!r}")


def main() -> None:
    cli = OpenAI(api_key=_key(), base_url=BASE)
    for m in MODELS:
        probe(cli, m, True)
    for m in MODELS:
        probe(cli, m, False)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
