"""用阿里云百炼 DeepSeek（无推理）跑 evaluator_v2 主流程（通用封装）。

支持百炼上任意 DeepSeek 非推理模型（deepseek-v3 / v3.1 / v3.2 等），
默认使用 **deepseek-v3.2**。

关键：百炼部分 DeepSeek 模型默认开启思考模式，而 evaluator_v2 用非流式调用，
不关思考会直接报错（'enable_thinking must be set to false for non-streaming calls'）。
本脚本继承官方 OpenAICompatibleAgent，在每次 chat.completions.create 时
硬性注入 extra_body={"enable_thinking": False}，实现真正的"无推理"。

- 不改动 evaluator_v2.py / tools_v2.py 任何源文件（新增本文件即可）
- 复用 evaluate_dataset 主循环 + TOOL_SCHEMAS_V2 + _score_answer，流程与 ds_run 一致
- 百炼 DeepSeek 系列支持 Function Calling，多轮工具调用正常

用法（默认 v3.2）：
  python run_bailian_deepseek.py dataset --max-samples 50 --output ds_run9.json --verbose
指定其它模型：
  python run_bailian_deepseek.py dataset --model deepseek-v3.1 --max-samples 50 --output xxx.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator_v2 import evaluate_dataset, OpenAICompatibleAgent


class NoThinkDeepSeekAgent(OpenAICompatibleAgent):
    """继承官方 agent，强制关闭思考模式（无推理）。"""

    def _create(self, **kwargs):
        # extra_body 透传百炼的非标准参数 enable_thinking
        extra = dict(kwargs.pop("extra_body") or {}) if kwargs.get("extra_body") else {}
        extra["enable_thinking"] = False
        kwargs["extra_body"] = extra
        # 注意：关掉思考后不再传 reasoning_effort（两者冲突）
        return self.client.chat.completions.create(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="用百炼 DeepSeek（无推理）跑 evaluator_v2 主流程")
    parser.add_argument("dataset_dir")
    parser.add_argument("--model", default="deepseek-v3.2")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="eval_report_deepseek.json")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    agent = NoThinkDeepSeekAgent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
    )

    report = evaluate_dataset(
        args.dataset_dir,
        agent=agent,
        max_samples=args.max_samples,
        progress=not args.no_progress,
        verbose=args.verbose,
    )

    report["agent_info"] = {
        "model": args.model,
        "enable_thinking": False,
        "base_url": args.base_url,
        "note": "百炼 DeepSeek 无推理模式，通过 extra_body enable_thinking=False 实现",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
