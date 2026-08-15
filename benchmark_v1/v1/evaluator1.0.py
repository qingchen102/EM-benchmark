"""Benchmark an OpenAI-compatible LLM agent on an EM signal dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

try:  # Supports both ``python simulation/evaluator.py`` and package imports.
    from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
except ImportError:
    from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


SYSTEM_PROMPT = """You are an RF signal analyst. Diagnose the jamming type from the supplied tool measurements.
Return JSON with keys jamming_type and mitigation (a short actionable recommendation).
Allowed jamming_type values: none, single_tone, swept, pulse, broadband."""


def _label(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from plain text or a Markdown fenced code block."""
    raw = "" if text is None else str(text).strip()
    # Prefer the contents of a fenced block (```json ... ```), while allowing
    # optional whitespace and a language tag other than json.
    fenced = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", raw)
    candidate = fenced.group(1).strip() if fenced else raw
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(candidate[start : end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                pass
        return {}


class OpenAICompatibleAgent:
    """Small adapter for OpenAI and Qwen-compatible chat-completions APIs."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use API evaluation") from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))
        self.model = model

    def diagnose(self, sample_path: Path) -> tuple[dict[str, Any], int, int]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Analyze sample at {sample_path}. Use both tools before diagnosing."}]
        calls = successes = 0
        for _ in range(5):
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto")
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return _extract_json(message.content or ""), calls, successes
            messages.append(message.model_dump() if hasattr(message, "model_dump") else {"role": "assistant", "content": message.content, "tool_calls": [tc.model_dump() for tc in tool_calls]})
            for call in tool_calls:
                calls += 1
                try:
                    args = json.loads(call.function.arguments or "{}")
                    args["sample_path"] = str(sample_path)
                    result = TOOL_FUNCTIONS[call.function.name](**args)
                    successes += 1
                except Exception as exc:
                    result = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.function.name, "content": json.dumps(result)})
        return {}, calls, successes


def evaluate_dataset(dataset_dir: str | Path, agent: OpenAICompatibleAgent | None = None) -> dict[str, Any]:
    """Evaluate every metadata record and return a JSON-serialisable report."""
    root = Path(dataset_dir)
    records = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    results, correct, tool_calls, tool_successes = [], 0, 0, 0
    snr_stats = {"Low SNR": [0, 0], "Mid SNR": [0, 0], "High SNR": [0, 0]}
    for record in records:
        truth = record.get("ground_truth", {})
        sample = root / truth.get("file", "")
        if agent is None:
            diagnosis, calls, successes = {"jamming_type": _label(truth.get("jamming_type")), "mitigation": "offline reference"}, 0, 0
        else:
            diagnosis, calls, successes = agent.diagnose(sample)
        predicted = _label(diagnosis.get("jamming_type"))
        expected = _label(truth.get("jamming_type"))
        correct += int(predicted == expected)
        snr_value = truth.get("snr_db", record.get("parameters", {}).get("snr_db"))
        try:
            snr = float(snr_value)
        except (TypeError, ValueError):
            snr = None
        if snr is not None:
            group = "Low SNR" if snr < 0 else ("Mid SNR" if snr <= 10 else "High SNR")
            snr_stats[group][1] += 1
            snr_stats[group][0] += int(predicted == expected)
        tool_calls += calls; tool_successes += successes
        results.append({"file": sample.name, "expected_jamming_type": expected, "predicted_jamming_type": predicted, "correct": predicted == expected, "diagnosis": diagnosis, "tool_calls": calls, "tool_successes": successes})
    total = len(results)
    accuracy_by_snr_group = {
        group: (values[0] / values[1] if values[1] else 0.0)
        for group, values in snr_stats.items()
    }
    return {"dataset": str(root), "num_samples": total, "jamming_classification_accuracy": correct / total if total else 0.0, "accuracy_by_snr_group": accuracy_by_snr_group, "snr_group_counts": {group: values[1] for group, values in snr_stats.items()}, "tool_call_success_rate": tool_successes / tool_calls if tool_calls else 0.0, "total_tool_calls": tool_calls, "successful_tool_calls": tool_successes, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir"); parser.add_argument("--output", default="eval_report.json")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini")); parser.add_argument("--base-url", default=None)
    parser.add_argument("--offline", action="store_true", help="Use ground truth labels to validate the pipeline without an API")
    args = parser.parse_args()
    agent = None if args.offline else OpenAICompatibleAgent(args.model, args.base_url)
    report = evaluate_dataset(args.dataset_dir, agent)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("num_samples", "jamming_classification_accuracy", "tool_call_success_rate")}, indent=2))


if __name__ == "__main__":
    main()
