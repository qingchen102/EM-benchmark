"""v3 评估器：城市低空干扰分析 Agent 评估。

复用 v2 的 tools_v2 工具集、OpenAICompatibleAgent、evaluate_dataset 循环（存档/断点/进度/repair 全继承），
通过 monkey-patch 三个模块级对象注入 v3 扩展，不修改 v2 冻结代码：
- SYSTEM_PROMPT  → 追加敌我意图 + 辐射体溯源两轴任务
- _score_answer → 加 intent / emitter 评分
- _aggregate    → 报告加 intent_accuracy / emitter_accuracy

用法（与 v2 完全一致，可 --resume / 存档）：
    python eval_v3.py dataset_v3 --model deepseek-v3.2 --max-samples 50 --output v3_report.json
    python eval_v3.py dataset_v3 --offline        # 离线回放验证评分管道（应满分）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_V2 = _ROOT.parent / "benchmark_v2"
for _p in (str(_ROOT), str(_ROOT / "simulation"), str(_V2), str(_V2 / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import evaluator_v2 as ev2
from evaluator_v2 import (OpenAICompatibleAgent, _greedy_match, _to_float,  # noqa: E402
                          evaluate_dataset, _aggregate as _v2_aggregate)

# ---- 新答案轴合法值 ----
INTENTS = ("malicious", "incidental", "benign")
EMITTERS = ("Bluetooth_FHSS", "WiFi_burst", "LTE_QPSK", "DECT", "Zigbee",
            "UAV_video_OFDM", "UAV_RC_hopping", "Radar_pulsed_LFM", "GPS_DSSS",
            "GNSS_jam", "GNSS_spoof", "Microwave", "TV_broadcast",
            "Chirp_jam", "Noise_jam", "Tone_jam")

# ---- v3 增强 SYSTEM_PROMPT ----
SYSTEM_PROMPT_V3 = ev2.SYSTEM_PROMPT + (
    "\n\nADDITIONAL v3 TASKS — for EACH interferer, also output:\n"
    "- **intent**: one of malicious|incidental|benign (friend/foe). "
    "malicious = deliberately disruptive (jamming/spoofing/ECM); "
    "incidental = non-targeted coexistence/casual interferer (microwave/DECT/Zigbee); "
    "benign = normal legal signal / background.\n"
    "- **emitter**: most likely source device, one of: " + ", ".join(EMITTERS) + ".\n"
    'Add "intent" and "emitter" keys to each interferer object in your JSON answer.'
)


_V2_SCORE = ev2._score_answer   # 捕获原始 v2 评分函数（monkey-patch 前）


def _score_v3(gt: dict, pred: dict) -> dict:
    """v2 评分基础上加 intent / emitter 两轴（在贪婪匹配对上评估）。"""
    base = _V2_SCORE(gt, pred)
    gt_srcs = [s for s in gt.get("sources", []) if s.get("role") == "interferer"]
    pred_srcs = pred.get("interferers") or []
    pairs, _, _ = _greedy_match(gt_srcs, pred_srcs)

    intent_hit = intent_total = 0
    emitter_hit = emitter_total = 0
    intent_by_gt = {}
    for gi, pi in pairs:
        g = gt_srcs[gi]; p = pred_srcs[pi]
        gi_ = g.get("intent")
        if gi_ in INTENTS:
            intent_total += 1
            intent_hit += int(str(p.get("intent", "")).lower() == gi_)
            intent_by_gt.setdefault(gi_, [0, 0])
            intent_by_gt[gi_][1] += 1
            intent_by_gt[gi_][0] += int(str(p.get("intent", "")).lower() == gi_)
        ge_ = g.get("emitter")
        if ge_ in EMITTERS:
            emitter_total += 1
            emitter_hit += int(str(p.get("emitter", "")).lower() == str(ge_).lower())

    out = dict(base)
    out["intent_accuracy"] = (intent_hit / intent_total) if intent_total else None
    out["emitter_accuracy"] = (emitter_hit / emitter_total) if emitter_total else None
    out["intent_by_gt"] = {k: v for k, v in intent_by_gt.items()}
    return out


def _aggregate_v3(results, total):
    rep = _v2_aggregate(results, total)
    rep["intent_accuracy"] = round(float(np.mean(
        [r["intent_accuracy"] for r in results if r.get("intent_accuracy") is not None])), 3) \
        if any(r.get("intent_accuracy") is not None for r in results) else None
    rep["emitter_accuracy"] = round(float(np.mean(
        [r["emitter_accuracy"] for r in results if r.get("emitter_accuracy") is not None])), 3) \
        if any(r.get("emitter_accuracy") is not None for r in results) else None
    # 意图混淆（按 GT）
    ibg = {}
    for r in results:
        for k, v in (r.get("intent_by_gt") or {}).items():
            ibg.setdefault(k, [0, 0])
            ibg[k][0] += v[0]; ibg[k][1] += v[1]
    rep["intent_by_gt"] = {k: v for k, v in ibg.items()}
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir", nargs="?", default="dataset")
    ap.add_argument("--model", default="deepseek-v3.2")
    ap.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--output", default="v3_report.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # monkey-patch v3 扩展（不修改 v2 冻结代码）
    ev2.SYSTEM_PROMPT = SYSTEM_PROMPT_V3
    ev2._score_answer = _score_v3
    ev2._aggregate = _aggregate_v3

    prior = None
    if args.resume:
        p = Path(args.output)
        if p.exists():
            try:
                prior = (json.loads(p.read_text(encoding="utf-8")).get("results") or [])
            except Exception:
                prior = None

    agent = None if args.offline else OpenAICompatibleAgent(
        args.model, base_url=args.base_url, api_key=args.api_key, use_tools=True)
    rep = evaluate_dataset(args.dataset_dir, agent, max_samples=args.max_samples,
                           progress=not args.no_progress, verbose=args.verbose,
                           checkpoint_path=args.output, prior_results=prior)
    rep["code_fingerprint"] = ev2._code_fingerprint()
    Path(args.output).write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {args.output}")

    # 打印 v3 新增指标
    print("\n=== v3 扩展指标 ===")
    print(f"intent_accuracy : {rep.get('intent_accuracy')}")
    print(f"emitter_accuracy: {rep.get('emitter_accuracy')}")
    print(f"intent_by_gt    : {rep.get('intent_by_gt')}")


if __name__ == "__main__":
    import json
    main()
