"""v2 Benchmark 评估器：干扰检测 / 参数估计 / 分类 / DOA 定位。

数据隔离：真值只从 ground_truth.json 读取；Agent 只拿到 observations.json 中的
先验（接收机参数 + 目标调制类型），与 sample .npy 路径。

任务（填空/选择题，Agent 输出 JSON）：
    {
      "num_interferers": 0|1|2,                      # 选择题：干扰源数量
      "interferers": [{
        "category": "co_channel|adjacent|blocking|pulse|none",   # 选择题
        "modulation": "QPSK|OFDM|...|single_tone|...",            # 选择题（可选）
        "freq_offset": <归一化频偏>,                   # 填空题
        "bandwidth": <归一化带宽 OBW99>,               # 填空题
        "doa": <度>                                   # 填空题
      }, ...]
    }

评分：
- 源数准确率；类别准确率（贪心匹配后）；参数命中率（容差内）+ MAE/RMSE；
- 按 SNR 分组；类别混淆矩阵。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# 根目录运行：本文件与 tools_v2.py 同级，信号模型库在 simulation/ 子目录
_ROOT = Path(__file__).resolve().parent
_SIM_DIR = _ROOT / "simulation"
for _p in (_ROOT, _SIM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from tools_v2 import TOOL_FUNCTIONS_V2, TOOL_SCHEMAS_V2
except ImportError:  # 兼容旧位置（simulation/em_signal_simulator/tools_v2.py）
    from em_signal_simulator.tools_v2 import TOOL_FUNCTIONS_V2, TOOL_SCHEMAS_V2

# 参数评估容差
FREQ_TOL = 0.03          # 归一化频偏（旧 ±0.02 与工具 MAE 0.026 卡边缘，校准到物理可达精度）
DOA_TOL_DEG = 10.0       # DOA（度）
BW_REL_TOL = 0.25        # 带宽相对误差
CATEGORIES = ["co_channel", "adjacent", "blocking", "pulse", "none"]

SYSTEM_PROMPT = """You are an RF signal analyst for a multi-antenna (4-element ULA) IQ benchmark.

You are given a sample (4, 1024) complex IQ array and the receiver prior (sampling rate,
antenna spacing, center frequency, target modulation type). The sample contains 1 target
signal and 0~2 interferers.

Use the provided tools to gather measurements, then answer:
- analyze_spectrum / estimate_num_sources / estimate_doa / detect_time_domain /
  estimate_modulation_features

Measurement-to-answer mapping (use the tool values directly):
- freq_offset   <- analyze_spectrum sources_candidates freq (per source), normalized
- bandwidth     <- analyze_spectrum sources_candidates bandwidth (per source)
- doa           <- estimate_doa doa_estimates_deg
- num_interferers <- num_sources_estimate MINUS the target (1 source is the target),
                     adjusted ONLY per the source-count decision tree below
- modulation    <- estimate_modulation_features per_peak_template_distances. For each
                   interferer, pick the source slice whose peak_freq is closest to the
                   interferer's freq_offset, then choose the template with the SMALLEST
                   feature_distance. Rules:
                   * candidates are: BPSK / QPSK / GFSK / LFM / OFDM / single_tone /
                     swept / pulse / broadband / nfm — answer ONLY one of these exact names
                   * per-slice distances are bandwidth-matched; global_template_distances
                     are dominated by the target and are only a fallback
                   * if the two smallest distances are within 0.15 of each other, prefer
                     the simpler/lower-order one (e.g. QPSK over OFDM); NEVER fall back to
                     a specific QAM order without evidence, and never guess 64QAM by default

Source count — estimate_num_sources returns num_sources_estimate (raw MDL) AND
final_suggestion: the MDL value AFTER an over/under-count correction has already been
applied by the tool, using merged spectral peaks and cross-order MUSIC peaks as
evidence. Use final_suggestion as your source count (= 1 target + N interferers).
Do NOT re-derive the correction yourself; override final_suggestion ONLY when
multiple independent tools clearly contradict it.

DOA usage — estimate_doa returns doa_estimates_deg for the requested num_sources, plus
stable_peaks_deg (angles seen in BOTH the order-2 and order-3 runs, tolerance 5 deg;
real sources are stable) and peaks_new_at_order3_deg (angles seen only at order 3).
Peaks appearing at only one MUSIC order are usually over-resolution artifacts on
wideband sources — do not treat them as confirmed sources on their own.

Interference category definition — use the per-source measurements that
analyze_spectrum already computed (ratio_to_target_bw, power_ratio_approx_db):
1. pulse:      requires ALL of detect_time_domain: papr_db >= 8 AND
               pulse_duty_cycle <= 0.3 AND pulse_edge_count >= 4 (periodic
               bursts). High PAPR alone does NOT imply pulse — strong
               out-of-band or fading signals also elevate it; without burst
               evidence, prefer blocking/adjacent/none over pulse.
2. co_channel: ratio_to_target_bw < 0.5 (interferer inside the target band)
3. blocking:   ratio_to_target_bw >= 0.5 AND power_ratio_approx_db >= 10 dB
               (strong out-of-band signal -> receiver LNA saturation)
4. adjacent:   0.5 <= ratio_to_target_bw < 2.0 with weaker power
5. none:       ratio_to_target_bw >= 2.0 with weaker power
Apply the first matching rule (pulse has priority).

Output ONLY valid JSON with schema:
{
  "num_interferers": <0 or 1 or 2>,
  "interferers": [
    {
      "category": "co_channel|adjacent|blocking|pulse|none",
      "modulation": "<best guess, e.g. QPSK / OFDM / LFM / single_tone / nfm / swept / pulse / broadband>",
      "freq_offset": <normalized frequency offset of the interferer, relative to receiver band>,
      "bandwidth": <normalized occupied bandwidth (99% OBW) of the interferer>,
      "doa": <DOA in degrees>
    }, ...
  ]
}
Rules:
- num_interferers must match the number of interferers in the interferers list.
- freq_offset/bandwidth are normalized to sampling rate (range [-0.5, 0.5] / [0, 1]).
- If a parameter cannot be estimated, put a reasonable estimate; do not omit keys.
- The target signal is centered near 0 Hz and 0 deg; interferers are displaced in
  frequency and/or DOA.
- After gathering enough measurements, ALWAYS output the final JSON answer. Do not
  keep calling tools without producing the answer; you may stop calling tools once
  you have reasonable estimates for all fields.
- The final message must contain ONLY the JSON answer object and nothing else
  (no prose, no markdown fences, no second JSON block).
"""

# repair 轮：模型最终回复缺有效 JSON 时（长推理截断/纯散文），追加一次无工具的
# 强制格式回复。ds_run8 中 8/50 样本因此丢掉全部指标，此轮可挽回。
REPAIR_PROMPT = (
    "Your previous reply did not contain a valid final JSON answer. Reply NOW with "
    "ONLY the final JSON object matching this schema — no prose, no markdown, no "
    "reasoning, no tool calls:\n"
    '{"num_interferers": <0|1|2>, "interferers": [{"category": "co_channel", '
    '"modulation": "QPSK", "freq_offset": 0.1, "bandwidth": 0.05, "doa": 30.0}, ...]}'
)


def _extract_json(text: str) -> dict[str, Any]:
    raw = "" if text is None else str(text).strip()
    fenced = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", raw)
    candidate = fenced.group(1).strip() if fenced else raw
    # 尝试 1: 整个字符串直接解析
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except (TypeError, json.JSONDecodeError):
        pass
    # 尝试 2: 扫描文本中所有完整 JSON 对象，取最后一个
    # （模型常在推理文本后附最终答案，或多段 JSON 并存；旧的 first-{-to-last-}
    #   会在花括号混入推理文本/多段 JSON 时整体解析失败）
    objs = []
    idx = 0
    n = len(candidate)
    decoder = json.JSONDecoder()
    while idx < n:
        start = candidate.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
        idx = start + max(int(end), 1)
    if objs:
        return objs[-1]
    return {}


def _to_float(value, default=float("nan")):
    """把模型输出中的任意值（None / 字符串 / NaN / Inf）清洗为 float，失败返回 default。"""
    if value is None:
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _has_final_answer(pred: Any) -> bool:
    """最终答案有效性：num_interferers 可解析为非负整数且 interferers 是 list。

    缺 num_interferers（长推理无 JSON / 截断）的回复按无效处理，触发 repair 轮。
    """
    if not isinstance(pred, dict):
        return False
    n = _to_float(pred.get("num_interferers"), float("nan"))
    if not np.isfinite(n) or n < 0:
        return False
    return isinstance(pred.get("interferers", []), list)


def _format_observation(obs: dict) -> str:
    """把 observation（先验信息）转成人类可读描述，明确归一化约定。

    只含 Agent 已知的设备/任务先验，不含任何干扰真值。
    """
    r = obs.get("receiver", {})
    fc = float(r.get("center_frequency_hz", 0.0)) / 1e6
    fs = float(r.get("sampling_rate_hz", 0.0)) / 1e6
    d = float(r.get("antenna_spacing_m", 0.0))
    lam = float(r.get("wavelength_m", 0.0))
    lines = [
        f"Receiver center frequency: {fc:.1f} MHz",
        f"Sampling rate: {fs:.1f} MHz",
        f"Antenna array: 4-element ULA, element spacing {d:.4f} m "
        f"(= {d / lam:.2f} wavelength, half-wavelength)",
        f"Target signal modulation: {obs.get('target_modulation', 'unknown')}",
        f"Target occupied bandwidth (OBW99, normalized): "
        f"{obs.get('target_bandwidth_normalized', 'unknown')}",
        f"Observed samples per antenna: {obs.get('num_samples', 1024)}",
        "Normalization conventions: freq_offset and bandwidth are normalized to the "
        "sampling rate (freq in [-0.5, 0.5], bandwidth in [0, 1]); DOA is in degrees.",
    ]
    return "\n".join(lines)


class OpenAICompatibleAgent:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 use_tools: bool = True, timeout: float = 300.0,
                 reasoning_effort: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use API evaluation") from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                             base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                             timeout=float(timeout))
        self.model = model
        self.use_tools = bool(use_tools)
        self.reasoning_effort = reasoning_effort

    def _create(self, **kwargs):
        """chat.completions.create 封装：
        - 推理模型可传 reasoning_effort 控制思考强度（low 大幅减少 token/耗时），
          API 不支持该参数时自动降级重试；
        - 网络抖动 / 限流 / 服务端 5xx 指数退避重试 —— 数小时长跑必须扛得住
          短暂断网（ds_run13 曾因一次 DNS 抖动整跑崩溃）。"""
        import time as _t
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                if self.reasoning_effort:
                    try:
                        return self.client.chat.completions.create(
                            **kwargs, reasoning_effort=self.reasoning_effort)
                    except Exception:
                        pass  # 参数不被支持 → 去掉该参数重试
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                retryable = type(exc).__name__ in (
                    "APIConnectionError", "APITimeoutError", "InternalServerError",
                    "RateLimitError", "APIStatusError")
                if not retryable or attempt == max_attempts - 1:
                    raise
                wait = min(60.0, 5.0 * (2 ** attempt))
                print(f"[retry] {type(exc).__name__}: {wait:.0f}s 后第 "
                      f"{attempt + 1}/{max_attempts - 1} 次重试...")
                _t.sleep(wait)

    def diagnose(self, sample_path: Path, observation: dict) -> tuple[dict[str, Any], int, str, list]:
        """返回 (prediction, tool_call_count, raw_output, round_log)。

        round_log: 每轮诊断 [{"elapsed_s", "content_len", "tool_calls", "names"}]，
        用于定位"跑得慢 / 不收敛 / 空回复"的根因。
        raw_output 为模型最后一轮的内容文本（即使解析失败也返回，供诊断）。
        """
        import time as _time
        obs_text = _format_observation(observation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Analyze sample at {sample_path}.\n"
                f"Receiver prior (observation):\n{obs_text}"},
        ]
        calls = 0
        raw = ""
        round_log: list[dict] = []
        if not self.use_tools:
            # 纯推理模式：不带工具，模型仅依据观测先验与常识回答
            # （此模式无 tools 冲突，可安全启用 JSON 结构化输出）
            try:
                t0 = _time.time()
                response = self._create(
                    model=self.model, messages=messages,
                    temperature=0.2, seed=42, max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                round_log.append({"elapsed_s": round(_time.time() - t0, 1),
                                  "content_len": len(raw), "tool_calls": 0, "names": []})
                return _extract_json(raw), 0, raw, round_log
            except Exception:
                return {}, 0, "", round_log
        best_cand: dict[str, Any] = {}
        for _ in range(10):
            # 工具模式不加 max_tokens：ds-v4-flash 无限制时生成量巨大（极慢），
            # 加了又截断导致空答案 —— 先用 round_log 诊断每轮耗时/输出量，
            # 定位后再决定是否/如何限制
            t0 = _time.time()
            response = self._create(
                model=self.model, messages=messages,
                tools=TOOL_SCHEMAS_V2, tool_choice="auto",
                temperature=0.2, seed=42,
            )
            message = response.choices[0].message
            raw = message.content or ""      # 记录每轮模型原文（诊断用）
            tool_calls = getattr(message, "tool_calls", None) or []
            names = [tc.function.name for tc in tool_calls]
            round_log.append({
                "elapsed_s": round(_time.time() - t0, 1),
                "content_len": len(raw),
                "tool_calls": len(tool_calls),
                "names": names,
            })
            if not tool_calls:
                best_cand = _extract_json(raw)
                if _has_final_answer(best_cand):
                    return best_cand, calls, raw, round_log
                break                        # 无效答案（纯推理无 JSON）→ repair 轮
            calls += len(tool_calls)
            messages.append(message.model_dump() if hasattr(message, "model_dump")
                            else {"role": "assistant", "content": message.content,
                                  "tool_calls": [tc.model_dump() for tc in tool_calls]})
            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    args["sample_path"] = str(sample_path)
                    # 自动注入目标带宽先验（模型无需自己传）：频谱/源数工具用于
                    # 谱谷分裂的目标保护区，调制工具用于切片选择
                    if call.function.name in ("analyze_spectrum", "estimate_num_sources",
                                              "estimate_modulation_features"):
                        args["target_bandwidth_normalized"] = observation.get(
                            "target_bandwidth_normalized")
                    result = TOOL_FUNCTIONS_V2[call.function.name](**args)
                except Exception as exc:
                    result = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "name": call.function.name,
                                 "content": json.dumps(result)})
        # 轮数耗尽仍无最终答案（工具调用循环不收敛）或最终回复缺 JSON：
        # repair 轮 —— 不带工具参数强制文本回复（模型无法继续调工具），限
        # max_tokens 防止再次长推理。失败则回退最后一次解析结果。
        try:
            messages.append({"role": "user", "content": REPAIR_PROMPT})
            t0 = _time.time()
            response = self._create(
                model=self.model, messages=messages,
                temperature=0.2, seed=42, max_tokens=2048,
            )
            raw_rep = response.choices[0].message.content or ""
            round_log.append({"phase": "repair",
                              "elapsed_s": round(_time.time() - t0, 1),
                              "content_len": len(raw_rep),
                              "tool_calls": 0, "names": []})
            cand = _extract_json(raw_rep)
            if _has_final_answer(cand):
                return cand, calls, raw_rep, round_log
            if cand:
                best_cand = cand
        except Exception:
            pass
        return best_cand, calls, raw, round_log


def _greedy_match(gt_interferers, pred_interferers):
    """按频偏最近做贪心匹配，返回 (matched_pairs, unmatched_gt, unmatched_pred)。

    matched_pairs: [(gt_idx, pred_idx)]；匹配要求 |freq| 差 < 0.1 且 |doa| 差 < 40
    （作为"同一干扰"的对应判据），否则视为漏检/虚检。
    """
    pairs, used_p, used_g = [], set(), set()
    # 按 GT 顺序，为每个 GT 找频偏最近的未用预测（模型输出可能为 null/字符串，需清洗）
    for gi, g in enumerate(gt_interferers):
        best, best_d = None, 1e9
        g_freq = _to_float(g.get("freq_offset_normalized"), 1e9)
        g_doa = _to_float(g.get("doa_degree"), 1e9)
        for pi, p in enumerate(pred_interferers):
            if pi in used_p:
                continue
            p_freq = _to_float(p.get("freq_offset"), 1e9)
            p_doa = _to_float(p.get("doa"), 1e9)
            df = abs(p_freq - g_freq)
            dd = abs(p_doa - g_doa)
            if df < 0.1 and dd < 40.0 and df + dd * 0.01 < best_d:
                best, best_d = pi, df + dd * 0.01
        if best is not None:
            pairs.append((gi, best))
            used_p.add(best)
            used_g.add(gi)
    unmatched_g = [i for i in range(len(gt_interferers)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred_interferers)) if i not in used_p]
    return pairs, unmatched_g, unmatched_p


def _score_answer(gt: dict, pred: dict) -> dict[str, Any]:
    gt_srcs = gt["sources"][1:]                       # 干扰源（GT）
    gt_n = len(gt_srcs)
    try:
        pred_n = int(_to_float(pred.get("num_interferers"), -1)) if isinstance(pred, dict) else -1
    except (TypeError, ValueError):
        pred_n = -1
    pred_srcs = pred.get("interferers", []) if isinstance(pred, dict) else []
    if not isinstance(pred_srcs, list):
        pred_srcs = []

    num_ok = (pred_n == gt_n)
    if gt_n == 0:
        # 无干扰样本：源数答对即全对（无参数可评估）
        return {
            "num_interferers_ok": bool(num_ok),
            "num_interferers_gt": 0,
            "num_interferers_pred": pred_n,
            "matched": 0, "missed": 0, "false_alarms": int(pred_n) if pred_n > 0 else 0,
            "category_accuracy": 1.0, "freq_accuracy": 1.0,
            "bandwidth_accuracy": 1.0, "doa_accuracy": 1.0,
            "modulation_accuracy": None,
            "mod_e2e_accuracy": None,
            "mod_hits": 0, "mod_matched_total": 0, "mod_gt_total": 0,
            "freq_mae": None, "doa_mae": None, "bandwidth_mae_rel": None,
        }
    pairs, unmatched_g, unmatched_p = _greedy_match(gt_srcs, pred_srcs)

    cat_hit = freq_hit = bw_hit = doa_hit = mod_hit = mod_total = 0
    freq_err, doa_err, bw_relerr = [], [], []
    for gi, pi in pairs:
        g, p = gt_srcs[gi], pred_srcs[pi]
        if not isinstance(p, dict):
            continue
        if str(p.get("category", "")).lower() == str(g["v2_category"]).lower():
            cat_hit += 1
        if abs(_to_float(p.get("freq_offset"), 1e9) - g["freq_offset_normalized"]) < FREQ_TOL:
            freq_hit += 1
        if abs(_to_float(p.get("bandwidth"), 1e9) - g["bandwidth_normalized"]) < \
                BW_REL_TOL * max(g["bandwidth_normalized"], 1e-3):
            bw_hit += 1
        if abs(_to_float(p.get("doa"), 1e9) - g["doa_degree"]) < DOA_TOL_DEG:
            doa_hit += 1
        # 调制仅对真实调制类干扰评估（波形类干扰的 modulation 是类型名，不强制）。
        # 注：循环只遍历贪心匹配对（pairs），未匹配的干扰不计入分母 —— 该指标
        # 已是"仅匹配干扰"口径，与源数/匹配误差天然解耦。
        if g.get("waveform_type") is None:
            mod_total += 1
            if str(p.get("modulation", "")).upper() == str(g["modulation"]).upper():
                mod_hit += 1
        freq_err.append(abs(_to_float(p.get("freq_offset"), 1e9) - g["freq_offset_normalized"]))
        doa_err.append(abs(_to_float(p.get("doa"), 1e9) - g["doa_degree"]))
        bw_relerr.append(abs(_to_float(p.get("bandwidth"), 1e9) - g["bandwidth_normalized"]) /
                         max(g["bandwidth_normalized"], 1e-3))

    n_gt = max(gt_n, 1)
    # 端到端调制口径：分母 = 该样本全部真实调制类 GT 干扰（未匹配计为错）。
    # 现有 modulation_accuracy 分母只含匹配对 —— 跨 run 匹配数变化时不可比
    # （ds_run6→8 matched 25→17，调制指标被分母假象抬高）。
    mod_gt_total = sum(1 for g in gt_srcs if g.get("waveform_type") is None)
    return {
        "num_interferers_ok": bool(num_ok),
        "num_interferers_gt": gt_n,
        "num_interferers_pred": pred_n,
        "matched": len(pairs),
        "missed": len(unmatched_g),
        "false_alarms": len(unmatched_p),
        "category_accuracy": cat_hit / n_gt,
        "freq_accuracy": freq_hit / n_gt,
        "bandwidth_accuracy": bw_hit / n_gt,
        "doa_accuracy": doa_hit / n_gt,
        "modulation_accuracy": (mod_hit / mod_total) if mod_total else None,
        "mod_e2e_accuracy": (mod_hit / mod_gt_total) if mod_gt_total else None,
        "mod_hits": int(mod_hit),
        "mod_matched_total": int(mod_total),
        "mod_gt_total": int(mod_gt_total),
        "freq_mae": float(np.mean(freq_err)) if freq_err else None,
        "doa_mae": float(np.mean(doa_err)) if doa_err else None,
        "bandwidth_mae_rel": float(np.mean(bw_relerr)) if bw_relerr else None,
    }


def evaluate_dataset(dataset_dir: str | Path, agent=None, max_samples: int | None = None,
                     progress: bool = True, verbose: bool = False,
                     checkpoint_path: str | Path | None = None,
                     checkpoint_every: int = 10,
                     prior_results: list[dict] | None = None):
    """prior_results: 断点续跑 —— 上次部分完成的逐样本结果，原样前置，
    已完成的文件跳过不再调用 API。"""
    root = Path(dataset_dir)
    ground_truths = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    observations = json.loads((root / "observations.json").read_text(encoding="utf-8"))
    obs_by_file = {r["file"]: r["observation"] for r in observations}

    results = list(prior_results) if prior_results else []
    done_files = {r["file"] for r in results}
    if done_files:
        print(f"[resume] 已完成 {len(done_files)} 个样本，跳过续跑")
    total = len(ground_truths)
    if max_samples is not None:
        ground_truths = ground_truths[:max_samples]
        total = len(ground_truths)

    if progress:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
    else:
        tqdm = None
    iterator = tqdm(ground_truths, desc="Evaluating Agent", unit="sample") if tqdm else ground_truths

    for rec in iterator:
        file_name = rec["file"]
        if file_name in done_files:
            continue
        gt = rec["ground_truth"]
        obs = obs_by_file.get(file_name, {})
        sample = root / file_name

        if agent is None:
            # 离线回放：用 GT 构造"完美答案"验证评分管道
            pred = {
                "num_interferers": len(gt["sources"]) - 1,
                "interferers": [
                    {"category": s["v2_category"], "modulation": s["modulation"],
                     "freq_offset": s["freq_offset_normalized"],
                     "bandwidth": s["bandwidth_normalized"], "doa": s["doa_degree"]}
                    for s in gt["sources"][1:]
                ],
            }
            tool_calls = 0
            raw_output = ""
            round_log = []
        else:
            pred, tool_calls, raw_output, round_log = agent.diagnose(sample, obs)

        score = _score_answer(gt, pred)
        results.append({
            "file": file_name,
            "snr_db": gt.get("snr_db"),
            "num_sources": gt.get("num_sources"),
            "tool_calls": int(tool_calls),
            "raw_output": str(raw_output)[:500],   # 诊断：模型最后一轮原文
            "round_log": round_log,                # 诊断：每轮耗时/输出量/工具调用
            "prediction": pred,
            **score,
        })
        n_done = len(results)
        # 定期存档：长跑（全量 500 约 6h）中途崩溃/断网不丢已完成样本
        if checkpoint_path and n_done % max(int(checkpoint_every), 1) == 0:
            try:
                partial = {**_aggregate(results, total), "partial": True,
                           "num_completed": n_done}
                Path(checkpoint_path).write_text(
                    json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")
                msg = f"[checkpoint] saved {n_done}/{total} -> {checkpoint_path}"
                (tqdm.write(msg) if tqdm else print(msg))
            except OSError:
                pass
        if tqdm:
            iterator.set_postfix({
                "src_acc": f"{sum(r['num_interferers_ok'] for r in results)/n_done:.2f}",
                "cat_acc": f"{sum(r['category_accuracy'] for r in results)/n_done:.2f}",
            })
        if verbose:
            line = (f"[{file_name}] GT_src={score['num_interferers_gt']} "
                    f"pred_src={score['num_interferers_pred']} "
                    f"cat={score['category_accuracy']:.2f} "
                    f"({'OK' if score['num_interferers_ok'] else 'FAIL'})")
            if tqdm:
                tqdm.write(line)
            else:
                print(line)
            # 诊断：每轮耗时与输出量（定位"慢 / 不收敛 / 空回复"）
            for rnd in round_log:
                detail = (f"    round: {rnd['elapsed_s']:6.1f}s "
                          f"content={rnd['content_len']:5d} "
                          f"tools={rnd['tool_calls']} {rnd['names']}")
                if tqdm:
                    tqdm.write(detail)
                else:
                    print(detail)
            try:
                pred_str = json.dumps(pred, ensure_ascii=False)
            except (TypeError, ValueError):
                pred_str = str(pred)
            if tqdm:
                tqdm.write("    pred: " + pred_str[:400])
            else:
                print("    pred: " + pred_str[:400])

    return _aggregate(results, total)


def _code_fingerprint() -> str:
    """评估侧代码指纹（md5 前 12 位）：写入报告，跨 run 对比可精确归因版本。"""
    h = hashlib.md5()
    for name in ("evaluator_v2.py", "tools_v2.py"):
        p = _ROOT / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _aggregate(results, total):
    def mean(key, skip_none=True):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _pooled_ratio(rows, num_key, den_key):
        n = sum(int(r.get(num_key) or 0) for r in rows)
        d = sum(int(r.get(den_key) or 0) for r in rows)
        return (n / d) if d else None

    # 按 SNR 分组
    groups = {"Low": [], "Mid": [], "High": []}
    for r in results:
        snr = r["snr_db"]
        g = "Low" if snr < 0 else ("Mid" if snr <= 8 else "High")
        groups[g].append(r)
    acc_by_snr = {g: (float(np.mean([r["category_accuracy"] for r in rs])) if rs else None)
                  for g, rs in groups.items()}

    return {
        "dataset": str(Path(results[0]["file"]).parent) if results else "",
        "code_fingerprint": _code_fingerprint(),
        "num_samples": int(total),
        "num_interferers_accuracy": float(np.mean([r["num_interferers_ok"] for r in results])),
        "category_accuracy": mean("category_accuracy"),
        "freq_accuracy": mean("freq_accuracy"),
        "bandwidth_accuracy": mean("bandwidth_accuracy"),
        "doa_accuracy": mean("doa_accuracy"),
        "modulation_accuracy": mean("modulation_accuracy"),
        # 端到端调制：全局 hits / 全局真实调制 GT 总数（跨 run 可比，不受匹配数影响）
        "modulation_accuracy_e2e": _pooled_ratio(results, "mod_hits", "mod_gt_total"),
        # 匹配对池化口径（对照 mean-of-ratios 的 modulation_accuracy，诊断分母效应用）
        "modulation_accuracy_matched_pooled": _pooled_ratio(results, "mod_hits", "mod_matched_total"),
        "freq_mae": mean("freq_mae"),
        "doa_mae": mean("doa_mae"),
        "bandwidth_mae_rel": mean("bandwidth_mae_rel"),
        "category_accuracy_by_snr": acc_by_snr,
        "snr_group_counts": {g: len(rs) for g, rs in groups.items()},
        "missed_sources_total": int(sum(r["missed"] for r in results)),
        "false_alarm_total": int(sum(r["false_alarms"] for r in results)),
        "avg_tool_calls_per_sample": float(np.mean([r["tool_calls"] for r in results])),
        "tool_call_rate": float(np.mean([r["tool_calls"] > 0 for r in results])),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("--output", default="eval_report_v2.json")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None,
                        help="OpenAI 兼容端点，如 Ollama: http://localhost:11434/v1")
    parser.add_argument("--api-key", default=None,
                        help="API Key（Ollama 可任意填，如 'ollama'）")
    parser.add_argument("--offline", action="store_true",
                        help="Replay ground truth as the answer (validate scoring pipeline)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只评估前 N 个样本（快速试跑）")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    parser.add_argument("--verbose", action="store_true", help="逐样本打印结果与模型输出")
    parser.add_argument("--no-tools", action="store_true",
                        help="不给 Agent 提供工具（纯推理兜底；无测量值，精度通常很低）")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["low", "medium", "high"],
                        help="推理模型思考强度（v4 系列可用 low 大幅减少推理 token/耗时；"
                             "API 不支持时自动忽略）")
    parser.add_argument("--resume", action="store_true",
                        help="从输出文件中的部分结果续跑（配合定期存档；已完成的"
                             "样本跳过，不重复调用 API）")
    args = parser.parse_args()

    prior_results = None
    if args.resume:
        out_path = Path(args.output)
        if out_path.exists():
            try:
                old = json.loads(out_path.read_text(encoding="utf-8"))
                prior_results = old.get("results") or []
                if not prior_results:
                    print(f"[resume] {out_path} 无已完成结果，从头开始")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[resume] 存档读取失败（{exc}），从头开始")
        else:
            print(f"[resume] 找不到 {out_path}，从头开始")

    agent = None if args.offline else OpenAICompatibleAgent(
        args.model, base_url=args.base_url, api_key=args.api_key,
        use_tools=not args.no_tools, reasoning_effort=args.reasoning_effort)
    report = evaluate_dataset(args.dataset_dir, agent, max_samples=args.max_samples,
                              progress=not args.no_progress, verbose=args.verbose,
                              checkpoint_path=args.output,
                              prior_results=prior_results)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    keys = ("code_fingerprint", "num_samples", "num_interferers_accuracy",
            "category_accuracy", "freq_accuracy", "bandwidth_accuracy",
            "doa_accuracy", "modulation_accuracy", "modulation_accuracy_e2e",
            "freq_mae", "doa_mae", "bandwidth_mae_rel",
            "avg_tool_calls_per_sample", "tool_call_rate")
    print(json.dumps({k: report.get(k) for k in keys}, indent=2, ensure_ascii=False))
    print(f"Report -> {args.output}")


if __name__ == "__main__":
    main()
