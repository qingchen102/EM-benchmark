"""一次性（幂等）：把 2026-10-09 的 D2 归因结论写进 dataset_d2/BASELINE.json。

背景：BASELINE 里的 agent_full_50 = 0.86 是「工具返回值泄漏」的产物，不是模型能力；
今天又跑了两轮 50 样本（layer1 / layer1b，均在泄漏修复之前），需一并登记。
写盘口径与既有文件一致：indent=1、ensure_ascii=False、结尾换行。
"""
from __future__ import annotations

import json
import pathlib

P = pathlib.Path(__file__).resolve().parents[1] / "dataset_d2" / "BASELINE.json"
d = json.loads(P.read_text(encoding="utf-8"))

d["timestamp"] = "2026-10-09"

d["agent_full_50"]["note"] = (
    "**【2026-10-09 更正】这 0.86 不是模型能力，是工具返回值泄漏的产物。**"
    "capture_tools.estimate_band_occupancy_by_correlation 为 D1 虚警根治同时返回两条清单"
    "（对外门控版 residual_peaks / 仅供聚类内部定带的 residual_peaks_legacy），"
    "而 evaluator_v2.py:367-372 把工具返回值整体 json.dumps 回灌给模型 → agent 直接抄旧宽带清单："
    "本目录 answers_S2_agent_full 的答案频率与 legacy 清单前两名逐位相同"
    "（d2_00001 −678400/520100、d2_00009 661200/−207700、d2_00000 328300/−483200，"
    "见 benchmark_v3/_an_d2_oldnew_trace.txt）。泄漏已由 52a2f42 修复"
    "（capture_tools.AGENT_HIDDEN_KEYS + _agent_facing）。"
)

d["agent_layer1_2026_10_09"] = {
    "answers_dir": "answers_S2_agent_layer1",
    "date": "2026-10-09",
    "n": 50,
    "model": "deepseek-v3.2（百炼）",
    "detection_hit": 0.48,
    "detection_within1": 0.57,
    "detect_abs_err": 0.86,
    "kind_hits": 0.4821,
    "freq_hits": 0.2143,
    "freq_aes": 0.1787,
    "pos_hits": 0.5357,
    "pos_errs": 232.0,
    "intent_hits": 0.768,
    "spoof_fa": 0.0,
    "layer1b": {
        "answers_dir": "answers_S2_agent_layer1b",
        "detection_hit": 0.48,
        "detection_within1": 0.57,
        "detect_abs_err": 0.86,
        "kind_hits": 0.4464,
        "freq_hits": 0.1964,
        "freq_aes": 0.1822,
        "pos_hits": 0.5,
        "pos_errs": 273.72,
        "intent_hits": 0.768,
        "spoof_fa": 0.0,
    },
    "note": (
        "数据层已排除：聚类 A/B 上 K 33/50 相同、ΣK 119→109；全量证据 A/B 上门控峰条数相同 43/50、"
        "逐条强度差 ≤1.0 dB 41/50、时频指纹 ≤5% 41/50（见 _diag_d2_cluster_ab.txt / "
        "_diag_d2_evidence_magnitude.txt）。"
        "真瓶颈是仪器缺「同源并合」：estimate_sources_by_spatial_clustering 的 K = len(sources) "
        "与 GT 源数精确匹配仅 13/50（旧数据）/ 16/50（新数据），而 prompt 却要求照抄 K；"
        "agent 的应对是拒答（layer1b 50 样本里 15 个把 num_interferers 写成被禁哨兵 -1，"
        "再被 run_capture_agent._sanitize 静默清成 0）。"
        "注意：layer1 与 layer1b 都跑在泄漏修复（capture_tools.py mtime 14:04）之前"
        "（layer1 12:02–13:14、layer1b 13:41–14:16）；**修复后的 D2 从未跑过 agent**。"
        "layer1b 与 layer1 的唯一差别是去掉事件日志 prompt 段落，结果同为 0.48 → 段落不是变量。"
        "eval_report_S2.json 现为 layer1b 的报告（layer1 的同名报告已被覆盖）。"
    ),
}

d["gate_toolview"]["reverified_2026_10_09"] = (
    "同日在两套数据上各跑一次 oracle16.py 复核（_gate_d2_postnoise.txt / _gate_d2_prenoise.txt）："
    "postnoise = 0.510/0.612/0.480/0.543/0.245（与本节正式值逐位相同）、"
    "prenoise = 0.561/0.623/0.473/0.591/0.265（与 _history_2026_10_08 逐位相同）→ "
    "噪声菜单没有动到 gate 口径。gate 不走 evaluator_v2，故与峰清单泄漏无关。"
)

P.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"[ok] 已写回 {P}")
print("[keys]", list(d.keys()))
