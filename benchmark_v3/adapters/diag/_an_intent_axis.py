"""意图轴的独立性核查（只读、零 API）。

问题（user m09114）：Agent 的 intent 是"自己分析判断"，还是被（表/工具）告知的？

三问：
  ① 表是否唯一      capture_tools.INTENT_OF  vs  emitters.INTENT 逐键比较
  ② GT 是否查表     真值 intent 是否恒等于 INTENT[GT kind]（= 生成期写死映射）
  ③ Agent 是否抄表   answer.intent 是否恒等于 INTENT_OF[answer.kind]
    —— 若比例≈100%，则意图轴与体制轴同源：intent 命中 ≈ "kind 落在同一 intent 桶"，
       它不再携带独立信息（与 kind_hit 的差只来自 19 类分错后桶没变）。

用法（在 benchmark_v3/adapters 下）：
    python -X utf8 -u diag\\_an_intent_axis.py
输出同时写到 benchmark_v3/evidence/_an_intent_axis.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_sys, _pl = sys, Path
_sys.path.insert(0, str(_pl(__file__).resolve().parents[1]))

import capture_tools as ct            # noqa: E402
import eval_dataset as ed             # noqa: E402
from em_signal_simulator_v3.emitters import INTENT as EMITTER_INTENT  # noqa: E402

ROOT = _pl(__file__).resolve().parents[2]      # benchmark_v3/
OUT = ROOT / "evidence" / "_an_intent_axis.txt"

# 现行注册口径的答案目录（2026-10-11 拒答修复后）
SPECS = (
    ("d1", "S0", "answers_S0_agent_fix2"),
    ("d2", "S2", "answers_S2_agent_fix2"),
    ("d3", "S2", "answers_S2_agent_fix2"),
    ("d4", "S2", "answers_S2_agent_fix2"),
)


def _report_row(ds: str, slevel: str) -> dict:
    p = ROOT / f"dataset_{ds}" / f"eval_report_{slevel}.json"
    if not p.is_file():
        return {}
    rep = json.loads(p.read_text(encoding="utf-8"))
    return rep.get(slevel) or {}


def main() -> None:
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("== ① 两张表是否一致 ==")
    keys = sorted(set(ct.INTENT_OF) | set(EMITTER_INTENT))
    diff = [(k, EMITTER_INTENT.get(k), ct.INTENT_OF.get(k)) for k in keys
            if EMITTER_INTENT.get(k) != ct.INTENT_OF.get(k)]
    say(f"capture_tools.INTENT_OF: {len(ct.INTENT_OF)} 键   "
        f"emitters.INTENT: {len(EMITTER_INTENT)} 键   差异 {len(diff)} 项")
    for k, a, b in diff:
        say(f"   ! {k}: emitters={a}  capture_tools={b}")
    say(f"   值域: {sorted(set(EMITTER_INTENT.values()))}")
    say()

    for ds, slevel, adir in SPECS:
        dpath = ROOT / f"dataset_{ds}"
        apath = dpath / adir
        if not apath.is_dir():
            say(f"-- {ds}×{slevel}: 缺答案目录 {apath}，跳过")
            continue
        subsets = json.loads((dpath / "index.json").read_text(encoding="utf-8"))["observation_subsets"]
        stations = subsets[slevel]

        # ② GT 是否查表
        gt_tot = gt_echo = 0
        gt_off: list[str] = []
        # ③ Agent 是否抄表
        ans_items = ans_echo = ans_no_intent = 0
        ans_off: list[str] = []
        n_ans_files = 0

        for sp in sorted(dpath.glob(f"{ds}_*")):
            sid = sp.name
            if not (sp / "capture" / "private" / "d1_truth.json").is_file():
                continue
            truth = ed.load_truth(dpath, sid)
            for g in ed.gt_at_slevel(truth, stations):
                gk, gi = g.get("kind"), g.get("intent")
                if gk is None:
                    continue
                gt_tot += 1
                if gi == EMITTER_INTENT.get(gk):
                    gt_echo += 1
                elif len(gt_off) < 6:
                    gt_off.append(f"{sid} kind={gk} intent={gi} 表值={EMITTER_INTENT.get(gk)}")

            ap = apath / f"{sid}.json"
            if not ap.is_file():
                continue
            n_ans_files += 1
            a = json.loads(ap.read_text(encoding="utf-8"))
            for it in (a.get("interferers") or []):
                k = it.get("kind")
                iv = it.get("intent")
                ans_items += 1
                if iv is None or iv == "":
                    ans_no_intent += 1
                elif iv == ct.INTENT_OF.get(k):
                    ans_echo += 1
                elif len(ans_off) < 6:
                    ans_off.append(f"{sid} kind={k} 填={iv} 表值={ct.INTENT_OF.get(k)}")

        row = _report_row(ds, slevel)
        say(f"== {ds}×{slevel}  ({adir}) ==")
        say(f"   答案文件 {n_ans_files} 个；答案条目 {ans_items} 条")
        say(f"   ② GT: intent == INTENT[kind]  {gt_echo}/{gt_tot}"
            + ("  ← 生成期就是查表" if gt_tot and gt_echo == gt_tot else ""))
        for s in gt_off:
            say(f"       例外 {s}")
        pct = (100.0 * ans_echo / ans_items) if ans_items else 0.0
        say(f"   ③ Agent: intent == INTENT_OF[自己报的 kind]  {ans_echo}/{ans_items} = {pct:.1f}%"
            f"   （未填 intent {ans_no_intent} 条）")
        for s in ans_off:
            say(f"       例外 {s}")
        if row:
            say(f"   报告轴: kind_hits={row.get('kind_hits')!r}  "
                f"intent_accuracy={row.get('intent_accuracy')!r}  n={row.get('n')}")
        say()

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
