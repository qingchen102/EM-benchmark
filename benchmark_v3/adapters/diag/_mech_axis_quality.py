"""机理轴质量分析（零 API，只读）：混淆矩阵 + 逐类召回 + 宏平均 + 参考基线。

背景：首轮只报了 mech_accuracy 与"常数多数类基线"，结论写成"只比瞎蒙好"太糙——
（一）机理只在**已配对成功**的样本上计分（关联本身是更难的轴，模型先得把源找到、对上号）；
（二）类别极不平衡时 accuracy 会被多数类主导，必须看逐类召回与宏平均。
本脚本按评测器**同一套**配对规则（eval_dataset.match）复算，输出：

  1. 关联覆盖：GT 干扰数 / 配对数（机理的分母为什么小于 GT 总数）
  2. 预测分布 vs 真值分布、混淆矩阵
  3. 逐类 recall / precision / support、accuracy、macro-recall、macro-F1
  4. 两个参考基线：常数多数类（不看输入的最优猜测）、按预测先验抽签（Σ p_gt·p_pred）

用法：
  python -X utf8 diag/_mech_axis_quality.py                    # 四档 mech 臂
  python -X utf8 diag/_mech_axis_quality.py <dataset> <SLEVEL> <answers_dir>
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys

_PLUG = pathlib.Path(__file__).resolve()
_HERE = _PLUG.parent
_ADAPTERS = _HERE.parent
_V3 = _ADAPTERS.parent
sys.path.insert(0, str(_ADAPTERS))

import eval_dataset as ed  # noqa: E402
import mechanism as mk  # noqa: E402

DEFAULT_ARMS = (
    ("dataset_d1", "S0", "answers_S0_agent_fix2_mech"),
    ("dataset_d2", "S2", "answers_S2_agent_fix2_mech"),
    ("dataset_d3", "S2", "answers_S2_agent_fix2_mech"),
    ("dataset_d4", "S2", "answers_S2_agent_fix2_mech"),
)
CLASSES = ("co_channel", "adjacent", "pulse", "spoof")
EMPTY = "（缺失）"


def analyse(dataset: str, slevel: str, answers_dir: str, out: list[str]) -> None:
    root = _V3 / dataset
    ans_root = root / answers_dir
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    subsets = idx["observation_subsets"]
    stations = subsets[slevel]

    gt_of_pair: list[str] = []
    pred_of_pair: list[str] = []
    gt_all = collections.Counter()
    n_samples = n_missing = n_gt_total = 0
    own_of_pair: list[str] = []
    n_oob = kind_ok = 0
    df_abs: list[float] = []
    consist_zero = consist_any = 0
    for s in idx["samples"]:
        ans_p = ans_root / f"{s['id']}.json"
        if not ans_p.exists():
            continue
        n_samples += 1
        truth = ed.load_truth(root, s["id"])
        gt_list = ed.gt_at_slevel(truth, stations)
        for g in gt_list:
            gt_all[ed.gt_mechanism(g)] += 1
        n_gt_total += len(gt_list)
        ans = json.loads(ans_p.read_text(encoding="utf-8"))
        pred = [p for p in (ans.get("interferers") or [])
                if slevel != "S0" or "doa_deg" in p]
        for p, g in ed.match(pred, gt_list):
            gm = ed.gt_mechanism(g)
            pm = str(p.get("mechanism", "")).strip().lower() or EMPTY
            try:
                m_own = mk.derive(p.get("kind"), float(p.get("freq_offset_hz") or 0.0),
                                  float(g.get("bandwidth_hz") or 0.0))
            except (ValueError, TypeError):
                m_own = "越界"
                n_oob += 1
            own_of_pair.append(m_own)
            try:
                m_zero = mk.derive(p.get("kind"), float(p.get("freq_offset_hz") or 0.0), 0.0)
            except (ValueError, TypeError):
                m_zero = "越界"
            if m_zero == pm:
                consist_zero += 1
            if pm in (m_own, m_zero):
                consist_any += 1
            if str(p.get("kind")) == str(g.get("kind")):
                kind_ok += 1
            df_abs.append(abs(float(p.get("freq_offset_hz") or 0.0)
                              - float(g.get("freq_offset_hz") or 0.0)))
            if pm == EMPTY:
                n_missing += 1
            gt_of_pair.append(gm)
            pred_of_pair.append(pm)

    dom = sorted(set(gt_all) | set(pred_of_pair) | set(CLASSES))
    conf = collections.Counter(zip(gt_of_pair, pred_of_pair))
    n = len(gt_of_pair)
    out.append(f"\n{'=' * 78}")
    out.append(f"{dataset}  {slevel}  {answers_dir}")
    out.append(f"{'=' * 78}")
    out.append(f"样本 {n_samples} / 50    GT 干扰总数 {n_gt_total}    配对成功 {n}"
               f"（关联覆盖 {n / n_gt_total:.1%}）    预测未填机理 {n_missing}")
    if not n:
        out.append("（没有配对成功样本，机理轴无分母）")
        return

    out.append(f"\n真值分布： " + "  ".join(f"{c}={gt_all.get(c, 0)}" for c in dom if gt_all.get(c)))
    pred_all = collections.Counter(pred_of_pair)
    out.append(f"预测分布： " + "  ".join(f"{c}={pred_all.get(c, 0)}" for c in dom if pred_all.get(c)))

    out.append("\n混淆矩阵（行=真值，列=预测）：")
    header = "  {:>11} " + " ".join(f"{c[:11]:>11}" for c in dom) + f" {'support':>8}"
    out.append(header.format("GT \\ PRED"))
    for g in dom:
        if not gt_all.get(g):
            continue
        row = "  {:>11} ".format(g)
        cols = []
        for p in dom:
            v = conf.get((g, p), 0)
            cols.append(f"{'·' if v == 0 else v:>11}")
        out.append(row + " ".join(cols) + f" {gt_all[g]:>8}")

    out.append("\n逐类指标：")
    out.append(f"  {'类别':<11}{'support':>8}{'recall':>9}{'pred':>7}{'precision':>11}{'F1':>8}")
    recalls, f1s = [], []
    for c in dom:
        sup = gt_all.get(c, 0)
        if not sup:
            continue
        rec = conf.get((c, c), 0) / sup
        npred = pred_all.get(c, 0)
        prec = conf.get((c, c), 0) / npred if npred else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec)
        f1s.append(f1)
        out.append(f"  {c:<11}{sup:>8}{rec:>9.3f}{npred:>7}{prec:>11.3f}{f1:>8.3f}")

    acc = sum(conf.get((c, c), 0) for c in dom) / n
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    maj = max(gt_all.values()) / n if n else 0.0
    chance = sum((gt_all.get(c, 0) / n) * (pred_all.get(c, 0) / n) for c in dom)
    ncls = len([c for c in dom if gt_all.get(c)])
    out.append(f"\n  accuracy（逐对）      = {acc:.3f}   ← 报告里的 mech_accuracy 同口径")
    out.append(f"  macro-recall（逐类等权）= {macro_rec:.3f}")
    out.append(f"  macro-F1               = {macro_f1:.3f}")
    out.append(f"  参考：常数多数类        = {maj:.3f}   ← 完全不看输入的最优猜测")
    out.append(f"  参考：按预测先验抽签    = {chance:.3f}")
    out.append(f"  参考：K 类等权抽签 1/K  = {1 / ncls:.3f}   （本档实际出现 {ncls} 类）")

    own_acc = sum(1 for a, b in zip(own_of_pair, gt_of_pair) if a == b) / n
    consist = sum(1 for a, b in zip(own_of_pair, pred_of_pair) if a == b) / n
    med_df = statistics.median(df_abs) if df_abs else 0.0
    out.append("\n  拆解（机理是 kind + 频率偏置的**派生量**，不是独立能力）：")
    out.append(f"    模型填的 mechanism 与真值一致          = {acc:.3f}")
    out.append(f"    用**它自己报的** kind/f0 推机理再比真值 = {own_acc:.3f}"
               f"   （kind 逐对命中 {kind_ok / n:.1%}，|Δf| 中位 {med_df / 1e3:.1f} kHz）")
    out.append(f"    它填的 mechanism 与自己的 kind/f0 自洽  = {consist:.3f}"
               f"   （不自洽 = 机理字段不是从自己报的证据推的）")
    if n_oob:
        out.append(f"    按它自己的 kind/f0 推到窗外（报错）的对数 = {n_oob}")
    out.append(f"    自洽性上界（带宽当 0，只看 |f0| 是否 ≤500 kHz）= {consist_zero / n:.3f}")
    out.append(f"    自洽性宽松口径（两种带宽假设任一命中）       = {consist_any / n:.3f}")


def main(argv: list[str]) -> int:
    arms = ([(argv[0], argv[1], argv[2])] if len(argv) >= 3 else DEFAULT_ARMS)
    out: list[str] = [f"机理轴质量分析（零 API；配对规则与 eval_dataset.match 一致，gate=0.45）"]
    for dataset, slevel, answers_dir in arms:
        analyse(dataset, slevel, answers_dir, out)
    text = "\n".join(out) + "\n"
    print(text, end="")
    import os
    dst = pathlib.Path(os.environ.get("MECH_AXIS_OUT")
                       or (_V3 / "evidence" / "mech_axis_quality_2026-10-12.txt"))
    dst.write_text(text, encoding="utf-8")
    print(f"\n[已写入] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
