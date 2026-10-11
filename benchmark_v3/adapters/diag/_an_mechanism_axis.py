"""机理轴体检（只读、零 API）——如果给评分加"机理"这一轴，题目撑不撑得住？

输出四块：
  ① 四档机理普查（按 mechanism.derive 的规则派生），并与**现行真值标签**交叉
     （现行 gen_dataset 只为 D3/D4 的 preset 恶意源写 jam/spoof，其它源连事件都没有）
  ② 每类机理的正样本数 / 涉及样本数 / ≥3 站可见数（per_station.inr_db ≥ 3）
  ③ co_channel 与 adjacent 的边界体检：重叠宽度分布（判断"擦边重叠"会不会把两类搅在一起）
  ④ 每类"要判定它需要什么证据 / 现有工具给不给得出"的对照表（缺口单独标出）

用法（在 benchmark_v3/adapters 下）：
    python -X utf8 -u diag\\_an_mechanism_axis.py [--samples N]
输出写到 benchmark_v3/evidence/_an_mechanism_axis.txt
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

_sys, _pl = sys, Path
_sys.path.insert(0, str(_pl(__file__).resolve().parents[1]))

import eval_dataset as ed          # noqa: E402
import mechanism as mk             # noqa: E402

ROOT = _pl(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "_an_mechanism_axis.txt"
DATASETS = ("d1", "d2", "d3", "d4")

# ④ 判据 → 现有工具 → 现状（依据 2026-10-11 已知实测结论，非本轮新测）
EVIDENCE_MAP = (
    ("co_channel", "干扰带 ∩ 目标带 > 0；目标带 = 目标载频 ±0.5 MHz（公开先验 target_bw_hz）",
     "analyze_spectrum_station（给 band_hz）+ 目标先验", "可判，但共道多源不分流：D2 两源同带宽 851 kHz"),
    ("adjacent", "干扰带在 ±810 kHz 通带内、但与目标带不重叠",
     "同上（与 co_channel 同一组测量）", "可判；需目标带位置正确（目标 freq_offset=0 是公开先验）"),
    ("pulse", "时域脉冲结构：占空比 / 脉冲沿计数 / PAPR",
     "detect_time_domain_station（duty/edge/PAPR）", "工具在；脉冲型辐射体 4 类（Welder/VFD_noise/Corona/Radar_pulsed_LFM）"),
    ("spoof", "GNSS 类结构 + 参数错误（伪距/星历不自洽）",
     "identify_modulation_features / CNN 体制分类", "只有 kind 层面的间接证据（CNN 判 GNSS_spoof）；无参数级判据"),
    ("jam", "（保留但不启用）兜底：带内但频带信息缺失/类别未知", "—", "仅兜底，不作为主答案；当前四值域里没有它"),
    ("blocking/imd/harmonic", "窗外强源前端注入 → 灵敏度塌方 / 三阶互调产物",
     "平台无此能力（前端注入未实现）", "**零正样本**：当前前端口 −79 dBm 量级，比 IIP3 低 60+ dB"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0, help="每档最多看多少样本（0=全部）")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    grand: Counter = Counter()
    per_ds: dict[str, Counter] = {}
    samples_of: dict[str, set] = defaultdict(set)
    visible_of: dict[str, int] = Counter()
    overlap_pct_of: dict[str, list[float]] = defaultdict(list)
    kinds_of: dict[str, Counter] = defaultdict(Counter)
    legacy_cross: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)

    for ds in DATASETS:
        dpath = ROOT / f"dataset_{ds}"
        cnt: Counter = Counter()
        n = 0
        for sp in sorted(dpath.glob(f"{ds}_*")):
            if not (sp / "capture" / "private" / "d1_truth.json").is_file():
                continue
            if args.samples and n >= args.samples:
                break
            n += 1
            sid = sp.name
            truth = ed.load_truth(dpath, sid)
            # 现行真值标签必须在 annotate 之前快照！
            # （annotate(overwrite=True) 会把事件表的 mechanism 也改成派生值，
            #   先 annotate 再读 legacy 会把派生值当成本来就有的标签。）
            legacy_ev = {str(e.get("source_id")): str(e.get("mechanism"))
                         for e in (truth.get("interference_events") or [])}
            legacy_def = "有事件表" if truth.get("interference_events") is not None else "无事件表"
            mk.annotate(truth, overwrite=True)
            for i, rec in enumerate(truth.get("interferers") or []):
                m = rec["mechanism"]
                kind = str(rec.get("kind"))
                cnt[m] += 1
                grand[m] += 1
                samples_of[m].add(sid)
                kinds_of[m][kind] += 1
                f0 = float(rec.get("freq_offset_hz", 0.0) or 0.0)
                bw = float(rec.get("bandwidth_hz", 0.0) or 0.0)
                ov = mk.band_overlap_hz(f0, bw)
                if m in ("co_channel", "adjacent"):
                    overlap_pct_of[m].append(100.0 * ov / bw if bw else 0.0)
                ps = rec.get("per_station") or {}
                if sum(1 for v in ps.values() if float(v.get("inr_db", -99)) >= 3.0) >= 3:
                    visible_of[m] += 1
                src = f"interferer_{i + 1}"
                legacy = legacy_ev.get(src) or (f"{legacy_def}·未标注" if legacy_def == "无事件表" else "事件表未含此源")
                legacy_cross[m][legacy] += 1
                if len(examples[m]) < 3:
                    examples[m].append(
                        f"{sid}/{src} kind={kind} f0={f0/1e3:+.0f}kHz bw={bw/1e3:.0f}kHz "
                        f"overlap={ov/1e3:.0f}kHz legacy={legacy}")
        per_ds[ds] = cnt
        say(f"[{ds}] 样本 {n}，干扰源 {sum(cnt.values())}：{dict(cnt)}")
    say()

    say("== ① 机理普查（派生）× 现行真值标签（annotate 前快照）==")
    order = [m for m in mk.DERIVABLE if grand[m]]
    for m in order:
        cross = ", ".join(f"{k}={v}" for k, v in legacy_cross[m].most_common())
        kinds = ", ".join(f"{k}×{v}" for k, v in kinds_of[m].most_common(6))
        say(f"  {m:11s} {grand[m]:4d} 个源，涉及 {len(samples_of[m]):2d} 样本，≥3 站可见 {visible_of[m]:4d}"
            f"\n              现行标签: {cross}"
            f"\n              辐射体:   {kinds}")
    say()


    say("== ② 每类典型实例 ==")
    for m in order:
        for e in examples[m]:
            say(f"  {m:11s} {e}")
    say()

    say("== ③ co_channel / adjacent 边界体检（重叠占干扰带宽的百分比）==")
    for m in ("co_channel", "adjacent"):
        v = overlap_pct_of[m]
        if not v:
            continue
        v2 = sorted(v)
        p10 = v2[max(0, int(0.10 * len(v2)) - 1)]
        say(f"  {m:11s} n={len(v2):3d}  中位 {st.median(v2):6.1f}%  p10 {p10:6.1f}%  最小 {v2[0]:6.1f}%")
    edge = [x for x in overlap_pct_of["co_channel"] if x < 20.0]
    say(f"  → co_channel 里重叠 <20% 的「擦边」实例 {len(edge)} 个"
        f"（这些在测量误差下极易被判成 adjacent）")
    say()

    say("== ④ 判据 / 现有工具 / 缺口 ==")
    for name, crit, tool, gap in EVIDENCE_MAP:
        say(f"  · {name}\n      判据: {crit}\n      工具: {tool}\n      现状: {gap}")
    say()
    say(f"派生域（{len(mk.DERIVABLE)}）：{mk.DERIVABLE}")
    say(f"未覆盖（{len(mk.NOT_COVERED)}，零正样本）：{mk.NOT_COVERED}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
