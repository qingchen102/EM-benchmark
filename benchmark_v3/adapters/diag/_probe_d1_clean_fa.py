"""干净样本虚警的形态统计（零 API，只读）。

背景：D1 的干净样本（真值 `interferers == []`）上，agent 偶尔会报 1~2 个源。抽查
`d1_00012` 发现那两条报告的 **DOA 与位置完全相同**，都等于合作目标自己的方位/位置
—— 看起来是把目标自身的滚降当成了邻道源。

本脚本把这条"签名"做成统计：对每个真值源数为 0、而答案非空的样本，
  * 报了几条；
  * 这些条的 DOA 是否彼此一致（max pairwise ≤ `doa_tol`）、位置是否彼此一致（≤ `pos_tol`）；
  * 一致的样本里，报的 `kind` / `mechanism` 分布。
再给出干净样本虚警率与"同源签名"占比，用于评估是否值得做题面修订（明说：若门控清单为空
且所见结构与目标同向同址，就报空列表）。

用法：
    python -X utf8 diag/_probe_d1_clean_fa.py [dataset_d1] [answers_S0_agent_quota]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ADAPTERS = HERE.parent
if str(_ADAPTERS) not in sys.path:
    sys.path.insert(0, str(_ADAPTERS))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eval_dataset as ed  # noqa: E402

_V3 = _ADAPTERS.parent


def _pairwise_consistent(items: list[dict], doa_tol: float, pos_tol: float) -> bool:
    """所有条目的 DOA 与位置两两一致（缺字段则跳过该维度的检查）。"""
    doas = [float(i["doa_deg"]) for i in items if i.get("doa_deg") is not None]
    poss = [tuple(i["position_m"][:2]) for i in items if i.get("position_m")]
    if len(doas) < 2 or len(poss) < 2:
        return False
    if max(doas) - min(doas) > doa_tol:
        return False
    for i in range(len(poss)):
        for j in range(i + 1, len(poss)):
            if abs(poss[i][0] - poss[j][0]) > pos_tol or abs(poss[i][1] - poss[j][1]) > pos_tol:
                return False
    return True


def main(argv: list[str]) -> int:
    deck = argv[0] if argv else "dataset_d1"
    arm = argv[1] if len(argv) > 1 else "answers_S0_agent_quota"
    root = Path(deck)
    if not root.is_absolute():
        cand = _V3 / deck
        # deck 可以是相对 benchmark_v3/ 的路径（如 _pre_quota_20261012/dataset_d1），也可以是 CWD 相对路径
        root = cand if (cand / "index.json").is_file() else (Path.cwd() / deck)
    adir = root / arm
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    sids = [s["id"] for s in idx["samples"]]
    slevel = "S0" if "_S0_" in arm else "S2"
    stations = idx["observation_subsets"][slevel]

    n_eval = n_clean = n_fa = n_sig = 0
    kinds: dict[str, int] = {}
    mechs: dict[str, int] = {}
    lines: list[str] = []
    for sid in sids:
        ans_p = adir / f"{sid}.json"
        if not ans_p.is_file():
            continue
        truth = ed.load_truth(root, sid)
        gt_list = ed.gt_at_slevel(truth, stations)
        if gt_list:
            continue
        n_clean += 1
        pred = json.loads(ans_p.read_text(encoding="utf-8"))
        items = pred.get("interferers") or []
        n_fa += 1 if items else 0
        if not items:
            continue
        ok = _pairwise_consistent(items, doa_tol=3.0, pos_tol=20.0)
        n_sig += 1 if ok else 0
        for it in items:
            kinds[str(it.get("kind"))] = kinds.get(str(it.get("kind")), 0) + 1
            mechs[str(it.get("mechanism"))] = mechs.get(str(it.get("mechanism")), 0) + 1
        lines.append(
            f"  {sid}: 报 {len(items)} 条"
            + ("  [同源签名: DOA/位置一致]" if ok else "  [条目不自洽]")
            + "  " + ", ".join(
                f"{i.get('kind')}@{i.get('freq_offset_hz')}/{(i.get('mechanism'))}" for i in items
            )
        )
    n_eval = n_clean

    out = [
        f"干净样本虚警形态统计（零 API，只读）  {root.name}/{arm}  S 级 = {slevel}",
        "=" * 92,
        f"已评样本            {sum(1 for s in sids if (adir / f'{s}.json').is_file())}/{len(sids)}",
        f"干净样本（GT=0）    {n_clean}",
        f"其中报非空（虚警）  {n_fa}"
        + (f"    虚警率 {n_fa / n_clean:.1%}" if n_clean else ""),
        f"同源签名（DOA+位置一致） {n_sig}"
        + (f"    占虚警 {n_sig / n_fa:.0%}" if n_fa else ""),
        f"报出的 kind 分布    {kinds or '（无）'}",
        f"报出的 mechanism    {mechs or '（无）'}",
        "",
        "逐样本（仅列干净却报非空的）：",
    ]
    out += lines or ["  （无）"]
    text = "\n".join(out) + "\n"
    print(text, end="")
    suffix = ("d1" if "_S0_" in arm else "d2") + "_" + arm.replace("answers_", "")
    dst = _V3 / "evidence" / f"_probe_clean_fa_{suffix}.txt"
    dst.write_text(text, encoding="utf-8")
    print(f"[已写入] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
