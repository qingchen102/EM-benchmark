"""零成本：把「目标在本站是否 0 径（残差工具不可用）」与 agent 实测命中率对齐，
量化 0-path 站的代价，并看这些样本在别的站上是否还有可答的干扰。

只读。用法：
    python _an_zeropath_impact.py
输出：
    benchmark_v3/_an_zeropath_impact.txt
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH = Path(__file__).resolve().parents[1]          # benchmark_v3
sys.path.insert(0, str(BENCH / "adapters"))
import eval_dataset as ed  # noqa: E402

ROOT = BENCH
OUT = BENCH / "_an_zeropath_impact.txt"

# (档, S 级, agent 答案目录)  —— 答案目录里的 JSON 与 dataset_<档> 同级
RUNS = [
    ("d1", "S0", "answers_S0_agent_revert"),
    ("d2", "S2", "answers_S2_agent_bwfix"),
    ("d3", "S2", "answers_S2_agent_bwfix"),
    ("d4", "S2", "answers_S2_agent_bwfix"),
]


def station_state(root: Path, sid: str, st: str) -> str:
    """与 oracle16._target_ref_state 同判据：缺文件=missing，0 径=invisible，否则 ok。"""
    p = root / sid / "capture" / "private" / f"channel_target_{st}.npz"
    if not p.exists():
        return "missing"
    try:
        with np.load(p) as z:
            d = z.get("delays_s")
        return "invisible" if d is None or d.size == 0 else "ok"
    except Exception:
        return "loaderr"


def main() -> int:
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 78)
    say("0-path 站（目标 0 径 → 该站残差工具返回 error）对 agent 命中率的代价")
    say("=" * 78)
    say("判据：capture/private/channel_target_<st>.npz 缺=missing；delays_s.size==0=invisible")
    say("detect = 报数与 GT 完全一致（注册口径 detection_hit）")
    say()

    for diff, slevel, adir in RUNS:
        root = ROOT / f"dataset_{diff}"
        if not root.exists():
            say(f"[{diff}] 目录不存在，跳过")
            continue
        samples = sorted(p for p in root.glob(f"{diff}_[0-9]*") if p.is_dir())
        if not samples:
            say(f"[{diff}] 无样本，跳过")
            continue
        t0 = ed.load_truth(root, samples[0].name)
        stations: list[str] = t0["station_subsets"][slevel]
        st_a = stations[0]

        rows = []
        for d in samples:
            sid = d.name
            t = ed.load_truth(root, sid)
            gt = len(ed.gt_at_slevel(t, stations))
            ap = root / adir / f"{sid}.json"
            n_ifw, abst, has_ans = None, False, False
            if ap.exists():
                try:
                    o = json.loads(ap.read_text(encoding="utf-8"))
                    has_ans = True
                    n_ifw = o.get("num_interferers")
                    abst = bool(o.get("abstained"))
                except Exception:
                    pass
            states = {st: station_state(root, sid, st) for st in stations}
            # 别的站上是否还有可答的干扰（GT 在 stations 上算，可见性已含"收不到不计入"）
            rows.append(dict(sid=sid, gt=gt, n=n_ifw, abst=abst, has_ans=has_ans,
                             a=states[st_a], states=states))

        def grp(name: str, pred) -> str:
            sub = [r for r in rows if pred(r)]
            if not sub:
                return f"    {name:<22} n=0"
            ok = [r for r in sub if r["has_ans"] and r["n"] == r["gt"]]
            ab = [r for r in sub if r["abst"]]
            nn = [r["n"] for r in sub if r["has_ans"] and r["n"] is not None]
            mean_rep = float(np.mean(nn)) if nn else float("nan")
            return (f"    {name:<22} n={len(sub):<3} detect={len(ok)}/{len(sub)}"
                    f"={len(ok)/len(sub):.3f}  拒答={len(ab):<3} 平均报数={mean_rep:.2f}"
                    f"  平均GT={np.mean([r['gt'] for r in sub]):.2f}")

        say(f"--- {diff} × {slevel}（{adir}，n={len(rows)}，主站={st_a}）---")
        say(grp(f"{st_a}=ok", lambda r: r["a"] == "ok"))
        say(grp(f"{st_a}=invisible", lambda r: r["a"] == "invisible"))
        say(grp(f"{st_a}=missing", lambda r: r["a"] == "missing"))
        sub = [r for r in rows if r["a"] != "ok"]
        if sub:
            allinv = [r for r in sub if all(v != "ok" for v in r["states"].values())]
            say(f"    其中{st_a}不可用的 {len(sub)} 个样本里，四站全不可用 {len(allinv)} 个"
                f"（这些样本仪器无残差工具可用，但题目答案仍由 GT/其他工具决定）")
            nz = [r for r in sub if r["gt"] > 0]
            say(f"    其中 GT>0（确实有干扰可答）{len(nz)} 个，"
                f"其 detect={sum(1 for r in nz if r['has_ans'] and r['n'] == r['gt'])}/{len(nz)}")
        say()

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[写出] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
