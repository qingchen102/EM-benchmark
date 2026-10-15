"""新增场景（d5–d8）数据体检：零 API、只读。

与 `_audit_regen_v2.py` 的区别：那支针对 d1–d4（含 QA 约束 4 的配额判定），
这支针对**参数重排**出来的新场景档——它们没有档级机理配额，所以只做统计体检：
结构完整性、源数/机理/意图分布、可见性、IQ 大小、公开层字段。

用法：
    python -X utf8 diag/_audit_scenes.py                          # 默认 ../_regen_scenes，全四档
    python -X utf8 diag/_audit_scenes.py ../_regen_scenes d5,d6
"""
from __future__ import annotations

import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ADP = _HERE.parent
sys.path.insert(0, str(_ADP))
sys.path.insert(0, str(_ADP / "rad_cls"))

import eval_dataset as ed          # noqa: E402
import mechanism as mk             # noqa: E402
from em_signal_simulator_v3.emitters import INTENT  # noqa: E402


def _dist(counter: dict) -> str:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return "  ".join(f"{k}:{v}" for k, v in items)


def audit(root: pathlib.Path, deck: str) -> bool:
    d = root / f"dataset_{deck}"
    idx_p = d / "index.json"
    if not idx_p.is_file():
        print(f"[{deck}] 缺 index.json（跳过）")
        return False
    idx = json.loads(idx_p.read_text(encoding="utf-8"))
    subs = idx["observation_subsets"]
    n_src, mech, intent, snr_lo, snr_hi = {}, {}, {}, [], []
    vis = {lv: 0 for lv in subs}
    n_oow = 0
    n_visible_tot = 0
    iq_sizes, missing = set(), 0
    for s in idx["samples"]:
        cap = d / s["id"] / "capture"
        for need in (cap / "private" / "d1_truth.json",
                     cap / "public" / "observations.json"):
            if not need.is_file():
                missing += 1
        t = ed.load_truth(d, s["id"])
        # 窗外强源（D-25）不在答案键里：源数/机理/意图/SNR 统计均不计入，只单独计数，
        # 避免污染该档的分布口径（它们没有窗内谱峰证据，偏移/带宽也不代表窗内语义）。
        n_oow += len([g for g in t.get("interferers", []) if g.get("out_of_window")])
        ifs = [g for g in t.get("interferers", [])
               if g.get("kind") in mk_all_kinds() and not g.get("out_of_window")]
        n_src[len(ifs)] = n_src.get(len(ifs), 0) + 1
        for g in ifs:
            try:
                m = mk.derive(g["kind"], float(g.get("freq_offset_hz", 0.0)),
                              float(g.get("bandwidth_hz", 0.0)))
            except Exception:
                m = "未定义"
            mech[m] = mech.get(m, 0) + 1
        for g in t.get("interferers", []):
            if g.get("out_of_window"):
                continue
            it = g.get("intent") or INTENT.get(g.get("kind"), "?")
            intent[it] = intent.get(it, 0) + 1
        for ent in ([t.get("target") or {}]
                    + [g for g in t.get("interferers", []) if not g.get("out_of_window")]):
            for _st, rec in (ent.get("per_station") or {}).items():
                if isinstance(rec, dict) and rec.get("snr_db") is not None:
                    snr_lo.append(float(rec["snr_db"]))
        for lv, stations in subs.items():
            if ed.gt_at_slevel(t, stations):
                vis[lv] += 1
        n_visible_tot += 1
        for st in ("a", "b", "c", "d"):
            p = cap / "public" / f"iq_station_{st}.npy"
            if p.is_file():
                iq_sizes.add(p.stat().st_size)
    print(f"[{deck}] 样本 {len(idx['samples'])}  缺失文件 {missing}  IQ 大小 {sorted(iq_sizes)}")
    print(f"      源数分布 {_dist(n_src)}")
    print(f"      机理分布 {_dist(mech)}")
    print(f"      意图分布 {_dist(intent)}")
    if n_oow:
        print(f"      窗外强源 {n_oow} 个（D-25：不进答案键，不计入以上分布）")
    print(f"      可见源样本数 S0/S1/S2 = "
          + " / ".join(f"{vis[lv]}" for lv in subs))
    if snr_lo:
        print(f"      站级 SNR {min(snr_lo):.1f}~{max(snr_lo):.1f} dB")
    return True


_KINDS: set | None = None


def mk_all_kinds() -> set:
    global _KINDS
    if _KINDS is None:
        from em_signal_simulator_v3.emitters import EMITTERS
        _KINDS = set(EMITTERS)
    return _KINDS


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path("../_regen_scenes")
    decks = (argv[2].replace(",", " ").split() if len(argv) > 2
             else ["d5", "d6", "d7", "d8"])
    root = (pathlib.Path.cwd() / root).resolve() if not root.is_absolute() else root
    ok = [audit(root, dk) for dk in decks]
    print(f"[done] 审计 {sum(ok)}/{len(decks)} 档   root={root}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv))
