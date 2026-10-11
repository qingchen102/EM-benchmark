"""A/B 对比两次 D2×S2 跑的「答案契约健康度」（只读，零 API）。

对比对象默认：
  OLD = dataset_d2/answers_S2_agent_bwfix   （2026-10-10 零点前后：旧 prompt，无 confidence）
  NEW = dataset_d2/answers_S2_agent_fix1    （2026-10-10 拒答修复后：confidence + 防丢条目）

输出：两边的 拒答数 / count_field_invalid 数 / low-confidence 数 / 注册口径 detection /
排除拒答 detection / 报数分布，并逐样本列出计数发生变化的条目。

用法：
  python -X utf8 _cmp_d2_abstain.py
  python -X utf8 _cmp_d2_abstain.py --old <dir> --new <dir> --root ../dataset_d2
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_dataset as ed  # noqa: E402


def load_dir(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:                      # noqa: BLE001
            out[p.stem] = {"__error__": str(e)}
    return out


def summarize(d: Path, root: Path, slevel: str = "S2") -> dict:
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    stations = idx["observation_subsets"][slevel]
    ans = load_dir(d)
    log_ids = ed._load_abstain_ids(d)     # 旧目录无 abstained 字段 → 回退读 _run_log.json
    gts = {}
    for s in idx["samples"]:
        truth = ed.load_truth(root, s["id"])
        gts[s["id"]] = len(ed.gt_at_slevel(truth, stations))
    rows = []
    for sid in sorted(gts):
        a = ans.get(sid)
        if a is None:
            rows.append((sid, gts[sid], None, False, False, None))
            continue
        n = a.get("num_interferers")
        rows.append((sid, gts[sid], n,
                     a.get("abstained") is True or sid in log_ids,
                     a.get("count_field_invalid") is True,
                     a.get("confidence")))
    n_have = sum(1 for r in rows if r[2] is not None)
    hits = [1 if r[2] == r[1] else 0 for r in rows if r[2] is not None]
    abst = [r[3] for r in rows if r[2] is not None]
    cfi = [r[4] for r in rows if r[2] is not None]
    low = [r[5] == "low" for r in rows if r[2] is not None]
    keep = [h for h, a in zip(hits, abst) if not a]
    n_dist = Counter(r[2] for r in rows if r[2] is not None)
    return {"dir": str(d), "n": len(rows), "n_answered": n_have,
            "detection_hit": round(sum(hits) / max(n_have, 1), 3),
            "abstain_n": int(sum(abst)), "abstain_rate": round(sum(abst) / max(n_have, 1), 3),
            "excl_detection": round(sum(keep) / max(len(keep), 1), 3),
            "denom_excl": len(keep),
            "count_field_invalid_n": int(sum(cfi)),
            "low_conf_n": int(sum(low)),
            "count_dist": dict(sorted(n_dist.items(), key=lambda kv: (kv[0] is None, kv[0]))),
            "gt_dist": dict(sorted(Counter(gts.values()).items())),
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../dataset_d2")
    ap.add_argument("--old", default="answers_S2_agent_bwfix")
    ap.add_argument("--new", default="answers_S2_agent_fix1")
    ap.add_argument("--slevel", default="S2")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / args.root).resolve()
    old = root / args.old
    new = root / args.new

    rep = {}
    for tag, d in (("old", old), ("new", new)):
        if not d.exists():
            print(f"[warn] {tag} 目录不存在: {d}")
            continue
        r = summarize(d, root, args.slevel)
        rep[tag] = r
        print(f"=== {tag}: {d.name} （{r['n_answered']}/{r['n']} 有答案）")
        for k in ("detection_hit", "abstain_n", "abstain_rate", "excl_detection",
                  "denom_excl", "count_field_invalid_n", "low_conf_n"):
            print(f"    {k:24s} {r[k]}")
        print(f"    GT 源数分布              {r['gt_dist']}")
        print(f"    报数分布                 {r['count_dist']}")

    changed: list = []
    if "old" in rep and "new" in rep:
        o = {r[0]: r for r in rep["old"]["rows"]}
        n = {r[0]: r for r in rep["new"]["rows"]}
        changed = [(sid, o[sid][2], n[sid][2], o[sid][1], o[sid][3], n[sid][3])
                   for sid in sorted(o) if sid in n and o[sid][2] != n[sid][2]]
        print(f"\n=== 计数发生变化 {len(changed)} 个样本（sid, 旧报数, 新报数, GT, 旧拒答, 新拒答）")
        for c in changed:
            print("    %-10s %-5s -> %-5s  GT=%s  abst %s->%s" % c)

    out = Path(__file__).resolve().parents[3] / "evidence" / "_cmp_d2_abstain.json"
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in rep.items()}
    slim["changed"] = changed
    out.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
