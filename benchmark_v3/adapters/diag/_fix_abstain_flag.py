# -*- coding: utf-8 -*-
"""Repair the `abstained` flag written before the 2026-10-11 semantics fix.

Old rule: empty interferers list  -> abstained=True (wrong: an empty list is a legal
answer meaning "measured nothing").  New rule needs the tool evidence, which is not
stored in the answer file, so recompute offline with the reliable proxy:

    empty list AND the sample never entered the repair round  ->  legitimate empty
    empty list AND the last round_log phase == "repair"       ->  genuine refusal

(With the fixed validator an empty list is rejected whenever candidates were measured,
so a legitimate empty answer always ends without a repair round.)

usage: python _fix_abstain_flag.py <dataset> <SLEVEL> [answers_dir_name] [--dry-run]
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # benchmark_v3/（与 CWD 无关）

ds, lvl = sys.argv[1], sys.argv[2]
args = [a for a in sys.argv[3:] if not a.startswith("--")]
dry = "--dry-run" in sys.argv
aname = args[0] if args else ("answers_S0_agent_fix2" if lvl == "S0" else "answers_S2_agent_fix2")
A = ROOT / ds / aname

log = A / "_run_log.json"
rounds = {}
if log.exists():
    recs = json.loads(log.read_text(encoding="utf-8"))
    for r in recs if isinstance(recs, list) else recs.values():
        rl = r.get("round_log") or []
        rounds[r.get("sample_id")] = (rl[-1].get("phase") if rl and isinstance(rl[-1], dict) else None)

changed = []
for p in sorted(A.glob("*.json")):
    if p.name.startswith("_"):
        continue
    a = json.loads(p.read_text(encoding="utf-8"))
    if a.get("abstained") is not True:
        continue
    if a.get("interferers"):
        continue
    phase = rounds.get(p.stem)
    if phase == "repair":
        print(f"  keep   {p.stem}: empty list + repair round -> genuine refusal")
        continue
    a["abstained"] = False
    changed.append((p.stem, phase))
    if not dry:
        p.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"[{ds}/{aname}] fixed {len(changed)} flags" + (" (dry-run)" if dry else ""))
for sid, phase in changed:
    print(f"  fixed  {sid}: empty list, last phase={phase!r} -> abstained=false")
