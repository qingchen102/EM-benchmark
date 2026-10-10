"""只读：对比 D1 两次跑（layer1 vs noneg）的拒答与报数分布。

用法：python -u _an_d1_abstain.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def scan(d: str) -> None:
    p = Path(d)
    if not p.exists():
        print(f"{d}: MISSING")
        return
    files = sorted(p.glob("d1_*.json"))
    abst = 0
    dist: dict[object, int] = {}
    for f in files:
        o = json.loads(f.read_text(encoding="utf-8"))
        if o.get("abstained") is True:
            abst += 1
        dist[o.get("num_interferers")] = dist.get(o.get("num_interferers"), 0) + 1
    print(f"{d}: answers={len(files)}  abstained_flag={abst}  dist={dict(sorted(dist.items(), key=lambda kv: str(kv[0])))}")
    rl = p / "_run_log.json"
    if rl.exists():
        recs = json.loads(rl.read_text(encoding="utf-8"))
        if isinstance(recs, dict):
            recs = recs.get("records") or recs.get("runs") or []
        n_raw = 0
        n_neg = 0
        for r in recs:
            raw = (r or {}).get("raw_output") or ""
            n_raw += 1
            if '"num_interferers": -' in raw or '"num_interferers":-' in raw:
                n_neg += 1
        print(f"    run_log records={n_raw}  raw_output 里出现负值={n_neg}")


for d in (
    "../dataset_d1/answers_S0_agent_layer1",
    "../dataset_d1/answers_S0_agent_noneg",
):
    scan(d)
