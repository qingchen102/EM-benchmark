"""一次性：把 D1/D2 现有评测报告与基线汇总成一张紧凑表（供现状报告引用）。只读。"""
import json
from pathlib import Path

BV3 = Path("..")
FILES = [
    ("D1", "eval_report.json"),
    ("D1", "eval_report_S0.json"),
    ("D1", "eval_report_S1.json"),
    ("D1", "eval_report_S2.json"),
    ("D2", "eval_report_S2.json"),
    ("D2", "eval_report_S2_agent_full.json"),
]
KEYS = ["n_samples", "n_answers", "detection_hit", "detection_within1", "kind_hits",
        "pos_hits", "freq_hits", "freq_ae_khz", "intent_accuracy", "intent_pairs"]

for ds, fn in FILES:
    p = BV3 / f"dataset_{ds.lower()}" / fn
    if not p.exists():
        print(f"[missing] {p}")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    print("=" * 90)
    print(f"{p.as_posix()}   ({p.stat().st_size} B)")
    top = {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
    print("  top:", json.dumps(top, ensure_ascii=False))
    for sub in ("S0", "S1", "S2", "overall", "by_slevel"):
        if sub in d and isinstance(d[sub], dict):
            row = {k: d[sub].get(k) for k in KEYS if k in d[sub]}
            print(f"  [{sub}]", json.dumps(row, ensure_ascii=False))
    # 有些报告是 {slevel: {...}} 直接平铺
    for k in ("S0", "S1", "S2"):
        if k in d and isinstance(d[k], dict):
            continue
    print("  keys:", list(d.keys()))

for ds in ("d2",):
    p = BV3 / "dataset_d2" / "BASELINE.json"
    if p.exists():
        print("=" * 90)
        print(p.as_posix())
        print(p.read_text(encoding="utf-8"))
