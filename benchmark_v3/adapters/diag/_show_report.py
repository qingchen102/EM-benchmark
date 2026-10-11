"""打印某档（或某个）eval_report*.json 的关键指标（零 API，只读）。

用法：
    python -X utf8 diag/_show_report.py d2                 # 列出 dataset_d2 下所有报告
    python -X utf8 diag/_show_report.py d2 eval_report_S2.json
    python -X utf8 diag/_show_report.py --file ../dataset_d2/_eval_report_S2_nothink_20261012.json

输出：文件名 + mtime + 每个 S 级的 detection/within1/kind/freq/空间/intent/机理 + 分母。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_V3 = HERE.parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KEYS = ("detection_hit", "detection_within1", "kind_hits", "freq_hits", "freq_aes",
        "doa_hits", "pos_hits", "pos_errs", "intent_accuracy", "mech_accuracy",
        "spoof_fa_rate", "n")


def show(path: Path) -> None:
    j = json.loads(path.read_text(encoding="utf-8"))
    mt = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")
    print(f"\n=== {path}   ({mt})")
    for s in ("S0", "S1", "S2"):
        m = j.get(s)
        if not isinstance(m, dict):
            continue
        parts = []
        for k in KEYS:
            v = m.get(k)
            if v is None:
                continue
            if k == "pos_errs":
                parts.append(f"pos_err={v:.0f}m")
            elif k == "freq_aes":
                parts.append(f"freq_ae={v * 2e6 / 1e3:.1f}kHz")
            elif k == "spoof_fa_rate":
                parts.append(f"spoof_fa={v:.3f}")
            elif isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            else:
                parts.append(f"{k}={v}")
        print(f"  [{s}] " + "  ".join(parts))


def main(argv: list[str]) -> int:
    if "--file" in argv:
        i = argv.index("--file")
        for f in argv[i + 1:]:
            p = Path(f)
            if not p.is_absolute():
                p = (HERE.parent / f).resolve()
            show(p)
        return 0
    deck = argv[0] if argv and not argv[0].startswith("-") else "d2"
    root = _V3 / f"dataset_{deck}"
    want = argv[1] if len(argv) > 1 else None
    files = sorted(root.glob("eval_report*.json")) + sorted(root.glob("_eval_report*.json"))
    if want:
        files = [p for p in files if p.name == want]
    for p in files:
        show(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
