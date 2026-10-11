"""对照同一个 D2 样本在 10-07（加噪前+旧仪器/prompt）与 10-09（加噪后+新仪器/prompt）两次 Agent 运行中的
工具序列、耗时、最终自述，用来定位「数源行为为何改变」。只读。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]   # benchmark_v3
OLD = ROOT / "_pre_noise_20261010" / "dataset_d2" / "answers_S2_agent_full" / "_run_log.json"
NEW = ROOT / "dataset_d2" / "answers_S2_agent_layer1" / "_run_log.json"


def load(p: Path) -> dict:
    if not p.exists():
        return {}
    return {e["sample_id"]: e for e in json.loads(p.read_text(encoding="utf-8"))}


def seq(e: dict) -> str:
    names = [n for r in (e.get("round_log") or []) for n in (r.get("names") or [])]
    out, prev = [], None
    for n in names:
        out.append(n if n != prev else ".")
        prev = n
    return " > ".join(out)


def main() -> int:
    sids = sys.argv[1:] or ["d2_00000", "d2_00001", "d2_00009", "d2_00037"]
    old, new = load(OLD), load(NEW)
    for sid in sids:
        truth = json.loads((ROOT / "dataset_d2" / sid / "capture" / "private" / "d1_truth.json")
                           .read_text(encoding="utf-8"))
        kinds = [i["kind"] for i in truth["interferers"]]
        f0 = [round(i["freq_offset_hz"] / 1e3, 1) for i in truth["interferers"]]
        print("=" * 100)
        print(f"{sid}  GT={len(kinds)} {kinds}  f0(kHz)={f0}")
        for tag, d in (("OLD 10-07", old), ("NEW 10-09", new)):
            e = d.get(sid)
            if not e:
                print(f"  {tag}: 无记录")
                continue
            pred = e.get("prediction") or {}
            print(f"  {tag}: 报 {pred.get('num_interferers')}  工具 {e.get('tool_calls')} 次  "
                  f"耗时 {e.get('elapsed_s')}s  自述 {len(e.get('raw_output') or '')} 字")
            print(f"    序列: {seq(e)}")
            print(f"    答案: {json.dumps(pred, ensure_ascii=False)[:300]}")
            print(f"    自述: {(e.get('raw_output') or '').strip()[:600].replace(chr(10), ' | ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
