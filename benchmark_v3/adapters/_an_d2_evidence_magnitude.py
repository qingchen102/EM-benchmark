"""分析 _diag_d2_evidence_all.json：证据差异到底是「数值噪声」还是「真变了」。

逐样本把 OLD/NEW 的门控峰按频率就近配对（5 kHz 内），报告：
  - 峰数差、只在单侧出现的峰
  - 配对峰的最大 |Δover_floor_db|
  - 时频指纹各键的相对差
再给全 50 样本的分布，用来判断「加噪改了多少 agent 可见证据」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
data = json.loads((HERE.parent / "_diag_d2_evidence_all.json").read_text(encoding="utf-8"))
TOL_KHZ = 5.0
BIG_DB = 1.0

rows = []
for sid, s in sorted(data.items()):
    o, n = s.get("OLD", {}), s.get("NEW", {})
    go, gn = o.get("gated") or [], n.get("gated") or []
    lo, ln = o.get("legacy") or [], n.get("legacy") or []
    used = set()
    max_d = 0.0
    n_pairs = 0
    for (f1, d1) in go:
        best, bj = 1e9, None
        for j, (f2, d2) in enumerate(gn):
            if j in used:
                continue
            if abs(f2 - f1) < best:
                best, bj = abs(f2 - f1), j
        if bj is not None and best <= TOL_KHZ:
            used.add(bj)
            max_d = max(max_d, abs(gn[bj][1] - d1))
            n_pairs += 1
    only_o = len(go) - n_pairs
    only_n = len(gn) - len(used)
    tf_o, tf_n = o.get("tf") or {}, n.get("tf") or {}
    tf_rel = {}
    for k in sorted(set(tf_o) | set(tf_n)):
        a, b = tf_o.get(k), tf_n.get(k)
        if a is None or b is None:
            tf_rel[k] = None
        elif abs(a) < 1e-9 and abs(b) < 1e-9:
            tf_rel[k] = 0.0
        else:
            tf_rel[k] = abs(b - a) / max(abs(a), abs(b), 1e-9)
    rows.append(dict(sid=sid, gt=n.get("gt_n"), go=len(go), gn=len(gn), pairs=n_pairs,
                     only_o=only_o, only_n=only_n, max_db=max_d,
                     lgo=len(lo), lgn=len(ln), tf=tf_rel))

same_num = sum(1 for r in rows if r["go"] == r["gn"] and r["only_o"] == 0 and r["only_n"] == 0 and r["max_db"] <= 0.05)
same_loose = sum(1 for r in rows if r["go"] == r["gn"] and r["only_o"] == 0 and r["only_n"] == 0 and r["max_db"] <= BIG_DB)
same_pos = sum(1 for r in rows if r["go"] == r["gn"])
tf_same_keys = sum(1 for r in rows if all((v or 0) <= 0.05 for v in r["tf"].values()))
tf_all = sum(1 for r in rows if all((v or 0) <= 0.05 for v in r["tf"].values()) and len(r["tf"]) == 4)

out = ["=" * 90,
       "D2 门控峰 OLD vs NEW：差异幅度分析（容差 5 kHz 就近配对）",
       "=" * 90,
       f"峰数相同                : {same_pos}/50",
       f"峰数相同且逐条 |Δover| ≤ 0.05 dB : {same_num}/50",
       f"峰数相同且逐条 |Δover| ≤ 1.0 dB  : {same_loose}/50",
       f"时频指纹四键相对差全 ≤ 5%        : {tf_all}/50",
       "",
       f"{'sid':10s} {'GT':>2s} {'门控 O→N':>10s} {'配对':>4s} {'只O':>3s} {'只N':>3s} {'最大ΔdB':>8s} "
       f"{'legacy O→N':>10s} {'duty':>7s} {'cv':>7s} {'burst':>7s} {'hop':>7s}"]
for r in rows:
    tf = r["tf"]
    def g(k):
        v = tf.get(k)
        return "  -  " if v is None else f"{v * 100:6.1f}%"
    out.append(f"{r['sid']:10s} {r['gt']:>2} {r['go']:>4}→{r['gn']:<5} {r['pairs']:>4} "
               f"{r['only_o']:>3} {r['only_n']:>3} {r['max_db']:>8.2f} "
               f"{r['lgo']:>4}→{r['lgn']:<5} {g('duty_cycle')} {g('frame_power_cv')} "
               f"{g('burst_count_per_window')} {g('hop_channel_count')}")

path = HERE.parent / "_diag_d2_evidence_magnitude.txt"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out[:12]))
print(f"\n写出 {path}")
