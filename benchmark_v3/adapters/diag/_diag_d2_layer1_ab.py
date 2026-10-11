"""D2 同一批 50 个样本上：加噪前 agent 答案 vs 加噪后 agent 答案 vs GT 可见源数。

只读。用途：判断 D2 detection 0.86 -> 0.48 是「真源被噪声吃掉」还是「agent 计数行为变了」。
口径与 eval_dataset.score_answer 完全一致：detection_hit = num_interferers == len(gt_at_slevel(truth, S2 站集))。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # adapters（迁移后 HERE 下沉一级）
import eval_dataset as ed  # noqa: E402

ROOT = (HERE.parent / "dataset_d2").resolve()
OLD_ANS = (HERE.parent / "_pre_noise_20261010" / "dataset_d2" / "answers_S2_agent_full").resolve()
NEW_ANS = (HERE.parent / "dataset_d2" / "answers_S2_agent_layer1").resolve()

idx = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
stations_s2 = idx["observation_subsets"]["S2"]

rows = []
for s in idx["samples"]:
    sid = s["id"]
    cap = ROOT / sid / "capture"
    truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    gt = ed.gt_at_slevel(truth, stations_s2)
    n_gt = len(gt)
    n_all = len(truth["interferers"])

    def _n(p: Path) -> int | None:
        if not p.exists():
            return None
        a = json.loads(p.read_text(encoding="utf-8"))
        return int(a.get("num_interferers", 0))

    n_old = _n(OLD_ANS / f"{sid}.json")
    n_new = _n(NEW_ANS / f"{sid}.json")
    rows.append((sid, n_all, n_gt, n_old, n_new))

print(f"样本数 {len(rows)}   旧答案 {OLD_ANS.name}   新答案 {NEW_ANS.name}")
print(f"GT 可见源数(S2)分布: {sorted(set(r[2] for r in rows))}")


def _hit(r, i) -> int | None:
    return None if r[i] is None else int(r[i] == r[2])


old_hits = [h for h in (_hit(r, 3) for r in rows) if h is not None]
new_hits = [h for h in (_hit(r, 4) for r in rows) if h is not None]
print(f"旧 detection_hit = {sum(old_hits)}/{len(old_hits)} = {sum(old_hits)/len(old_hits):.3f}")
print(f"新 detection_hit = {sum(new_hits)}/{len(new_hits)} = {sum(new_hits)/len(new_hits):.3f}")

new_err = [r[4] - r[2] for r in rows if r[4] is not None]
old_err = [r[3] - r[2] for r in rows if r[3] is not None]
print(f"新 计数偏差 mean={sum(new_err)/len(new_err):+.2f}  (欠报 {sum(1 for e in new_err if e<0)} / 准 {sum(1 for e in new_err if e==0)} / 过报 {sum(1 for e in new_err if e>0)})")
print(f"旧 计数偏差 mean={sum(old_err)/len(old_err):+.2f}  (欠报 {sum(1 for e in old_err if e<0)} / 准 {sum(1 for e in old_err if e==0)} / 过报 {sum(1 for e in old_err if e>0)})")

zero_gt = sum(1 for r in rows if r[2] > 0 and r[4] == 0)
print(f"新答案：GT 有源却报 0 源的样本数 = {zero_gt}")
print()
print("逐样本（n_all=全部干扰, n_gt=S2 可见, 旧/新=agent 报的源数）")
print("id          n_all n_gt  old new   flip")
for sid, n_all, n_gt, n_old, n_new in rows:
    if _hit((sid, n_all, n_gt, n_old, n_new), 3) != _hit((sid, n_all, n_gt, n_old, n_new), 4):
        flip = "** 变差" if _hit((sid, n_all, n_gt, n_old, n_new), 3) == 1 else "   改善"
    else:
        flip = ""
    print(f"{sid}  {n_all:5d} {n_gt:4d}  {n_old if n_old is not None else -1:3d} {n_new if n_new is not None else -1:3d}   {flip}")
