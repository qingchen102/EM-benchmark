"""只读核实（第四轮）：D4 与 D3 的真实差异 + 结果面口径核对（零 token）。

要查：
  A) D4 相对 D3 到底改了什么——逐字段对比 frontend 全键（含 correlated_noise /
     velocity / phase_noise），并把 truth 的 envelope/difficulty 文本打出来。
     （claim: D4 = 楼后NLoS + 恶意压制 + GNSS欺骗 + IMD3 + 阵元相关噪声 + 机动多普勒）
  B) D2 那批 HANDOVER 数字（det 0.860 等）是从哪来的：数 answers 目录样本数，
     看 eval_report_S2.json 覆盖了几个。
  C) D1 修后 20 样本 agent 跑出 det=1.0 但 S1/S2 pos=0.0（旧 50 样本跑是 0.61）——
     到底是真退化还是口径问题。逐样本比对 answers_S0_agent_fix20 与真值。

用法：python _audit_gaps.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
OUT = HERE / "_audit_gaps.txt"
_buf: list[str] = []


def w(s: str = "") -> None:
    _buf.append(s)


def main() -> None:
    # ---------- A. D3 vs D4 frontend 逐字段 ----------
    w("=" * 78)
    w("A. D3 vs D4 的真实差异（frontend 全字段 + envelope 文本）")
    w("=" * 78)
    for d in ("d3", "d4"):
        root = _V3 / f"dataset_{d}"
        sp = root / f"{d}_00000" / "capture" / "private" / "scenario.json"
        sc = json.loads(sp.read_text(encoding="utf-8"))
        rx = (sc.get("receivers") or [{}])[0]
        fe = rx.get("frontend") or {}
        w(f"\n[{d}] frontend 全字段:")
        for k in sorted(fe):
            w(f"    {k} = {fe[k]}")
        # 发射机 / 接收机运动
        ems = sc.get("emitters") or []
        w(f"  emitters 数 {len(ems)}；字段: {sorted(ems[0].keys()) if ems else 'N/A'}")
        for e in ems:
            if any(kk in e for kk in ("velocity_mps", "velocity", "speed_mps")):
                w(f"    emitter role={e.get('role')} velocity={e.get('velocity_mps') or e.get('velocity')}")
        if "velocity_mps" in rx:
            w(f"  receiver velocity_mps = {rx['velocity_mps']}")
        tr = json.loads((root / f"{d}_00000" / "capture" / "private" / "d1_truth.json")
                        .read_text(encoding="utf-8"))
        w(f"  truth.difficulty = {tr.get('difficulty')}")
        w(f"  truth.envelope   = {tr.get('envelope')}")
        w(f"  truth.noise      = {json.dumps(tr.get('noise'), ensure_ascii=False)[:300]}")
        ie = tr.get("interference_events")
        if ie:
            w(f"  interference_events ({len(ie)}): {json.dumps(ie[:3], ensure_ascii=False)[:400]}")

    # ---------- B. D2 结果面 ----------
    w("\n" + "=" * 78)
    w("B. D2 现有 agent 结果的真实覆盖（HANDOVER 的 det 0.860 从哪来）")
    w("=" * 78)
    root = _V3 / "dataset_d2"
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    all_ids = [s["id"] for s in idx["samples"]]
    w(f"  D2 数据集样本数: {len(all_ids)}")
    for ad in sorted([p for p in root.glob("answers*") if p.is_dir()]):
        files = list(ad.glob("*.json"))
        sids = {f.stem for f in files}
        w(f"    {ad.name}: {len(files)} 个答案文件")
    rep = json.loads((root / "eval_report_S2.json").read_text(encoding="utf-8"))
    w(f"  eval_report_S2.json 顶层键: {sorted(rep.keys())}")
    for sk in ("S0", "S1", "S2"):
        v = rep.get(sk) or {}
        ids = v.get("sample_ids") or v.get("ids") or []
        w(f"    [{sk}] n={v.get('n')} 覆盖样本: {ids if ids else '(报告未存 id 列表)'}")
    w(f"  → 结论：D2 有据可查的 agent 评测只有 {rep.get('S0',{}).get('n')} 个样本；"
      f"HANDOVER 记录的 0.860/0.412/... 若无对应 answers 目录即为**已丢失的历史数字**。")

    # ---------- C. D1 修后 20 样本 ----------
    w("\n" + "=" * 78)
    w("C. D1 修后 20 样本 agent：det=1.0 但 S1/S2 pos=0.0 的核对")
    w("=" * 78)
    d1 = _V3 / "dataset_d1"
    ad = d1 / "answers_S0_agent_fix20"
    files = sorted(ad.glob("*.json"))
    w(f"  answers 文件数: {len(files)}")
    det_ok = 0; pos_ok = 0
    rows = []
    for f in files:
        if f.name.startswith("_"):
            continue
        a = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(a, dict):
            continue
        tp = d1 / f.stem / "capture" / "private" / "d1_truth.json"
        t = json.loads(tp.read_text(encoding="utf-8"))
        gt_ifw = t.get("interferers") or []
        got = a.get("interferers") or a.get("detections") or []
        rows.append((f.stem, len(gt_ifw), len(got),
                     [g.get("kind") for g in got],
                     [g.get("kind") for g in gt_ifw]))
    w(f"\n  {'样本':<12} {'GT数':>4} {'报数':>4}  报的kind / GT kind")
    for sid, ng, no, kg, kt in rows:
        flag = "" if ng == no else "  <-- 计数不符"
        w(f"  {sid:<12} {ng:>4} {no:>4}  {kg} / {kt}{flag}")
    w(f"\n  计数相符: {sum(1 for r in rows if r[1]==r[2])}/{len(rows)}")
    for sid in ("d1_00000", "d1_00003"):
        p = ad / f"{sid}.json"
        if p.is_file():
            w(f"\n  [{sid}] 原文: {p.read_text(encoding='utf-8')}")
    rl = ad / "_run_log.json"
    if rl.is_file():
        r = json.loads(rl.read_text(encoding="utf-8"))
        w(f"\n  _run_log.json 类型 {type(r).__name__} 长度 {len(r)}")
        if isinstance(r, list) and r and isinstance(r[0], dict):
            w(f"    首条键: {sorted(r[0].keys())}")
            w(f"    首条内容: {json.dumps(r[0], ensure_ascii=False)[:500]}")

    text = "\n".join(_buf)
    OUT.write_text(text, encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
