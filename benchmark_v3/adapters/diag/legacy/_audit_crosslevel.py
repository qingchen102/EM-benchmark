"""只读核实（第三轮）：跨难度同源 + D4 新增难度是否真的落在数据里（零 token）。

要查三件事：
  A) D1–D4 是否共享同一批目标位置/种子（gen_dataset: seed = seed0 + i，seed0 默认
     全是 31000 → 同 index 的四档共享同一 seed）。
  B) D4 声称的"三件新增困难"（IMD3 非线性 / 阵元相关噪声 ρ=0.3 / 机动多普勒）
     **是否真的写进了数据**（查 scenario.json / truth.json 的字段）。
  C) 结果面：现有 answers/eval_report 覆盖了哪些 (难度 × 站数) 格。

用法：python _audit_crosslevel.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
OUT = HERE / "_audit_crosslevel.txt"
_buf: list[str] = []


def w(s: str = "") -> None:
    _buf.append(s)


def load(root: Path, fname="d1_truth.json"):
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    out = {}
    for s in idx["samples"]:
        p = root / s["id"] / "capture" / "private" / fname
        if p.is_file():
            out[s["id"]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def main() -> None:
    L = {d: load(_V3 / f"dataset_{d}") for d in ("d1", "d2", "d3", "d4")}

    # ---------- A. 跨难度目标位置同源 ----------
    w("=" * 78)
    w("A. 跨难度目标位置是否同一批（gen_dataset: seed = seed0 + i，seed0 默认 31000）")
    w("=" * 78)
    keys = sorted(L["d1"])
    n = len(keys)
    for a, b in (("d1", "d2"), ("d2", "d3"), ("d3", "d4"), ("d1", "d4")):
        ka, kb = sorted(L[a]), sorted(L[b])
        m = min(len(ka), len(kb))
        same = tot = 0
        for i in range(m):
            pa = (L[a][ka[i]].get("target") or {}).get("position_m") or []
            pb = (L[b][kb[i]].get("target") or {}).get("position_m") or []
            if len(pa) >= 2 and len(pb) >= 2:
                tot += 1
                if np.allclose(pa[:2], pb[:2], atol=0.5):
                    same += 1
        w(f"  {a} vs {b}: 目标位置相同 {same}/{tot}")
    seeds = {d: [L[d][k].get("seed") for k in sorted(L[d])] for d in L}
    w(f"\n  各档 seed 序列前 5: " + " | ".join(f"{d}={seeds[d][:5]}" for d in ("d1", "d2", "d3", "d4")))
    w(f"  seed 序列是否逐位相同(d1 vs d2): {seeds['d1'] == seeds['d2']}")
    w(f"  seed 序列是否逐位相同(d3 vs d4): {seeds['d3'] == seeds['d4']}")

    # ---------- B. D4 的新增困难是否在数据里 ----------
    w("\n" + "=" * 78)
    w("B. D4 声称新增的困难（IMD3 / 阵元相关噪声 / 机动多普勒）是否落地")
    w("=" * 78)
    for d in ("d3", "d4"):
        root = _V3 / f"dataset_{d}"
        k0 = sorted(L[d])[0]
        sp = root / k0 / "capture" / "private" / "scenario.json"
        w(f"\n[{d} {k0}]")
        if sp.is_file():
            sc = json.loads(sp.read_text(encoding="utf-8"))
            fe = None
            for r in (sc.get("receivers") or []):
                fe = r.get("frontend") or fe
                break
            w(f"  frontend 键: {sorted(fe.keys()) if fe else 'N/A'}")
            if fe:
                for kk in ("imd3_enabled", "iip3_dbm", "element_noise_rho",
                           "phase_noise", "iq_imbalance"):
                    if kk in fe:
                        w(f"    {kk} = {fe[kk]}")
            # 接收机带宽 / 阵元
            r0 = (sc.get("receivers") or [{}])[0]
            w(f"  receiver 键: {sorted(r0.keys())}")
            # 目标速度/多普勒
            for e in (sc.get("emitters") or []):
                if e.get("role") == "target":
                    w(f"  target emitter 键: {sorted(e.keys())}")
                    for kk in ("velocity_mps", "speed_mps", "doppler_hz", "motion"):
                        if kk in e:
                            w(f"    {kk} = {e[kk]}")
        tr = L[d][k0]
        w(f"  d1_truth 顶层键: {sorted(tr.keys())}")
        tg = tr.get("target") or {}
        w(f"  target 键: {sorted(tg.keys())}")
        for kk in ("velocity_mps", "doppler_hz", "speed_mps"):
            if kk in tg:
                w(f"    target.{kk} = {tg[kk]}")
        ifw = tr.get("interferers") or []
        if ifw:
            w(f"  interferer[0] 键: {sorted(ifw[0].keys())}")
        # 直接扫关键字，别只查我猜的键名
        raw = json.dumps(tr, ensure_ascii=False)
        sc_raw = ""
        if sp.is_file():
            sc_raw = sp.read_text(encoding="utf-8")
        for kw in ("imd3", "iip3", "rho", "doppler", "velocity", "speed", "correlat"):
            hit_t = kw in raw.lower()
            hit_s = kw in sc_raw.lower()
            w(f"    关键字 '{kw}': d1_truth={hit_t}  scenario.json={hit_s}")

    # ---------- C. 结果面覆盖 ----------
    w("\n" + "=" * 78)
    w("C. 现有结果覆盖的 (难度 × 站数) 格")
    w("=" * 78)
    for d in ("d1", "d2", "d3", "d4"):
        root = _V3 / f"dataset_{d}"
        reps = sorted(root.glob("eval_report*.json"))
        ansdirs = sorted([p.name for p in root.glob("answers*") if p.is_dir()])
        w(f"\n[{d}]")
        w(f"  eval_report: {[p.name for p in reps] or '无'}")
        w(f"  answers 目录: {ansdirs or '无'}")
        for p in reps:
            r = json.loads(p.read_text(encoding="utf-8"))
            for sk in ("S0", "S1", "S2"):
                if sk in r and isinstance(r[sk], dict):
                    v = r[sk]
                    w(f"    {p.name} [{sk}] n={v.get('n')} det={v.get('detection_hit')}"
                      f" kind={v.get('kind_hits')} freq={v.get('freq_hits')}"
                      + (f" pos={v.get('pos_hits')}" if "pos_hits" in v else "")
                      + (f" doa={v.get('doa_hits')}" if "doa_hits" in v else "")
                      + (f" intent={v.get('intent_accuracy')}" if "intent_accuracy" in v else ""))

    text = "\n".join(_buf)
    OUT.write_text(text, encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
