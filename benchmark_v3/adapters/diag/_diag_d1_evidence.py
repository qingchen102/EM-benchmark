"""只读诊断（零 token）：D1/S0 的工具证据 vs GT vs 三次 agent 跑（layer1 / noneg / revert）。

目的：判断 D1 从 0.96 掉到 ~0.5 能不能归因于「泄漏修复 52a2f42 移除了 residual_peaks_legacy」。
做法：对全部 50 个 D1 样本，用**当前代码**重建 ctx，调用
`estimate_band_occupancy_by_correlation("station_a")`，同时取出
  residual_peaks        —— 门控后、agent 可见（修复前后逐位相同）
  residual_peaks_legacy —— 旧宽带口径，**修复后不再回灌给 agent**
再把两条清单的峰数、GT 干扰数、以及三次 agent 跑的报数/拒答并排对照。

用法：python _diag_d1_evidence.py [n_samples]
输出：_diag_d1_evidence.txt / _diag_d1_evidence.json（仓库根 benchmark_v3/ 下）
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
import tempfile
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

D1 = (Path(sys.argv[2]) if len(sys.argv) > 2 else _V3 / "dataset_d1")
RUNS = {
    "layer1(旧工具+原prompt)": D1 / "answers_S0_agent_layer1",
    "noneg(现工具+加固prompt)": D1 / "answers_S0_agent_noneg",
    "revert(现工具+原prompt)": D1 / "answers_S0_agent_revert",
}
TOL_HZ = 120e3


def read_run(d: Path, sid: str) -> dict:
    p = d / f"{sid}.json"
    if not p.exists():
        return {"missing": True}
    try:
        o = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return {"missing": True}
    n = o.get("num_interferers")
    return {"n": n, "abstained": bool(o.get("abstained")),
            "freqs": [i.get("freq_offset_hz") for i in (o.get("interferers") or [])]}


def main(n_samples: int) -> None:
    idx = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    rows = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", []) or []
        gt_f = [g.get("freq_offset_hz") for g in ifw]
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                        # noqa: BLE001
            refs = {}
        tmp = Path(tempfile.mkdtemp(prefix="d1ev_"))
        try:
            ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                                 allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                                 fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                                 target_bw_normalized=0.25, aoi_center=[0, 0],
                                 target_reference=refs)
            ct.set_context(ctx)
            occ = ct.estimate_band_occupancy_by_correlation("station_a")
            gated = occ.get("residual_peaks") or []
            legacy = occ.get("residual_peaks_legacy") or []
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        g_f = [p["freq_offset_hz"] for p in gated]
        l_f = [p["freq_offset_hz"] for p in legacy]
        # 峰清单是否覆盖每个 GT 源（±120 kHz）
        gt_cov_g = sum(1 for g in gt_f if any(abs(g - f) <= TOL_HZ for f in g_f))
        gt_cov_l = sum(1 for g in gt_f if any(abs(g - f) <= TOL_HZ for f in l_f))
        r = {"sid": sid, "gt_n": len(gt_f), "gt_f": gt_f,
             "gated_n": len(gated), "legacy_n": len(legacy),
             "gt_cov_gated": gt_cov_g, "gt_cov_legacy": gt_cov_l,
             "gated_f": [round(f / 1e3, 1) for f in g_f],
             "legacy_f": [round(f / 1e3, 1) for f in l_f],
             "runs": {k: read_run(d, sid) for k, d in RUNS.items()}}
        rows.append(r)
        print(f"{sid}  GT={len(gt_f)}  门控峰={len(gated)}  旧宽带峰={len(legacy)}  "
              f"覆盖GT {gt_cov_g}/{len(gt_f)} vs {gt_cov_l}/{len(gt_f)}  "
              + "  ".join(f"{k.split('(')[0]}={v.get('n')}{'*' if v.get('abstained') else ''}"
                          for k, v in r["runs"].items()))

    # ---------------- 汇总 ----------------
    print("\n========== 汇总 ==========")
    print(f"n = {len(rows)} 样本（S0 = 仅 station_a）")
    eq_g = sum(1 for r in rows if r["gated_n"] == r["gt_n"])
    eq_l = sum(1 for r in rows if r["legacy_n"] == r["gt_n"])
    same = sum(1 for r in rows if r["gated_n"] == r["legacy_n"])
    print(f"峰数 == GT 数：门控 {eq_g}/{len(rows)}   旧宽带 {eq_l}/{len(rows)}")
    print(f"两条清单峰数相同的样本：{same}/{len(rows)}")
    tot_gt = sum(r["gt_n"] for r in rows)
    print(f"GT 源覆盖：门控 {sum(r['gt_cov_gated'] for r in rows)}/{tot_gt}   "
          f"旧宽带 {sum(r['gt_cov_legacy'] for r in rows)}/{tot_gt}")
    print(f"GT 数分布 {dict(Counter(r['gt_n'] for r in rows))}")
    print(f"门控峰数分布 {dict(sorted(Counter(r['gated_n'] for r in rows).items()))}")
    print(f"旧宽带峰数分布 {dict(sorted(Counter(r['legacy_n'] for r in rows).items()))}")
    for k in RUNS:
        hit = abst = 0
        match_g = match_l = 0
        for r in rows:
            v = r["runs"][k]
            if v.get("missing"):
                continue
            if v.get("abstained"):
                abst += 1
            if v.get("n") == r["gt_n"]:
                hit += 1
            if v.get("n") == r["gated_n"]:
                match_g += 1
            if v.get("n") == r["legacy_n"]:
                match_l += 1
        print(f"{k}: 报数==GT {hit}/{len(rows)}  拒答 {abst}  "
              f"报数==门控峰数 {match_g}  报数==旧宽带峰数 {match_l}")
    # 拒答样本的证据长什么样
    for k in RUNS:
        bad = [r for r in rows
               if not r["runs"][k].get("missing") and r["runs"][k].get("abstained")]
        if bad:
            print(f"\n[{k}] 拒答 {len(bad)} 例，其证据：")
            for r in bad:
                print(f"   {r['sid']}  GT={r['gt_n']} 门控峰={r['gated_n']} "
                      f"旧宽带峰={r['legacy_n']} 覆盖GT={r['gt_cov_gated']}/{r['gt_n']}")

    (_V3 / "_diag_d1_evidence.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
