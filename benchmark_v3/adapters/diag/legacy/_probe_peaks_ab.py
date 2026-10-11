"""临时探针（零 token）：加噪到底改变了工具链的哪一环？

已有证据：
  - 50 条 D2 的 `matched_frac`（iq vs 目标参考）在 old/new 上**几乎逐位相同**
    → 目标参考直减没被破坏，相位噪声不是通过这条路径发作的。
  - b74 的 3 样本审计（只看**新判据** `residual_peaks`）显示抬底 ±19 dB 峰数不变。

但**聚类定带**用的是 legacy 宽带判据（全局中位底 + ≥5 bin 连段），它对噪声底敏感。
本探针逐样本对比 old/new 的**两个判据**在 station_a 上选出的峰：
  gated  = `residual_peaks`（局部中位底 + 相干门控）
  legacy = `residual_peaks_legacy`（全局中位底 + 连段，聚类定带用的那个）
"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD))

import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402

ROOTS = {
    "old": _AD.parent / "_pre_noise_20261010" / "dataset_d2",
    "new": _AD.parent / "dataset_d2",
}
CACHE = _AD / "_probe_peaks_cache"


def count(root: Path, sid: str, st: str = "station_a") -> dict | None:
    cap = root / sid / "capture"
    priv, pub = cap / "private", cap / "public"
    iq_p = pub / f"iq_{st}.npy"
    if not iq_p.exists():
        return None
    obs = json.loads((pub / "observations.json").read_text(encoding="utf-8"))
    ns = obs["receivers"][0]["shape"][1]
    refs = rca._build_target_reference(priv, ns)
    if not refs:
        return None
    key = st if st in refs else sorted(refs)[0]
    cdir = CACHE / root.name / sid
    if cdir.exists():
        for f in cdir.glob("*.npy"):
            f.unlink()
    cdir.mkdir(parents=True, exist_ok=True)
    ctx = ct.ToolContext(
        capture_dir=pub, cache_dir=cdir, allowed_stations=[st],
        fs=obs["receivers"][0]["sample_rate_hz"],
        fc=obs["receivers"][0]["center_frequency_hz"],
        target_bw_normalized=0.25, aoi_center=[0, 0],
        target_reference={key: refs[key]})
    ct.set_context(ctx)
    out = ct.estimate_band_occupancy_by_correlation(st)
    g = out.get("residual_peaks") or []
    lg = out.get("residual_peaks_legacy") or []
    return {
        "n_gated": len(g), "n_legacy": len(lg),
        "g_over": [round(float(p["over_floor_db"]), 1) for p in g],
        "l_over": [round(float(p["over_floor_db"]), 1) for p in lg],
        "l_bw": [round(float(p["bandwidth_hz"]) / 1e3, 1) for p in lg],
    }


def main() -> None:
    sids = sorted(p.name for p in ROOTS["new"].iterdir() if p.name.startswith("d2_"))
    print(f"n={len(sids)} 样本 station_a\n")
    print(f"{'sid':12s} {'old_g':>5s} {'new_g':>5s} {'old_leg':>7s} {'new_leg':>7s}   legacy over_floor (old → new)")
    agg = {"og": [], "ng": [], "ol": [], "nl": []}
    for sid in sids:
        a = count(ROOTS["old"], sid)
        b = count(ROOTS["new"], sid)
        if a is None or b is None:
            print(f"{sid:12s}  (跳过)")
            continue
        agg["og"].append(a["n_gated"]); agg["ng"].append(b["n_gated"])
        agg["ol"].append(a["n_legacy"]); agg["nl"].append(b["n_legacy"])
        mark = "  <== legacy 变了" if a["n_legacy"] != b["n_legacy"] else ""
        print(f"{sid:12s} {a['n_gated']:5d} {b['n_gated']:5d} {a['n_legacy']:7d} {b['n_legacy']:7d}   "
              f"{a['l_over']} → {b['l_over']}{mark}")
    for k, lbl in (("og", "gated old"), ("ng", "gated new"), ("ol", "legacy old"), ("nl", "legacy new")):
        v = np.array(agg[k], dtype=float)
        print(f"{lbl:12s} 均值={v.mean():6.2f} 中位={np.median(v):6.2f} 总和={int(v.sum()):4d}")
    og, ng, ol, nl = (np.array(agg[k], float) for k in ("og", "ng", "ol", "nl"))
    print(f"\ngated  每样本差 (new-old)：均值={np.mean(ng - og):+.2f}  完全相同的样本={int(np.sum(ng == og))}/{len(og)}")
    print(f"legacy 每样本差 (new-old)：均值={np.mean(nl - ol):+.2f}  完全相同的样本={int(np.sum(nl == ol))}/{len(ol)}")


if __name__ == "__main__":
    main()
