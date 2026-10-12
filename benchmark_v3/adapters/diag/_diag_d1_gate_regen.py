"""只读仪表检查：给定 D1 数据集根，量「虚警 / 覆盖 / 频率命中」三项（零 API）。

动机：数据换成配额 v2（四档各 100）后，D1 的仪表侧能力必须重新量一遍——
`matched_energy_fraction` 只说明参考对不对得上，不代表门控判据能报出真源。

口径（与 `capture_tools.estimate_band_occupancy_by_correlation` 一致，station_a）：
  · 干净样本（GT 干扰数 = 0）：门控峰 ≥1 即**虚警**；
  · 有干扰样本：门控峰 ≥1 记**覆盖**；存在峰的 |f_peak − f_GT| ≤ 40 kHz 记**频率命中**
    （40 kHz = 注册口径 `eval_dataset.FREQ_TOL`）。

用法：python -X utf8 diag\\_diag_d1_gate_regen.py <dataset_root> [--legacy]
      （--legacy 用 DSH_PEAK_LEGACY=1 跑旧判据做对照）
"""
from __future__ import annotations

import json as _json
import os as _os
import pathlib as _pl
import shutil
import sys
import tempfile

_HERE = _pl.Path(__file__).resolve().parent
_V3 = _HERE.parent
for _p in (str(_V3), str(_V3 / "simulation"), str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

FREQ_TOL_HZ = 40e3


def one(root: _pl.Path, sid: str) -> dict:
    """单样本仪表检查。**每个样本用自己的临时缓存目录**（cache_dir 里文件名只按站点分，
    复用同一目录会读到上一个样本的残差/波束文件）。"""
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = _json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    tr = _json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    refs = rca._build_target_reference(priv, n)
    tmp = _pl.Path(tempfile.mkdtemp(prefix=f"d1gate_{sid}_"))
    try:
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=float(obs["receivers"][0]["sample_rate_hz"]),
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        peaks = occ.get("residual_peaks") or []
        gts = list(tr.get("interferers") or [])
        hit = any(abs(p["freq_offset_hz"] - g["freq_offset_hz"]) <= FREQ_TOL_HZ
                  for p in peaks for g in gts) if (gts and peaks) else False
        return {"sid": sid, "n_gt": len(gts), "n_pk": len(peaks),
                "matched": occ.get("matched_energy_fraction"), "hit": hit,
                "gt_f": [round(g["freq_offset_hz"], 1) for g in gts],
                "pk_f": [round(p["freq_offset_hz"], 1) for p in peaks[:4]]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    if "--legacy" in argv:
        _os.environ["DSH_PEAK_LEGACY"] = "1"
    root = _pl.Path(argv[0]).resolve()
    idx = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    clean = {"n": 0, "fa": 0, "pk": 0}
    dirty = {"n": 0, "cov": 0, "hit": 0, "pk": 0}
    for s in idx["samples"]:
        r = one(root, s["id"])
        if r["n_gt"] == 0:
            clean["n"] += 1
            clean["pk"] += r["n_pk"]
            clean["fa"] += 1 if r["n_pk"] else 0
            if r["n_pk"]:
                print(f"  虚警 {r['sid']}  门控峰 {r['n_pk']}  pk_f={r['pk_f']}"
                      f"  matched={r['matched']}")
        else:
            dirty["n"] += 1
            dirty["pk"] += r["n_pk"]
            dirty["cov"] += 1 if r["n_pk"] else 0
            dirty["hit"] += 1 if r["hit"] else 0
            if not r["hit"]:
                print(f"  漏报/偏频 {r['sid']}  GT_f={r['gt_f']}  pk_f={r['pk_f']}"
                      f"  matched={r['matched']}")

    tag = "旧判据(DSH_PEAK_LEGACY=1)" if _os.environ.get("DSH_PEAK_LEGACY") else "现行判据"
    print(f"\n== {root.name}（station_a，{tag}）==")
    print(f"  干净样本 {clean['n']}：虚警样本 {clean['fa']}"
          f"（{100 * clean['fa'] / max(clean['n'], 1):.1f}%），峰合计 {clean['pk']}")
    print(f"  有干扰样本 {dirty['n']}：覆盖 {dirty['cov']}"
          f"（{100 * dirty['cov'] / max(dirty['n'], 1):.1f}%），"
          f"频率命中 {dirty['hit']}（{100 * dirty['hit'] / max(dirty['n'], 1):.1f}%），"
          f"峰合计 {dirty['pk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
