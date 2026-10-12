"""只读：D1 门控带清单 vs 真值——"多余峰"到底长什么样（位置/宽度/与 GT 的关系）。

动机（2026-10-12 新档体检）：新档 D1 每个有干扰样本的多余峰约为旧档两倍
（28/51 vs 8/27）。D1 的 detection 是**严格计数**（多报即错），所以这些多余峰
比"漏报"更可能压分数。本脚本只统计形态，不改任何东西：

  · 每个 band 与最近 GT 的频距（>40 kHz 记多余）；
  · 多余带的中心偏移分布（分桶）+ 带宽；
  · 多余带是否出现在**通带边缘**（±845 kHz 附近，D1 的 brick 边界）；
  · 真值为 tone（窄带） vs noise（宽带）时多余峰的差别。

用法：python -X utf8 diag\\_probe_d1_extra_bands.py <dataset_root>
"""
from __future__ import annotations

import json as _json
import pathlib as _pl
import shutil
import sys
import tempfile

import numpy as np

_HERE = _pl.Path(__file__).resolve().parent
_V3 = _HERE.parent
for _p in (str(_V3), str(_V3 / "simulation"), str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

TOL_HZ = 40e3
EDGE_BAND_HZ = 845e3


def load(root: _pl.Path, sid: str):
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = _json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    tr = _json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    tmp = _pl.Path(tempfile.mkdtemp(prefix=f"xb_{sid}_"))
    try:
        refs = rca._build_target_reference(priv, n)
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=float(obs["receivers"][0]["sample_rate_hz"]),
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        ct.estimate_band_occupancy_by_correlation("station_a")
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        resid = np.asarray(ct._strip_target_at_station("station_a"), np.complex128)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return resid, float(occ["band_hz"][1]), list(tr.get("interferers") or [])


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    root = _pl.Path(argv[0]).resolve()
    idx = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    n_dirty = n_extra = n_near = 0
    bucket = {"0-300k": 0, "300-700k": 0, ">700k": 0}
    edge = 0
    by_kind = {"tone": [0, 0], "noise": [0, 0]}       # [多余, 总数]
    bw_extra: list[float] = []
    for s in idx["samples"]:
        resid, rx_half, gts = load(root, s["id"])
        if not gts:
            continue
        bands = ct._residual_bands(resid, 4e6, rx_half)
        kmap = [g.get("bandwidth_hz") or 0.0 for g in gts]
        ktyp = "noise" if (kmap and max(kmap) > 5e3) else "tone"
        by_kind[ktyp][1] += len(gts)
        pairs = [(b, min(gts, key=lambda g: abs(g["freq_offset_hz"] - b["freq_offset_hz"])))
                 for b in bands]
        near = [b for b, g in pairs
                if abs(b["freq_offset_hz"] - g["freq_offset_hz"]) <= TOL_HZ]
        extra = [b for b, g in pairs
                 if abs(b["freq_offset_hz"] - g["freq_offset_hz"]) > TOL_HZ]
        n_dirty += 1
        n_near += len(near)
        n_extra += len(extra)
        by_kind[ktyp][0] += len(extra)
        for b in extra:
            f = abs(b["freq_offset_hz"])
            bucket["0-300k" if f < 300e3 else ("300-700k" if f < 700e3 else ">700k")] += 1
            if f > EDGE_BAND_HZ - 60e3:
                edge += 1
            bw_extra.append(float(b.get("bandwidth_hz") or 0.0))
    print(f"== {root.parent.name}/{root.name}  D1 多余峰形态（station_a，门控判据 "
          f"窗={ct.PEAK_COH_WIN_HZ/1e3:.1f}kHz）==")
    print(f"有干扰样本 {n_dirty}：近真源带 {n_near}，多余带 {n_extra}"
          f"（{n_extra/max(n_dirty,1):.2f}/样本）")
    print(f"多余带中心 |f| 分布：{bucket}；落在通带边缘(|f|>{EDGE_BAND_HZ-60e3:.0f}Hz) {edge} 个")
    if bw_extra:
        print(f"多余带带宽 中位 {np.median(bw_extra)/1e3:.1f} kHz / 最大 {max(bw_extra)/1e3:.1f} kHz")
    print("按真值类型（tone=窄带 / noise=宽带，>5kHz 记 noise）：")
    for k, (e, t) in by_kind.items():
        print(f"  {k:5s} GT {t:3d} 源 → 多余带 {e:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
