"""只读微探针（零 token）：跨档检查「运行时合成的目标参考」是否与**盘上 IQ** 对得上。

关键指标 `matched_energy_fraction`（目标被剥离的比例）：
  ≈0.99 → 参考与数据里的目标分量逐位一致，参考直减有效；
  ≈0.0  → 残差能量 ≥ 原始能量 = 参考与数据**不是同一个目标波形**（会把目标自身
          判成"跨阵元相干源"，干净样本也报满 6 个门控峰）。

D2/D3/D4 的 IQ 是 2026-10-09 夜用**当前波形代码**重放换装的；D1 的 IQ 是
2026-09-27 生成的（**波形带限修复之前**）。故这里并排测四档，验证
「D1 参考失配 = 波形带限修复只换了参考、没换数据」这一假设。

用法：python _probe_refmatch.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
import tempfile
import shutil
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

# 用法：python _probe_refmatch.py [数据集目录|档位名] [逗号分隔的样本 id]
PLAN = {
    "d1": ["d1_00001", "d1_00020", "d1_00000"],
    "d2": ["d2_00000", "d2_00001", "d2_00005"],
    "d3": ["d3_00000", "d3_00001"],
    "d4": ["d4_00000", "d4_00001"],
}
if len(sys.argv) > 1:
    _key = sys.argv[1]
    _sids = (sys.argv[2].split(",") if len(sys.argv) > 2
             else [p.name for p in sorted(Path(_key).glob("*/capture"))][:3])
    PLAN = {_key: _sids}
# 可选第 3 个参数：测哪个站（默认 station_a）
STATION = sys.argv[3] if len(sys.argv) > 3 else "station_a"


def probe(root: Path, sid: str) -> dict:
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    ifw = tr.get("interferers", []) or []
    refs = rca._build_target_reference(priv, n)
    tmp = Path(tempfile.mkdtemp(prefix="rm_"))
    try:
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        occ = ct.estimate_band_occupancy_by_correlation(STATION)
        peaks = occ.get("residual_peaks") or []
        return {"sid": sid, "gt_n": len(ifw), "refs": sorted(refs),
                "matched": occ.get("matched_energy_fraction"),
                "floor": occ.get("residual_noise_floor_db"),
                "n_gated": len(peaks), "n_legacy": len(occ.get("residual_peaks_legacy") or []),
                "peaks": [(round(p["freq_offset_hz"] / 1e3, 1), p["over_floor_db"],
                           p.get("coherence")) for p in peaks[:3]]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    print(f"{'sample':10s} {'GT':>3s} {'matched':>8s} {'floor':>7s} "
          f"{'gated':>5s} {'legacy':>6s}  前3峰(f kHz, over dB, coh)")
    rows = []
    for d, sids in PLAN.items():
        root = (Path(d) if ("/" in d or "\\" in d)
                else _V3 / f"dataset_{d}")
        for sid in sids:
            try:
                r = probe(root, sid)
            except Exception as exc:                            # noqa: BLE001
                print(f"{sid:10s}  ERROR {type(exc).__name__}: {str(exc)[:70]}")
                continue
            rows.append(r)
            print(f"{sid:10s} {r['gt_n']:3d} {str(r['matched']):>8s} "
                  f"{str(r['floor']):>7s} {r['n_gated']:5d} {r['n_legacy']:6d}  {r['peaks']}")
    (_V3 / "_probe_refmatch.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
