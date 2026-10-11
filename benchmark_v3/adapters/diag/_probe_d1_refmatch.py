"""只读微探针（零 token）：D1 的「目标参考是否还对得上盘上数据」。

动机：D1 的盘上 IQ 是 2026-09-27 生成的（**波形带限修复之前**），而
`run_capture_agent._build_target_reference` 是**运行时用当前代码**合成参考的。
若 2026-10-09 夜的波形带限修复改变了目标波形，参考就与数据里的目标分量不再逐位
一致 → 参考直减留下目标残影 → 门控判据会把**目标自己**当成「跨阵元相干」的源，
于是干净样本也报 4~6 个峰。

本脚本对若干样本打印：
  matched_energy_fraction  —— 目标被剥离的比例（修复期实测干净样本 ≈0.998）
  residual_noise_floor_db
  前几个 residual_peaks 的 (freq, over_floor_db, coherence, bandwidth)

若干净样本的 matched 明显 < 0.99 且峰很密 → 参考失配成立。

用法：python _probe_d1_refmatch.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
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

D1 = _V3 / "dataset_d1"
SIDS = ["d1_00001", "d1_00020", "d1_00045",     # GT=0（干净）
        "d1_00000", "d1_00003", "d1_00043"]     # GT=1


def main() -> None:
    for sid in SIDS:
        cap = D1 / sid / "capture" / "public"
        priv = D1 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", []) or []
        refs = rca._build_target_reference(priv, n)
        tmp = Path(tempfile.mkdtemp(prefix="d1rm_"))
        try:
            ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                                 allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                                 fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                                 target_bw_normalized=0.25, aoi_center=[0, 0],
                                 target_reference=refs)
            ct.set_context(ctx)
            occ = ct.estimate_band_occupancy_by_correlation("station_a")
            peaks = occ.get("residual_peaks") or []
            print(f"\n=== {sid}  GT={len(ifw)}  refs_stations={sorted(refs)} ===")
            print(f"    matched_energy_fraction = {occ.get('matched_energy_fraction')}"
                  f"   floor = {occ.get('residual_noise_floor_db')} dB"
                  f"   门控峰 {len(peaks)} / 旧宽带峰 {len(occ.get('residual_peaks_legacy') or [])}")
            for p in peaks[:4]:
                print(f"    PK {p['freq_offset_hz']/1e3:9.1f}k  over={p['over_floor_db']:5.1f} dB"
                      f"  coh={p.get('coherence')}  bw={p['bandwidth_hz']/1e3:7.1f}k")
            for g in ifw:
                print(f"    GT {g.get('kind'):>10s} f={g.get('freq_offset_hz')/1e3:9.1f}k"
                      f" bw={g.get('bandwidth_hz', 0)/1e3:7.1f}k")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
