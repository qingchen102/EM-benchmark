"""只读扫描：把门控相干窗 PEAK_COH_WIN_HZ 从 20 kHz 收窄，看 D1 判据能否既保住真源又不虚警。

动机（2026-10-12 新代数据体检）：真源门控覆盖率从旧代 92.6% 掉到 80.4%，逐样本取证
（`_probe_d1_peakmiss.py`）显示真源峰 **高出局部门限 60 dB**，但 ±20 kHz 窗内的空间相干
只有 0.39（门限 0.6）→ 被判据滤掉。原因：相干度量把 ±20 kHz 窗内 3200 个 bin 一起做
协方差，纯音只占 1~8 个 bin，被窗内的非相干失配底稀释；旧代数据因目标剥离不干净
（matched≈0.02，目标残留本身相干）反而"救"了相干度。

本脚本对每个候选窗，用 `_residual_bands` 复算同一残差上的判定，给出：
  · 有干扰样本：真源是否被报出（带中心与 GT 相差 ≤40 kHz）；
  · 干净样本：是否虚警（≥1 带）。
用法：python -X utf8 diag\\_sweep_d1_cohwin.py <dataset_root> [win_khz,win_khz,...]
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
WINS_KHZ = [20, 10, 5, 3, 2]


def load(root: _pl.Path, sid: str):
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = _json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    tr = _json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    fc = float(obs["receivers"][0]["center_frequency_hz"])
    tmp = _pl.Path(tempfile.mkdtemp(prefix=f"sw_{sid}_"))
    try:
        refs = rca._build_target_reference(priv, n)
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=fc, target_bw_normalized=0.25,
                             aoi_center=[0, 0], target_reference=refs)
        ct.set_context(ctx)
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        resid = np.asarray(ct._strip_target_at_station("station_a"), np.complex128)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return occ, resid, fs, float(occ["band_hz"][1]), list(tr.get("interferers") or [])


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    root = _pl.Path(argv[0]).resolve()
    wins = ([float(x) * 1e3 for x in argv[1].split(",")] if len(argv) > 1 else
            [w * 1e3 for w in WINS_KHZ])
    idx = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    stat = {w: {"dirty": 0, "hit": 0, "clean": 0, "fa": 0, "coh_true": [],
                "bands": 0, "near": 0, "bw": []} for w in wins}
    for s in idx["samples"]:
        occ, resid, fs, rx_half, gts = load(root, s["id"])
        for w in wins:
            ct.PEAK_COH_WIN_HZ = w
            bands = ct._residual_bands(resid, fs, rx_half)
            st = stat[w]
            if gts:
                st["dirty"] += 1
                near = [b for b in bands
                        if any(abs(b["freq_offset_hz"] - g["freq_offset_hz"]) <= TOL_HZ
                               for g in gts)]
                if near:
                    st["hit"] += 1
                st["bands"] += len(bands)
                st["near"] += len(near)
                st["bw"] += [float(b.get("bandwidth_hz") or 0.0) for b in bands]
                for g in gts:
                    st["coh_true"].append(
                        ct._band_coherence(resid, fs, g["freq_offset_hz"], w))
            else:
                st["clean"] += 1
                st["fa"] += 1 if bands else 0
    print(f"== {root.parent.name}  D1 相干窗扫描（station_a；其余门限不变 "
          f"over≥{ct.PEAK_OVER_DB}dB / coh≥{ct.PEAK_COH_MIN}）==")
    print(f"{'窗(kHz)':>8}{'真源覆盖':>12}{'虚警率':>12}{'真源处相干度中位':>16}"
          f"{'多余峰/总峰':>12}{'带宽中位(kHz)':>14}")
    for w in wins:
        st = stat[w]
        cov = 100 * st["hit"] / max(st["dirty"], 1)
        fa = 100 * st["fa"] / max(st["clean"], 1)
        med = float(np.median(st["coh_true"])) if st["coh_true"] else float("nan")
        cov_s = f"{st['hit']}/{st['dirty']} ({cov:.1f}%)"
        fa_s = f"{st['fa']}/{st['clean']} ({fa:.1f}%)"
        extra = st["bands"] - st["near"]
        mbw = float(np.median(st["bw"])) / 1e3 if st["bw"] else float("nan")
        pk_s = f"{extra}/{st['bands']}"
        print(f"{w/1e3:8.1f}{cov_s:>12}{fa_s:>12}{med:16.3f}{pk_s:>12}{mbw:14.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
