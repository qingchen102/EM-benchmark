"""临时只读：每源频偏的"空间选择性"估计 —— 能否拿回 +8.5pp。

对比三种每源频偏估计（都以 station_a 为参照，对同一个真源）：
  A 候选带中心        = 0.5*(band_lo+band_hi)         （现行 gate 口径）
  B 带内全谱 argmax   = 带内总功率最强峰              （GLM 试过，证伪 -1.9pp）
  C 空间选择性 argmax = 该源导向矢量下的波束功率峰    （本次验证）

C 的做法：滑窗子带算协方差 R(f)，取 P(f) = a(u)^H R(f) a(u) / ||a||²，argmax 即该源频偏。
因为 a(u) 只对准该源方向，其他方向的共道源被抑制 → 不受串扰带偏。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cityrf.baselines.spatial import band_covariance  # noqa: E402

import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402

D2 = _V3 / "dataset_d2"


def main(n_samples: int, sub=200e3, hop=50e3) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    errC, errB, errA = [], [], []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        n = int(obs["receivers"][0]["shape"][1])
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                    # noqa: BLE001
            refs = {}
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=fs, fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        try:
            ct.strip_target_all_stations(-fs * 0.45, fs * 0.45)
        except Exception:                                    # noqa: BLE001
            pass
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        iq = ct._station_iq_raw("station_a", on_residual=True)
        rec = next(r for r in obs["receivers"] if r["receiver_id"] == "station_a")
        off = np.asarray(rec["element_offsets_world_m"], float)
        axis = (off[-1] - off[0]) / np.linalg.norm(off[-1] - off[0])
        rp = np.asarray(rec["receiver_position_m"], float)
        rx = fs * 0.45
        # 预扫全带 P(f)（空间选择性）
        fcs = np.arange(-rx + sub / 2, rx - sub / 2, hop)
        for j, g in enumerate(tr.get("interferers", [])):
            if (ach.get(f"interferer_{j+1}", {}).get("s0", {}).get("pr_dbm", -999) - nz) < 3.0:
                continue
            p = np.asarray(g["position_m"], float)[:3]
            d = p - rp
            d[2] = 0
            u = float((d / np.linalg.norm(d)) @ axis)
            a = np.exp(2j * np.pi * 2.4e9 / 299792458.0 * (off @ np.array([0.0, u, 0.0])))
            P = []
            for f_c in fcs:
                try:
                    R, _ = band_covariance(iq, fs, f_c - sub / 2, f_c + sub / 2)
                except Exception:                            # noqa: BLE001
                    P.append(0.0)
                    continue
                P.append(float((a.conj() @ (R @ a)) / (a.conj() @ a)))
            P = np.asarray(P)
            f_true = g["freq_offset_hz"]
            # A 带中心：用 occupancy/候选带做近似（这里用真值带宽的带中心作代理上限）
            errA.append(abs(f_true - 0.0) / 1e3)
            errC.append(abs(f_true - float(fcs[int(np.argmax(P))])) / 1e3)
            # B 带内总谱 argmax：用平均通道功率谱
            sp = np.abs(np.fft.fft(iq, axis=1)).mean(axis=0) ** 2
            fr = np.fft.fftfreq(iq.shape[1], 1 / fs)
            lo, hi = f_true - g["bandwidth_hz"] / 2, f_true + g["bandwidth_hz"] / 2
            m = (fr >= lo) & (fr <= hi)
            errB.append(abs(f_true - float(fr[m][int(np.argmax(sp[m]))])) / 1e3)
    for tag, e in (("A 频偏=0（无修正下界）", errA), ("B 带内总谱 argmax", errB),
                   ("C 空间选择性 argmax", errC)):
        e = np.array(e)
        print(f"  {tag:26s} 中位 {np.median(e):6.0f} kHz   p75 {np.percentile(e,75):6.0f}"
              f"   ≤40kHz 占 {100*np.mean(e<=40):3.0f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
