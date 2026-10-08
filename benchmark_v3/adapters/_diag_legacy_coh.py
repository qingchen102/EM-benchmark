"""只读：方案 G —— 保留旧判据（D2 已验证的带宽/召回），只给每个峰**加相干标注**，
由 agent 在阈值处自行剔除（零 token）。

动机：新判据（局部底+相干门控）在 D1 上 0 虚警，但在 D2 上召回 54%->36%
（窄带候选让 RANSAC 失去几何支撑）。旧判据的宽峰虽然"口径难看"，却是 D2 现有
基线（召回 55.1%/假簇 63.0%）的组成部分。最小侵入 = 判据不动、只加判别量。

本脚本验证：旧判据的峰，在峰频处算相干度，真/假 是否可分（若可分，agent 一句
规则即可去掉 D1 虚警而完全不动 D2）。

用法：python _diag_legacy_coh.py
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

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

D1 = _V3 / "dataset_d1"
D2 = _V3 / "dataset_d2"


def survey(root: Path, sids, station="station_a"):
    real, fake, clean_fa = [], [], 0
    for sid in sids:
        cap = root / sid / "capture" / "public"
        priv = root / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ifw = tr.get("interferers", [])
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception:                                     # noqa: BLE001
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=root / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        st = station if station in allowed else allowed[0]
        try:
            ct.estimate_band_occupancy_by_correlation(st)
            resid = np.load(ctx.cache_dir / f"resid_{st}.npy")
        except Exception:                                     # noqa: BLE001
            continue
        peaks = ct._residual_peaks_legacy(resid, fs, 0.45 * fs)
        if not ifw:
            if peaks:
                clean_fa += 1
            for p in peaks:
                c = ct._band_coherence(resid, fs, p["freq_offset_hz"], 20e3)
                fake.append((c, p["over_floor_db"], p["bandwidth_hz"], sid, p["freq_offset_hz"]))
            continue
        for p in peaks:
            c = ct._band_coherence(resid, fs, p["freq_offset_hz"], 20e3)
            hit = any(abs(p["freq_offset_hz"] - g["freq_offset_hz"]) <= 250e3 for g in ifw)
            (real if hit else fake).append(
                (c, p["over_floor_db"], p["bandwidth_hz"], sid, p["freq_offset_hz"]))
    return real, fake, clean_fa


def main() -> None:
    idx1 = json.loads((D1 / "index.json").read_text(encoding="utf-8"))
    idx2 = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    s1 = [s["id"] for s in idx1["samples"][:12]]
    s2 = [s["id"] for s in idx2["samples"][:12]]
    type("X", (), {})  # noop
    for label, root, sids, st in (("D1", D1, s1, "station_a"),
                                  ("D2", D2, s2, "station_b")):
        real, fake, cfa = survey(root, sids, st)
        print(f"\n===== {label}（旧判据 + 相干标注，站 {st}）=====")
        print(f"  干净样本有峰的样本数 {cfa}；真峰 {len(real)} 假峰 {len(fake)}")
        for nm, rows in (("真峰", real), ("假峰", fake)):
            if not rows:
                print(f"  {nm}: 无")
                continue
            c = np.array([r[0] for r in rows])
            print(f"  {nm}: coh 中位 {np.median(c):.2f} P10 {np.percentile(c,10):.2f}"
                  f" P90 {np.percentile(c,90):.2f}  最大 {c.max():.2f}")
        R = np.array([r[0] for r in real])
        F = np.array([r[0] for r in fake])
        print("  阈值扫描（留真/留假）：")
        for thr in (0.5, 0.6, 0.7, 0.8):
            print(f"    thr={thr:.2f}: 留真 {int((R>=thr).sum())}/{len(R)}"
                  f"  留假 {int((F>=thr).sum())}/{len(F)}")
        if label == "D1":
            print("  假峰明细（假峰若在通带边缘则是失配能量）:")
            for r in sorted(fake, key=lambda x: -x[0])[:12]:
                print(f"    coh={r[0]:.2f} f={r[4]/1e3:8.1f}k over={r[1]:5.1f}"
                      f" bw={r[2]/1e3:7.1f}k {r[3]}")


if __name__ == "__main__":
    main()
