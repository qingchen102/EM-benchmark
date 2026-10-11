"""只读：D2 上"相干门控"漏了谁——逐真源归因（零 token）。

A/B 结果：D2 前 20 样本 真源召回 54%(旧判据) -> 36%(新判据)，假簇率 53%->63%。
本脚本对每个真源回答：它所在频率处
   (a) 覆盖窗子带功率是否 over 局部底 ≥8 dB？（能量够不够）
   (b) 覆盖窗 lambda1/Σλ 是否 ≥0.6？（相干门控卡不卡）
   并给出 (a)(b) 的四象限计数，判定漏因归属。

用法：python _diag_d2_gate.py [n_samples]
"""
from __future__ import annotations

import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

D2 = _V3 / "dataset_d2"


def sub_power_over(iq, fs, center, win, floor_win_hz=300e3):
    n = iq.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    psd = (np.abs(np.fft.fft(iq, axis=-1)) ** 2).mean(axis=0)
    db = 10 * np.log10(psd + 1e-30)
    k = max(int(round(floor_win_hz / (fs / n))) | 1, 5)
    k = min(k, (db.size // 2) | 1)
    order = np.argsort(f)
    db_s = db[order]
    f_s = f[order]
    floor = median_filter(db_s, size=k, mode="nearest")
    m = np.abs(f - center) <= win
    over = (db - floor)[m]
    return float(np.max(over)) if over.size else float("nan")


def coh(iq, fs, center, win):
    n = iq.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = np.abs(f - center) <= win
    if int(m.sum()) < 2:
        return float("nan")
    x = np.fft.fft(iq, axis=-1)[:, m]
    x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-30)
    r = x @ x.conj().T / x.shape[1]
    w = np.linalg.eigvalsh(r)[::-1].real
    return float(w[0] / (w.sum() + 1e-30))


def main(n_samples: int) -> None:
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    quad = {}
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        nz = tr["noise"]["noise_floor_dbm"]
        ach = tr.get("achieved") or {}
        try:
            refs = rca._build_target_reference(priv, n)
        except Exception as exc:                              # noqa: BLE001
            print(f"{sid}: ref fail {str(exc)[:40]}")
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=refs)
        ct.set_context(ctx)
        print(f"\n{sid}")
        for j, g in enumerate(tr.get("interferers", [])):
            eid = f"interferer_{j+1}"
            vis = [f"station_{chr(ord('a') + i)}"
                   for i in range(len(allowed))
                   if (ach.get(eid, {}).get(f"s{i}", {}).get("pr_dbm", -999) - nz) >= 3.0]
            if not vis:
                continue
            f0 = g["freq_offset_hz"]
            bw = g.get("bandwidth_hz", 100e3)
            print(f"   {eid} {g['kind']:>12s} f={f0/1e3:8.1f}k bw={bw/1e3:6.1f}k"
                  f" 可见站 {len(vis)} {vis}")
            for st in vis:
                try:
                    ct.estimate_band_occupancy_by_correlation(st)
                    resid = np.load(ctx.cache_dir / f"resid_{st}.npy")
                except Exception as exc:                      # noqa: BLE001
                    print(f"      {st}: strip fail {str(exc)[:40]}")
                    continue
                ov = sub_power_over(resid, fs, f0, max(bw / 2, 30e3))
                c20 = coh(resid, fs, f0, 20e3)
                c10 = coh(resid, fs, f0, 10e3)
                c60 = coh(resid, fs, f0, 60e3)
                key = ("over>=8" if ov >= 8 else "over<8") + "/" + \
                      ("coh20>=0.6" if c20 >= 0.6 else "coh20<0.6")
                quad[key] = quad.get(key, 0) + 1
                print(f"      {st}: over(局部底)={ov:6.1f}dB  coh10={c10:.2f}"
                      f" coh20={c20:.2f} coh60={c60:.2f}   [{key}]")
    print("\n===== 真源×可见站 四象限 =====")
    for k in sorted(quad):
        print(f"  {k}: {quad[k]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
