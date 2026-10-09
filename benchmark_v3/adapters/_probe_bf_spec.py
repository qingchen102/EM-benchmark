"""只读：把接收波束指向每个 GT 源的位置，看该源**实际**的谱线/占用带宽。

回答的问题：真值 freq_offset_hz / bandwidth_hz 与波形实际占用是否一致
（即这些标签在数据里到底可不可观测）。
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _diag_d2_extract as dx  # noqa: E402

B = HERE.parent


def profile(p, f, nbin=16, half=1.6e6):
    m = np.abs(f) <= half
    fp, pp = f[m], p[m]
    edges = np.linspace(-half, half, nbin + 1)
    out = []
    for i in range(nbin):
        sel = (fp >= edges[i]) & (fp < edges[i + 1])
        out.append(10 * np.log10(pp[sel].sum() / max(sel.sum(), 1) + 1e-30))
    return edges, out


def run(ds, sid):
    cap = B / ds / sid / "capture"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    fc = float(obs["receivers"][0]["center_frequency_hz"])
    rec = next(r for r in obs["receivers"] if r["receiver_id"] == "station_a")
    iq = np.load(cap / "public" / "iq_station_a.npy")
    n = iq.shape[1]
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    X = np.fft.fftshift(np.fft.fft(iq, axis=1), axes=1)

    tg = truth.get("target") or {}
    rows = [("target", tg.get("kind"), tg.get("position_m"), tg.get("freq_offset_hz"),
             tg.get("bandwidth_hz"))]
    for i, g in enumerate(truth.get("interferers", [])):
        rows.append((f"ifw{i+1}", g.get("kind"), g.get("position_m"),
                     g.get("freq_offset_hz"), g.get("bandwidth_hz")))

    print(f"\n===== {ds}/{sid}  station_a  n={n} fs={fs/1e6:.1f}MHz =====")
    for name, kind, pos, f0, bw in rows:
        if pos is None:
            continue
        try:
            a = dx.steering(rec, [pos], True, fc)[:, 0]
        except Exception as e:  # noqa: BLE001
            print(f"  {name:7s} {kind:16s} steering 失败: {type(e).__name__}: {e}")
            continue
        a = a / (np.linalg.norm(a) + 1e-12)
        p = np.abs(np.conj(a) @ X) ** 2
        ker = np.ones(max(int(round(20e3 / (fs / n))), 1))
        ps = np.convolve(p, ker / ker.size, "same")
        m = np.abs(f) <= 1.9e6
        pk = f[m][np.argmax(ps[m])]
        # 99% 占用带宽（相对指向后的谱）
        c = np.cumsum(ps[m])
        c = c / c[-1]
        lo = f[m][np.searchsorted(c, 0.005)]
        hi = f[m][np.searchsorted(c, 0.995)]
        sf0 = "None" if f0 is None else f"{f0/1e3:9.1f}kHz"
        sbw = "None" if bw is None else f"{bw/1e3:8.1f}kHz"
        print(f"  {name:7s} 真值 kind={str(kind):16s} f0={sf0:>14s} "
              f"bw={sbw:>13s} | 指向后 峰={pk/1e3:9.1f}kHz "
              f"99%占用={((hi-lo)/1e3):8.1f}kHz [{lo/1e3:8.1f},{hi/1e3:8.1f}]")
        if name.startswith("ifw1"):
            edges, prof = profile(ps, f)
            print("          16 段剖面(dB, 相对该剖面中位):",
                  " ".join(f"{v - np.median(prof):+.0f}" for v in prof))
            print("          段边界(kHz):",
                  " ".join(f"{e/1e3:+.0f}" for e in edges))


for ds, sids in (("dataset_d4", ["d4_00000"]), ("dataset_d3", ["d3_00005"]),
                 ("dataset_d2", ["d2_00001"])):
    for sid in sids:
        try:
            run(ds, sid)
        except Exception as e:  # noqa: BLE001
            print(f"{ds}/{sid} 失败: {type(e).__name__}: {e}")
