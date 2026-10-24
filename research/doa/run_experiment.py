"""
Final experiment for the preliminary study.

Three questions:
  Q1  how close in angle must a coherent echo be before azimuth estimation degrades?
  Q2  how much SNR robustness does wideband incoherent averaging buy?
  Q3  does instantaneous bandwidth help azimuth at all (delay decorrelation)?

Outputs: results/fig_*.png and results/summary.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wideband_doa as W

OUT = Path("results")
OUT.mkdir(exist_ok=True)
GRID = np.deg2rad(np.arange(-90.0, 90.0 + 1e-9, 0.25))
KEYS = ("NB", "ISM", "CSM", "SS")
LABEL = {"NB": "NB-MUSIC (centre band)", "ISM": "ISM (incoherent wideband)",
         "CSM": "CSM (coherent focusing)", "SS": "SS-MUSIC (spatial smoothing)"}
STYLE = {"NB": dict(color="#1f77b4", marker="o"), "ISM": dict(color="#2ca02c", marker="s"),
         "CSM": dict(color="#d62728", marker="^"), "SS": dict(color="#9467bd", marker="D")}
MC = 20
SEED = 7


def two_path(bw, snr, dphi, dtau):
    sc = W.Scenario(bandwidth_mhz=bw, snr_db=snr, n_mc=MC, velocity_mps=15.0)
    sc.paths = [W.Path(30.0, 0.0, 1.0, 0.0), W.Path(30.0 + dphi, dtau, 0.6, 50.0)]
    return W.run_point(sc, GRID, seed=SEED)


def plot(xs, curves, xlabel, title, fname, logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    # grid step is 0.25 deg, so an exact hit means "within +-0.125 deg"; floor the
    # plotted value there instead of letting a log axis dive to 1e-15.
    floor = 0.1
    for k in KEYS:
        ys = [max(float(y), floor) for y in curves[k]]
        ax.plot(xs, ys, label=LABEL[k], lw=1.8, ms=5, **STYLE[k])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("azimuth RMSE  [deg]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    print(f"  -> {OUT / fname}")


summary = {}

# ---------------------------------------------------------------- Q1
print("Q1: coherent echo angular offset (B=40 MHz, dtau=20 ns, SNR=-5 dB)")
dphis = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0]
q1 = {k: [] for k in KEYS}
for d in dphis:
    r = two_path(40.0, -5.0, d, 20.0)
    for k in KEYS:
        q1[k].append(r[k])
    print(f"   dphi={d:5.1f}  " + "  ".join(f"{k}={r[k]:7.2f}" for k in KEYS))
summary["q1_angle_offset"] = {"x": dphis, **q1}
plot(dphis, q1, "coherent echo angular offset  [deg]",
     "Q1  Single coherent echo 0.6x, 20 ns: azimuth error vs angular offset",
     "fig_q1_angle_offset.png", logy=True)

# ---------------------------------------------------------------- Q2
print("Q2: SNR (B=40 MHz, dphi=3 deg, dtau=20 ns)")
snrs = [-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 10.0]
q2 = {k: [] for k in KEYS}
for s in snrs:
    r = two_path(40.0, s, 3.0, 20.0)
    for k in KEYS:
        q2[k].append(r[k])
    print(f"   SNR={s:6.1f}  " + "  ".join(f"{k}={r[k]:7.2f}" for k in KEYS))
summary["q2_snr"] = {"x": snrs, **q2}
plot(snrs, q2, "per-element SNR  [dB]",
     "Q2  Azimuth error vs SNR (B = 40 MHz, echo 3 deg away)",
     "fig_q2_snr.png", logy=True)

# ---------------------------------------------------------------- Q3
print("Q3: bandwidth (dphi=3 deg, dtau=13 ns, SNR=-5 dB)")
bws = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
q3 = {k: [] for k in KEYS}
for b in bws:
    r = two_path(b, -5.0, 3.0, 13.0)
    for k in KEYS:
        q3[k].append(r[k])
    print(f"   B={b:6.1f}  " + "  ".join(f"{k}={r[k]:7.2f}" for k in KEYS))
summary["q3_bandwidth"] = {"x": bws, **q3}
plot(bws, q3, "instantaneous bandwidth  [MHz]",
     "Q3  Does bandwidth help azimuth?  (echo 3 deg away, 13 ns delay)",
     "fig_q3_bandwidth.png", logx=True, logy=True)

with open(OUT / "summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print("\nwrote", OUT / "summary.json")
