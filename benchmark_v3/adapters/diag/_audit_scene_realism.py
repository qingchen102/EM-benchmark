"""只读审计（零 token）：场景"真实感"核对 —— 多径到底有没有、AOI 是不是落在空地。

三件事：
A. 冻结信道 taps 普查：每条链路有几条径（0 = 被完全遮挡；≥2 = 真多径），
   以及多径时延扩展。数据来源是 `capture/private/channel_*.npz` 的 `delays_s`，
   即建库时 Sionna RT 实际算出来的路径表，与网页显示无关。
B. 各档每个样本的"站位周围建筑密度"：用 `priors/hangzhou_geometry/occupancy.npz`
   数 ±200 m 内的建筑格数；用于发现"整网落在湖面/空地"的样本。
C. 00000 号样本（= 平台预设 `examples/benchmark_v3/` 的来源）里每台电台到
   最近建筑的距离。

背景：2026-10-11 用户看平台时发现「D2 的站和源全在城市之外」，本审计即为排查。
结论（2026-10-11 实测）：
  - 多径确实存在：D2 76.2% / D3 48.9% / D4 53.7% 的链路 ≥2 条径，时延扩展中位
    2.7~3.7 µs；D1 是 free_space，100% 单径（设计如此）。
  - 但排布器只在"地图矩形 + inset"里抽 AOI，**不看建筑密度**，所以会抽到空地：
    D2 有 4/50 个样本整网落在空地（含 00000 号，它正是平台预设的来源），
    D3/D4 为 0/50。
用法：
  python _audit_scene_realism.py                  # 打印审计
  python _audit_scene_realism.py --plot out.png    # 另存四档底图（需 matplotlib/scipy）
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import numpy as np

_V3 = Path(__file__).resolve().parent.parent.parent
PLAT = _V3 / "城市低空电磁信号仿真平台"
DIFFS = ("d1", "d2", "d3", "d4")
CELL_M = 4.0


def load_grid():
    z = np.load(PLAT / "priors" / "hangzhou_geometry" / "occupancy.npz", allow_pickle=True)
    occ = np.asarray(z["occ"])
    origin = np.asarray(z["origin"], dtype=float)
    return occ, origin


def per_station_density(occ, origin, pts, radius_m=200.0):
    H, W = occ.shape
    r = int(radius_m / CELL_M)
    out = []
    for p in pts:
        col = int((p[0] - origin[0]) / CELL_M); row = int((p[1] - origin[1]) / CELL_M)
        r0, r1 = max(0, row - r), min(H, row + r + 1)
        c0, c1 = max(0, col - r), min(W, col + r + 1)
        out.append(int((occ[r0:r1, c0:c1] > 0).sum()))
    return out


def scenario_of(sample_dir: Path):
    p = sample_dir / "capture" / "private" / "scenario.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def audit_channels():
    print("=== A. 冻结信道：每条链路几条径（0=完全遮挡，>=2=真多径）===")
    for tag in DIFFS:
        root = _V3 / f"dataset_{tag}"
        samples = sorted(p for p in root.glob(f"{tag}_*") if p.is_dir())
        counts, spreads, zero_by_station, total = [], [], {}, 0
        for s in samples:
            for npz in sorted((s / "capture" / "private").glob("channel_*.npz")):
                d = np.asarray(np.load(npz, allow_pickle=True)["delays_s"], dtype=float).ravel()
                total += 1
                counts.append(d.size)
                if d.size == 0:
                    parts = npz.stem.split("_")
                    if len(parts) > 2 and parts[1] == "target":
                        zero_by_station[parts[-1]] = zero_by_station.get(parts[-1], 0) + 1
                elif d.size > 1:
                    spreads.append((d.max() - d.min()) * 1e9)
        if not total:
            print(f"  {tag}: 无信道文件"); continue
        c = np.asarray(counts)
        hist = {int(k): int(v) for k, v in zip(*np.unique(c, return_counts=True))}
        print(f"  {tag}: {total} 条链路 / {len(samples)} 样本   径数分布 {hist}")
        print(f"      0 径 {int((c==0).sum())} ({float((c==0).mean()):.1%})   "
              f"1 径 {int((c==1).sum())} ({float((c==1).mean()):.1%})   "
              f">=2 径 {int((c>=2).sum())} ({float((c>=2).mean()):.1%})")
        if spreads:
            sp = np.asarray(spreads)
            print(f"      多径时延扩展 中位 {np.median(sp):.0f} ns  p90 {np.percentile(sp, 90):.0f} ns  max {sp.max():.0f} ns")
        if zero_by_station:
            print(f"      target 链路 0 径按站 {zero_by_station}")


def audit_density(occ, origin):
    print("\n=== B. 各档每样本的站位建筑密度（±200 m 建筑格数，1 格 = 4 m）===")
    H, W = occ.shape
    full = 101 * 101
    avg = float((occ > 0).mean()) * full
    print(f"  参考：方格 {full} 格；全市平均建筑密度 {avg:.0f} 格（{float((occ>0).mean()):.1%}）")
    for tag in DIFFS:
        root = _V3 / f"dataset_{tag}"
        mins, meds, opens = [], [], 0
        for s in sorted(p for p in root.glob(f"{tag}_*") if p.is_dir()):
            sc = scenario_of(s)
            if not sc:
                continue
            c = per_station_density(occ, origin, [r["position_m"] for r in sc["receivers"]])
            mins.append(min(c)); meds.append(float(np.median(c)))
            if max(c) < avg * 0.3:
                opens += 1
        if not mins:
            print(f"  {tag}: 无场景文件"); continue
        mins = np.asarray(mins)
        print(f"  {tag}: n={len(mins)}  最稀站密度 中位 {np.median(mins):.0f} / min {mins.min():.0f} / max {mins.max():.0f}"
              f"    整网落空地 {opens}/{len(mins)} = {opens/len(mins):.0%}"
              f"    最稀站<500 格 {(mins<500).sum()}/{len(mins)}")


def audit_preset_case(occ, origin):
    print("\n=== C. 00000 号样本（平台预设来源）每台电台到最近建筑的距离 ===")
    from scipy import ndimage as ndi
    dist = ndi.distance_transform_edt(occ == 0) * CELL_M
    H, W = occ.shape
    for tag in DIFFS:
        sc = scenario_of(_V3 / f"dataset_{tag}" / f"{tag}_00000")
        if not sc:
            continue
        pts = [r["position_m"] for r in sc["receivers"]] + [e["position_m"] for e in sc["emitters"]]
        d = []
        for p in pts:
            col = int(np.clip((p[0] - origin[0]) / CELL_M, 0, W - 1))
            row = int(np.clip((p[1] - origin[1]) / CELL_M, 0, H - 1))
            d.append(float(dist[row, col]))
        print(f"  {tag}: backend={sc['propagation']['backend']:10s} 电台 {len(pts)} 台，"
              f"到最近建筑 {min(d):.0f}~{max(d):.0f} m")


def plot(occ, origin, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    H, W = occ.shape
    x0, y0 = origin[0], origin[1]
    x1, y1 = x0 + W * CELL_M, y0 + H * CELL_M
    titles = {"d1": "D1 物流航线（free_space，不用建筑）", "d2": "D2 居民区共存",
              "d3": "D3 城市峡谷", "d4": "D4 对抗演习"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 15), dpi=110)
    for ax, tag in zip(axes.ravel(), DIFFS):
        ax.imshow(occ > 0, origin="lower", extent=[x0, x1, y0, y1], cmap="Greys", vmin=0, vmax=1.6,
                  interpolation="nearest")
        sc = scenario_of(_V3 / f"dataset_{tag}" / f"{tag}_00000")
        if sc:
            rx = np.asarray([r["position_m"] for r in sc["receivers"]], dtype=float)
            tx = np.asarray([e["position_m"] for e in sc["emitters"]], dtype=float)
            ax.scatter(rx[:, 0], rx[:, 1], s=170, marker="s", facecolor="none", edgecolor="#129e87",
                       linewidth=2.6, label="监测站 rx", zorder=5)
            ax.scatter(tx[:, 0], tx[:, 1], s=180, marker="o", facecolor="none", edgecolor="#e98942",
                       linewidth=2.6, label="辐射源 tx", zorder=5)
            dens = per_station_density(occ, origin, list(rx) + list(tx))
            ax.set_title(f"{titles[tag]}\nbackend={sc['propagation']['backend']}  "
                         f"±200 m 内建筑格数 {min(dens)}~{max(dens)}", fontsize=13, pad=10)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.grid(alpha=.15)
    axes.ravel()[0].legend(loc="upper left", fontsize=11, framealpha=.9)
    fig.suptitle("四档 00000 样本：电台 vs 杭州建筑底图（灰=建筑，4 m/格）", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, facecolor="white")
    print(f"\n[saved] {out}")


def main():
    occ, origin = load_grid()
    audit_channels()
    audit_density(occ, origin)
    audit_preset_case(occ, origin)
    if "--plot" in _sys.argv:
        i = _sys.argv.index("--plot")
        plot(occ, origin, Path(_sys.argv[i + 1]))


if __name__ == "__main__":
    main()
