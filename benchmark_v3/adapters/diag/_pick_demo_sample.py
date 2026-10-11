"""给平台演示挑一个"整网都在建成区"的样本（只读；打印候选排序，可选出对比图）。

动机（2026-10-11）：平台预设 D2 原取 `d2_00000`，那一组电台全落在杭州图幅西南角的
无建筑区（水面/绿地，到最近建筑最远 792 m），Sionna RT 只给出单条直达径 →
在网页地图上看起来"站和源全在城市之外"。本脚本按"±200 m 建筑密度 / 到最近建筑距离 /
每条链路的径数"给样本排序，供人工拍板。

用法::

    python -X utf8 diag/_pick_demo_sample.py                     # d2，打印前 12 名
    python -X utf8 diag/_pick_demo_sample.py --difficulty d3
    python -X utf8 diag/_pick_demo_sample.py --plot d2_00000 d2_00017 \
        --out ../evidence/d2_preset_swap_2026-10-11.png

依赖：numpy（必需）；matplotlib（仅 --plot 时需要）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]          # EM-benchmark/
V3 = ROOT / "benchmark_v3"
PLAT = V3 / "城市低空电磁信号仿真平台"
CELL_M = 4.0
DEFAULT_DIFFICULTY = "d2"

_z = np.load(PLAT / "priors" / "hangzhou_geometry" / "occupancy.npz", allow_pickle=True)
OCC = np.asarray(_z["occ"])
ORIGIN = np.asarray(_z["origin"], dtype=float)
H, W = OCC.shape
EXTENT = [ORIGIN[0], ORIGIN[0] + W * CELL_M, ORIGIN[1], ORIGIN[1] + H * CELL_M]
_BY, _BX = np.nonzero(OCC)          # 建筑格坐标（只算一次）


def dens(p, radius_m: float = 200.0) -> int:
    """位置 p 的 ±radius_m 方格内建筑格数。"""
    r = int(radius_m / CELL_M)
    col = int((p[0] - ORIGIN[0]) / CELL_M)
    row = int((p[1] - ORIGIN[1]) / CELL_M)
    r0, r1 = max(0, row - r), min(H, row + r + 1)
    c0, c1 = max(0, col - r), min(W, col + r + 1)
    return int((OCC[r0:r1, c0:c1] > 0).sum())


def nearest_building_m(p) -> float:
    """到最近建筑格的直线距离（米）。"""
    col = int((p[0] - ORIGIN[0]) / CELL_M)
    row = int((p[1] - ORIGIN[1]) / CELL_M)
    if not _BY.size or not (0 <= row < H and 0 <= col < W):
        return float("inf")
    j = int(np.argmin((_BY - row) ** 2 + (_BX - col) ** 2))
    return float(np.hypot(_BY[j] - row, _BX[j] - col) * CELL_M)


def path_counts(sample: Path) -> dict[str, int]:
    """每条链路（target / interferer_N）× 站点的径数。"""
    out: dict[str, int] = {}
    for npz in sorted((sample / "capture" / "private").glob("channel_*_station_*.npz")):
        stem = npz.stem                      # channel_<src>_station_<site>
        src = stem[len("channel_"):stem.index("_station_")]
        site = stem.split("_")[-1]
        d = np.asarray(np.load(npz, allow_pickle=True)["delays_s"], dtype=float).ravel()
        out[f"{src}:{site}"] = int(d.size)
    return out


def survey(difficulty: str) -> list[dict]:
    rows = []
    for s in sorted((V3 / f"dataset_{difficulty}").glob(f"{difficulty}_*")):
        sc_p = s / "capture" / "private" / "scenario.json"
        if not sc_p.is_file():
            continue
        sc = json.loads(sc_p.read_text(encoding="utf-8"))
        rx = [r["position_m"] for r in sc["receivers"]]
        tx = [e["position_m"] for e in sc["emitters"]]
        d_rx = [dens(p) for p in rx]
        d_tx = [dens(p) for p in tx]
        nb = [nearest_building_m(p) for p in rx + tx]
        pc = path_counts(s)
        tgt = {k.split(":")[1]: v for k, v in pc.items() if k.startswith("target:")}
        itf = {k: v for k, v in pc.items() if not k.startswith("target:")}
        rows.append(dict(
            sid=s.name, min_rx=min(d_rx), min_all=min(d_rx + d_tx), max_nb=max(nb),
            tgt=tgt, n_multi=sum(1 for v in tgt.values() if v >= 2),
            itf_min=min(itf.values()) if itf else 0, itf=itf,
            kinds=[e.get("kind") for e in sc["emitters"]],
        ))
    rows.sort(key=lambda r: (-r["min_all"], -r["min_rx"], r["max_nb"]))
    return rows


def plot(old_sid: str, new_sid: str, difficulty: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.4))
    for ax, sid, title in zip(
        axes, (old_sid, new_sid),
        (f"旧预设 {old_sid}：整网在无建筑区（水面/绿地）", f"新预设 {new_sid}：整网在建成区"),
    ):
        ax.imshow(OCC, origin="lower", extent=EXTENT, cmap="Greys", alpha=.85,
                  interpolation="nearest")
        sc = json.loads((V3 / f"dataset_{difficulty}" / sid / "capture" / "private"
                         / "scenario.json").read_text(encoding="utf-8"))
        rx = np.array([r["position_m"] for r in sc["receivers"]], dtype=float)
        tx = np.array([e["position_m"] for e in sc["emitters"]], dtype=float)
        ax.scatter(rx[:, 0], rx[:, 1], marker="s", s=190, facecolors="none",
                   edgecolors="#129e87", linewidths=2.6, label="监测站 rx", zorder=5)
        ax.scatter(tx[:, 0], tx[:, 1], marker="o", s=150, facecolors="none",
                   edgecolors="#e98942", linewidths=2.6, label="辐射源 tx", zorder=5)
        for i, p in enumerate(rx):
            ax.annotate("abcd"[i], (p[0], p[1]), fontsize=13, color="#0d6f60",
                        ha="right", va="bottom")
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlim(-2300, 2300); ax.set_ylim(-2300, 2300)
        ax.set_aspect("equal"); ax.grid(alpha=.12)
    axes[0].legend(loc="upper left", fontsize=11)
    fig.suptitle(f"{difficulty.upper()} 演示样本替换：旧 {old_sid} → 新 {new_sid}"
                 f"（灰=建筑，{CELL_M:.0f} m/格）", fontsize=16, y=.97)
    fig.tight_layout(rect=(0, 0, 1, .95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"[plot] {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="挑演示样本（只读）")
    ap.add_argument("--difficulty", default=DEFAULT_DIFFICULTY)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--plot", nargs=2, metavar=("OLD", "NEW"), default=None)
    ap.add_argument("--out", default=str(V3 / "evidence" / "demo_sample_swap.png"))
    args = ap.parse_args()

    if args.plot:
        plot(args.plot[0], args.plot[1], args.difficulty, Path(args.out))

    rows = survey(args.difficulty)
    if not rows:
        print(f"[!] dataset_{args.difficulty} 下没有可用样本")
        return 1
    print(f"数据集 dataset_{args.difficulty}：{len(rows)} 个样本，按「全网最稀电台的 ±200 m 建筑密度」降序")
    print(f"{'样本':<12}{'最稀电台':>9}{'全网最稀':>9}{'最远建筑m':>10}{'目标多径站':>11}{'干扰最少径':>11}  "
          f"目标各站径数 / 干扰各站径数")
    for r in rows[:args.top]:
        print(f"{r['sid']:<12}{r['min_rx']:>9}{r['min_all']:>9}{r['max_nb']:>10.0f}"
              f"{r['n_multi']:>11}{r['itf_min']:>11}  {r['tgt']} / {r['itf']}")
    print("\n判据提示：min_rx/min_all 越大、max_nb 越小 = 越「城里」；"
          "n_multi 要多、itf_min 要 ≥1（否则某站看不到干扰源）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
