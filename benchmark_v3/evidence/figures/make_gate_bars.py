# -*- coding: utf-8 -*-
"""D2 仪器验收 gate：比率条形图 + 计数对照图。

数据来源（唯一口径）：benchmark_v3/evidence/_gate_d2_hygiene_run3.txt
（= BASELINE.json 的 gate_effect / README §6.3 表；run1..run4 逐字节相同）
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.dirname(os.path.abspath(__file__))


def pick_font() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in have:
            return name
    return "DejaVu Sans"


FONT = pick_font()
plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#1F4E79"      # 通过
RED = "#C00000"       # 未过
ORANGE = "#E36C0A"    # 反向指标（越低越好）
GREY = "#8496B0"      # 无门限
INK = "#333333"

# 名称, 比率, 分子, 分母, 分母单位, 门限, 方向(up/down), 判定
ROWS = [
    ("干净样本检测 (clean)", 0.878, 1001, 1140, "条波形", 0.85, "up", "通过", BLUE),
    ("GT 切片检测", 0.550, 44, 80, "个源切片", 0.50, "up", "通过", BLUE),
    ("工具召回", 0.653, 64, 98, "个可见干扰", 0.50, "up", "通过", BLUE),
    ("工具假簇率", 0.540, 75, 139, "个输出簇", 0.50, "down", "未过", ORANGE),
    ("工具条件分类", 0.438, 28, 64, "个已认领簇", 0.50, "up", "未过（唯一卡点）", RED),
    ("可解子集召回", 0.685, 63, 92, "个可见干扰", None, "up", "—", GREY),
    ("端到端", 0.286, 28, 98, "个可见干扰", None, "up", "— 派生", GREY),
]

# ---------------------------------------------------------------- 图 1：比率
fig, ax = plt.subplots(figsize=(11.6, 6.0))
ys = list(range(len(ROWS)))
INSIDE_FROM = 0.62
for y, (name, val, num, den, unit, gate, direc, verdict, color) in zip(ys, ROWS):
    ax.barh(y, val, height=0.58, color=color, zorder=3)
    frac = f"[{num}/{den} {unit}]"
    if val >= INSIDE_FROM:
        ax.text(val - 0.014, y, f"{val:.3f}", va="center", ha="right",
                fontsize=13, fontweight="bold", color="white", zorder=5)
        ax.text(val - 0.098, y, frac, va="center", ha="right",
                fontsize=9.5, color="white", zorder=5)
    else:
        ax.text(val + 0.012, y, f"{val:.3f}", va="center", ha="left",
                fontsize=13, fontweight="bold", color=color, zorder=5)
        ax.text(val + 0.085, y, frac, va="center", ha="left",
                fontsize=9.5, color=INK, zorder=5)
    txt = verdict + ("（越低越好）" if direc == "down" else "")
    ax.text(1.035, y, txt, va="center", ha="left", fontsize=10.5,
            color=color, fontweight="bold", zorder=6)

for g, lab in ((0.50, "0.50"), (0.85, "0.85")):
    ax.axvline(g, color="#7F7F7F", linestyle=(0, (3, 2)), linewidth=1.4, zorder=2)
    ax.text(g, len(ROWS) - 0.34, f"门限 {lab}", ha="center", va="bottom",
            fontsize=9, color="#7F7F7F", zorder=6)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=12.5, color=INK)
ax.set_xlim(0, 1.28)
ax.set_ylim(-0.6, len(ROWS) - 0.45)
ax.invert_yaxis()                       # 与表格同序：clean 在最上
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("比率（灰虚线 = 该指标门限；假簇率越低越好）", fontsize=11, color=INK)
ax.set_title("D2 仪器验收 gate（50 样本，2026-10-10 最终口径）",
             fontsize=15.5, fontweight="bold", color=BLUE, pad=30, loc="left")
ax.text(0, 1.055, "同一份数据、同一个分类器（19 类 CNN），只换喂料；分母单位各异，故并列标注分子 / 分母",
        transform=ax.transAxes, fontsize=10, color="#666666")
ax.grid(axis="x", color="#E6E6E6", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#BFBFBF")
ax.tick_params(axis="y", length=0)
fig.tight_layout()
p1 = os.path.join(OUT, "gate_d2_rate.png")
fig.savefig(p1, dpi=200, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 图 2：计数
fig, ax = plt.subplots(figsize=(11.6, 5.8))
labels = [r[0] for r in ROWS]
tot = [r[3] for r in ROWS]
hit = [r[2] for r in ROWS]
h = 0.33
for y, (t, v) in enumerate(zip(tot, hit)):
    ax.barh(y + h / 2 + 0.02, t, height=h, color="#D9D9D9", zorder=3,
            label="总样本数（分母）" if y == 0 else None)
    ax.barh(y - h / 2 - 0.02, v, height=h, color=BLUE, zorder=3,
            label="满足条件数（分子）" if y == 0 else None)
    ax.text(t * 1.07, y + h / 2 + 0.02, str(t), va="center", ha="left",
            fontsize=10, color="#7F7F7F", zorder=4)
    ax.text(v * 1.07, y - h / 2 - 0.02, str(v), va="center", ha="left",
            fontsize=10, fontweight="bold", color=BLUE, zorder=4)
    ax.text(1.035, y, f"{v}/{t}  =  {v / t:.3f}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold", zorder=5)
ax.set_xscale("log")
ax.set_xlim(8, 3400)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels(labels, fontsize=12, color=INK)
ax.invert_yaxis()
ax.set_xlabel("样本数（对数刻度；分母单位不同，勿横向比大小）", fontsize=11, color=INK)
ax.set_title("同口径的分子 / 分母", fontsize=15.5, fontweight="bold",
             color=BLUE, pad=14, loc="left")
ax.legend(loc="lower right", frameon=False, fontsize=10.5)
ax.grid(axis="x", color="#E6E6E6", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#BFBFBF")
ax.tick_params(axis="y", length=0)
fig.tight_layout()
p2 = os.path.join(OUT, "gate_d2_counts.png")
fig.savefig(p2, dpi=200, facecolor="white")
plt.close(fig)

print("font:", FONT)
print("wrote:", p1)
print("wrote:", p2)
