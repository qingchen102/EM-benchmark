"""Radio map 图件导出（PNG）——供汇报与报告直接使用。

只在 Web 侧存在：核心求解（:mod:`cityrf.radiomap`）不依赖 matplotlib。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cityrf.radiomap import METRIC_META, RAMP_STOPS


def _colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("cityrf", list(RAMP_STOPS), N=256)


def _extent(grid: dict) -> list[float]:
    cell = grid["cell_size_m"]
    x0, y0 = grid["origin_xy_m"]
    return [x0, x0 + grid["nx"] * cell, y0, y0 + grid["ny"] * cell]


#: 能画中文的字体族：先用系统已注册的，再试着借用 WSL 里可见的 Windows 字体。
_CJK_NAMES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
              "Source Han Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
              "PingFang SC", "Hiragino Sans GB", "AR PL UMing CN")
_CJK_FILES = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
#: 找不到中文字体时的英文回退（只影响 PNG，网页里的标签不受影响）。
_EN_LABEL = {"path_gain": "Path gain · best TX",
             "rss": "Received power · all TX",
             "sinr": "SINR · best TX"}


def _resolve_cjk_font() -> str | None:
    """找一个能渲染中文的字体族；找不到就返回 None，图件退化成英文标签。"""
    from matplotlib import font_manager
    registered = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_NAMES:
        if name in registered:
            return name
    for candidate in _CJK_FILES:
        if not Path(candidate).exists():
            continue
        try:
            font_manager.fontManager.addfont(candidate)
            return font_manager.FontProperties(fname=candidate).get_name()
        except Exception:
            continue
    return None


def render_png(result, payload: dict, path: Path, title: str) -> None:
    """把每个指标画成一张热力图，外加一张「最强源归属」图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cjk = _resolve_cjk_font()
    if cjk:
        plt.rcParams["font.family"] = [cjk, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    def t(zh: str, en: str) -> str:
        return zh if cjk else en

    cmap = _colormap()
    names = [n for n in result.metrics_requested if n in result.metrics]
    panels = names + ["association"]
    cols = 2 if len(panels) > 1 else 1
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.6 * cols, 5.7 * rows), squeeze=False)
    grid = result.grid
    extent = _extent(grid)
    tx_xy = np.array([e["position_m"][:2] for e in payload["emitters"]]) if payload["emitters"] else np.zeros((0, 2))
    rx_xy = np.array([r["position_m"][:2] for r in payload["receivers"]]) if payload["receivers"] else np.zeros((0, 2))

    for slot, name in enumerate(panels):
        ax = axes[slot // cols][slot % cols]
        if name == "association":
            data = result.association.astype(float)
            data[data < 0] = np.nan
            n_tx = max(len(result.tx_ids), 1)
            image = ax.imshow(data, origin="lower", extent=extent, interpolation="nearest",
                              cmap="tab20" if n_tx > 2 else "Set1", vmin=-0.5, vmax=n_tx - 0.5)
            bar = fig.colorbar(image, ax=ax, ticks=range(n_tx), fraction=0.046)
            bar.ax.set_yticklabels(result.tx_ids, fontsize=8)
            ax.set_title(t("最强发射源归属（空白 = 无覆盖）", "Strongest transmitter (blank = no coverage)"), fontsize=11)
        else:
            values = result.metrics[name]
            stats = result.ranges[name]
            vmin = stats["vmin"] if stats["vmin"] is not None else -140.0
            vmax = stats["vmax"] if stats["vmax"] is not None else -40.0
            if vmax - vmin < 1e-6:
                vmax = vmin + 1.0
            image = ax.imshow(values, origin="lower", extent=extent, interpolation="nearest",
                              cmap=cmap, vmin=vmin, vmax=vmax)
            bar = fig.colorbar(image, ax=ax, fraction=0.046)
            bar.set_label(f"{t(METRIC_META[name]['label'], _EN_LABEL.get(name, name))} · {METRIC_META[name]['unit']}", fontsize=9)
            ax.set_title(t(METRIC_META[name]["label"], _EN_LABEL.get(name, name)), fontsize=11)
        if tx_xy.size:
            ax.scatter(tx_xy[:, 0], tx_xy[:, 1], marker="^", s=90, c="white",
                       edgecolors="#1b2733", linewidths=1.2, zorder=5, label=t("发射源", "Transmitter"))
        if rx_xy.size:
            ax.scatter(rx_xy[:, 0], rx_xy[:, 1], marker="s", s=64, c="none",
                       edgecolors="#129e87", linewidths=1.6, zorder=5, label=t("监测站", "Monitoring station"))
        ax.set_xlabel("X · m", fontsize=9)
        ax.set_ylabel("Y · m", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="box")

    for slot in range(len(panels), rows * cols):
        axes[slot // cols][slot % cols].axis("off")

    axes[0][0].legend(loc="upper right", fontsize=8, framealpha=0.85)
    if cjk:
        head = f"{title} · 电磁态势图（Radio Map）"
        meta = (f"场景 {result.scene} · 测量高度 {grid['height_m']:g} m · 网格 {grid['cell_size_m']:g} m · "
                f"{grid['ny']}×{grid['nx']} 格 · 每源 {result.samples_per_tx:,} 条采样 · "
                f"最大交互深度 {result.max_depth} · 求解 {result.seconds:.1f} s")
    else:
        head = f"{title} · Radio Map"
        meta = (f"scene {result.scene} · height {grid['height_m']:g} m · cell {grid['cell_size_m']:g} m · "
                f"{grid['ny']}x{grid['nx']} cells · {result.samples_per_tx:,} samples/tx · "
                f"max depth {result.max_depth} · solved in {result.seconds:.1f} s")
    fig.suptitle(f"{head}\n{meta}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
