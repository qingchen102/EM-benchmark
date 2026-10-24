"""Radio map（电磁态势 / 覆盖图）——基于 Sionna RT 2.1 的测量平面求解。

Radio map 把一块水平测量平面切成网格，每个网格给出该格内平均的**信道量**，
并假设格点上放着一个理想各向同性接收机：

    g_i = (1/|C_i|) * ∫_{C_i} |h(s)|² ds

它是功率的**空间**分布，没有频率维——因此是覆盖图 / 电磁态势图，而不是严格
意义的「频谱地图」。求解成本主要由射线采样数决定，几乎与网格分辨率无关。

场景加载与天空可见性检查与 :mod:`cityrf.sionna_backend` 完全一致，保证
radio map 与逐径仿真描述的是同一个物理场景。本模块不依赖 Web 层。
"""
from __future__ import annotations

import importlib.metadata
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .config import Scenario
from .sionna_backend import _load_scene, _numpy

Metric = Literal["path_gain", "rss", "sinr"]

#: 指标显示名、单位与口径说明（前端与 PNG 图例共用）。
METRIC_META: dict[str, dict[str, str]] = {
    "path_gain": {
        "label": "路径增益 · 最佳源",
        "unit": "dB",
        "hint": "各发射源中最大的平均路径增益（线性功率，未含发射功率）。",
    },
    "rss": {
        "label": "接收信号强度 · 全部源",
        "unit": "dBm",
        "hint": "所有发射源接收功率之和，已按各源名义发射功率换算。",
    },
    "sinr": {
        "label": "信干噪比 · 最佳源",
        "unit": "dB",
        "hint": "各源 SINR 中的最大值：信号 /（其余源干扰 + 噪声）。",
    },
}

#: 网格上限。分辨率本身几乎不影响求解耗时，但浏览器要传输、渲染并做悬停查询。
MAX_CELLS = 300_000

#: 与前端共用的色带锚点（深紫 → 蓝 → 青 → 绿 → 黄 → 橙 → 红）。
RAMP_STOPS: tuple[tuple[float, float, float], ...] = (
    (0.19, 0.07, 0.23),
    (0.27, 0.31, 0.71),
    (0.10, 0.60, 0.79),
    (0.10, 0.84, 0.60),
    (0.55, 0.94, 0.27),
    (0.93, 0.79, 0.20),
    (0.97, 0.50, 0.13),
    (0.80, 0.16, 0.08),
    (0.48, 0.02, 0.01),
)


class RadioMapRequest(BaseModel):
    """一次 radio map 计算的参数（与 Scenario 分开，因为它不是仿真产物）。"""

    height_m: float = Field(default=1.5, ge=-1000, le=5000)
    cell_size_m: float = Field(default=5.0, gt=0, le=500)
    size_m: tuple[float, float] | None = None        # None = 覆盖整个场景
    center_xy_m: tuple[float, float] | None = None   # None = 场景中心
    samples_per_tx: int = Field(default=10_000_000, ge=1_000, le=100_000_000)
    max_depth: int = Field(default=3, ge=0, le=10)
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    metrics: list[Metric] = Field(default_factory=lambda: ["path_gain", "rss", "sinr"])

    @model_validator(mode="after")
    def _clean(self):
        seen, ordered = set(), []
        for name in self.metrics:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        if not ordered:
            raise ValueError("metrics 不能为空，至少选择一个指标")
        self.metrics = ordered
        if self.size_m is not None and min(self.size_m) <= 0:
            raise ValueError("size_m 的两个分量必须为正")
        return self


@dataclass
class RadioMapResult:
    """求解结果：全部数值都在内存里，落盘由调用方决定。"""

    grid: dict
    metrics: dict[str, np.ndarray]   # 2D (ny, nx) float64，dB / dBm，无效格为 NaN
    association: np.ndarray          # 2D (ny, nx) int32，-1 表示该格没有任何源
    tx_ids: list[str]
    ranges: dict[str, dict]
    seconds: float
    scene: str
    samples_per_tx: int
    max_depth: int
    metrics_requested: list[str]


def _db(linear: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """线性功率 → dB，无效格给 NaN。"""
    out = np.full(linear.shape, np.nan, dtype=np.float64)
    good = valid & np.isfinite(linear) & (linear > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[good] = 10.0 * np.log10(linear[good])
    return out


def _dbm(watts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return _db(watts * 1000.0, valid)


def _measurement_plane(scene, request: RadioMapRequest) -> tuple[list, list, int, int]:
    """把「None = 整个场景」解析成显式 center/size，并校验网格规模。"""
    box = scene.mi_scene.bbox()
    lower, upper = np.array(box.min, dtype=np.float64), np.array(box.max, dtype=np.float64)
    if request.center_xy_m is None:
        cx, cy = (lower[0] + upper[0]) / 2.0, (lower[1] + upper[1]) / 2.0
    else:
        cx, cy = request.center_xy_m
    if request.size_m is None:
        sx, sy = float(upper[0] - lower[0]), float(upper[1] - lower[1])
    else:
        sx, sy = request.size_m
    cell = request.cell_size_m
    nx, ny = int(np.ceil(sx / cell)), int(np.ceil(sy / cell))
    if nx * ny > MAX_CELLS:
        raise ValueError(
            f"网格 {ny}×{nx} = {nx * ny:,} 格超过上限 {MAX_CELLS:,}；"
            f"请把网格尺寸调大，或缩小测量平面尺寸。"
        )
    return [float(cx), float(cy), float(request.height_m)], [float(sx), float(sy)], nx, ny


def _check_sky_visibility(scene, scenario: Scenario, mi) -> None:
    """与逐径后端同一套保守规则：端点必须在场景内、地面以上、且正上方无遮挡。"""
    if scenario.propagation.scene == "empty":
        return
    box = scene.mi_scene.bbox()
    lower, upper = np.array(box.min), np.array(box.max)
    for dev in scenario.emitters:
        x, y, z = dev.position_m
        if not (lower[0] < x < upper[0] and lower[1] < y < upper[1] and z > 0):
            raise ValueError(f"{dev.id}: outside scene XY bounds or below ground")
        hit = scene.mi_scene.ray_intersect(mi.Ray3f(o=mi.Point3f(x, y, z), d=mi.Vector3f(0, 0, 1)))
        if bool(_numpy(hit.is_valid()).reshape(-1)[0]):
            raise ValueError(f"{dev.id}: endpoint is not sky-visible (interior/roof/overhang)")


def compute_radio_map(
    scenario: Scenario,
    request: RadioMapRequest,
    emit: Callable[[str], None] | None = None,
) -> RadioMapResult:
    """在 ``scenario`` 描述的场景上求解 ``request`` 指定的 radio map。

    只使用发射源：测量面上的每个格点本身就是接收位置，因此场景里的监测站
    不参与求解，只用于前端叠加显示。
    """
    log = emit or (lambda _message: None)
    started = time.perf_counter()

    try:
        import mitsuba as mi
        mi.set_variant(scenario.propagation.rt_variant)
        import sionna.rt as rt
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Sionna RT unavailable: install the rt extra and configure the requested Mitsuba backend"
        ) from exc
    if importlib.metadata.version("sionna-rt") != "2.1.0":
        raise RuntimeError("Sionna RT adapter requires version 2.1.0")

    p = scenario.propagation
    if p.backend != "sionna_rt":
        raise ValueError(
            f"电磁态势图需要 Sionna RT 后端（当前 backend='{p.backend}'）；"
            "自由空间与抽头后端不利用建筑，无法给出城市电磁态势。"
        )

    log(f"[1/4] 加载场景 {p.scene}，装配 {len(scenario.emitters)} 个发射源")
    scene = _load_scene(p.scene)
    scene.frequency = scenario.carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for tx in scenario.emitters:
        device = rt.Transmitter(tx.id, position=tx.position_m, orientation=tx.orientation_rad)
        device.power_dbm = tx.power_dbm
        scene.add(device)
    _check_sky_visibility(scene, scenario, mi)

    center, size, nx, ny = _measurement_plane(scene, request)
    log(
        f"[2/4] 求解测量平面 {size[0]:.0f}×{size[1]:.0f} m @ {request.cell_size_m:g} m，"
        f"高度 {request.height_m:g} m，{ny}×{nx} 格，每源 {request.samples_per_tx:,} 条采样"
    )
    solver = rt.RadioMapSolver()
    radio_map = solver(
        scene,
        center=center,
        orientation=[0.0, 0.0, 0.0],
        size=size,
        cell_size=[request.cell_size_m, request.cell_size_m],
        samples_per_tx=request.samples_per_tx,
        max_depth=request.max_depth,
        los=request.los,
        specular_reflection=request.specular_reflection,
        diffuse_reflection=request.diffuse_reflection,
        refraction=request.refraction,
        diffraction=request.diffraction,
        seed=scenario.seed,
    )
    # drjit 是惰性求值：求解真正发生在下面这几次取值里，计时必须包住它们。
    path_gain = np.asarray(_numpy(radio_map.path_gain), dtype=np.float64)
    rss = np.asarray(_numpy(radio_map.rss), dtype=np.float64)
    sinr = np.asarray(_numpy(radio_map.sinr), dtype=np.float64)
    association = np.asarray(_numpy(radio_map.tx_association()), dtype=np.int32)
    centers = np.asarray(_numpy(radio_map.cell_centers), dtype=np.float64)

    log("[3/4] 汇总指标（最佳源路径增益 / 全源接收功率 / 最佳源 SINR）")
    with warnings.catch_warnings():
        # 全遮挡的格会让 nanmax 报「All-NaN slice encountered」，这里是有意为之。
        warnings.simplefilter("ignore", RuntimeWarning)
        if path_gain.ndim == 3:
            best_gain = np.nanmax(np.where(path_gain > 0, path_gain, np.nan), axis=0)
            total_rss = np.nansum(rss, axis=0)
            best_sinr = np.nanmax(np.where(sinr > 0, sinr, np.nan), axis=0)
        else:
            best_gain, total_rss, best_sinr = path_gain, rss, sinr

    valid = association >= 0
    values: dict[str, np.ndarray] = {}
    if "path_gain" in request.metrics:
        values["path_gain"] = _db(best_gain, valid)
    if "rss" in request.metrics:
        values["rss"] = _dbm(total_rss, valid)
    if "sinr" in request.metrics:
        values["sinr"] = _db(best_sinr, valid)

    # 归一到「行 = y 递增、列 = x 递增」，前端与 PNG 都按这个约定取网格。
    xs, ys = centers[:, :, 0], centers[:, :, 1]
    row_order, col_order = np.argsort(ys[:, 0]), np.argsort(xs[0, :])
    if not (
        np.array_equal(row_order, np.arange(row_order.size))
        and np.array_equal(col_order, np.arange(col_order.size))
    ):
        index = np.ix_(row_order, col_order)
        for name in values:
            values[name] = values[name][index]
        association = association[index]
        xs, ys = xs[index], ys[index]

    cell = float(request.cell_size_m)
    grid = {
        "origin_xy_m": [float(xs.min() - cell / 2.0), float(ys.min() - cell / 2.0)],
        "cell_size_m": cell,
        "nx": int(xs.shape[1]),
        "ny": int(xs.shape[0]),
        "height_m": float(request.height_m),
        "x_min_m": float(xs.min()), "x_max_m": float(xs.max()),
        "y_min_m": float(ys.min()), "y_max_m": float(ys.max()),
    }

    ranges: dict[str, dict] = {}
    for name, array in values.items():
        flat = array[np.isfinite(array)]
        if flat.size:
            ranges[name] = {
                "min": float(flat.min()), "max": float(flat.max()),
                "vmin": float(np.percentile(flat, 2)), "vmax": float(np.percentile(flat, 98)),
                "valid_count": int(flat.size), "total_count": int(array.size),
            }
        else:
            ranges[name] = {"min": None, "max": None, "vmin": None, "vmax": None,
                            "valid_count": 0, "total_count": int(array.size)}

    seconds = time.perf_counter() - started
    covered = int(valid.sum())
    log(
        f"[4/4] 求解完成：{seconds:.2f} s，覆盖 {covered:,}/{valid.size:,} 格"
        f"（{100.0 * covered / max(valid.size, 1):.1f}%），"
        f"{len(scenario.emitters)} 个源"
    )
    return RadioMapResult(
        grid=grid, metrics=values, association=association,
        tx_ids=[tx.id for tx in scenario.emitters], ranges=ranges,
        seconds=seconds, scene=p.scene, samples_per_tx=request.samples_per_tx,
        max_depth=request.max_depth, metrics_requested=list(request.metrics),
    )


def _encode(values: np.ndarray) -> list:
    """NaN → null 的 JSON 友好列表。Python 的 json 默认会写裸 NaN，浏览器解析不了。"""
    flat = np.round(values, 1).reshape(-1)
    items = flat.tolist()
    for i in np.flatnonzero(~np.isfinite(flat)):
        items[int(i)] = None
    return items


def to_payload(result: RadioMapResult, scenario: Scenario) -> dict:
    """浏览器与 PNG 共用的可序列化载荷。"""
    association = result.association.reshape(-1)
    return {
        "grid": result.grid,
        "metrics": {
            name: dict(METRIC_META[name], data=_encode(result.metrics[name]), **result.ranges[name])
            for name in result.metrics_requested
            if name in result.metrics
        },
        "association": {
            "tx_ids": result.tx_ids,
            "data": [None if int(v) < 0 else int(v) for v in association],
        },
        "emitters": [
            dict(id=tx.id, role=tx.role, power_dbm=tx.power_dbm,
                 position_m=list(tx.position_m), preset=tx.preset)
            for tx in scenario.emitters
        ],
        "receivers": [
            dict(id=rx.id, position_m=list(rx.position_m))
            for rx in scenario.receivers
        ],
        "scene": result.scene,
        "samples_per_tx": result.samples_per_tx,
        "max_depth": result.max_depth,
        "seconds": result.seconds,
    }


def write_npz(result: RadioMapResult, path: Path) -> None:
    """完整精度落盘；浏览器载荷是四舍五入到 0.1 的显示副本。"""
    payload = {f"metric_{name}": values for name, values in result.metrics.items()}
    payload["association"] = result.association
    payload["cell_size_m"] = np.float64(result.grid["cell_size_m"])
    payload["origin_xy_m"] = np.asarray(result.grid["origin_xy_m"], dtype=np.float64)
    payload["height_m"] = np.float64(result.grid["height_m"])
    np.savez_compressed(Path(path), **payload)
