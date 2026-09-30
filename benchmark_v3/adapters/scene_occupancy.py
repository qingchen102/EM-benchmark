"""从 mesh.npz 构建楼宇占用栅格，供设备布点做"不在楼内"拒绝采样。

用法：
    python scene_occupancy.py            # 构建/刷新 priors/hangzhou_geometry/occupancy.npz
API（供数据集生成器 import）：
    occ = SceneOccupancy.load(<platform>/priors/hangzhou_geometry/occupancy.npz)
    occ.is_free(x, y)                   # 点是否在楼外（含墙体安全边距）
    occ.free_point(rng)                 # 随机采一个楼外点
    occ.height_at(x, y)                 # 该处楼高（楼外为 0）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CELL = 4.0          # 栅格分辨率 m
_MARGIN = 5.0        # 离墙安全边距 m


def build_from_mesh(mesh_npz: str | Path, out_npz: str | Path) -> None:
    d = np.load(mesh_npz)
    tris, ids = d["triangles"], d["object_ids"]
    bt = tris[ids > 0]                                   # 仅建筑
    allv = tris.reshape(-1, 3)
    lo, hi = allv[:, :2].min(0), allv[:, :2].max(0)
    nx = int(np.ceil((hi[0] - lo[0]) / _CELL)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / _CELL)) + 1
    occ = np.zeros((nx, ny), dtype=np.uint8)
    hmax = np.zeros((nx, ny), dtype=np.float32)
    for tri in bt:                                       # 屋顶+墙面 xy 投影
        v = tri[:, :2]
        z = tri[:, 2]
        i0 = np.maximum(((v.min(0) - lo) / _CELL).astype(int) - 1, 0)
        i1 = np.minimum(((v.max(0) - lo) / _CELL).astype(int) + 2, [nx, ny])
        if i1[0] <= i0[0] or i1[1] <= i0[1]:
            continue
        gx, gy = np.meshgrid(np.arange(i0[0], i1[0]), np.arange(i0[1], i1[1]), indexing="ij")
        cx = lo[0] + (gx + .5) * _CELL
        cy = lo[1] + (gy + .5) * _CELL
        d0 = (v[1, 0] - v[0, 0]) * (cy - v[0, 1]) - (v[1, 1] - v[0, 1]) * (cx - v[0, 0])
        d1 = (v[2, 0] - v[1, 0]) * (cy - v[1, 1]) - (v[2, 1] - v[1, 1]) * (cx - v[1, 0])
        d2 = (v[0, 0] - v[2, 0]) * (cy - v[2, 1]) - (v[0, 1] - v[2, 1]) * (cx - v[2, 0])
        inside = ((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) | ((d0 <= 0) & (d1 <= 0) & (d2 <= 0))
        sub_occ = occ[i0[0]:i1[0], i0[1]:i1[1]]
        sub_h = hmax[i0[0]:i1[0], i0[1]:i1[1]]
        sub_occ[inside] = 1
        sub_h[inside] = np.maximum(sub_h[inside], z.max())
    # 膨胀 margin（近似：按格数）
    mcells = int(np.ceil(_MARGIN / _CELL))
    if mcells:
        from scipy.ndimage import binary_dilation
        occ = binary_dilation(occ, iterations=mcells).astype(np.uint8)
    lo_full = np.r_[lo, 0.0]
    hi_full = np.r_[hi, float(hmax.max())]
    np.savez_compressed(out_npz, occ=occ, hmax=hmax, origin=lo_full.astype(np.float64),
                        cell=_CELL)
    print(f"occupancy: {nx}x{ny} cells, 占用 {occ.mean()*100:.1f}%, 楼高 max {hmax.max():.0f} m")


class SceneOccupancy:
    def __init__(self, occ, hmax, origin, cell):
        self.occ, self.hmax, self.origin, self.cell = occ, hmax, origin, cell
        self.nx, self.ny = occ.shape
        self.extent = (origin[0], origin[1], origin[0] + (self.nx - 1) * cell,
                       origin[1] + (self.ny - 1) * cell)

    @classmethod
    def load(cls, path: str | Path) -> "SceneOccupancy":
        d = np.load(path)
        return cls(d["occ"], d["hmax"], d["origin"], float(d["cell"]))

    def _cell(self, x, y):
        ix = int((x - self.origin[0]) / self.cell)
        iy = int((y - self.origin[1]) / self.cell)
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return None
        return ix, iy

    def is_free(self, x, y) -> bool:
        c = self._cell(x, y)
        return c is not None and self.occ[c] == 0

    def height_at(self, x, y) -> float:
        c = self._cell(x, y)
        return float(self.hmax[c]) if c else 0.0

    def free_point(self, rng, *, z_low=5.0, z_high=80.0, tries=200) -> tuple | None:
        """随机楼外点，返回 (x, y, z)；z 为离地高度。"""
        for _ in range(tries):
            x = float(rng.uniform(self.extent[0] + 20, self.extent[2] - 20))
            y = float(rng.uniform(self.extent[1] + 20, self.extent[3] - 20))
            if self.is_free(x, y):
                return (x, y, float(rng.uniform(z_low, z_high)))
        return None

    def free_point_near(self, rng, cx, cy, r_low, r_high, *, z_low=5.0, z_high=60.0,
                        tries=300) -> tuple | None:
        """距 (cx,cy) r_low~r_high 的楼外随机点。"""
        for _ in range(tries):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(r_low, r_high)
            x, y = cx + r * np.cos(a), cy + r * np.sin(a)
            if self.is_free(x, y):
                return (x, y, float(rng.uniform(z_low, z_high)))
        return None

    def los_blocked(self, p1, p2) -> bool:
        """连线是否被楼体遮挡（射线在该点的高度低于当地楼高）。"""
        a = np.array(p1[:2], float)
        b = np.array(p2[:2], float)
        d = float(np.linalg.norm(b - a))
        n = max(int(d / (self.cell / 2)), 2)
        ts = np.linspace(0.0, 1.0, n)
        xs = p1[0] + (b[0] - a[0]) * ts
        ys = p1[1] + (b[1] - a[1]) * ts
        zs = p1[2] + (p2[2] - p1[2]) * ts
        for x, y, z in zip(xs[1:-1], ys[1:-1], zs[1:-1]):
            c = self._cell(x, y)
            if c is not None and self.occ[c] and self.hmax[c] > z:
                return True
        return False

    def target_nlos(self, rng, stations, *, z_low=30, z_high=80, min_blocked=1,
                    tries=60) -> tuple | None:
        """D3/D4：把目标放到相对监测网的楼后（至少 min_blocked 站 NLoS）。"""
        for _ in range(tries):
            p = self.free_point(rng, z_low=z_low, z_high=z_high, tries=80)
            if p is None:
                continue
            blocked = sum(self.los_blocked(p, s) for s in stations)
            if blocked >= min_blocked:
                return p
        return None


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    mesh = here.parent / "城市低空电磁信号仿真平台" / "priors" / "hangzhou_geometry" / "mesh.npz"
    out = here.parent / "城市低空电磁信号仿真平台" / "priors" / "hangzhou_geometry" / "occupancy.npz"
    build_from_mesh(mesh, out)