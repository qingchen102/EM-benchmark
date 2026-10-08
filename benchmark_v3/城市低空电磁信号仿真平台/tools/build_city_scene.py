"""OSM Overpass → 城市 AOI LoD1 场景（Sionna RT + 网页地图）。

用法（在本平台目录下，任意 python，只需 urllib+numpy）：
    python tools/build_city_scene.py --name hangzhou --bbox 30.270,120.120,30.278,120.130

产物（-name 即 <name>）：
    priors/<name>_geometry/scene.xml + meshes/*.ply     # Sionna RT 自定义场景（ITU 混凝土）
    priors/<name>_geometry/mesh.npz + geometry.json     # 几何校验用（与 export_map.py 同约定）
    webui/assets/<name>_mesh.bin/.json                  # 网页 2D/3D 地图

坐标：AOI 中心为原点的 ENU（x 东、y 北、z 天，米），与场景里设备 position_m 同一世界系。
高度：OSM height 标签优先，其次 building:levels×3 m，缺省 10 m。
数据：© OpenStreetMap contributors（ODbL）；发布衍生图需注明来源。
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # 平台根目录
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = [OVERPASS,
                    "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter"]
UA = "urban-lowalt-rf-city-scene/0.1 (research; benchmark AOI)"
GROUND_MARGIN_M = 50.0
DEFAULT_HEIGHT_M = 10.0
LEVEL_HEIGHT_M = 3.0
MIN_BUILDING_HEIGHT_M = 2.0


# --------------------------------------------------------------------------- #
#  Overpass 下载与解析
# --------------------------------------------------------------------------- #
def fetch_buildings(bbox) -> list[dict]:
    s, w, n, e = bbox
    q = (f"[out:json][timeout:90];way['building']({s},{w},{n},{e});(._;>;);out body;")
    best = None                                   # 镜像不稳定：全部都试，取结果最多的一份
    cache = ROOT / "priors" / "overpass_cache.json"
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("bbox") == list(bbox) and "elements" in cached:
                print("  使用本地 Overpass 缓存（--refresh 可强制重新下载）")
                return _parse_buildings(cached), cached
        except (json.JSONDecodeError, OSError):
            pass
    for endpoint in OVERPASS_MIRRORS:
        data = None
        req = urllib.request.Request(endpoint, data=q.encode(),
                                     headers={"User-Agent": UA,
                                              "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                data = json.load(r)
        except Exception as exc:
            print(f"  urllib 失败（{exc}），尝试 curl …")
            data = _fetch_via_curl(endpoint, q)
        if not data or "elements" not in data:
            continue
        cand, raw = _parse_buildings(data)
        print(f"  {endpoint.split('/')[2]}: {len(cand)} 栋")
        if best is None or len(cand) > len(best[0]):
            best = (cand, raw)
    if not best:
        raise RuntimeError("所有 Overpass 镜像均失败或结果为空，稍后重试或缩小 bbox")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"bbox": list(bbox), **best[1]}, ensure_ascii=False), encoding="utf-8")
    return best[0], best[1]


def _parse_buildings(data) -> tuple[list[dict], dict]:
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"] == "node"}
    out = []
    for el in data["elements"]:
        if el["type"] != "way" or "building" not in el.get("tags", {}):
            continue
        ring = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
        if len(ring) < 4:                       # 未闭合或退化
            continue
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        tags = el.get("tags", {})
        h = _height_from_tags(tags)
        if 0 < h < MIN_BUILDING_HEIGHT_M:
            continue
        out.append({"ring": ring, "height_m": h, "osm_id": el["id"],
                    "name": tags.get("name", "")})
    return out, data


def _fetch_via_curl(endpoint: str, query: str):
    """部分网络下 python 直连被掐，curl 能走通时自动回退。"""
    import subprocess
    try:
        proc = subprocess.run(["curl", "-s", "--max-time", "150", "-A", UA,
                               "-d", f"data={query}", endpoint],
                              capture_output=True, timeout=160)
        return json.loads(proc.stdout.decode("utf-8"))
    except Exception as exc:
        print(f"  curl 也失败（{exc}）")
        return None


def _height_from_tags(tags) -> float:
    raw = tags.get("height")
    if raw:
        try:
            return float(str(raw).strip().rstrip("m"))
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return float(levels) * LEVEL_HEIGHT_M
        except ValueError:
            pass
    return 0.0                                   # 交给合成模型


def _synthetic_height(ring, osm_id, rng) -> float:
    """确定性合成高度：按占地大小分层 + 少量高楼（中国城市形态），可复现。

    真实楼高在开放数据里对中国城市基本缺失（OSM ~99% 缺、Overture 2/7726 有值），
    故未标注的楼用此模型；几何完全真实，高度为合理合成并在元数据中标注。
    """
    area = abs(_signed_area(ring))
    p = rng.random()
    if rng.random() < 0.04:                      # 少量高层/塔楼
        return float(rng.uniform(40, 120))
    if area > 5000:
        return float(rng.uniform(15, 35))        # 大型公共/商业
    if area > 1500:
        return float(rng.uniform(9, 24))         # 多层
    if p < 0.15:
        return float(rng.uniform(12, 27))        # 小高层
    return float(rng.uniform(5, 12))             # 低层住宅


# --------------------------------------------------------------------------- #
#  ENU 投影与三角化
# --------------------------------------------------------------------------- #
def to_enu(rings, lat0, lon0):
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    return [[((lon - lon0) * kx, (lat - lat0) * ky) for lon, lat in ring]
            for ring in rings]


def _signed_area(pts) -> float:
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                     - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


def ear_clip(pts) -> list[tuple[int, int, int]]:
    """简单多边形耳切三角化（无孔洞）。pts: [(x,y)]，返回顶点索引三角。"""
    idx = list(range(len(pts)))
    if _signed_area([pts[i] for i in idx]) < 0:
        idx.reverse()
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10 * len(idx):
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if cross <= 1e-9:                        # 非凸耳
                continue
            bad = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts[j]
                if _in_tri(p, a, b, c):
                    bad = True
                    break
            if not bad:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break                                    # 数值退化，放弃剩余
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _in_tri(p, a, b, c) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


# --------------------------------------------------------------------------- #
#  网格组装
# --------------------------------------------------------------------------- #
def build_meshes(buildings, bbox, lat0, lon0) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (triangles[N,3,3] f32 米/ENU, object_ids[N] i32: 0=地面 1..=建筑, stats)。"""
    s, w, n, e = bbox
    rings_enu = to_enu([b["ring"] for b in buildings], lat0, lon0)
    tris: list[tuple] = []
    ids: list[int] = []
    n_synth = n_tag = 0

    minx = (w - lon0) * 111320.0 * math.cos(math.radians(lat0)) - GROUND_MARGIN_M
    maxx = (e - lon0) * 111320.0 * math.cos(math.radians(lat0)) + GROUND_MARGIN_M
    miny = (s - lat0) * 110540.0 - GROUND_MARGIN_M
    maxy = (n - lat0) * 110540.0 + GROUND_MARGIN_M
    g = [(minx, miny, 0.0), (maxx, miny, 0.0), (maxx, maxy, 0.0), (minx, maxy, 0.0)]
    for t in ((0, 1, 2), (0, 2, 3)):
        tris.append(tuple(g[i] for i in t))
        ids.append(0)

    for bi, (b, ring) in enumerate(zip(buildings, rings_enu), start=1):
        if b["height_m"] > 0:
            h = b["height_m"]
            n_tag += 1
        else:
            h = _synthetic_height(ring, b["osm_id"],
                                  np.random.default_rng(b["osm_id"] & 0x7FFFFFFF))
            n_synth += 1
        b["height_m"] = h
        pts2 = [(x, y) for x, y in ring]
        base = [(x, y, 0.0) for x, y in pts2]
        top = [(x, y, h) for x, y in pts2]
        for i0, i1, i2 in ear_clip(pts2):
            tris.append((top[i0], top[i1], top[i2]))
            ids.append(bi)
        m = len(pts2)
        for k in range(m):
            a0, b0 = base[k], base[(k + 1) % m]
            a1, b1 = top[k], top[(k + 1) % m]
            tris.append((a0, b0, b1))
            ids.append(bi)
            tris.append((a0, b1, a1))
            ids.append(bi)
    stats = {"buildings": len(buildings), "height_tagged": n_tag, "height_synthetic": n_synth}
    return np.asarray(tris, dtype=np.float32), np.asarray(ids, dtype=np.int32), stats


# --------------------------------------------------------------------------- #
#  写文件
# --------------------------------------------------------------------------- #
def write_ply(path: Path, tris: np.ndarray) -> None:
    """Mitsuba PLY（二进制小端）：顶点 float32 + 面(uchar 3, int32×3)。"""
    verts, lut = [], {}
    for t in tris:
        for v in t:
            key = tuple(np.round(v, 4))
            if key not in lut:
                lut[key] = len(verts)
                verts.append(key)
    faces = [tuple(lut[tuple(np.round(v, 4))] for v in t) for t in tris]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
              "end_header\n").encode("ascii")
    vbuf = b"".join(struct.pack("<fff", *v) for v in verts)
    fbuf = b"".join(struct.pack("<B3i", 3, *f) for f in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + vbuf + fbuf)


def write_scene_xml(path: Path, name: str) -> None:
    """Mitsuba 场景（格式对齐慕尼黑 scene 2.1.0）：建筑与地面均为 ITU 混凝土。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<scene version="2.1.0">
	<bsdf type="itu-radio-material" id="concrete">
		<string name="type" value="concrete" />
		<float name="thickness" value="0.1" />
	</bsdf>
	<shape type="ply" id="mesh-{name}-ground">
		<string name="filename" value="meshes/ground.ply" />
		<ref id="concrete" name="bsdf" />
	</shape>
	<shape type="ply" id="mesh-{name}-buildings">
		<string name="filename" value="meshes/buildings.ply" />
		<ref id="concrete" name="bsdf" />
	</shape>
</scene>
""", encoding="utf-8")


def write_web_mesh(path_bin: Path, path_json: Path, tris: np.ndarray, name: str) -> None:
    path_bin.parent.mkdir(parents=True, exist_ok=True)
    tris.tofile(str(path_bin))
    meta = {"name": name, "format": "float32 little endian triangles[N,3,3]",
            "triangle_count": int(len(tris)),
            "bounds": [tris.reshape(-1, 3).min(0).tolist(), tris.reshape(-1, 3).max(0).tolist()],
            "source": "© OpenStreetMap contributors (ODbL); LoD1 extrusion"}
    path_json.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def write_priors(path_npz: Path, path_json: Path, tris: np.ndarray, ids: np.ndarray,
                 name: str, lat0: float, lon0: float, bbox, n_buildings: int,
                 stats: dict | None = None) -> None:
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_npz, triangles=tris, object_ids=ids)
    meta = {"schema_version": "1.0", "scene": name,
            "coordinate_frame": "ENU at AOI centre (x east, y north, z up), metres, not georeferenced",
            "origin_latlon": [lat0, lon0], "bbox_swne": list(bbox),
            "building_count": n_buildings,
            "triangle_count": int(len(tris)),
            "materials": {"buildings": "itu-radio-material concrete",
                          "ground": "itu-radio-material concrete"},
            "heights": stats or {},
            "height_model": "OSM height/building:levels 优先；缺失部分按占地分层合成"
                            "（低层5-12m/多层9-24m/大型15-35m/4%塔楼40-120m），按 osm_id 可复现",
            "source": "© OpenStreetMap contributors (ODbL)",
            "note": "RT 使用同目录 scene.xml；本文件仅供几何校验与网页显示"}
    path_json.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="hangzhou")
    ap.add_argument("--bbox", default="30.270,120.120,30.278,120.130",
                    help="south,west,north,east（度）")
    ap.add_argument("--limit", type=int, default=0, help="最多建筑数（按面积取大者优先），0=不限")
    ap.add_argument("--default-height", type=float, default=DEFAULT_HEIGHT_M)
    args = ap.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3], "bbox 应为 south,west,north,east"
    lat0 = (bbox[0] + bbox[2]) / 2
    lon0 = (bbox[1] + bbox[3]) / 2

    print(f"下载 OSM 建筑 …（{bbox}）")
    buildings, raw = fetch_buildings(bbox)
    if args.limit and len(buildings) > args.limit:
        buildings.sort(key=lambda b: -abs(_signed_area(to_enu([b["ring"]], lat0, lon0)[0])))
        buildings = buildings[: args.limit]
    print(f"建筑 {len(buildings)} 栋；拉伸并三角化 …")
    tris, ids, stats = build_meshes(buildings, bbox, lat0, lon0)
    rings_enu = to_enu([b["ring"] for b in buildings], lat0, lon0)
    footprints = [{"poly": [[round(x, 2), round(y, 2)] for x, y in ring],
                   "height_m": round(b["height_m"], 2), "osm_id": b["osm_id"]}
                  for b, ring in zip(buildings, rings_enu)]
    (priors / "footprints.json").write_text(
        json.dumps({"coordinate_frame": "ENU at AOI centre, metres",
                    "origin_latlon": [lat0, lon0], "buildings": footprints},
                   ensure_ascii=False), encoding="utf-8")
    print(f"三角面 {len(tris)}，顶点范围 "
          f"{tris.reshape(-1,3).min(0).round(1)} ~ {tris.reshape(-1,3).max(0).round(1)}")

    priors = ROOT / "priors" / f"{args.name}_geometry"
    write_ply(priors / "meshes" / "buildings.ply", tris[ids > 0])
    ground = tris[ids == 0]
    write_ply(priors / "meshes" / "ground.ply", ground)
    write_scene_xml(priors / "scene.xml", args.name)
    write_priors(priors / "mesh.npz", priors / "geometry.json", tris, ids,
                 args.name, lat0, lon0, bbox, len(buildings), stats)
    write_web_mesh(ROOT / "webui" / "assets" / f"{args.name}_mesh.bin",
                   ROOT / "webui" / "assets" / f"{args.name}_mesh.json",
                   tris, args.name)
    kb = sum(f.stat().st_size for f in priors.rglob("*") if f.is_file()) / 1024
    print(f"完成：priors/{args.name}_geometry/（{kb:.0f} KB）+ webui/assets/{args.name}_mesh.*")
    print(f"RT 场景文件：priors/{args.name}_geometry/scene.xml（Propagation.scene 填这个路径）")


if __name__ == "__main__":
    main()