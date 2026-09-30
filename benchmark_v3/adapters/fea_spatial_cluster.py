"""多站空间聚类可行性验证（零 token，先验证再接入）。

对重度 crosstalk 的 D2 样本：每站对残差测向（u 空间多峰）→ RANSAC 跨站交会聚类
→ 得到 K 个源簇。对比 K vs GT 源数 vs 单站 estimate_num_sources。

用法：
    python fea_spatial_cluster.py ../dataset_d2 d2_00002
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_capture_agent as rca  # noqa: E402
import capture_tools as ct  # noqa: E402
from cityrf.baselines.localization import intersect_rays  # noqa: E402


def _u_to_rays(u_list, pos, axis):
    """u→世界 3D 单位方向（单臂锥面，含镜象两分支）。"""
    rays = []
    for u in u_list:
        az = np.arcsin(np.clip(u, -1, 1))
        # 阵轴=axis，方向 = cos(el)*cos/sin(...) 近似 el≈0；两镜像（u 与 180-u 同响应）
        d = axis * np.cos(az) + 0.0
        # 构造世界方向：绕垂直轴的锥面，取两个候选方位
        # 简化：axis=(a_hat)，u=cos(θ) 其中 θ 是相对 axis 的角度
        # 方向在"过 axis 且与世界任意垂直向量张成的平面"上取两支
        v_ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v1 = np.cross(axis, v_ref)
        v1 = v1 / (np.linalg.norm(v1) + 1e-12)
        v2 = np.cross(axis, v1)
        sin_t = np.sqrt(max(1 - u * u, 0.0))
        rays.append(pos + (axis * u + v1 * sin_t))          # 支1
        rays.append(pos + (axis * u - v1 * sin_t))          # 支2（镜象）
        # 也用 v2 平面扫一遍（全锥面）——主支取相对正 z 的两支即可
    return rays


def cluster(root: Path, sid: str) -> dict:
    cap = root / sid / "capture"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    private = cap / "private"
    num_samples = obs["receivers"][0]["shape"][1]
    truth = json.loads((private / "d1_truth.json").read_text(encoding="utf-8"))
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    ref = rca._build_target_reference(private, num_samples)
    ctx = ct.ToolContext(
        capture_dir=cap / "public", cache_dir=root / "_agent_cache" / sid,
        allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
        fs=obs["receivers"][0]["sample_rate_hz"],
        fc=obs["receivers"][0]["center_frequency_hz"],
        target_bw_normalized=0.25, aoi_center=aoi, target_reference=ref)
    ct.set_context(ctx)
    # 对全部站剥离目标
    rx_half = ctx.fs * 0.45
    tgt_half = min(0.25 * ctx.fs / 2, rx_half)
    strip = ct.strip_target_all_stations(-tgt_half, tgt_half)
    print(f"剥离成功站: {list(strip.keys())}")
    # 用 station_a 残差找干扰带
    occ = ct.estimate_band_occupancy_by_correlation('station_a')
    peaks = occ['residual_peaks']
    print(f"station_a 残差峰值带: {[(round(p['freq_offset_hz']), p['over_floor_db']) for p in peaks[:4]]}")
    # 对第一个强带做多站测向+交会聚类
    if not peaks:
        return {"K_est": 0, "GT_n": len(truth["interferers"]), "single_n": 0}
    f0 = peaks[0]["freq_offset_hz"]
    bw = peaks[0]["bandwidth_hz"]
    results = []
    for st in ctx.allowed_stations:
        try:
            scan = ct._u_scan_cached(st, f0, bw, on_residual=True)
        except Exception:
            continue
        for u, rel in scan["u_peaks"][:3]:
            pos, axis = ct._station_geometry(st)
            for ray in _u_to_rays([u], pos, axis):
                results.append({"station": st, "u": u, "origin": pos, "ray": ray})
    # RANSAC：随机抽3站做交会，收集有效位置，聚类
    positions = []
    rng = np.random.default_rng(7)
    per_st = {}
    for r in results:
        per_st.setdefault(r["station"], []).append(r)
    sts = [s for s in ctx.allowed_stations if s in per_st and len(per_st[s]) > 0]
    if len(sts) < 3:
        return {"K_est": len(sts), "GT_n": len(truth["interferers"]), "single_n": len(sts),
                "note": "不足3站"}
    for _ in range(800):
        trio = rng.choice(sts, 3, replace=False)
        rays = []
        for s in trio:
            c = rng.choice(per_st[s])
            rays.append((c["origin"], c["ray"] - c["origin"]))
        hit = intersect_rays([r[0] for r in rays], [r[1] for r in rays])
        if hit and hit["position_m"][2] > 0 and max(hit["ranges_m"]) < 2000:
            positions.append(np.array(hit["position_m"]))
    # 空间聚类（<40m）
    clusters = []
    for p in positions:
        if any(np.linalg.norm(p - c) < 40 for c in clusters):
            continue
        clusters.append(p)
    K = min(len(clusters), 6)
    return {"K_est": K, "GT_n": len(truth["interferers"]),
            "single_n": ct.estimate_num_sources_station('station_a',
                                                        on_residual=True)["num_spatial_components"],
            "n_intersections": len(positions), "n_clusters": len(clusters),
            "cluster_pos": [c.round(0).tolist() for c in clusters[:4]]}


def main():
    root = Path(sys.argv[1])
    sids = sys.argv[2:] or ["d2_00000", "d2_00001", "d2_00002"]
    for sid in sids:
        r = cluster(root, sid)
        print(f"\n== {sid}: 空间聚类 K={r.get('K_est')} vs GT源数={r.get('GT_n')} "
              f"vs 单站源数={r.get('single_n')}")
        print(f"   交会命中={r.get('n_intersections')} 簇={r.get('n_clusters')} 位置={r.get('cluster_pos')}")


if __name__ == "__main__":
    main()