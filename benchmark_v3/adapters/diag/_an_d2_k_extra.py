"""损失分解：聚类候选板里那些「对不上任何 GT 源」的候选是什么来路？（只读，零 API）

对 dataset_d2 全 50 样本跑 `ct.estimate_sources_by_spatial_clustering(on_residual=True)`：
 - GT 可见源用与 agent 评分一致的 `eval_dataset.gt_at_slevel(truth, S2 站集)`；
 - 50 m 贪心匹配（与 gate / eval_dataset 同口径）分出 matched / extra；
 - 每个 extra 记录：到最近 GT 的距离、到最近 matched 候选的位置距离、与它的频带是否重叠、
   导向互相关（复刻 capture_tools `_steers_nms` 的 u_r 公式，用 observations.json 的站址）、
   n_explained / max_du / deflation_db / 带宽；
 - 每个未被匹配的 GT 记录到最近候选的距离（欠报侧）。
输出 benchmark_v3/_an_d2_k_extra.txt
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # adapters（迁移后 HERE 下沉一级）
import capture_tools as ct  # noqa: E402
import eval_dataset as ed  # noqa: E402
import run_capture_agent as rca  # noqa: E402

ROOT = HERE.parent.parent          # benchmark_v3
D2 = ROOT / "dataset_d2"
STATIONS = ["station_a", "station_b", "station_c", "station_d"]
POS_M = 50.0
OUT = ROOT / "_an_d2_k_extra.txt"

_lines: list[str] = []


def say(msg: str = "") -> None:
    _lines.append(msg)
    print(msg)


def ctx_for(sid: str, cache: Path):
    cap = D2 / sid / "capture"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    ns = obs["receivers"][0]["shape"][1]
    refs = rca._build_target_reference(cap / "private", ns)
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    ct.set_context(ct.ToolContext(capture_dir=cap / "public", cache_dir=cache / sid,
                                  allowed_stations=allowed,
                                  fs=obs["receivers"][0]["sample_rate_hz"],
                                  fc=obs["receivers"][0]["center_frequency_hz"],
                                  target_bw_normalized=0.25, aoi_center=aoi,
                                  target_reference=refs))
    return {r["receiver_id"]: np.array(r["receiver_position_m"][:2], dtype=float)
            for r in obs["receivers"]}


def gt_pos2(g: dict) -> np.ndarray:
    for key in ("position_m", "position_xy_m"):
        if key in g:
            p = g[key]
            return np.array([float(p[0]), float(p[1])])
    raise KeyError(sorted(g))


def _steer(cand_pos2: np.ndarray, st_pos: dict) -> dict:
    cols = {}
    for s, p in st_pos.items():
        d = np.array([cand_pos2[0] - p[0], cand_pos2[1] - p[1], 0.0])
        u_r = d[1] / (np.linalg.norm(d) + 1e-12)
        cols[s] = np.exp(2j * np.pi * 0.5 * u_r * np.arange(8))
    return cols


def steer_corr(pos1: np.ndarray, pos2: np.ndarray, st_pos: dict) -> float:
    s1, s2 = _steer(pos1, st_pos), _steer(pos2, st_pos)
    return float(np.mean([abs(np.vdot(s1[s], s2[s])) / 8 for s in s1]))


def bands_overlap(b1, b2) -> bool:
    return not (b1[1] < b2[0] or b1[0] > b2[1])


def main() -> int:
    cache = Path(tempfile.mkdtemp(prefix="_an_k_extra_"))
    tot_gt = tot_matched = tot_extra = tot_gt_miss = 0
    ex_overlap = ex_corr_hi = ex_near40 = ex_near100 = ex_near200 = 0
    ex_gt_dist: list[float] = []
    ex_pos_to_matched: list[float] = []
    ex_sup_ok = 0          # extra 自身是否过了支持门（n_explained/max_du）
    ex_bw: list[float] = []
    miss_gt_dist: list[float] = []

    for i in range(50):
        sid = f"d2_{i:05d}"
        st_pos = ctx_for(sid, cache)
        truth = json.loads((D2 / sid / "capture" / "private" / "d1_truth.json")
                           .read_text(encoding="utf-8"))
        gt = ed.gt_at_slevel(truth, STATIONS)
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cands = clu.get("candidates") or []
        n_usable = (clu.get("support_threshold") or {}).get("n_usable_stations") or 0
        need = max(n_usable - 1, 1)

        gt_p = [gt_pos2(g) for g in gt]
        cand_p = [np.array(c["position_xy_m"], dtype=float) for c in cands]

        # 50 m 贪心最近匹配（复刻 gate / eval_dataset 口径）
        pairs = sorted(((float(np.linalg.norm(cp - gp)), ig, ic)
                        for ig, gp in enumerate(gt_p) for ic, cp in enumerate(cand_p)))
        claimed_c: dict[int, int] = {}
        claimed_g: dict[int, int] = {}
        for d, ig, ic in pairs:
            if d < POS_M and ig not in claimed_g and ic not in claimed_c:
                claimed_g[ig] = ic
                claimed_c[ic] = ig

        matched_idx = sorted(claimed_c)
        extra_idx = [k for k in range(len(cands)) if k not in claimed_c]
        miss_idx = [g for g in range(len(gt)) if g not in claimed_g]

        tot_gt += len(gt)
        tot_matched += len(matched_idx)
        tot_extra += len(extra_idx)
        tot_gt_miss += len(miss_idx)

        for g in miss_idx:
            if cand_p:
                miss_gt_dist.append(min(float(np.linalg.norm(cp - gt_p[g])) for cp in cand_p))

        rows: list[str] = []
        for k in extra_idx:
            c = cands[k]
            cp = cand_p[k]
            d_gt = min((float(np.linalg.norm(cp - gp)) for gp in gt_p), default=float("nan"))
            ex_gt_dist.append(d_gt)
            bw = float(c["band_hz"][1]) - float(c["band_hz"][0])
            ex_bw.append(bw)
            if int(c["n_explained_stations"]) >= need and float(c["max_du"]) <= 0.3:
                ex_sup_ok += 1
            if matched_idx:
                nk = min(matched_idx, key=lambda j: float(np.linalg.norm(cp - cand_p[j])))
                d_m = float(np.linalg.norm(cp - cand_p[nk]))
                ex_pos_to_matched.append(d_m)
                ov = bands_overlap(c["band_hz"], cands[nk]["band_hz"])
                co = steer_corr(cp, cand_p[nk], st_pos)
                ex_overlap += ov
                ex_corr_hi += co > 0.85
                ex_near40 += d_m < 40
                ex_near100 += d_m < 100
                ex_near200 += d_m < 200
                rows.append(f"    extra k={k} pos=({cp[0]:.0f},{cp[1]:.0f}) bw={bw/1e3:.0f}k "
                            f"n_exp={c['n_explained_stations']} maxdu={c['max_du']:.2f} "
                            f"def={c['deflation_db']:.1f} | 最近 GT {d_gt:6.1f}m | "
                            f"最近matched {d_m:6.1f}m overlap={'Y' if ov else 'n'} corr={co:.2f} "
                            f"(k'={nk})")
            else:
                rows.append(f"    extra k={k} pos=({cp[0]:.0f},{cp[1]:.0f}) bw={bw/1e3:.0f}k "
                            f"n_exp={c['n_explained_stations']} maxdu={c['max_du']:.2f} "
                            f"| 最近 GT {d_gt:6.1f}m | 无 matched 候选")

        if extra_idx or miss_idx:
            say(f"[{sid}] GT={len(gt)} matched={len(matched_idx)} extra={len(extra_idx)} "
                f"未匹配GT={len(miss_idx)}  cand={len(cands)} K={len(clu.get('sources') or [])}")
            for r in rows:
                say(r)
            for g in miss_idx:
                if cand_p:
                    dmin = min(float(np.linalg.norm(cp - gt_p[g])) for cp in cand_p)
                    say(f"    GT#{g} 位置=({gt_p[g][0]:.0f},{gt_p[g][1]:.0f}) "
                        f"kind={gt[g].get('kind')} 最近候选 {dmin:.1f}m")

    def stat(xs: list[float]) -> str:
        if not xs:
            return "n=0"
        a = np.array(xs, dtype=float)
        a = a[np.isfinite(a)]
        return (f"n={len(a)} 中位 {np.median(a):.1f} p90 {np.percentile(a, 90):.1f} "
                f"max {a.max():.1f}")

    say("")
    say("=== 汇总（dataset_d2 全 50 样本，S2 四站口径）===")
    say(f"GT 可见源 {tot_gt}  matched {tot_matched} ({tot_matched/max(tot_gt,1):.3f})  "
        f"extra {tot_extra}  未匹配 GT {tot_gt_miss}")
    say(f"extra 中：过支持门 {ex_sup_ok}/{tot_extra}；"
        f"与最近 matched 候选 位置<40m {ex_near40}、<100m {ex_near100}、<200m {ex_near200}；"
        f"频带重叠 {ex_overlap}；导向互相关>0.85 {ex_corr_hi}")
    say(f"extra 到最近 GT 距离：{stat(ex_gt_dist)}")
    say(f"extra 到最近 matched 候选距离：{stat(ex_pos_to_matched)}")
    say(f"extra 带宽：{stat(ex_bw)}")
    say(f"未匹配 GT 到最近候选距离：{stat(miss_gt_dist)}")

    OUT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"[ok] 已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
