"""D2 决定性对照：Agent 数源真正读的那份证据（聚类候选板 / K=len(sources)）。

背景：prompt 的 Round 3 是 estimate_sources_by_spatial_clustering(on_residual=true)
→ 候选板 + K = len(sources)；Round 4/5 对候选逐个提取+分类，答案"逐字复制"。
所以 D2/S2 的报数主要由这个工具的 candidates/sources 决定，而不是 residual_peaks
（那是 S0 回退路径）。

本脚本对同一 50 个样本，在**今天这套代码**（含 model C 参考构造）下分别跑
OLD(加噪前) 与 NEW(加噪后) 数据集，比较 n_intersections / len(candidates) /
len(sources) / 各站的 peak_pool 是否有峰（n_usable），从而判定：
    「Agent 数字掉下来」到底是数据（噪声菜单）造成的，还是别的东西。

只读数据；缓存在系统临时目录。输出 _diag_d2_cluster_ab.txt / .json
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
sys.path.insert(0, str(HERE))
_PLAT = HERE.parent / "城市低空电磁信号仿真平台" / "src"
if str(_PLAT) not in sys.path:
    sys.path.insert(0, str(_PLAT))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402

ROOTS = {
    "OLD": (HERE.parent / "_pre_noise_20261010" / "dataset_d2").resolve(),
    "NEW": (HERE.parent / "dataset_d2").resolve(),
}


def probe(tag: str, root: Path, sid: str, tmp: Path) -> dict:
    rec = {"sid": sid, "tag": tag}
    cap = root / sid / "capture"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    ns = obs["receivers"][0]["shape"][1]
    refs = rca._build_target_reference(cap / "private", ns)
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    ctx = ct.ToolContext(capture_dir=cap / "public",
                         cache_dir=tmp / tag / sid,
                         allowed_stations=allowed,
                         fs=obs["receivers"][0]["sample_rate_hz"],
                         fc=obs["receivers"][0]["center_frequency_hz"],
                         target_bw_normalized=0.25, aoi_center=aoi,
                         target_reference=refs)
    ct.set_context(ctx)
    rec["gt_n"] = len(truth["interferers"])
    try:
        out = ct.estimate_sources_by_spatial_clustering(on_residual=True)
    except Exception as exc:                                  # noqa: BLE001
        rec["error"] = repr(exc)[:300]
        return rec
    if "error" in out:
        rec["error"] = out["error"]
        return rec
    rec["n_intersections"] = out.get("n_intersections")
    rec["n_candidates"] = out.get("n_candidates")
    rec["n_sources"] = len(out.get("sources") or [])
    rec["n_bands"] = len(out.get("bands") or [])
    rec["support"] = out.get("support_threshold")
    rec["src_bands_khz"] = [[round(c["band_hz"][0] / 1e3), round(c["band_hz"][1] / 1e3)]
                            for c in (out.get("sources") or [])]
    rec["cand_bands_khz"] = [[round(c["band_hz"][0] / 1e3), round(c["band_hz"][1] / 1e3),
                              c.get("n_explained_stations"), c.get("max_du")]
                             for c in (out.get("candidates") or [])]
    return rec


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="d2cluster_"))
    rows: dict[str, dict] = {}
    sids = sorted(p.name for p in ROOTS["NEW"].iterdir() if p.name.startswith("d2_"))
    for i, sid in enumerate(sids, 1):
        rows[sid] = {}
        for tag, root in ROOTS.items():
            try:
                rows[sid][tag] = probe(tag, root, sid, tmp)
            except Exception as exc:                          # noqa: BLE001
                rows[sid][tag] = {"sid": sid, "tag": tag, "fatal": repr(exc)[:300]}
        o, n = rows[sid]["OLD"], rows[sid]["NEW"]
        print(f"[{i:2d}/50] {sid} GT={n.get('gt_n')}  "
              f"K {o.get('n_sources', '?')}→{n.get('n_sources', '?')}  "
              f"cand {o.get('n_candidates', '?')}→{n.get('n_candidates', '?')}  "
              f"xing {o.get('n_intersections', '?')}→{n.get('n_intersections', '?')}  "
              f"err={(n.get('error') or o.get('error') or '-')[:40]}", flush=True)

    same_k = sum(1 for s in rows.values() if s["OLD"].get("n_sources") == s["NEW"].get("n_sources"))
    same_c = sum(1 for s in rows.values() if s["OLD"].get("n_candidates") == s["NEW"].get("n_candidates"))
    sum_ko = sum(s["OLD"].get("n_sources") or 0 for s in rows.values())
    sum_kn = sum(s["NEW"].get("n_sources") or 0 for s in rows.values())
    sum_co = sum(s["OLD"].get("n_candidates") or 0 for s in rows.values())
    sum_cn = sum(s["NEW"].get("n_candidates") or 0 for s in rows.values())
    zer_o = sum(1 for s in rows.values() if s["OLD"].get("n_sources") == 0)
    zer_n = sum(1 for s in rows.values() if s["NEW"].get("n_sources") == 0)

    lines = ["=" * 92,
             "D2 聚类候选板对照（今天这套代码，OLD=加噪前 vs NEW=加噪后）",
             "=" * 92,
             f"K=len(sources) 相同的样本: {same_k}/50    （OLD 报 0 的 {zer_o} 个 / NEW 报 0 的 {zer_n} 个）",
             f"len(candidates) 相同的样本: {same_c}/50",
             f"ΣK  OLD {sum_ko} → NEW {sum_kn}",
             f"Σcand OLD {sum_co} → NEW {sum_cn}", "",
             f"{'sid':10s} {'GT':>3s} {'K O→N':>9s} {'cand O→N':>11s} {'xing O→N':>13s} {'bands O→N':>11s}  support(N)"]
    for sid, s in sorted(rows.items()):
        o, n = s["OLD"], s["NEW"]
        lines.append(f"{sid:10s} {n.get('gt_n', '?'):>3} "
                     f"{o.get('n_sources', '?'):>3}→{n.get('n_sources', '?'):<4} "
                     f"{o.get('n_candidates', '?'):>4}→{n.get('n_candidates', '?'):<5} "
                     f"{o.get('n_intersections', '?'):>5}→{n.get('n_intersections', '?'):<6} "
                     f"{o.get('n_bands', '?'):>4}→{n.get('n_bands', '?'):<5}  {n.get('support')}")
        for tag in ("OLD", "NEW"):
            r = s[tag]
            if r.get("error") or r.get("fatal"):
                lines.append(f"    {tag}: {r.get('error') or r.get('fatal')}")
            else:
                lines.append(f"    {tag}: sources={r.get('src_bands_khz')} "
                             f"cands={r.get('cand_bands_khz')}")
    lines.append(f"\n临时缓存目录（可删）：{tmp}")
    out = HERE.parent / "_diag_d2_cluster_ab.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE.parent / "_diag_d2_cluster_ab.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines[:9]))
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
