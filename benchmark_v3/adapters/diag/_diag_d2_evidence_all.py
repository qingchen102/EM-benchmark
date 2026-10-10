"""D2 加噪前/加噪后：全 50 样本的 Agent 可见证据对照（只读，缓存在系统临时目录）。

动机：同样的 50 个样本，agent 报源数从 43/50 掉到 24/50，而仪器侧 gate 只变 −5pp。
要判定「数据变了导致证据变少」还是「agent 判断变保守」，就要逐样本比较喂给 agent 的
那份证据：① station_a 门控残差峰；② legacy 宽带峰（聚类定带用）；③ 时频指纹；
④ 目标参考是否可用（station_a 参考缺失会让工具直接少一条腿）。

输出：benchmark_v3/_diag_d2_evidence_all.txt + _diag_d2_evidence_all.json
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

TF_KEYS = ("duty_cycle", "frame_power_cv", "burst_count_per_window",
           "hop_channel_count", "mean_active_span", "n_frames")


def _peaks_fp(peaks, ndig=1):
    return sorted((round(p["freq_offset_hz"] / 1e3, ndig),
                   round(p["over_floor_db"], 1)) for p in (peaks or []))


def probe(tag: str, root: Path, sid: str, tmp: Path) -> dict:
    rec = {"sid": sid, "tag": tag}
    cap = root / sid / "capture"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    ns = obs["receivers"][0]["shape"][1]
    refs = rca._build_target_reference(cap / "private", ns)
    rec["ref_stations"] = sorted(k for k in refs.keys()) if isinstance(refs, dict) else []
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
    rec["gt_kinds"] = [g["kind"] for g in truth["interferers"]]
    rec["gt_f0_khz"] = [round(g["freq_offset_hz"] / 1e3, 1) for g in truth["interferers"]]
    try:
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        rec["gated"] = _peaks_fp(occ.get("residual_peaks"))
        rec["legacy"] = _peaks_fp(occ.get("residual_peaks_legacy"))
    except Exception as exc:                                  # noqa: BLE001
        rec["gated"], rec["legacy"], rec["occ_error"] = None, None, repr(exc)[:200]
    try:
        tf = ct.analyze_time_frequency_station("station_a")
        rec["tf"] = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else None)
                     for k, v in tf.items() if k in TF_KEYS}
        rec["tf_error"] = None
    except Exception as exc:                                  # noqa: BLE001
        rec["tf"], rec["tf_error"] = None, repr(exc)[:200]
    return rec


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="d2evidence_all_"))
    rows: dict[str, dict] = {}
    sids = sorted(p.name for p in ROOTS["NEW"].iterdir() if p.name.startswith("d2_"))
    for i, sid in enumerate(sids, 1):
        rows[sid] = {}
        for tag, root in ROOTS.items():
            try:
                rows[sid][tag] = probe(tag, root, sid, tmp)
            except Exception as exc:                          # noqa: BLE001
                rows[sid][tag] = {"sid": sid, "tag": tag, "fatal": repr(exc)[:300]}
        o, n = rows[sid].get("OLD", {}), rows[sid].get("NEW", {})
        same_g = o.get("gated") == n.get("gated")
        same_l = o.get("legacy") == n.get("legacy")
        same_t = o.get("tf") == n.get("tf")
        print(f"[{i:2d}/50] {sid}  GT={n.get('gt_n', '?')}  "
              f"门控峰 {len(o.get('gated') or [])}→{len(n.get('gated') or [])} {'==' if same_g else '变'}  "
              f"legacy {len(o.get('legacy') or [])}→{len(n.get('legacy') or [])} {'==' if same_l else '变'}  "
              f"TF {'==' if same_t else '变'}  ref={','.join(o.get('ref_stations') or []) or '无'}"
              f"→{','.join(n.get('ref_stations') or []) or '无'}", flush=True)

    n_same_g = sum(1 for s in rows.values() if s["OLD"].get("gated") == s["NEW"].get("gated"))
    n_same_l = sum(1 for s in rows.values() if s["OLD"].get("legacy") == s["NEW"].get("legacy"))
    n_same_t = sum(1 for s in rows.values() if s["OLD"].get("tf") == s["NEW"].get("tf"))
    n_ref_lost = sum(1 for s in rows.values()
                     if set(s["OLD"].get("ref_stations") or []) - set(s["NEW"].get("ref_stations") or []))
    n_ref_gain = sum(1 for s in rows.values()
                     if set(s["NEW"].get("ref_stations") or []) - set(s["OLD"].get("ref_stations") or []))
    tot_o = sum(len(s["OLD"].get("gated") or []) for s in rows.values())
    tot_n = sum(len(s["NEW"].get("gated") or []) for s in rows.values())
    tot_lo = sum(len(s["OLD"].get("legacy") or []) for s in rows.values())
    tot_ln = sum(len(s["NEW"].get("legacy") or []) for s in rows.values())

    lines = ["=" * 78,
             "D2 Agent 可见证据：加噪前(OLD) vs 加噪后(NEW)，50 样本",
             "=" * 78,
             f"门控残差峰逐条相同（频率 0.1 kHz / 强度 0.1 dB 内）: {n_same_g}/50",
             f"legacy 宽带峰逐条相同                              : {n_same_l}/50",
             f"时频指纹完全相同                                  : {n_same_t}/50",
             f"门控峰总数 OLD {tot_o} → NEW {tot_n}（差 {tot_n - tot_o:+d}）",
             f"legacy 峰总数 OLD {tot_lo} → NEW {tot_ln}（差 {tot_ln - tot_lo:+d}）",
             f"NEW 少了参考的样本: {n_ref_lost}/50    NEW 多了参考的样本: {n_ref_gain}/50", ""]
    for sid, s in rows.items():
        o, n = s["OLD"], s["NEW"]
        lines.append(f"{sid}  GT={n.get('gt_n')}  kinds={n.get('gt_kinds')}  f0={n.get('gt_f0_khz')}")
        for tag in ("OLD", "NEW"):
            r = s[tag]
            if r.get("fatal"):
                lines.append(f"   {tag}: FATAL {r['fatal']}")
                continue
            lines.append(f"   {tag}: ref={r.get('ref_stations')}  "
                         f"gated={r.get('gated')}  legacy_n={len(r.get('legacy') or [])}  "
                         f"tf={r.get('tf')}")
            if r.get("occ_error"):
                lines.append(f"        occ_error={r['occ_error']}")
            if r.get("tf_error"):
                lines.append(f"        tf_error={r['tf_error']}")
    lines.append(f"\n临时缓存目录（可删）：{tmp}")

    out = HERE.parent / "_diag_d2_evidence_all.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    (HERE.parent / "_diag_d2_evidence_all.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
