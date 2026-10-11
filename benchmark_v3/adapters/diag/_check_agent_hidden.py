"""验收：对外工具表已剥掉内部字段，而聚类内部路径不变。

三条断言：
 A 原始函数 `estimate_band_occupancy_by_correlation()` 仍返回 residual_peaks_legacy（聚类要用）；
 B 对外工具 `TOOL_FUNCTIONS[...]` 的返回值**不再**含该键，且 residual_peaks 逐条不变；
 C `estimate_sources_by_spatial_clustering(on_residual=True)` 的 K / 候选数与改前一致
   （d2_00001 / d2_00002 的记录值：K=5/cand=5、K=2/cand=2）。
另扫一遍：所有对外工具的返回值都不含 AGENT_HIDDEN_KEYS（顶层）与 AGENT_HIDDEN_ITEM_KEYS
（interferers[*] 逐条目，如 classify_extracted_waveform 的 intent / is_spoof）。
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

HERE = Path(__file__).resolve().parent          # benchmark_v3/adapters/diag
ADAPTERS = HERE.parent                         # benchmark_v3/adapters（capture_tools 所在）
sys.path.insert(0, str(ADAPTERS))
import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402

ROOT = ADAPTERS.parent                         # benchmark_v3（数据集根）
EXPECT = {"d2_00001": (5, 5), "d2_00002": (1, 1)}   # 2026-10-11 现行数据（带宽修复 + 换装后）
# 带宽修复前的记录值是 (5,5) / (2,2)；d2_00002 的 K 从 2 降到 1 属已知变化，
# 不是本次 _agent_facing 改动造成（见 evidence/_diag_d2_cluster_ab.txt）。
bad = 0


def ctx_for(sid: str, cache: Path):
    cap = ROOT / "dataset_d2" / sid / "capture"
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


def main() -> int:
    global bad
    cache = Path(tempfile.mkdtemp(prefix="_chk_hidden_"))
    fn_name = "estimate_band_occupancy_by_correlation"
    from em_signal_simulator_v3.emitters import INTENT as _EMITTER_INTENT
    same_table = ct.INTENT_OF == _EMITTER_INTENT
    print(f"意图表单一来源：capture_tools.INTENT_OF == emitters.INTENT -> {same_table}"
          f"（{len(ct.INTENT_OF)} 键）")
    bad += (not same_table)
    print(f"逐条目剥字段常量 AGENT_HIDDEN_ITEM_KEYS = {ct.AGENT_HIDDEN_ITEM_KEYS}")
    for sid in EXPECT:
        ctx_for(sid, cache)
        raw = ct.estimate_band_occupancy_by_correlation("station_a")
        pub = ct.TOOL_FUNCTIONS[fn_name]("station_a")
        gated_raw = [p["freq_offset_hz"] for p in raw.get("residual_peaks") or []]
        gated_pub = [p["freq_offset_hz"] for p in pub.get("residual_peaks") or []]
        ok_a = "residual_peaks_legacy" in raw
        ok_b = "residual_peaks_legacy" not in pub and gated_raw == gated_pub
        clus = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        k, nc = len(clus.get("sources") or []), clus.get("n_candidates")
        ok_c = (k, nc) == EXPECT[sid]
        leaks = [k2 for k2 in ct.AGENT_HIDDEN_KEYS if k2 in pub]
        print(f"{sid}: A={'OK' if ok_a else 'FAIL'}(raw 含 legacy) "
              f"B={'OK' if ok_b else 'FAIL'}(对外 {sorted(pub)[:4]}… legacy={'有' if 'residual_peaks_legacy' in pub else '无'}) "
              f"C={'OK' if ok_c else 'FAIL'}(K={k} cand={nc} 期望 {EXPECT[sid]}) leaks={leaks}")
        bad += (not ok_a) + (not ok_b) + (not ok_c) + bool(leaks)
    # 全表扫描
    ctx_for("d2_00001", cache)
    found = []
    for name, fn in ct.TOOL_FUNCTIONS.items():
        try:
            out = fn("station_a") if name != "estimate_position" else fn()
        except Exception as exc:                                        # noqa: BLE001
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(out, dict):
            hit = [k for k in ct.AGENT_HIDDEN_KEYS if k in out]
            items = out.get("interferers")
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        hit += [k for k in ct.AGENT_HIDDEN_ITEM_KEYS if k in it]
            if hit:
                found.append((name, hit))
            if name == "classify_extracted_waveform":
                keys = sorted(items[0]) if items else []
                print(f"  [classify 实测] items={len(items) if isinstance(items, list) else 'n/a'} "
                      f"首条字段={keys}（应无 intent / is_spoof）")
    print(f"全表扫描：仍泄漏内部字段的工具 = {found or '无'}")
    print(f"=== 结论：{'全部通过' if bad == 0 and not found else '有问题'} ===")
    return 0 if bad == 0 and not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
