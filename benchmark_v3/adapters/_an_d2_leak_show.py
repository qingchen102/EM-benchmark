"""举证：estimate_band_occupancy_by_correlation 的**返回值**（原样回灌给 Agent 的那个 dict）
里同时含 residual_peaks（门控）与 residual_peaks_legacy（旧宽带口径），而工具说明只描述前者。

用今天的代码对指定样本跑一次，打印键名与两条清单的逐项内容。只读（写 tempfile 缓存）。
"""
from __future__ import annotations

import json
import sys
import tempfile

import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402

ROOT = (HERE / "..").resolve()


def main() -> int:
    root = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "dataset_d2")
    sids = sys.argv[2:] or ["d2_00001", "d2_00002"]
    cache = Path(tempfile.mkdtemp(prefix="_an_leak_"))
    for sid in sids:
        cap = root / sid / "capture"
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        ns = obs["receivers"][0]["shape"][1]
        refs = rca._build_target_reference(cap / "private", ns)
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ct.set_context(ct.ToolContext(capture_dir=cap / "public",
                                      cache_dir=cache / sid,
                                      allowed_stations=allowed,
                                      fs=obs["receivers"][0]["sample_rate_hz"],
                                      fc=obs["receivers"][0]["center_frequency_hz"],
                                      target_bw_normalized=0.25, aoi_center=aoi,
                                      target_reference=refs))
        out = ct.estimate_band_occupancy_by_correlation("station_a")
        print("=" * 100)
        print(f"{sid}  GT={[i['kind'] for i in truth['interferers']]}")
        print(f"  返回键: {sorted(out.keys())}")
        for key in ("residual_peaks", "residual_peaks_legacy"):
            lst = out.get(key) or []
            print(f"  {key}: {len(lst)} 条")
            for p in lst:
                print("     " + json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                                            for k, v in p.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
