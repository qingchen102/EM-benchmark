"""核查 estimate_sources_by_spatial_clustering 输出里到底有没有 `sources` 键（只读，零 API）。

用法：python -X utf8 _probe_srckeys.py d2_00001 d2_00000 d2_00003
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "城市低空电磁信号仿真平台" / "src"))
import capture_tools as ct  # noqa: E402
import eval_dataset as ed  # noqa: E402
import run_capture_agent as rca  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
for sid in (sys.argv[1:] or ["d2_00001"]):
    diff = "d" + sid.split("_")[0][-1]
    root = ROOT / "benchmark_v3" / f"dataset_{diff}"
    cap = root / sid / "capture"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    rx = obs["receivers"][0]
    refs = rca._build_target_reference(cap / "private", rx["shape"][1])
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    ct.set_context(ct.ToolContext(capture_dir=cap / "public",
                                  cache_dir=pathlib.Path(tempfile.mkdtemp(prefix="sk_")),
                                  allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                                  fs=rx["sample_rate_hz"], fc=rx["center_frequency_hz"],
                                  target_bw_normalized=0.25, aoi_center=aoi, target_reference=refs))
    ct.estimate_band_occupancy_by_correlation("station_a")
    out = ct.estimate_sources_by_spatial_clustering()
    truth = ed.load_truth(root, sid)
    gt = ed.gt_at_slevel(truth, truth["station_subsets"]["S2"])
    print(f"\n### {sid}  GT={len(gt)}")
    print("keys      :", list(out.keys()))
    print("len(cand) :", len(out.get("candidates") or []))
    print("sources?  :", "sources" in out, " len(sources) =", len(out.get("sources") or []) if "sources" in out else "N/A")
    print("bands     :", [b.get("band_center_hz") for b in (out.get("bands") or [])])
