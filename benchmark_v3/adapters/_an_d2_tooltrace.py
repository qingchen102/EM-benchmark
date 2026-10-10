"""把 agent 在 D2/D3/D4 上会看到的工具输出原样打出来（只读，零 API）。

用途：判断「模型为什么说不出答案」——是工具互相矛盾、还是某个工具报错、还是候选板为空。

用法：
  python -X utf8 _an_d2_tooltrace.py d2 d2_00001,d2_00002,d2_00000 --max 300
  python -X utf8 _an_d2_tooltrace.py d4 d4_00006
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "城市低空电磁信号仿真平台" / "src"))
import capture_tools as ct  # noqa: E402
import eval_dataset as ed  # noqa: E402
import run_capture_agent as rca  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def make_ctx(cap: pathlib.Path):
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    rx = obs["receivers"][0]
    ns = rx["shape"][1] if isinstance(rx.get("shape"), (list, tuple)) else rx.get("num_samples")
    refs = rca._build_target_reference(cap / "private", ns)
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    # 必须转 list：capture_tools.estimate_doa_station 里写的是 `if ctx.aoi_center:`，
    # 传 numpy 数组会 `ValueError: truth value of an array...`（生产路径 run_capture_agent 里
    # 是 .tolist()，所以线上没暴露——探针必须对齐）。
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
    tmp = tempfile.mkdtemp(prefix="tt_")
    ctx = ct.ToolContext(capture_dir=cap / "public", cache_dir=pathlib.Path(tmp),
                         allowed_stations=allowed, fs=rx["sample_rate_hz"],
                         fc=rx["center_frequency_hz"], target_bw_normalized=0.25,
                         aoi_center=aoi, target_reference=refs)
    ct.set_context(ctx)


def brief(o, maxlen: int) -> str:
    try:
        s = json.dumps(o, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        s = repr(o)
    if len(s) <= maxlen:
        return s
    return s[:maxlen] + f" …(+{len(s) - maxlen} chars)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("difficulty", help="d2 / d3 / d4")
    ap.add_argument("sids", help="逗号分隔的样本 id")
    ap.add_argument("--max", type=int, default=320, help="每个工具输出最多打印多少字符")
    args = ap.parse_args()

    root = ROOT / "benchmark_v3" / f"dataset_{args.difficulty}"
    for sid in [s.strip() for s in args.sids.split(",") if s.strip()]:
        sdir = root / sid
        cap = sdir / "capture"
        if not cap.exists():
            print(f"!! 找不到 {cap}")
            continue
        truth = ed.load_truth(root, sid)
        gt = ed.gt_at_slevel(truth, truth["station_subsets"]["S2"])
        ans_p = None
        for cand in sorted(root.glob(f"answers_*{args.difficulty[-1]}*")):
            p = cand / f"{sid}.json"
            if p.exists():
                ans_p = p
                break
        print(f"\n############ {sid}  GT(S2)={len(gt)}  "
              f"GT 源={[g.get('kind') for g in gt]}  "
              f"f0={[round(g.get('freq_offset_hz', 0) / 1e3) for g in gt]} kHz")
        if ans_p:
            print(f"     agent 答案: {ans_p.read_text(encoding='utf-8').strip()[:200]}")
        make_ctx(cap)

        def call(name, fn):
            try:
                out = fn()
            except Exception as exc:  # noqa: BLE001
                out = {"__exception__": repr(exc)}
            print(f"  -- {name}")
            print(f"     {brief(out, args.max)}")
            return out

        call("analyze_spectrum_station('station_a')",
             lambda: ct.analyze_spectrum_station("station_a"))
        occ = call("estimate_band_occupancy_by_correlation('station_a')",
                   lambda: ct.estimate_band_occupancy_by_correlation("station_a"))
        cl = call("estimate_sources_by_spatial_clustering()",
                  lambda: ct.estimate_sources_by_spatial_clustering())
        call("extract_source_waveforms()", lambda: ct.extract_source_waveforms())
        call("classify_extracted_waveform('station_a')",
             lambda: ct.classify_extracted_waveform("station_a"))
        call("estimate_doa_station('station_a')",
             lambda: ct.estimate_doa_station("station_a"))
        f0 = None
        if isinstance(occ, dict) and occ.get("residual_peaks"):
            f0 = occ["residual_peaks"][0].get("freq_offset_hz")
        if f0 is None and isinstance(cl, dict) and cl.get("sources"):
            f0 = (cl["sources"][0] or {}).get("freq_offset_hz")
        if f0 is not None:
            call(f"estimate_position(freq_offset_hz={f0:.0f})",
                 lambda f=f0: ct.estimate_position(float(f)))
        call("list_radiator_catalog()", lambda: ct.list_radiator_catalog())
    return 0


if __name__ == "__main__":
    sys.exit(main())
