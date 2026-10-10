"""拒绝答 vs 工具证据（只读，零 API）。

对每档每个样本：
  - GT 可见源数（S0/S2）
  - 主站 station_a 的目标参考状态（ok / invisible / missing）
  - estimate_band_occupancy_by_correlation("station_a") 的门控峰数 / 旧宽带峰数
  - estimate_sources_by_spatial_clustering() 的 K=len(sources) / len(candidates)
  - agent 答案是否拒答（答案文件 abstained 字段，或 run-log 原文里的负数哨兵）
输出按「拒答 / 未拒答」分组统计。
"""
import json
import pathlib
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, "benchmark_v3/adapters")
sys.path.insert(0, "benchmark_v3/城市低空电磁信号仿真平台/src")
import capture_tools as ct  # noqa: E402
import eval_dataset as ed  # noqa: E402
import run_capture_agent as rca  # noqa: E402
ROOT = pathlib.Path(__file__).resolve().parents[2]  # 仓库根：脚本在 benchmark_v3/adapters/
NEG = re.compile(r'"num_interferers"\s*:\s*-\d+')


def station_state(priv: pathlib.Path, st: str) -> str:
    f = priv / f"channel_target_{st}.npz"
    if not f.exists():
        return "missing"
    try:
        d = np.load(f)
    except Exception:  # noqa: BLE001
        return "missing"
    return "invisible" if np.asarray(d["delays_s"]).size == 0 else "ok"


def make_ctx(cap: pathlib.Path):
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    rx = obs["receivers"][0]
    ns = rx["shape"][1] if isinstance(rx.get("shape"), (list, tuple)) else rx.get("num_samples")
    refs = rca._build_target_reference(cap / "private", ns)
    allowed = [r["receiver_id"] for r in obs["receivers"]]
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0)
    tmp = tempfile.mkdtemp(prefix="ev_")
    ctx = ct.ToolContext(capture_dir=cap / "public", cache_dir=pathlib.Path(tmp),
                         allowed_stations=allowed, fs=rx["sample_rate_hz"],
                         fc=rx["center_frequency_hz"], target_bw_normalized=0.25,
                         aoi_center=aoi, target_reference=refs)
    ct.set_context(ctx)
    return obs


def refusal_set(root: pathlib.Path, diff: str, ans: pathlib.Path) -> set[str]:
    out = set()
    for p in ans.glob(f"{diff}_*.json"):
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if o.get("abstained") is True:
            out.add(p.stem)
    lp = ans / "_run_log.json"
    if lp.exists():
        try:
            raw = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = []
        for x in raw if isinstance(raw, list) else []:
            if isinstance(x, dict) and NEG.search(x.get("raw_output") or ""):
                out.add(x.get("sample_id", ""))
    return out


def main() -> None:
    tiers = [("d2", "S2", "answers_S2_agent_bwfix"), ("d3", "S2", "answers_S2_agent_bwfix"),
             ("d4", "S2", "answers_S2_agent_bwfix")]
    for diff, slevel, ansname in tiers:
        root = ROOT / "benchmark_v3" / f"dataset_{diff}"
        ans = root / ansname
        ref = refusal_set(root, diff, ans)
        rows = []
        for d in sorted(root.glob(f"{diff}_[0-9]*")):
            sid = d.name
            cap = d / "capture"
            priv = cap / "private"
            truth = ed.load_truth(root, sid)
            gt = ed.gt_at_slevel(truth, truth["station_subsets"][slevel])
            st = station_state(priv, "station_a")
            try:
                obs = make_ctx(cap)
                occ = ct.estimate_band_occupancy_by_correlation("station_a")
                gated = len(occ.get("residual_peaks") or []) if "error" not in occ else -1
                leg = len(occ.get("residual_peaks_legacy") or []) if "error" not in occ else -1
                cl = ct.estimate_sources_by_spatial_clustering()
                if "error" in cl:
                    k, nc = -1, -1
                else:
                    k, nc = len(cl.get("sources") or []), len(cl.get("candidates") or [])
            except Exception as exc:  # noqa: BLE001
                print(f"  [{sid}] 工具异常: {exc!r}")
                gated = leg = k = nc = -9
            rows.append({"sid": sid, "gt": len(gt), "st": st, "gated": gated, "leg": leg,
                         "k": k, "nc": nc, "refuse": sid in ref,
                         "ans_n": json.loads((ans / f"{sid}.json").read_text(encoding="utf-8")).get("num_interferers") if (ans / f"{sid}.json").exists() else None})
            r = rows[-1]
            print(f"[{diff}] {sid} GT={r['gt']} st_a={r['st']:9s} gated={r['gated']:2d} legacy={r['leg']:2d} K={r['k']:2d} cands={r['nc']:2d} refuse={int(r['refuse'])} ans_n={r['ans_n']}")

        def avg(sel, key):
            v = [x[key] for x in rows if x["refuse"] == sel and x[key] >= 0]
            return (len(v), float(np.mean(v)) if v else float("nan"))

        print(f"===== {diff} 汇总 =====")
        for sel, name in ((True, "拒答"), (False, "未拒答")):
            n = sum(1 for x in rows if x["refuse"] == sel)
            if not n:
                print(f"  {name}: 0 个")
                continue
            parts = []
            for key in ("gt", "gated", "leg", "k", "nc"):
                m, a = avg(sel, key)
                parts.append(f"{key}={a:.2f}(n={m})")
            print(f"  {name}: {n} 个  " + "  ".join(parts))
        print(f"  主站状态: ok={sum(1 for x in rows if x['st']=='ok')} "
              f"invisible={sum(1 for x in rows if x['st']=='invisible')} "
              f"missing={sum(1 for x in rows if x['st']=='missing')}")
        out = ROOT / "benchmark_v3" / f"_an_refuse_evidence_{diff}.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
