"""临时只读：分类准确率 vs 提取时的候选集大小（K）——验证"多候选污染 LS 提取"假说。

对每个"认领到真源"的候选，用三种候选集提取同一路波形再分类：
  all  = 全部 candidates 的导向（= 现行 gate 口径）
  own  = 只用该候选自己的导向（单源匹配滤波）
  top2 / top3 = 按打分排序取前 2/3 个导向
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (str(HERE), str(_V3 / "simulation"), str(_HERE := str(HERE / "rad_cls")),
           str(_V3.parent / "benchmark_v2"), str(_V3.parent / "benchmark_v2" / "simulation"),
           str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct  # noqa: E402
import numpy as _np
_orig_iq = ct._station_iq_raw
def _iq_fb(station, on_residual=False):
    try:
        return _orig_iq(station, on_residual=on_residual)
    except RuntimeError:
        if not on_residual:
            raise
        _c = ct._ctx()
        _x = _orig_iq(station, on_residual=False)
        _n = _x.shape[1]
        _f = _np.fft.fftfreq(_n, 1 / _c.fs)
        _m = _np.abs(_f) <= 0.45 * _c.fs
        return _np.fft.ifft(_np.fft.fft(_x, axis=1) * _m, axis=1).astype(_np.complex128)
ct._station_iq_raw = _iq_fb
from data_gen16 import CLS_IDX  # noqa: E402
from oracle16 import _pred, load_model  # noqa: E402

D2 = _V3 / "dataset_d2"
C = 299792458.0


def extract(resid, off, rp, positions):
    A = np.zeros((off.shape[0], len(positions)), complex)
    for k, p in enumerate(positions):
        d = np.array([p[0] - rp[0], p[1] - rp[1], 0.0])
        d = d / (np.linalg.norm(d) + 1e-12)
        A[:, k] = np.exp(2j * np.pi * 2.4e9 / C * (off @ d))
    if not np.isfinite(A).all() or float(np.linalg.cond(A)) > 1e6:
        return None
    return np.linalg.pinv(A) @ resid


def main(n_samples: int) -> None:
    model, _ = load_model()
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    import run_capture_agent as rca
    acc = {k: [0, 0] for k in ("all", "own", "top2", "top3")}
    ks = []
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture" / "public"
        priv = D2 / sid / "capture" / "private"
        obs = json.loads((cap / "observations.json").read_text(encoding="utf-8"))
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        n = int(obs["receivers"][0]["shape"][1])
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        try:
            ref = rca._build_target_reference(priv, n)
        except Exception:                                  # noqa: BLE001
            continue
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=allowed, fs=fs,
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=ref)
        ct.set_context(ctx)
        tr = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
        ach = tr.get("achieved") or {}
        nz = tr["noise"]["noise_floor_dbm"]
        gt = [g for j, g in enumerate(tr.get("interferers", []))
              if any((ach.get(f"interferer_{j+1}", {}).get(f"s{i}", {}).get("pr_dbm", -999)
                      - nz) >= 3.0 for i in range(len(allowed)))]
        res = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        cands = res.get("candidates") or []
        if not cands:
            continue
        ks.append(len(cands))
        rec = next(r for r in obs["receivers"] if r["receiver_id"] == "station_a")
        off = np.asarray(rec["element_offsets_world_m"], float)
        rp = np.asarray(rec["receiver_position_m"], float)
        resid = ct._station_iq_raw("station_a", on_residual=True)
        t = np.arange(resid.shape[1])
        pos_all = [np.asarray(c["position_xy_m"], float) for c in cands]
        used = set()
        for k, p in enumerate(pos_all):
            tgt = None
            for i, g in enumerate(gt):
                if i in used:
                    continue
                if np.linalg.norm(p - np.asarray(g["position_m"], float)[:2]) < 50:
                    tgt = (i, g)
                    break
            if tgt is None:
                continue
            used.add(tgt[0])
            variants = {"all": pos_all,
                        "own": [p],
                        "top2": pos_all[:2],
                        "top3": pos_all[:3]}
            for tag, ps in variants.items():
                if k >= len(ps):
                    continue
                S = extract(resid, off, rp, ps)
                if S is None:
                    continue
                w = S[k] * np.exp(-1j * 2 * np.pi * tgt[1]["freq_offset_hz"] * t / fs)
                w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)
                pred = _pred(model, [w.astype(np.complex64)])[0]
                acc[tag][0] += int(pred == CLS_IDX[tgt[1]["kind"]])
                acc[tag][1] += 1
    print(f"\n== 分类准确率 vs 提取候选集大小（{n_samples} 样本）==")
    print(f"  每样本候选数 中位 {np.median(ks):.0f}，最大 {max(ks)}")
    for tag in ("all", "own", "top2", "top3"):
        h, n_ = acc[tag]
        print(f"  {tag:5s}: {h}/{n_} = {h/max(n_,1):.0%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
