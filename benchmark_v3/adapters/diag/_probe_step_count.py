"""探针：D2 残差里"阶跃点个数"和"源个数"到底是什么关系。

回答的命题（用户提法）：频谱里有 4 处阶跃点 → 除掉目标残留那两处，只剩 1 个源，
取中间两处之间的片段就是要切的窗。

对每个有干扰样本、指定站点，按 C 法的阶跃判据（|d| ≥ 0.35 dB/kHz、边处相干 ≥ 0.8）数三类：
  raw     : 全部强阶跃局部极大（未做"源内结构"过滤）
  outside : 再加"两侧不能同时相干"（真边界；两侧都相干 = 源内部结构）
  kept    : NMS（同一条边 25 kHz 内只留最强）后接受的真边界数
并与 GT 对照：源数 n_gt、GT 边数（每源 2 条，仅带宽 ≥ 20 kHz 的源）。

用法：python -X utf8 diag/_probe_step_count.py <root> [n] [station]
产物：print 到 stdout（重定向存 evidence/）。
"""
from __future__ import annotations

import json
import pathlib as _pl
import shutil
import sys

import numpy as np

_HERE = _pl.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _sweep_step_edges as sse  # noqa: E402

STATION = "station_a"


def counts_for(resid, fs, rx_half):
    f, over, sm, X = sse._prep(resid, fs, rx_half)
    if f.size < 7:
        return 0, 0, 0
    f_full = np.fft.fftfreq(resid.shape[1], 1 / fs)
    coh = sse._coh_curve(X, f_full, f, sse.ct.PEAK_COH_WIN_HZ)
    d = np.gradient(sm, f) * 1e3
    ad = np.abs(d)
    cand = (ad[1:-1] >= ad[:-2]) & (ad[1:-1] >= ad[2:]) & (ad[1:-1] >= sse.EDGE_STEP_MIN)
    raw = outside = 0
    for i in (np.where(cand)[0] + 1):
        i = int(i)
        c = coh[i]
        if not np.isfinite(c) or c < sse.EDGE_COH_MIN:
            continue
        raw += 1
        jl = int(np.argmin(np.abs(f - (f[i] - sse.EDGE_OUTSIDE_HZ))))
        jr = int(np.argmin(np.abs(f - (f[i] + sse.EDGE_OUTSIDE_HZ))))
        cl, cr = coh[jl], coh[jr]
        if (not np.isfinite(cl)) or (not np.isfinite(cr)) or (min(cl, cr) < sse.EDGE_COH_MIN):
            outside += 1
    _bands, edges = sse.bands_steps(f, over, coh, sm)
    return raw, outside, len(edges)


def main(argv):
    root = _pl.Path(argv[1] if len(argv) > 1 else "../dataset_d2").resolve()
    n = int(argv[2]) if len(argv) > 2 else 40
    station = argv[3] if len(argv) > 3 else STATION
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    rows = []
    print(f"# root={root}  station={station}  判据: |d|>={sse.EDGE_STEP_MIN} dB/kHz, coh>={sse.EDGE_COH_MIN}, NMS {sse.EDGE_NMS_HZ/1e3:.0f} kHz")
    print(f"{'sid':<12}{'n_gt':>5}{'gt_edge':>8}{'raw':>6}{'outside':>8}{'kept':>6}   每源 kept")
    for s in idx["samples"]:
        sid = s["id"]
        truth_p = root / sid / "capture" / "private" / "d1_truth.json"
        if not truth_p.exists():
            continue
        tr = json.loads(truth_p.read_text(encoding="utf-8"))
        gts = list(tr.get("interferers") or [])
        if not gts:
            continue
        try:
            _ctx, resid, _tr, _obs, tmp = sse.residual_of(root, sid, station)
        except Exception as exc:
            print(f"{sid:<12} 跳过 {type(exc).__name__}: {exc}")
            continue
        try:
            fs = _ctx.fs
            raw, outside, kept = counts_for(resid, fs, fs * 0.45)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        n_gt = len(gts)
        gt_edge = 2 * sum(1 for g in gts if float(g.get("bandwidth_hz") or 0) >= 20e3)
        rows.append((n_gt, gt_edge, raw, outside, kept))
        print(f"{sid:<12}{n_gt:>5}{gt_edge:>8}{raw:>6}{outside:>8}{kept:>6}"
              f"   {kept / max(n_gt, 1):.1f}")
        if len(rows) >= n:
            break
    if not rows:
        print("没有可用样本")
        return 1
    a = np.array(rows, dtype=float)
    print("\n== 汇总（%d 个有干扰样本） ==" % len(rows))
    print(f"源数 n_gt: 均值 {a[:,0].mean():.2f}  GT 边数均值 {a[:,1].mean():.1f}")
    print(f"raw 强阶跃: 均值 {a[:,2].mean():.1f}（= {a[:,2].mean()/max(a[:,0].mean(),1e-9):.1f}/源）")
    print(f"outside 真边界候选: 均值 {a[:,3].mean():.1f}（= {a[:,3].mean()/max(a[:,0].mean(),1e-9):.1f}/源）")
    print(f"kept 接受边: 均值 {a[:,4].mean():.1f}（= {a[:,4].mean()/max(a[:,0].mean(),1e-9):.1f}/源）")
    exact = int(np.sum(np.abs(a[:, 4] - a[:, 1]) <= 1))
    over = int(np.sum(a[:, 4] > a[:, 1] + 1))
    under = int(np.sum(a[:, 4] < a[:, 1] - 1))
    print(f"kept 与 GT 边数（±1 内算准）: 准 {exact}/{len(rows)}  偏多 {over}  偏少 {under}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
