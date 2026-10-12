"""只读原型：D1 门控判据的两个候选改法（零 API，不改 capture_tools.py）。

动机（2026-10-12 新档体检，`diag/_probe_d1_extra_bands.py` + `_probe_d1_peakmiss.py`）：
  · `_residual_bands` 用**固定 10 kHz 扫描步长**取样点（`PEAK_STEP_HZ/fs_bin = 100 bin`），
    而纯音谱线只有 ~1 bin 宽 → 扫描网格可能整条**跨过**真源（`d1_00026`：真源 576.6 kHz，
    网格落在 570/580 kHz，over 分别为 −0.6/1.9 dB，于是零带返回）；
  · 宽带 `noise` 源（80~180 kHz）的谱有起伏 → 门控把它切成 2~3 个碎片，
    碎片中心离真中心 >40 kHz（`d1_00092` 返回 730k/620k/580k，真值 668k）；
    新档 28 个"多余带"**全部**来自 noise 源（tone 源 0 个）。

本脚本把三种候选实现与现行实现放在同一份残差上对跑：
  grid   —— 现行（固定网格 + 分组），用于**保真自校**（应与工具输出逐位一致）
  lm     —— 候选点改为 `over` 的**局部极大**（按 step 窗口贪心稀释），网格不再跨过谱线
  lm+h4  —— 在 lm 之上，把"内部 min(over) ≥ 4 dB"的相邻组并成一带（迟滞连接）
  lm+h6  —— 同上，低门限 6 dB

用法：python -X utf8 diag\\_sweep_d1_gridfix.py <dataset_root> [<more_roots>...]
"""
from __future__ import annotations

import json as _json
import pathlib as _pl
import shutil
import sys
import tempfile

import numpy as np
from scipy.ndimage import maximum_filter, median_filter

_HERE = _pl.Path(__file__).resolve().parent
_V3 = _HERE.parent
for _p in (str(_V3), str(_V3 / "simulation"), str(_V3 / "城市低空电磁信号仿真平台" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct        # noqa: E402
import run_capture_agent as rca   # noqa: E402

TOL_HZ = 40e3
MODES = [("grid", "grid", None), ("lm", "lm", None),
         ("lm+h4", "lm", 4.0), ("lm+h6", "lm", 6.0)]


def bands_proto(resid: np.ndarray, fs: float, rx_half: float,
                kind: str, low_db: float | None) -> list[dict]:
    """`capture_tools._residual_bands` 的逐行复刻 + 两个候选变体。"""
    n = resid.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    inb = np.where((f >= -rx_half) & (f <= rx_half))[0]
    if inb.size == 0:
        return []
    psd = (np.abs(np.fft.fft(resid, axis=-1)) ** 2).mean(axis=0)
    db = 10 * np.log10(psd[inb] + 1e-30)
    fs_bin = fs / n
    k = max(int(round(ct.PEAK_FLOOR_WIN_HZ / fs_bin)) | 1, 5)
    if k >= db.size:
        k = max((db.size // 2) | 1, 5)
    over = db - median_filter(db, size=k, mode="nearest")
    step = max(int(round(ct.PEAK_STEP_HZ / fs_bin)), 1)

    if kind == "grid":
        cand = list(range(0, inb.size, step))
    else:
        mx = maximum_filter(over, size=step, mode="nearest")
        hot = np.where((over >= ct.PEAK_OVER_DB) & (over >= mx - 1e-9))[0]
        # 贪心稀释：同窗内只留 over 最大者（防止平台/起伏造出过多候选）
        keep: list[int] = []
        for ci in sorted(hot.tolist(), key=lambda i: -over[i]):
            if all(abs(ci - j) > step // 2 for j in keep):
                keep.append(ci)
        cand = sorted(keep)

    marks: list[tuple[int, float, float, float]] = []
    for ci in cand:
        o = float(over[ci])
        if o < ct.PEAK_OVER_DB:
            continue
        c = ct._band_coherence(resid, fs, float(f[inb[ci]]), ct.PEAK_COH_WIN_HZ)
        if not np.isfinite(c) or c < ct.PEAK_COH_MIN:
            continue
        marks.append((ci, float(f[inb[ci]]), o, c))
    if not marks:
        return []

    groups: list[list[tuple[int, float, float, float]]] = [[marks[0]]]
    for prev, item in zip(marks, marks[1:]):
        if abs(item[1] - prev[1]) <= 2 * step * fs_bin + 1.0:
            groups[-1].append(item)
        else:
            groups.append([item])
    if low_db is not None and len(groups) > 1:
        merged = [groups[0]]
        for g in groups[1:]:
            ia, ib = merged[-1][-1][0], g[0][0]
            if ib > ia and float(over[ia:ib + 1].min()) >= low_db:
                merged[-1].extend(g)
            else:
                merged.append(g)
        groups = merged

    bands: list[dict] = []
    for g in groups:
        best = max(g, key=lambda t: t[2])
        lo = min(t[1] for t in g) - ct.PEAK_COH_WIN_HZ
        hi = max(t[1] for t in g) + ct.PEAK_COH_WIN_HZ
        bands.append({"freq_offset_hz": round(best[1], 0), "over_floor_db": round(best[2], 1),
                      "bandwidth_hz": round(hi - lo, 0), "band_hz": [round(lo, 0), round(hi, 0)],
                      "coherence": round(best[3], 3)})
    bands.sort(key=lambda p: -p["over_floor_db"])
    return bands[:ct.PEAK_MAX_N]


def one(root: _pl.Path, sid: str) -> dict:
    cap = root / sid / "capture" / "public"
    priv = root / sid / "capture" / "private"
    obs = _json.loads((cap / "observations.json").read_text(encoding="utf-8"))
    tr = _json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    tmp = _pl.Path(tempfile.mkdtemp(prefix=f"gridfix_{sid}_"))
    try:
        ctx = ct.ToolContext(capture_dir=cap, cache_dir=tmp,
                             allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                             fs=float(obs["receivers"][0]["sample_rate_hz"]),
                             fc=float(obs["receivers"][0]["center_frequency_hz"]),
                             target_bw_normalized=0.25, aoi_center=[0, 0],
                             target_reference=rca._build_target_reference(priv, n))
        ct.set_context(ctx)
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        resid = np.asarray(ct._strip_target_at_station("station_a"), np.complex128)
        fs = float(obs["receivers"][0]["sample_rate_hz"])
        rx_half = float(occ["band_hz"][1])
        gts = [float(g["freq_offset_hz"]) for g in (tr.get("interferers") or [])]
        out = {"gts": gts, "tool": [float(p["freq_offset_hz"])
                                    for p in (occ.get("residual_peaks") or [])]}
        for name, kind, low in MODES:
            bb = bands_proto(resid, fs, rx_half, kind, low)
            out[name] = [float(b["freq_offset_hz"]) for b in bb]
            out[name + ":bw"] = [float(b["bandwidth_hz"]) for b in bb]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    names = ["tool"] + [m[0] for m in MODES]
    for raw in argv:
        root = _pl.Path(raw).resolve()
        idx = _json.loads((root / "index.json").read_text(encoding="utf-8"))
        st = {nm: {"clean_n": 0, "fa": 0, "cpk": 0, "dirty": 0, "cov": 0, "hit": 0, "dpk": 0,
                   "extra": 0, "bw": []} for nm in names}
        for s in idx["samples"]:
            r = one(root, s["id"])
            for nm in names:
                pk = r[nm]
                if not r["gts"]:
                    st[nm]["clean_n"] += 1
                    st[nm]["cpk"] += len(pk)
                    st[nm]["fa"] += 1 if pk else 0
                else:
                    st[nm]["dirty"] += 1
                    st[nm]["dpk"] += len(pk)
                    st[nm]["cov"] += 1 if pk else 0
                    near = [p for p in pk
                            if min(abs(p - g) for g in r["gts"]) <= TOL_HZ]
                    st[nm]["hit"] += 1 if near else 0
                    st[nm]["extra"] += len(pk) - len(near)
                for b in r.get(nm + ":bw", []):
                    st[nm]["bw"].append(b)
        print(f"\n==== {root.parent.name}/{root.name} ====")
        print(f"{'判据':8s} {'干净虚警':>10s} {'覆盖':>10s} {'频率命中':>10s} "
              f"{'多余带':>7s} {'带中位宽':>9s}")
        for nm in names:
            d = st[nm]
            bw = float(np.median(d["bw"])) / 1e3 if d["bw"] else float("nan")
            print(f"{nm:8s} {d['fa']:>4d}/{d['clean_n']:<5d} "
                  f"{d['cov']:>4d}/{d['dirty']:<5d} {d['hit']:>4d}/{d['dirty']:<5d} "
                  f"{d['extra']:>7d} {bw:>8.1f}k")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
