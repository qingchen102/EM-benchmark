#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""时域特征能不能区分源？（零 API，只读）

动机：CNN 的错分落点在 Welder / Corona（Zigbee 0/28、WiFi_burst 0/24 全错），
      而这两类的时域形态（占空比 / 突发结构）与 Zigbee / WiFi 差别极大。
      本探针直接量：**只用时域（+少量谱）特征**，在真实提取波形上能把类分到多少。

做法：对每档前 N 个样本、station_a、每个可见 GT 源，用与 instrument gate 相同的
      GT 切片口径（真值位置导向 + pinv 解混 + 按真值载频变频）取逐源波形 w，
      再算特征：
        duty      活跃占比（平滑包络 ≥ 20% 峰值）
        bursts    活跃段个数
        burst_ms  活跃段中位长度（ms）
        cv        包络变异系数
        papr      峰均比
        ku        复峰度 mean|x|^4 / mean|x|^2^2
        obw99     99% 占用带宽（kHz）
      然后做标准化 + 留一法 1-NN，给出总体准确率与混淆矩阵（与 CNN 的 0/N 对照）。

用法：python -X utf8 diag/_probe_timedomain_features.py [deck=d2] [--n 100] [--feat-only]
输出：benchmark_v3/evidence/_timedomain_features_<deck>.txt / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADP = HERE.parent                               # benchmark_v3/adapters
for _p in (str(ADP), str(ADP / "rad_cls")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oracle16 as o16              # noqa: E402
import run_capture_agent as rca     # noqa: E402
import _probe_burst_gate as bg      # noqa: E402

SMOOTH = 200          # 包络平滑窗（样点）= 50 µs，远小于最短突发也远大于单样点闪烁
ACT_FRAC = 0.20


def _env(w: np.ndarray) -> np.ndarray:
    e = np.abs(w)
    if SMOOTH > 1:
        k = np.ones(SMOOTH, dtype=np.float64) / SMOOTH
        e = np.convolve(e, k, mode="same")
    return e


def features(w: np.ndarray, fs: float) -> dict:
    e = _env(w)
    mx = float(e.max()) or 1.0
    act = e >= ACT_FRAC * mx
    duty = float(np.mean(act))
    d = np.diff(act.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    if act.size and act[0]:
        starts = np.r_[0, starts]
    ends = np.flatnonzero(d == -1) + 1
    if act.size and act[-1]:
        ends = np.r_[ends, act.size]
    lens = (ends[: len(starts)] - starts[: len(ends)]) if len(starts) else np.array([0])
    bursts = int(len(lens))
    burst_ms = float(np.median(lens) / fs * 1e3) if bursts else 0.0
    cv = float(np.std(e) / (np.mean(e) + 1e-30))
    p = np.abs(w) ** 2
    papr = float(p.max() / (p.mean() + 1e-30))
    ku = float(np.mean(np.abs(w) ** 4) / (np.mean(np.abs(w) ** 2) ** 2 + 1e-30))
    X = np.fft.fftshift(np.abs(np.fft.fft(w)) ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(w.size, 1 / fs))
    c = np.cumsum(X)
    if c[-1] > 0:
        i0 = int(np.searchsorted(c, 0.005 * c[-1]))
        i1 = int(np.searchsorted(c, 0.995 * c[-1]))
        obw99 = float(f[min(i1, f.size - 1)] - f[max(i0, 0)]) / 1e3
    else:
        obw99 = 0.0
    return {
        "duty": duty, "bursts": bursts, "burst_ms": burst_ms,
        "cv": cv, "papr": papr, "ku": ku, "obw99_khz": obw99,
    }


FEATS = ["duty", "bursts", "burst_ms", "cv", "papr", "ku", "obw99_khz"]


def collect(deck: str, n_max: int) -> list[dict]:
    root = o16.D2_ROOT
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for s in idx["samples"][:n_max]:
        cap = root / s["id"] / "capture"
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        full = json.loads((cap / "private" / "truth.json").read_text(encoding="utf-8"))
        gts = [(i + 1, g) for i, g in enumerate(truth["interferers"]) if g["kind"] in o16.CLS_IDX]
        if not gts:
            continue
        rec0 = obs["receivers"][0]
        st = "station_a"
        rec = next(x for x in obs["receivers"] if x["receiver_id"] == st)
        stf = next((x for x in full["stations"] if x["receiver_id"] == st), None)
        rx = stf["receiver"] if stf else None
        try:
            refs = rca._build_target_reference(cap / "private", rec0["shape"][1])
        except Exception:                                    # noqa: BLE001
            refs = None
        try:
            r = bg._resid_for(cap, obs, s["id"], st, refs)
        except Exception:                                    # noqa: BLE001
            continue
        if r is None or np.asarray(r).ndim != 2:
            continue
        fs = bg.fs_of(rec0)
        A = bg._steering(rec, [g["position_m"] for _gi, g in gts])
        if not np.isfinite(np.linalg.cond(A)) or np.linalg.cond(A) > 1e6:
            continue
        S = np.linalg.pinv(A) @ r
        t = np.arange(r.shape[1])
        noise_dbm, pw = None, {}
        if rx is not None:
            try:
                noise_dbm = 10 * np.log10(rx.get("effective_output_noise_w")
                                          or rx["expected_output_noise_w"]) + 30
                pw = {x["emitter_id"]: 10 * np.log10(float(np.mean(x["mean_port_power_w"])) + 1e-30) + 30
                      for x in stf["sources"]}
            except Exception:                                # noqa: BLE001
                noise_dbm = None
        for k, (gi, g) in enumerate(gts):
            eid = f"interferer_{gi}"
            if noise_dbm is not None and pw:
                if (pw.get(eid, -999) - noise_dbm) < 3.0:
                    continue
            w = S[k] * np.exp(-1j * 2 * np.pi * g["freq_offset_hz"] * t / fs)
            f = features(w, fs)
            f.update({"sid": s["id"], "kind": g["kind"]})
            rows.append(f)
        print(f"  {s['id']}: 源 {len(gts)} -> 累计 {len(rows)}", flush=True)
    return rows


def loo_nn(rows: list[dict]) -> tuple[float, list[list[int]], list[str]]:
    labels = sorted({r["kind"] for r in rows})
    li = {c: i for i, c in enumerate(labels)}
    X = np.array([[r[k] for k in FEATS] for r in rows], dtype=float)
    y = np.array([li[r["kind"]] for r in rows])
    X = np.log1p(np.maximum(X, 0.0))          # 重尾特征取对数
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    n, C = len(rows), len(labels)
    conf = np.zeros((C, C), dtype=int)
    for i in range(n):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        j = int(np.argmin(d))
        conf[y[i], y[j]] += 1
    acc = float(np.trace(conf) / max(n, 1))
    return acc, conf.tolist(), labels


def main(deck: str, n_max: int, feat_only: bool) -> None:
    root = o16.D2_ROOT
    print(f"[deck] {root.name}  n_max={n_max}", flush=True)
    rows = collect(deck, n_max)
    lines = [f"时域特征 vs 体制（deck={root.name}, station_a, GT 切片口径, n 源 = {len(rows)}）", ""]
    kinds = sorted({r["kind"] for r in rows})
    hdr = f"{'kind':16s}{'n':>5s}" + "".join(f"{k:>12s}" for k in FEATS)
    lines += [hdr, "-" * len(hdr)]
    for kd in kinds:
        sub = [r for r in rows if r["kind"] == kd]
        cells = "".join(f"{float(np.median([r[f] for r in sub])):12.3f}" for f in FEATS)
        lines.append(f"{kd:16s}{len(sub):5d}{cells}")
    if not feat_only and rows:
        acc, conf, labels = loo_nn(rows)
        lines += ["", f"留一法 1-NN（对数+标准化，特征 {FEATS}）：总体准确率 {acc:.3f}（{len(rows)} 源）", ""]
        w = max(len(x) for x in labels) + 1
        lines.append(" "*w + "".join(f"{l[:8]:>9s}" for l in labels))
        for i, l in enumerate(labels):
            row = [conf[i][j] for j in range(len(labels))]
            lines.append(f"{l:<{w}s}" + "".join(f"{v:9d}" for v in row))
        lines.append("")
        lines.append("对照：同一 checkpoint 的 CNN 在 GT 切片上 Zigbee 0/28、WiFi_burst 0/24；clean 波形上 0.967 / 0.867。")
    text = "\n".join(lines) + "\n"
    out = ADP.parent / "evidence" / f"_timedomain_features_{root.name}.txt"
    out.write_text(text, encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)
    print(f"[写入] {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    deck = argv[0] if argv else "d2"
    n = 100
    for a in sys.argv[1:]:
        if a.startswith("--n="):
            n = int(a.split("=", 1)[1])
    main(deck, n, "--feat-only" in sys.argv)
