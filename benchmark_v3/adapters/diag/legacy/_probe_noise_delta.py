"""临时探针（零 token）：核对"换数据后 D2 的 IQ 到底变了多少"。

背景：gate A/B（同代码，仅换数据根）显示 D2 tool-view 召回 0.561→0.367，
但 d2_00001 的 |iq|^2 与 matched_frac 几乎逐位相同 —— 需要先确认
换数据后每个样本的 IQ 差异幅度与噪声底，再谈归因。

输出：每样本 old/new 的
  - d1_truth.json 的 noise 块
  - truth.json 每站 receiver 的 expected/effective 输出噪声
  - iq_station_*.npy 的 mean|iq|^2、max|Δ|、20log10(rms(Δ)/rms(iq))
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_AD = Path(__file__).resolve().parent

ROOTS = {
    "old": _AD.parent / "_pre_noise_20261010" / "dataset_d2",
    "new": _AD.parent / "dataset_d2",
}
d = lambda x: 10 * np.log10(x + 1e-300)  # noqa: E731


def noise_block(root: Path, sid: str) -> dict:
    p = root / sid / "capture" / "private" / "d1_truth.json"
    t = json.loads(p.read_text(encoding="utf-8"))
    nz = t.get("noise") or {}
    keep = ("model", "nf_db", "noise_floor_dbm", "profile")
    return {k: nz.get(k) for k in keep if k in nz}


def rx_noise(root: Path, sid: str) -> dict:
    p = root / sid / "capture" / "private" / "truth.json"
    t = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for st in t["stations"]:
        rx = st["receiver"]
        e = rx.get("expected_output_noise_w")
        f = rx.get("effective_output_noise_w")
        out[st["receiver_id"]] = (
            round(d(e) + 30, 2) if e else None,
            round(d(f) + 30, 2) if f else None,
        )
    return out


def probe(sid: str) -> None:
    print(f"\n=== {sid} ===")
    arrs = {}
    for tag, root in ROOTS.items():
        cap = root / sid / "capture"
        if not cap.exists():
            print(f"  {tag}: 缺失 {cap}")
            return
        print(f"  [{tag}] noise={noise_block(root, sid)}")
        print(f"  [{tag}] rx(expected,effective) dBm={rx_noise(root, sid)}")
        pub = cap / "public"
        for st in ("station_a", "station_b", "station_c", "station_d"):
            p = pub / f"iq_{st}.npy"
            if p.exists():
                arrs[(tag, st)] = np.asarray(np.load(p), complex)
    for st in ("station_a", "station_b", "station_c", "station_d"):
        a, b = arrs.get(("old", st)), arrs.get(("new", st))
        if a is None or b is None:
            continue
        pa = float(np.mean(np.abs(a) ** 2))
        pb = float(np.mean(np.abs(b) ** 2))
        dl = b - a
        md = float(np.max(np.abs(dl)))
        pd = float(np.mean(np.abs(dl) ** 2))
        print(f"  {st}: |old|^2={d(pa):8.2f} |new|^2={d(pb):8.2f} dB "
              f"Δ(|new|-|old|)={d(pb / pa):6.2f} dB | max|Δ|={md:.3e} "
              f"rms(Δ)/rms(iq)={d(pd / pb):7.2f} dB")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["d2_00001", "d2_00006", "d2_00010"]):
        probe(s)
