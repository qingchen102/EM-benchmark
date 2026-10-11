"""核对 mechanism.py 的接收通带常量（零 API，只读）。

发现：`mechanism.py:60  RX_HALF_HZ = 0.45 * 0.9 * 4e6` = **1.62e6**，注释写「±810 kHz」。
而 `gen_dataset.py:44  RX_FRAC = 0.45 * 0.9` 的语义是**全带宽占 fs 的比例**（`:241 rx_bw = fs * RX_FRAC`
= 1.62 MHz），平台接收机 FIR 用的是 `rx.bandwidth_hz / 2` 作为截止（`cityrf/receiver.py:83`
`firwin(f.filter_taps, rx.bandwidth_hz/2, fs=fs)`）→ 真实半通带 = **810 kHz**。

即 `RX_HALF_HZ` 少了 /2，宽了 2 倍 → |f0| ∈ (810 kHz, 1.62 MHz] 的源会被判成 `adjacent`（邻道），
而它的能量其实已被接收滤波切掉（本该报错/不该进机理轴）。

本脚本对四档真值做两遍普查，看是否有源会翻转。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

_PLUG = pathlib.Path(__file__).resolve()
_ADAPTERS = _PLUG.parent.parent
_V3 = _ADAPTERS.parent
sys.path.insert(0, str(_ADAPTERS))

import eval_dataset as ed  # noqa: E402
import mechanism as mk  # noqa: E402

ARMS = (("dataset_d1", "S0"), ("dataset_d2", "S2"), ("dataset_d3", "S2"), ("dataset_d4", "S2"))
OLD_HALF = 0.45 * 0.9 * 4e6      # 修正前的值（1.62 MHz，与自己的注释差 2×）
NEW_HALF = mk.RX_HALF_HZ         # 修正后（810 kHz，= OLD_HALF / 2）


def f0_bw(g: dict) -> tuple[float, float]:
    return float(g.get("freq_offset_hz") or 0.0), float(g.get("bandwidth_hz") or 0.0)


def main() -> int:
    out = [f"mechanism.RX_HALF_HZ = {mk.RX_HALF_HZ:.0f} Hz（修正后）  修正前的值 = {OLD_HALF:.0f} Hz"]
    flips = []
    for ds, sl in ARMS:
        root = _V3 / ds
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        stations = idx["observation_subsets"][sl]
        cur = collections.Counter()
        cor = collections.Counter()
        n = 0
        band = collections.Counter()
        for s in idx["samples"]:
            truth = ed.load_truth(root, s["id"])
            for g in ed.gt_at_slevel(truth, stations):
                n += 1
                a, b = f0_bw(g)
                cur[mk.derive(g.get("kind"), a, b, rx_half_hz=OLD_HALF)] += 1
                try:
                    cor[mk.derive(g.get("kind"), a, b)] += 1
                except ValueError:
                    cor["越界(报错)"] += 1
                    flips.append((ds, s["id"], g.get("kind"), a, b))
                if abs(a) > NEW_HALF:
                    band["> 810k"] += 1
        out.append(f"\n{ds} {sl}：GT 干扰 {n}")
        out.append("  旧口径(1.62M，未 /2)： " + "  ".join(f"{k}={v}" for k, v in sorted(cur.items()))")
        out.append("  新口径(810k，已 /2) ： " + "  ".join(f"{k}={v}" for k, v in sorted(cor.items()))")
        out.append(f"  |f0| > 810 kHz 的源： {band['> 810k']}")
    out.append(f"\n翻转样本数： {len(flips)}")
    for ds, sid, kind, a, b in flips:
        out.append(f"  {ds}/{sid}: kind={kind} f0={a:+.0f} Hz bw={b:.0f} Hz")
    text = "\n".join(out) + "\n"
    print(text, end="")
    dst = _V3 / "evidence" / "mech_passband_check_2026-10-12.txt"
    dst.write_text(text, encoding="utf-8")
    print(f"\n[已写入] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
