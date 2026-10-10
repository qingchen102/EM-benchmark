"""只读核实：D1 可见性口径 + D3/D4 是否同源（零 token）。

发现疑点：
  A) 审计里 D1 的"干扰可见站数"全是 0——大概率是我读错了字段（D1 真值可能没有
     achieved/noise_floor_dbm），需要用 per_station 的实测 SNR 重算。
  B) D3 与 D4 的干扰类型分布、目标统计**逐位相同**（Tone_jam 18 / Noise_jam 17 /
     Chirp_jam 15 / Bluetooth_FHSS 5 / DECT 4 / ...；目标 SNR 中位同为 6.3、
     范围同为 [-2.4, 31.4]）→ 疑似同 seed 同场景，D4 = D3 + GNSS_spoof。
     若成立，则"D3/D4 两档"区分度存疑。
本脚本把这两点查实。

用法：python _audit_d34_check.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
OUT = HERE / "_audit_d34_check.txt"
_buf: list[str] = []


def w(s: str = "") -> None:
    _buf.append(s)


def load(root: Path):
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    out = {}
    for s in idx["samples"]:
        p = root / s["id"] / "capture" / "private" / "d1_truth.json"
        if p.is_file():
            out[s["id"]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def main() -> None:
    d1 = load(_V3 / "dataset_d1")
    d2 = load(_V3 / "dataset_d2")
    d3 = load(_V3 / "dataset_d3")
    d4 = load(_V3 / "dataset_d4")

    # ---------- A. D1 可见性（用 per_station 的实测 SNR 判） ----------
    w("=" * 78)
    w("A. D1 干扰可见性（真值字段核查 + 用 per_station.snr_db 重算可见站数）")
    w("=" * 78)
    sid0 = sorted(d1)[0]
    t0 = d1[sid0]
    w(f"\n[{sid0}] 顶层键: {sorted(t0.keys())}")
    w(f"  有 achieved 吗: {'achieved' in t0}")
    w(f"  有 noise 吗: {'noise' in t0}")
    ifw0 = (t0.get("interferers") or [])
    if ifw0:
        w(f"  干扰[0] 键: {sorted(ifw0[0].keys())}")
        w(f"  干扰[0] per_station: {json.dumps(ifw0[0].get('per_station'), ensure_ascii=False)[:400]}")
    w(f"  目标 per_station: {json.dumps((t0.get('target') or {}).get('per_station'), ensure_ascii=False)[:300]}")

    hist = Counter()
    for sid, t in d1.items():
        for g in (t.get("interferers") or []):
            ps = g.get("per_station") or {}
            # 该站"能否看到"= 端口功率高出噪声底 >=3dB；D1 真值里若只有 snr_db
            # 则等价于 snr_db >= 3（信噪比口径与 INR 在单源场景同量级）
            nvis = 0
            for k, v in ps.items():
                if v.get("snr_db") is not None and float(v["snr_db"]) >= 3.0:
                    nvis += 1
                elif v.get("inr_db") is not None and float(v["inr_db"]) >= 3.0:
                    nvis += 1
            hist[nvis] += 1
    w(f"\n  D1 干扰的可见站数分布（用 per_station SNR>=3dB 重算）: {dict(sorted(hist.items()))}")

    # ---------- B. D3 / D4 同源性 ----------
    w("\n" + "=" * 78)
    w("B. D3 vs D4 是否同源（逐样本比对 GT）")
    w("=" * 78)

    def sig(t: dict):
        """场景指纹：目标位置/功率 + 干扰的 kind/位置/频偏/带宽（不含 spoof）"""
        tg = t.get("target") or {}
        parts = [tuple(round(float(x), 3) for x in (tg.get("position_m") or [])),
                 round(float(tg.get("power_dbm", 0)), 3),
                 tg.get("emitter")]
        for g in (t.get("interferers") or []):
            if g.get("kind") == "GNSS_spoof":
                continue
            parts.append((g.get("kind"),
                          tuple(round(float(x), 3) for x in (g.get("position_m") or [])),
                          round(float(g.get("freq_offset_hz", 0)), 1),
                          round(float(g.get("bandwidth_hz", 0)), 1)))
        return tuple(parts)

    n = min(len(d3), len(d4))
    same = 0
    diff_examples = []
    for i in range(n):
        s3, s4 = sorted(d3)[i], sorted(d4)[i]
        a, b = sig(d3[s3]), sig(d4[s4])
        if a == b:
            same += 1
        elif len(diff_examples) < 3:
            diff_examples.append((s3, s4, a, b))
    w(f"\n  同序比对（D3 第 i 个 vs D4 第 i 个，忽略 GNSS_spoof）: {same}/{n} 完全相同")
    w(f"  样本 id 是否同名: D3 前3 {sorted(d3)[:3]} | D4 前3 {sorted(d4)[:3]}")

    # 目标位置/SNR 逐样本相等？
    eq_pos = eq_snr = 0
    for i in range(n):
        t3, t4 = d3[sorted(d3)[i]], d4[sorted(d4)[i]]
        p3 = (t3.get("target") or {}).get("position_m") or []
        p4 = (t4.get("target") or {}).get("position_m") or []
        if p3 and p4 and np.allclose(p3[:2], p4[:2], atol=0.5):
            eq_pos += 1
        s3v = [(v.get("snr_db")) for v in (((t3.get("target") or {}).get("per_station")) or {}).values()]
        s4v = [(v.get("snr_db")) for v in (((t4.get("target") or {}).get("per_station")) or {}).values()]
        if s3v and s4v and np.allclose(s3v, s4v, atol=0.05):
            eq_snr += 1
    w(f"  目标位置逐样本相同: {eq_pos}/{n}")
    w(f"  目标逐站 SNR 逐样本相同: {eq_snr}/{n}")
    for e in diff_examples:
        w(f"\n  差异示例 {e[0]} vs {e[1]}:")
        w(f"    D3: {str(e[2])[:220]}")
        w(f"    D4: {str(e[3])[:220]}")

    # ---------- C. D2 干扰数（与 v3 README 声称的 2-3 比对） ----------
    w("\n" + "=" * 78)
    w("C. D2 每样本干扰数（v3 README 声称 '2-3 个共存辐射体'）")
    w("=" * 78)
    c = Counter(len(t.get("interferers") or []) for t in d2.values())
    w(f"  {dict(sorted(c.items()))}")

    text = "\n".join(_buf)
    OUT.write_text(text, encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
