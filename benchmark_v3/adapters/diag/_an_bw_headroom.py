"""只读统计：源带宽相对接收机通带的余量（回答"非目标源带宽理论上能不能测"）。

对每个样本的每个干扰源：
  - 源名义带宽 bandwidth_hz、频偏 freq_offset_hz（d1_truth.json 的 interferers）
  - 接收机通带带宽 bandwidth_hz / 滤波阶数（private/truth.json 的 stations[*].receiver）
  - 判定源谱 [f0-bw/2, f0+bw/2] 是否完整落在通带 [-fs_rx/2 或 ±passband/2] 内
另外统计工具口径：聚类候选的 band_hz 宽度是否等于共道带宽度（同带内所有候选同一数值）。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
FS = 4.0e6


def recv_block(truth_rt: dict) -> dict:
    sts = truth_rt.get("stations") or []
    return (sts[0].get("receiver") if sts else None) or {}
def public_passband(sdir: Path) -> float | None:
    """接收机通带从 public observations 取（truth.json 的 receiver 块不含带宽）。"""
    p = sdir / "capture" / "public" / "observations.json"
    if not p.exists():
        return None
    obs = json.loads(p.read_text(encoding="utf-8"))
    for r in (obs.get("receivers") or []):
        if r.get("receiver_bandwidth_hz"):
            return float(r["receiver_bandwidth_hz"])
    return None
def main() -> int:
    print(f"{'档':<4}{'样本':>5}{'源数':>5}{'通带±kHz':>11}{'滤波':>6}"
          f"{'完整落带内':>11}{'边缘被截':>9}{'最宽源kHz':>11}{'源BW中位kHz':>13}")
    for d in ("d1", "d2", "d3", "d4"):
        droot = ROOT / f"dataset_{d}"
        if not droot.is_dir():
            continue
        n_s = n_src = n_in = n_clip = 0
        base_bw: list[float] = []
        widest = 0.0
        pb = taps = None
        for sdir in sorted(p for p in droot.iterdir() if p.is_dir()):
            priv = sdir / "capture" / "private"
            j1, jr = priv / "d1_truth.json", priv / "truth.json"
            if not (j1.exists() and jr.exists()):
                continue
            t1 = json.loads(j1.read_text(encoding="utf-8"))
            tr = json.loads(jr.read_text(encoding="utf-8"))
            rb = recv_block(tr)
            if pb is None:
                pb = public_passband(sdir)
                taps = len(rb)
            n_s += 1
            for s in (t1.get("interferers") or []):
                bw = float(s.get("bandwidth_hz") or 0.0)
                f0 = float(s.get("freq_offset_hz") or 0.0)
                n_src += 1
                base_bw.append(bw)
                widest = max(widest, bw)
                if pb:
                    lo, hi = f0 - bw / 2.0, f0 + bw / 2.0
                    if lo >= -pb / 2.0 and hi <= pb / 2.0:
                        n_in += 1
                    else:
                        n_clip += 1
        if not n_src:
            print(f"{d:<4}{n_s:>5}{n_src:>5}{'—':>11}{'—':>6}{'—':>11}{'—':>9}{'—':>11}{'—':>13}")
            continue
        base_bw.sort()
        med = base_bw[len(base_bw) // 2] / 1e3
        pbs = f"{pb/2e3:.0f}" if pb else "?"
        print(f"{d:<4}{n_s:>5}{n_src:>5}{pbs:>11}{str(taps):>6}"
              f"{n_in:>11}{n_clip:>9}{widest/1e3:>11.1f}{med:>13.1f}")

    # 工具口径：候选 band_hz 宽度
    print("\n--- 工具口径：D2 聚类候选的 band_hz 宽度（同带候选是否同一数值）---")
    try:
        sdir0 = ROOT / "dataset_d2" / "d2_00001"
        rb0 = recv_block(json.loads((sdir0 / "capture" / "private" / "truth.json")
                                    .read_text(encoding="utf-8")))
        print(f"\n[参考] d2_00001 truth.receiver 键: {sorted(rb0.keys())}")
        print(f"[参考] d2_00001 public 接收机通带: {public_passband(sdir0)}")
        print(f"[参考] 采样率: {rb0.get('sample_rate_hz')}  中心频率: {rb0.get('center_frequency_hz')}")
    except Exception as e:  # noqa: BLE001
        print(f"  (参考块跳过：{type(e).__name__}: {e})")

    # 每样本源数分布 + 共道重叠（与同档另一个源频带重叠的源数）
    print("\n--- 每样本源数分布 / 共道重叠 ---")
    for d in ("d1", "d2", "d3", "d4"):
        droot = ROOT / f"dataset_{d}"
        if not droot.is_dir():
            continue
        hist: dict[int, int] = {}
        n_src2 = n_overlap = 0
        for sdir in sorted(p for p in droot.iterdir() if p.is_dir()):
            j1 = sdir / "capture" / "private" / "d1_truth.json"
            if not j1.exists():
                continue
            t1 = json.loads(j1.read_text(encoding="utf-8"))
            srcs = list(t1.get("interferers") or [])
            hist[len(srcs)] = hist.get(len(srcs), 0) + 1
            for i, a in enumerate(srcs):
                n_src2 += 1
                fa, wa = float(a["freq_offset_hz"]), float(a["bandwidth_hz"])
                for j, b in enumerate(srcs):
                    if i == j:
                        continue
                    fb, wb = float(b["freq_offset_hz"]), float(b["bandwidth_hz"])
                    if abs(fa - fb) < (wa + wb) / 2.0:
                        n_overlap += 1
                        break
        keys = ", ".join(f"{k}源:{v}" for k, v in sorted(hist.items()))
        print(f"  {d}: 每样本 {keys}   共道重叠源 = {n_overlap}/{n_src2}"
              f"   → 独占带源 = {n_src2 - n_overlap}/{n_src2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
