"""一次性探针：目标带宽先验的两条路径是否等价（public cooperative_sources vs private scenario 回退）。

只读。目的：判断把 gen_dataset 写的 `cooperative_sources` 恢复到 public（它被平台
replay 冲成了 []）会不会改变 `_target_bw_normalized` —— 即会不会动到残差提取口径。
等价 => 纯增量（只多出 link_quality_per_station 证据），不等价 => 先 A/B 再决定。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import run_capture_agent as rca  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
TIERS = ["d1", "d2", "d3", "d4"]


def main() -> int:
    bw = rca._preset_bw_table()
    print("preset 带宽受理表（accepts_bw, meas）——只列目标预设：")
    rows = []
    for tier in TIERS:
        ds = ROOT / f"dataset_{tier}"
        if not ds.is_dir():
            print(f"  [{tier}] 缺失：{ds}")
            continue
        for sid_dir in sorted(p for p in ds.iterdir() if p.is_dir()):
            sid = sid_dir.name
            if not sid.startswith(tier + "_"):
                continue
            cap = sid_dir / "capture"
            pub = cap / "public" / "observations.json"
            priv = cap / "private" / "scenario.json"
            if not (pub.is_file() and priv.is_file()):
                continue
            obs = json.loads(pub.read_text(encoding="utf-8"))
            sc = json.loads(priv.read_text(encoding="utf-8"))
            tgt = next(e for e in sc["emitters"] if e["id"] == "target")
            comp_bw = (tgt.get("components") or [{}])[0].get("bandwidth_hz")
            fs = float(obs["receivers"][0]["sample_rate_hz"])
            coop = (obs.get("cooperative_sources") or [{}])[0]
            n_pub = coop.get("nominal_bandwidth_hz")
            n_priv = comp_bw
            norm_pub = rca._target_bw_normalized(tgt["preset"], n_pub or 1e6, fs, bw)
            norm_priv = rca._target_bw_normalized(tgt["preset"], n_priv, fs, bw)
            rows.append((tier, sid, tgt["preset"], comp_bw, n_pub, fs, norm_pub, norm_priv))

    print(f"\n{'tier':4} {'sample':10} {'preset':18} {'comp_bw':>10} {'pub_nom':>8} "
          f"{'fs':>10} {'norm_pub':>9} {'norm_priv':>9} {'diff':>6}")
    diff = 0
    for tier, sid, preset, comp_bw, n_pub, fs, npub, npriv in rows:
        d = abs(npub - npriv)
        if d > 1e-12:
            diff += 1
        print(f"{tier:4} {sid:10} {preset:18} {str(comp_bw):>10} {str(n_pub):>8} "
              f"{fs:>10.0f} {npub:>9.4f} {npriv:>9.4f} {d:>6.1e}")
    print(f"\n不等价样本数 = {diff} / {len(rows)}")
    print("bw_table 中目标预设条目：")
    for _, _, preset, *_ in rows:
        print(f"  {preset:18} -> {bw.get(preset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
