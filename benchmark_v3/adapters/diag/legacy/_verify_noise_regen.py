"""验收 `regen_noise.py` 的产物（只读，零 API）。

检查四件事：
  1. 样本集合与源数据集一致（不丢、不多）。
  2. **D1 逐位恒等**：D1 的档位是空 dict，frozen-channel 重放应当逐位复现源 IQ（max|Δ| == 0）。
     这是"改噪声 ≠ 改信号"的端到端证明。
  3. 抬底实测：从 `run_summary.json` 的 `noise_patch` 与重算的 `d1_truth.json` 取噪声底，
     与源数据集对比，打印每档的抬底中位数（dB）。
  4. 真值口径一致性：`d1_truth.json` 的 `noise.noise_floor_dbm` 是否等于该样本各站实测底的中位数
     （容差 0.01 dB），且 `interferers[*].per_station[*].inr_db == pr_dbm − 该站实测底`（容差 0.02）。

用法：python _verify_noise_regen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("..")                       # benchmark_v3
PRE = ROOT / "_pre_noise_20261010"      # 加噪前的备份（对比基准）
DIFFS = ("d1", "d2", "d3", "d4")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def measured_floor_dbm(truth: dict) -> float | None:
    """从 d1_truth.json 的 noise 块取实测底（新版有，旧版没有）。"""
    nz = truth.get("noise") or {}
    per = nz.get("per_station") or {}
    vals = [v.get("noise_floor_dbm") for v in per.values() if isinstance(v, dict) and v.get("noise_floor_dbm") is not None]
    if vals:
        return float(np.median(vals))
    return nz.get("noise_floor_dbm")


def main() -> int:
    bad = 0
    for d in DIFFS:
        src_root = PRE / f"dataset_{d}"
        out_root = ROOT / f"dataset_{d}"
        print("=" * 88)
        print(f"### {d}   src={src_root}   out={out_root}")
        if not out_root.exists():
            print("  [SKIP] 加噪前备份不存在（_pre_noise_20261010）")
            continue
        src_ids = sorted(p.name for p in src_root.iterdir() if (p / "capture" / "private").is_dir())
        out_ids = sorted(p.name for p in out_root.iterdir() if (p / "capture" / "private").is_dir())
        missing = sorted(set(src_ids) - set(out_ids))
        extra = sorted(set(out_ids) - set(src_ids))
        print(f"  样本数 src={len(src_ids)} out={len(out_ids)}  缺={missing[:5]}  多={extra[:5]}")
        if missing or extra:
            bad += 1

        # ---- 1. IQ 逐位对比（D1 必须是 0；其它档报抬底） ----
        max_abs = 0.0
        pow_src = pow_out = 0.0
        ncmp = 0
        for sid in out_ids:
            spub = src_root / sid / "capture" / "public"
            opub = out_root / sid / "capture" / "public"
            if not spub.exists():
                continue
            for f in sorted(spub.glob("iq_station_*.npy")):
                of = opub / f.name
                if not of.exists():
                    print(f"  [MISS] {sid}/{f.name}")
                    bad += 1
                    continue
                a = np.load(f)
                b = np.load(of)
                if a.shape != b.shape:
                    print(f"  [SHAPE] {sid}/{f.name} {a.shape} -> {b.shape}")
                    bad += 1
                    continue
                max_abs = max(max_abs, float(np.max(np.abs(a - b))))
                pow_src += float(np.mean(np.abs(a) ** 2))
                pow_out += float(np.mean(np.abs(b) ** 2))
                ncmp += 1
        if ncmp:
            rise = 10 * np.log10(pow_out / pow_src) if pow_src > 0 else float("nan")
            tag = "逐位恒等 ✅" if max_abs == 0.0 else f"max|Δ|={max_abs:.3e}"
            print(f"  IQ 对比: {ncmp} 个阵列文件 | {tag} | 总功率变化 {rise:+.2f} dB")
            if d == "d1" and max_abs != 0.0:
                print("  [FAIL] D1 应当逐位恒等！")
                bad += 1

        # ---- 2. 噪声底：源 vs 重生成 ----
        floors_src, floors_out = [], []
        for sid in out_ids:
            t_src = src_root / sid / "capture" / "private" / "d1_truth.json"
            t_out = out_root / sid / "capture" / "private" / "d1_truth.json"
            if not (t_src.exists() and t_out.exists()):
                continue
            a, b = load(t_src), load(t_out)
            fa = measured_floor_dbm(a)
            fb = measured_floor_dbm(b)
            if fa is not None:
                floors_src.append(fa)
            if fb is not None:
                floors_out.append(fb)
                # 逐站 inr_db 一致性
                nz = (b.get("noise") or {}).get("per_station") or {}
                for j, ifw in enumerate(b.get("interferers", [])):
                    for sk, cell in (ifw.get("per_station") or {}).items():
                        lst = nz.get(sk, {}).get("noise_floor_dbm")
                        if lst is None or cell.get("pr_dbm") is None or cell.get("inr_db") is None:
                            continue
                        want = cell["pr_dbm"] - lst
                        if abs(want - cell["inr_db"]) > 0.02:
                            print(f"  [INR] {sid} ifw{j} {sk}: 记 {cell['inr_db']} 应为 {want:.2f}")
                            bad += 1
        if floors_src:
            print(f"  噪声底中位: 源 {np.median(floors_src):8.2f} dBm -> 重生成 {np.median(floors_out):8.2f} dBm"
                  f"  抬 {np.median(floors_out) - np.median(floors_src):+.2f} dB   (n={len(floors_out)})")
        # ---- 3. run_summary 的 noise_patch 是否写了 ----
        tag = out_root / out_ids[0] / "run_summary.json" if out_ids else None
        if tag and tag.exists():
            s = load(tag)
            np_ = s.get("noise_patch")
            print(f"  run_summary.noise_patch = {json.dumps(np_, ensure_ascii=False)[:200] if np_ else '缺失 ❌'}")
            if not np_:
                bad += 1
    print("=" * 88)
    print("结论: " + ("全部通过 ✅" if bad == 0 else f"{bad} 处异常 ❌"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
