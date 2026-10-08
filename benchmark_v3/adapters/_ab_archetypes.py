"""零 API：逐个"背景配方"（archetype）单独过官方 gate。

用户在 m03124 批准了噪声方案：热噪永开 + 按站随机配方 + 均衡分层。
按纪律"**每个配方都要先过一遍官方 gate，确认它不把题变成不可解**"（见 m03123），
本驱动把每个配方单独开到 50 个样本的**全部站**上，跑官方 `rad_cls/oracle16.py`。

对照锚点（已有数据，同代码只换数据根）：
  基线（纯热噪）            召回 0.561  假簇 0.623  可解 0.591
  V1 colored 0dB            召回 0.571  假簇 0.588  可解 0.568
  V2 corr ρ=0.2             召回 0.459  假簇 0.685  可解 0.484
  V3 phase 1.0°             召回 0.429  假簇 0.734  可解 0.441
  V4 phase 0.01°            召回 0.561  假簇 0.618  可解 0.591  ← 与基线逐项相同
本轮验证**最终刻度**下的三个非热噪配方（全部叠上 0.01° 相噪，即新方案里的固有项）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_AD = Path(__file__).resolve().parent
_SRC = _AD.parent / "_pre_noise_20261010" / "dataset_d2"
_OUT = _AD.parent / "_ab_d2"
_RAD = _AD / "rad_cls"

# 新方案里 d2–d4 的固有项：微小 LO 相噪（V4 已证明对 gate 无影响）
_BASE = dict(phase_noise_enabled=True, phase_noise_deg=0.01)

VARIANTS = {
    "A1_colored_m5": dict(_BASE, colored_noise_enabled=True, colored_noise_db=-5.0,
                          colored_bandwidth_hz=200e3),
    "A2_corr005": dict(_BASE, correlated_noise_enabled=True, noise_correlation=0.05),
    "A3_impulse25": dict(_BASE, impulse_noise_enabled=True, impulse_rate_hz=200.0,
                         impulse_power_db=25.0, impulse_width_us=5.0),
}


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    log = _OUT / "_driver3.log"
    for name, ov in VARIANTS.items():
        out = _OUT / name
        if out.exists():
            shutil.rmtree(out)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n########## {name}  override={json.dumps(ov)}\n")
            fh.flush()
            subprocess.run(
                [sys.executable, "-u", "regen_noise.py", "--src", str(_SRC),
                 "--out", str(out), "--difficulty", "d2", "--override", json.dumps(ov)],
                cwd=str(_AD), stdout=fh, stderr=subprocess.STDOUT, check=False)
            fh.write(f"---- {name}: regen 完成，开始官方 gate ----\n")
            fh.flush()
            env = {**os.environ, "DSH_D2_ROOT": str(out), "PYTHONIOENCODING": "utf-8"}
            subprocess.run([sys.executable, "-u", "oracle16.py"],
                           cwd=str(_RAD), stdout=fh, stderr=subprocess.STDOUT,
                           env=env, check=False)
            fh.write(f"---- {name}: gate 完成 ----\n")
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n########## 全部完成\n")
    print("done ->", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
