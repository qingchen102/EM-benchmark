"""零 API 的损失分解：D2 加噪到底是被哪个旋钮打坏的？

背景（`_probe_noise_power.py` 实测，d2_00001 零输入）：
  all off          -106.96 dBm  λ1/Σλ=0.128（=1/8，阵元独立）
  +colored 0dB     -101.51 dBm（**+5.45 dB**，远超解析预期的 +0.5 dB）λ 0.134
  +corr  ρ=0.2     -106.96 dBm（+0.00 dB）但 **λ1/Σλ 0.128→0.300**
  +phase 1.0°      -106.96 dBm（+0.00 dB）λ 0.128
  == D2 档位        -101.52 dBm（+5.44）λ 0.176
而 gate A/B（旧数据 vs 加噪数据，同代码）显示 tool-view 召回 0.561→0.367、
假簇 0.623→0.727。本驱动把三个旋钮**单独**跑一遍官方 gate，定位元凶。

做法：以加噪前的 `_pre_noise_20261010/dataset_d2` 为源，逐变体重放 50 样本，
再用 `DSH_D2_ROOT=<变体>` 跑官方 `rad_cls/oracle16.py`（同代码、只换数据根）。
全程零 API；输出 `benchmark_v3/_ab_d2/`（gitignore）。
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

VARIANTS = {
    "V1_colored": dict(colored_noise_enabled=True, colored_noise_db=0.0,
                       colored_bandwidth_hz=200e3,
                       phase_noise_enabled=False, correlated_noise_enabled=False),
    "V2_corr": dict(colored_noise_enabled=False, phase_noise_enabled=False,
                    correlated_noise_enabled=True, noise_correlation=0.2),
    "V3_phase": dict(colored_noise_enabled=False, phase_noise_enabled=True,
                     phase_noise_deg=1.0, correlated_noise_enabled=False),
}


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    log = _OUT / "_driver.log"
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
