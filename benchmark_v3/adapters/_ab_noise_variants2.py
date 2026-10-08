"""零 API 的损失分解（第二轮）：相位噪声的刻度扫描 + 最终候选档位。

第一轮（`_ab_noise_variants.py`，结果见 `_ab_d2/_driver.log`）结论：
  基线（不加噪）      召回 0.561  假簇 0.623  条件 0.473  e2e 0.265  可解 0.591
  V1_colored 0dB      召回 0.571  假簇 0.588  条件 0.429  e2e 0.245  可解 0.568
  V2_corr    ρ=0.2    召回 0.459  假簇 0.685  条件 0.444  e2e 0.204  可解 0.484
  V3_phase   1.0°     召回 0.429  假簇 0.734  条件 0.310  e2e 0.133  可解 0.441
  完整 D2 档位        召回 0.367  假簇 0.727  条件 0.444  e2e 0.163  可解 0.395

→ **相位噪声是单旋钮里最大的伤害源**（召回 −13.2pp、条件 −16.3pp），而
  `_probe_noise_power.py` 完全没测出来——因为那个探针是**零输入**测的，
  而相位噪声是**乘在信号上**的（不改噪声底、不改噪声的空间形状），
  零输入测试对它天然失明。这是该探针的口径漏洞，不是平台 bug。

相位噪声在这里的语义（`src/cityrf/receiver.py:50`）：
  `step = sigma * real(gaussian(n, seed))`，`common = exp(1j*cumsum(step))`
  → `phase_noise_deg` 是**逐采样点的相位步长**，不是整窗积分相位误差。
  逐点步长 sigma 对应瞬时频偏 sigma*fs/(2π)：
     1.0°  → 11.1 kHz/步，窗内累积 3.49 rad ≈ 200°（信号相位在整个窗里翻来覆去）
     0.05° → 0.56 kHz/步，累积 0.50°
     0.01° → 0.11 kHz/步，累积 0.10°
  真实相干阵列的 LO 相位噪声是**共模**的（不伤 DOA），整窗积分典型 0.1~1°；
  要做到"整窗积分 ~0.3°"，逐点步长应取 ~0.002~0.005°。
  平台还额外给每阵元叠 5% 的独立分量（`:53-54`），那部分才真的伤 DOA。

本轮扫三个相位刻度 + 一个"最终候选档位"，全部只重放且只跑官方 gate。
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

_OFF = dict(colored_noise_enabled=False, phase_noise_enabled=False,
            correlated_noise_enabled=False)


def _phase(deg: float) -> dict:
    return dict(_OFF, phase_noise_enabled=True, phase_noise_deg=deg)


VARIANTS = {
    "V4_phase01": _phase(0.01),
    "V5_phase005": _phase(0.05),
    "V6_phase02": _phase(0.2),
    "V7_final": dict(colored_noise_enabled=True, colored_noise_db=-5.0,
                     colored_bandwidth_hz=200e3,
                     phase_noise_enabled=True, phase_noise_deg=0.01,
                     correlated_noise_enabled=True, noise_correlation=0.05),
}


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    log = _OUT / "_driver2.log"
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
