"""零 API：验证"参考噪声泄漏"修复 —— 同一份数据，只换 harness。

背景（2026-10-10 发现）：
`run_capture_agent._build_target_reference()` 声称构造"无噪声、无干扰"的每站目标参考，
但它只置了 `frontend.noise_enabled = False`；而 `cityrf.receiver.receive` 里
**colored / impulse / phase 三个开关不受 `noise_enabled` 管辖**（`receiver.py:46/:60/:69`），
于是这三项漏进了参考 → 工具 8（合作目标剥离）用的参考本身带噪声。

推论：前几轮看到的"加噪打坏工具"其实主要走的是**参考被污染**这条路，不是仪器被噪声打坏。
本驱动把**同一批已经生成好的数据**重新过一遍官方 gate（修复后），与旧数字对比。

  旧数字（修复前，同数据）：pre_noise 0.561 | dataset_d2(全档位) 0.367 |
                            V3_phase(1.0°) 0.429 | V7_final 0.541
对照纪律：pre_noise 这一栏**必须逐项不变**（它的噪声开关本来就全关，修复对它应为 no-op）。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import os
import subprocess
import sys
from pathlib import Path

_AD = Path(__file__).resolve().parent
_V3 = _AD.parent
_OUT = _V3 / "_ab_d2"
_RAD = _AD / "rad_cls"

ROOTS = {
    "R0_pre_noise": _V3 / "_pre_noise_20261010" / "dataset_d2",
    "R1_full_profile": _V3 / "dataset_d2",
    "R2_phase_only": _OUT / "V3_phase",
    "R3_final_candidate": _OUT / "V7_final",
}


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    log = _OUT / "_driver4.log"
    for name, root in ROOTS.items():
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n########## {name}  root={root}\n")
            fh.flush()
            if not root.is_dir():
                fh.write(f"!! 数据根不存在，跳过\n")
                continue
            env = {**os.environ, "DSH_D2_ROOT": str(root), "PYTHONIOENCODING": "utf-8"}
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
