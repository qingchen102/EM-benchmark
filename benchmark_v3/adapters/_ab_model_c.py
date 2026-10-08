"""零 API：模型 C（参考=无加性噪声、保留本振相位噪声）下的同一批数据对照。

问题背景（2026-10-10）：
`run_capture_agent._build_target_reference()` 构造工具 8（合作目标剥离）用的
"无噪声、无干扰"目标参考。它原先只置 `frontend.noise_enabled = False`，而
`cityrf.receiver.receive` 里 **colored / impulse 两个开关不受 master 管辖**
（`receiver.py:60 / :69`），相位噪声也独立（`receiver.py:46`）→ 加噪后参考被污染。

更关键的是：那份参考和数据是**同一次 `receive()` 调用参数**（`sc.seed` + `rx.id`
+ 同样的 `indices`），所以噪声 realization 完全相同 —— 参考里"漏"进去的噪声
会与数据里的噪声**在直减中相消**。于是加噪前几轮看到的数字，量的是
"参考漏了多少噪声"，而不是"仪器抗不抗噪"。

两种模型：
  A（全关，包括相位）—— 物理上错：参考不带本振相位，目标项不再抵消，
    残差里留下相位残影。实测 D2 tool-view 召回 0.561→**0.173**、假簇 →**0.879**（作废）。
  C（本次采用）—— 只关**加性**噪声（热噪/共模/有色/脉冲），**保留相位噪声**：
    相位噪声是本振的乘性误差，数据与参考走同一台接收机、同一时刻、同一 realization，
    目标项精确抵消；加性噪声不再相消，**真实留在残差里**（这才是"噪声进入题目"）。
    残差里只剩：干扰项（带同一份相位调制）+ 完整的加性噪声。

本驱动在模型 C 下把同一批已有数据重过官方 gate，回答三件事：
  ① 加噪前基线必须逐项不变（no-op 自检，否则说明我改坏了别的路径）；
  ② 原档位（有色 0dB + ρ0.2 + 相噪 1.0°）在"噪声真的进残差"后掉多少；
  ③ 候选档位（有色 −5dB + ρ0.05 + 相噪 0.01°）能否把代价压到可接受。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_AD = Path(__file__).resolve().parent
_V3 = _AD.parent
_OUT = _V3 / "_ab_d2"
_RAD = _AD / "rad_cls"

ROOTS = {
    # 0) no-op 自检：加噪前的四档数据，噪声开关本来就全关，C 对它必须逐项不变
    "R0_pre_noise": _V3 / "_pre_noise_20261010" / "dataset_d2",
    # 1) 现在的 D2 数据集（= 有色0dB + ρ0.2 + 相噪1.0° 全档位）
    "R1_full_profile": _V3 / "dataset_d2",
    # 2) 候选档位（有色−5dB + ρ0.05 + 相噪0.01°）
    "R2_V7_final": _OUT / "V7_final",
    # 3) 只开有色 0dB（隔离有色）
    "R3_colored_only": _OUT / "V1_colored",
    # 4) 只开相噪 1.0°（隔离相位 FM）
    "R4_phase10": _OUT / "V3_phase",
    # 5) 只开相噪 0.2°（找安全上限）
    "R5_phase02": _OUT / "V6_phase02",
}


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    log = _OUT / "_driver5.log"
    for name, root in ROOTS.items():
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n########## {name}  root={root}\n")
            fh.flush()
            if not root.is_dir():
                fh.write("!! 数据根不存在，跳过\n")
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
