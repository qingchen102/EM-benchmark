"""临时探针（零 token）：每个噪声旋钮实际加了多少功率、加了多少"空间相干"。

动机：D2 档位（colored 0 dB + phase 1.0° + ρ=0.2）实测把噪声底抬了 **+5.4 dB**
（远超"200 kHz/1.62 MHz 有色 ≈ +0.5 dB"的解析预期），且加噪后 gate tool-view
**假簇率 0.623→0.727**。假簇是**空间**现象，所以本探针除了功率，还算
8 阵元空间协方差 R = y y^H / N 的**特征值谱**：
  - 纯热噪（阵元独立）→ 特征值近乎相等（λ1/Σλ ≈ 1/8 = 0.125）
  - 有共模（ρ>0）→ 出现一个大特征值（λ1/Σλ → 1）
共模噪声 = 一个"固定方向的假源"，正是聚类假簇的候选来源。

做法：取一个真实 D2 样本的 scenario.json，逐档位把 `Frontend` 旋钮打开，
**零输入**调 `cityrf.receiver.receive`（种子只由 (seed, rx.id) 决定，与 composite 无关），
量输出功率与特征值谱。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_AD = Path(__file__).resolve().parent
sys.path.insert(0, str(_AD.parent / "城市低空电磁信号仿真平台" / "src"))

from cityrf.config import Scenario  # noqa: E402
from cityrf.receiver import receive  # noqa: E402

CAP = _AD.parent / "dataset_d2" / "d2_00001" / "capture"

OFF = dict(colored_noise_enabled=False, phase_noise_enabled=False,
           correlated_noise_enabled=False, impulse_noise_enabled=False)

VARIANTS = [
    ("scenario as-is", None),          # 不加 OFF 遮盖：看数据集里实际写的是什么
    ("all off(thermal)", {}),
    ("+colored  0dB", dict(colored_noise_enabled=True, colored_noise_db=0.0, colored_bandwidth_hz=200e3)),
    ("+colored  5dB", dict(colored_noise_enabled=True, colored_noise_db=5.0, colored_bandwidth_hz=200e3)),
    ("+colored 10dB", dict(colored_noise_enabled=True, colored_noise_db=10.0, colored_bandwidth_hz=200e3)),
    ("+corr rho=.2 ", dict(correlated_noise_enabled=True, noise_correlation=0.2)),
    ("+corr rho=.5 ", dict(correlated_noise_enabled=True, noise_correlation=0.5)),
    ("+phase  1.0d ", dict(phase_noise_enabled=True, phase_noise_deg=1.0)),
    ("+imp 200/25  ", dict(impulse_noise_enabled=True, impulse_rate_hz=200.0,
                           impulse_power_db=25.0, impulse_width_us=5.0)),
    ("== D2 档位   ", dict(colored_noise_enabled=True, colored_noise_db=0.0, colored_bandwidth_hz=200e3,
                           phase_noise_enabled=True, phase_noise_deg=1.0,
                           correlated_noise_enabled=True, noise_correlation=0.2)),
    ("== D3 档位   ", dict(colored_noise_enabled=True, colored_noise_db=5.0, colored_bandwidth_hz=200e3,
                           phase_noise_enabled=True, phase_noise_deg=1.5,
                           impulse_noise_enabled=True, impulse_rate_hz=200.0,
                           impulse_power_db=25.0, impulse_width_us=5.0)),
]


def main() -> None:
    sc = json.loads((CAP / "private" / "scenario.json").read_text(encoding="utf-8"))
    s = Scenario.model_validate(sc)
    fs, n = s.sample_rate_hz, s.num_samples
    rx0 = s.receivers[0]
    me = rx0.antenna.elements
    guard = rx0.frontend.filter_taps // 2
    print(f"样本={CAP.parent.name}  fs={fs/1e6:.2f}MHz  n={n}  阵元={me}  fs 带宽={rx0.bandwidth_hz/1e6:.3f}MHz\n")
    base = None
    print(f"{'variante':14s} {'输出底 dBm':>11s} {'抬升 dB':>8s} {'λ1/Σλ':>7s}  特征值(前4, 归一)")
    for name, over in VARIANTS:
        r = rx0.model_copy(deep=True)
        if over is None:            # 不加遮盖：让 scenario.json 自带的档位生效
            front = {}
        else:
            front = dict(OFF)
            front.update(over)
        for k, v in front.items():
            setattr(r.frontend, k, v)
        x = np.zeros((me, n + 2 * guard), complex)
        y, _meta = receive(x, r, fs, s.seed, start_index=-guard)
        y = y[:, guard:guard + n]
        p = float(np.mean(np.abs(y) ** 2))
        p_dbm = 10 * np.log10(p + 1e-300) + 30
        R = y @ y.conj().T / y.shape[1]
        w = np.linalg.eigvalsh(R)[::-1].real
        wn = w / w.sum()
        if base is None:
            base = p_dbm
        print(f"{name:14s} {p_dbm:11.2f} {p_dbm - base:8.2f} {wn[0]:7.3f}  "
              + " ".join(f"{v:.3f}" for v in wn[:4]))


if __name__ == "__main__":
    main()
