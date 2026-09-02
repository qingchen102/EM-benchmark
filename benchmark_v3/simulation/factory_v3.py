"""v3 样本工厂骨架：低空辐射体驱动 + 环境信道 + v2 兼容输出。

与 v2 factory 的差异：
1. 干扰池 = 低空辐射体（emitters.EMITTERS），每个干扰增加 `emitter` 标签
   （辐射体溯源任务的答案轴；仅 ground_truth/metadata 可见，信息隔离不变）；
2. 信道链 = 阵列合成 → 低空信道（多径/IQ不平衡/相位噪声）→ LNA 饱和 → AWGN；
3. 输出 .npy (4, 1024) complex128 + 三 JSON（metadata/observations/ground_truth）
   与 v2 同构——v2 的 tools_v2 / evaluator 可直接消费 v3 样本。

状态：**骨架/草案**。辐射体清单、信道参数范围、溯源任务是否计分，
待与师兄确认后定稿；分离约束/防折叠等 v2 安全机制在 v3 定稿时补齐。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                     # benchmark_v3/
_V2SIM = _ROOT.parent / "benchmark_v2" / "simulation"
for _p in (str(_ROOT / "simulation"), str(_V2SIM), str(_V2SIM.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from em_signal_simulator.baseband import generate_baseband            # noqa: E402
from em_signal_simulator.channel import apply_freq_offset             # noqa: E402
from em_signal_simulator_v3 import channel_env                        # noqa: E402
from em_signal_simulator_v3.emitters import generate_emitter          # noqa: E402


def _steer_matrix(doa_deg: float, m: int) -> np.ndarray:
    idx = np.arange(m)
    return np.exp(1j * 2.0 * np.pi * 0.5 * np.sin(np.deg2rad(doa_deg)) * idx)[:, None]


def _apply_lna_saturation(x: np.ndarray, a_sat: float = 2.0, p: float = 2.0) -> np.ndarray:
    """Rapp 模型 LNA 饱和（与 v2 channel 同式）。"""
    return x / (1.0 + np.abs(x / a_sat) ** (2.0 * p)) ** (1.0 / (2.0 * p))


def build_sample(emitter_names: list[str], target_mod: str = "QPSK",
                 snr_db: float = 10.0, doas: list[float] | None = None,
                 inr_db: list[float] | None = None, rng=None,
                 length: int = 1024, m_ant: int = 4,
                 channel_v3: bool = True) -> dict:
    """生成一个 v3 样本（目标 + 低空辐射体干扰 + 低空信道）。

    emitter_names: 干扰辐射体名列表（0~2 个，来自 emitters.EMITTERS）
    doas:          各源 DOA（度），[0]=目标（默认 0°）；默认均匀取
    inr_db:        各干扰相对目标的功率比（dB），默认 [8, 14]
    """
    rng = rng or np.random.default_rng()
    doas = doas if doas is not None else [0.0] + [float(rng.uniform(-60, 60)) for _ in emitter_names]
    inr_db = inr_db if inr_db is not None else [8.0 + 6.0 * i for i in range(len(emitter_names))]

    n = length
    target_bw = float(rng.uniform(0.15, 0.60))
    target = generate_baseband(target_mod, n, rng=rng, bandwidth_normalized=target_bw)
    target = apply_freq_offset(target, 0.0)
    # 目标频率归一化（0dB 基准，修复 P1-3：v2 目标 RMS=1.0，v3 曾直接丢原始 target）
    target = target / np.sqrt(np.mean(np.abs(target) ** 2) + 1e-30)

    # 阵列合成：目标 0dB + 各干扰按 INR 缩放，各自 DOA 导向
    x = _steer_matrix(doas[0], m_ant) * target
    meta_interferers = []
    for k, ename in enumerate(emitter_names):
        e = generate_emitter(ename, rng, length=n)
        w = e["iq"] / (np.sqrt(np.mean(np.abs(e["iq"]) ** 2) + 1e-30))
        # 生成器返回带实测元数据，不再丢弃（含调制/带宽/意图）
        f_off = float(np.clip(rng.uniform(-0.45, 0.45), -0.45, 0.45))
        w = apply_freq_offset(w, f_off)
        scale = np.sqrt(10.0 ** (inr_db[k] / 10.0))
        x = x + _steer_matrix(doas[k + 1], m_ant) * (scale * w)[None, :]
        meta_interferers.append({
            "emitter": ename, "modulation": e["modulation"],
            "intent": e["intent"], "is_jamming": e["is_jamming"],
            "freq_offset_normalized": f_off,          # 修复 P0-1：用实际注入的偏移，非 0
            "doa_degree": float(doas[k + 1]), "inr_db": float(inr_db[k]),
            "bandwidth_normalized": e["bandwidth_normalized"],
        })

    # 低空环境信道（v3 新增）
    if channel_v3:
        x = channel_env.apply_multipath(x, rng)
        x = channel_env.apply_iq_imbalance(x, rng)
        x = channel_env.apply_phase_noise(x, rng)

    # LNA 饱和（blocking 物理机制，沿用 v2）+ AWGN（SNR = 目标/噪声）
    x = _apply_lna_saturation(x)
    p_t = float(np.mean(np.abs(x) ** 2)) + 1e-30
    noise_amp = np.sqrt(p_t * 10.0 ** (-snr_db / 10.0) / 2.0)
    x = x + noise_amp * (rng.standard_normal((m_ant, n)) + 1j * rng.standard_normal((m_ant, n)))

    return {
        "iq": x.astype(np.complex128),
        "ground_truth": {
            "num_sources": 1 + len(emitter_names),
            "target": {"modulation": target_mod, "doa_degree": float(doas[0]),
                       "bandwidth_normalized": target_bw},
            "interferers": meta_interferers,
            "snr_db": float(snr_db),
        },
    }


def write_sample(out_dir: Path, sample: dict, sid: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{sid}.npy", sample["iq"])
    gt = {"sample_id": sid, **sample["ground_truth"]}
    # v3 骨架：三 JSON 的完整同构（observations 等）待定稿时补齐
    (out_dir / f"{sid}.gt.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    s = build_sample(["Bluetooth_FHSS", "Radar_pulsed_LFM"], snr_db=10.0, rng=rng)
    print("iq:", s["iq"].shape, s["iq"].dtype)
    print(json.dumps(s["ground_truth"], ensure_ascii=False, indent=1))
