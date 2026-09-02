"""v3 样本工厂：城市低空辐射体 + 环境信道 + v2 兼容三 JSON 输出。

相对骨架版的升级：
- P1-5：移植 v2 的分离约束（Δfreq≥0.15 或 ΔDOA≥15°）、奈奎斯特防折叠
  （复用 v2 `_valid_offset_range`/`_sample_away_from`）、INR≥3 可检测性约束；
- P1-6：三 JSON 完整同构（metadata / observations / ground_truth），与 v2
  evaluator 可直接消费——样本含辐射体溯源 + 敌我意图两个新答案轴；
- P1-3/P0-1：目标 RMS 归一化、干扰频偏 GT 记录实际注入值（已在前一轮修复）。

输出 .npy (4,1024) complex128 + 三 JSON。
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
from em_signal_simulator.factory import (_valid_offset_range,         # noqa: E402
                                         _sample_away_from)
from em_signal_simulator_v3 import channel_env                        # noqa: E402
from em_signal_simulator_v3.emitters import generate_emitter          # noqa: E402

# 常量（与 v2 factory 对齐）
FREQ_MIN_SEP = 0.15
DOA_MIN_SEP = 15.0
DOA_RANGE = (-60.0, 60.0)
MIN_INR_DB = 3.0
CENTER_FREQ_HZ = 2.4e9
SAMPLING_RATE_HZ = 20e6
WAVELENGTH_M = 3e8 / CENTER_FREQ_HZ
ANTENNA_SPACING_M = WAVELENGTH_M / 2.0


def _steer_matrix(doa_deg: float, m: int) -> np.ndarray:
    idx = np.arange(m)
    return np.exp(1j * 2.0 * np.pi * 0.5 * np.sin(np.deg2rad(doa_deg)) * idx)[:, None]


def _apply_lna_saturation(x: np.ndarray, a_sat: float = 2.0, p: float = 2.0) -> np.ndarray:
    """Rapp 模型 LNA 饱和（与 v2 channel 同式）。"""
    return x / (1.0 + np.abs(x / a_sat) ** (2.0 * p)) ** (1.0 / (2.0 * p))


def _sample_doa(rng):
    return float(rng.uniform(*DOA_RANGE))


def _enforce_separation(interferers, rng):
    """移植 v2 `_correct_separation`：两两满足 Δfreq≥0.15 或 ΔDOA≥15°。

    interferers: list[dict]，含 freq_offset_normalized / doa_degree / bandwidth_normalized。
    就地修正。
    """
    if len(interferers) < 2:
        return
    for _ in range(20):
        ok = True
        for i in range(len(interferers)):
            for j in range(i + 1, len(interferers)):
                a, b = interferers[i], interferers[j]
                df = abs(a["freq_offset_normalized"] - b["freq_offset_normalized"])
                ddoa = abs(a["doa_degree"] - b["doa_degree"])
                if df >= FREQ_MIN_SEP or ddoa >= DOA_MIN_SEP:
                    continue
                ok = False
                others_f = [interferers[k]["freq_offset_normalized"]
                            for k in range(len(interferers)) if k != j]
                bounds = _valid_offset_range(b["bandwidth_normalized"])
                new_f = _sample_away_from(rng, others_f, bounds, FREQ_MIN_SEP)
                if new_f is not None:
                    b["freq_offset_normalized"] = new_f
                    continue
                others_d = [interferers[k]["doa_degree"]
                            for k in range(len(interferers)) if k != j]
                new_d = _sample_away_from(rng, others_d, DOA_RANGE, DOA_MIN_SEP)
                if new_d is not None:
                    b["doa_degree"] = new_d
        if ok:
            return
    # 兜底：确定性微调
    for i in range(len(interferers)):
        for j in range(i + 1, len(interferers)):
            a, b = interferers[i], interferers[j]
            df = abs(a["freq_offset_normalized"] - b["freq_offset_normalized"])
            ddoa = abs(a["doa_degree"] - b["doa_degree"])
            if df < FREQ_MIN_SEP and ddoa < DOA_MIN_SEP:
                sign = 1.0 if b["freq_offset_normalized"] >= 0 else -1.0
                lo, hi = _valid_offset_range(b["bandwidth_normalized"])
                b["freq_offset_normalized"] = float(np.clip(
                    b["freq_offset_normalized"] + sign * (FREQ_MIN_SEP - df + 1e-3), lo, hi))
                b["doa_degree"] = float(np.clip(
                    b["doa_degree"] + np.sign(b["doa_degree"] or 1.0)
                    * (DOA_MIN_SEP - ddoa + 0.5), *DOA_RANGE))


def _sample_inr(rng):
    """INR≥3dB 可检测性约束（干扰功率比 = 采样值，保证 INR=power+snr≥3）。"""
    return float(rng.uniform(3.0, 18.0))


def build_sample(emitter_names: list[str], target_mod: str = "QPSK",
                 snr_db: float = 10.0, doas: list[float] | None = None,
                 inr_db: list[float] | None = None, rng=None,
                 length: int = 1024, m_ant: int = 4,
                 channel_v3: bool = True) -> dict:
    """生成一个 v3 样本（目标 + 低空辐射体干扰 + 低空信道），含三 JSON 数据。

    emitter_names: 干扰辐射体名列表（0~2 个，来自 emitters.EMITTERS）
    输出含 `sources`（扁平，含 role/emitter/intent/is_jamming/v2_category...）
    """
    rng = rng or np.random.default_rng()
    n = length

    # ---- 采样干扰参数（防折叠 + 分离约束 + INR≥3）----
    interferers = []
    for k, ename in enumerate(emitter_names):
        e = generate_emitter(ename, rng, length=n)
        bw = e["bandwidth_normalized"]
        lo, hi = _valid_offset_range(bw)
        f_off = float(rng.uniform(lo, hi)) if hi > lo else 0.0
        interferers.append({
            "emitter": ename, "modulation": e["modulation"],
            "intent": e["intent"], "is_jamming": e["is_jamming"],
            "bandwidth_normalized": bw,
            "freq_offset_normalized": f_off,
            "doa_degree": _sample_doa(rng),
            "power_ratio_db": _sample_inr(rng),
            "scene": e["scene"],
        })
    _enforce_separation(interferers, rng)

    # ---- 目标（0dB，RMS 归一化）----
    target_bw = float(rng.uniform(0.15, 0.60))
    target = generate_baseband(target_mod, n, rng=rng, bandwidth_normalized=target_bw)
    target = target / np.sqrt(np.mean(np.abs(target) ** 2) + 1e-30)

    # ---- 阵列合成 ----
    x = _steer_matrix(0.0, m_ant) * target
    for itf in interferers:
        w = generate_emitter(itf["emitter"], rng, length=n)["iq"]
        w = w / np.sqrt(np.mean(np.abs(w) ** 2) + 1e-30)
        w = apply_freq_offset(w, itf["freq_offset_normalized"])
        scale = np.sqrt(10.0 ** (itf["power_ratio_db"] / 10.0))
        x = x + _steer_matrix(itf["doa_degree"], m_ant) * (scale * w)[None, :]

    # ---- 低空环境信道（v3 新增）----
    if channel_v3:
        x = channel_env.apply_multipath(x, rng)
        x = channel_env.apply_iq_imbalance(x, rng)
        x = channel_env.apply_phase_noise(x, rng)

    # ---- LNA 饱和 + AWGN（SNR = 目标/噪声）----
    x = _apply_lna_saturation(x)
    p_t = float(np.mean(np.abs(x) ** 2)) + 1e-30
    noise_amp = np.sqrt(p_t * 10.0 ** (-snr_db / 10.0) / 2.0)
    x = x + noise_amp * (rng.standard_normal((m_ant, n)) + 1j * rng.standard_normal((m_ant, n)))

    # ---- sources 扁平结构（与 v2 evaluator 消费兼容）----
    sources = [{"source_id": 0, "role": "target", "modulation": target_mod,
                "power_ratio_db": 0.0, "freq_offset_normalized": 0.0,
                "freq_offset_hz": 0.0, "doa_degree": 0.0,
                "bandwidth_normalized": target_bw}]
    for sid, itf in enumerate(interferers, start=1):
        # v2_category 兼容映射（敌我/溯源是新轴，四类为粗分类兼容）
        cat = "blocking" if itf["is_jamming"] else "co_channel"
        sources.append({
            "source_id": sid, "role": "interferer",
            "modulation": itf["modulation"], "emitter": itf["emitter"],
            "intent": itf["intent"], "is_jamming": itf["is_jamming"],
            "v2_category": cat, "scene": itf["scene"],
            "power_ratio_db": itf["power_ratio_db"],
            "freq_offset_normalized": itf["freq_offset_normalized"],
            "freq_offset_hz": itf["freq_offset_normalized"] * SAMPLING_RATE_HZ,
            "doa_degree": itf["doa_degree"],
            "bandwidth_normalized": itf["bandwidth_normalized"],
            "inr_db": itf["power_ratio_db"] + snr_db,
        })

    receiver = {"center_frequency_hz": CENTER_FREQ_HZ,
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "wavelength_m": WAVELENGTH_M, "antenna_spacing_m": ANTENNA_SPACING_M}

    return {
        "iq": x.astype(np.complex128),
        "sources": sources,
        "num_sources": len(sources),
        "snr_db": float(snr_db),
        "receiver": receiver,
        "target_bandwidth_normalized": target_bw,
        "target_modulation": target_mod,
        "channel_v3": channel_v3,
        "intents": [s["intent"] for s in sources[1:]],
        "emitters": [s["emitter"] for s in sources[1:]],
    }


def write_dataset(out_dir: Path, samples: list[dict], prefix: str = "sample_"):
    """写出一批样本 + 三 JSON（与 v2 同构：metadata 含真值 / observations Agent 可见 / ground_truth 评估用）。

    信息隔离：observations 不含 emitter/intent/真值参数；ground_truth 含全部真值（含新轴）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ground_truths, observations, metadatas = [], [], []
    for idx, s in enumerate(samples):
        sid = f"{prefix}{idx:05d}"
        fname = f"{sid}.npy"
        np.save(out_dir / fname, s["iq"])
        gt = {"sample_id": sid, "num_sources": s["num_sources"],
              "num_samples": s["iq"].shape[1], "snr_db": s["snr_db"],
              "bandwidth_definition": "obw99", "receiver": s["receiver"],
              "lna_saturation": {"applied": True, "model": "rapp", "a_sat": 2.0, "p": 2.0, "saturation_db": 6.0},
              "array_mismatch": {"gain_error_std": 0.0, "phase_error_std_deg": 0.0},
              "channel_v3": s["channel_v3"], "sources": s["sources"]}
        ground_truths.append({"file": fname, "ground_truth": gt})
        obs = {"sample_id": sid, "num_samples": s["iq"].shape[1],
               "receiver": s["receiver"], "target_modulation": s["target_modulation"],
               "array_mismatch_config": {"gain_error_std": 0.0, "phase_error_std_deg": 0.0},
               "target_bandwidth_normalized": s["target_bandwidth_normalized"]}
        observations.append({"file": fname, "observation": obs})
        metadatas.append({"file": fname, "sample_id": sid,
                          "ground_truth": gt})  # metadata 与 ground_truth 同构含真值
    (out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truths, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "observations.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadatas, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(samples)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    s = build_sample(["GNSS_jam", "Microwave"], snr_db=10.0, rng=rng)
    print("iq:", s["iq"].shape, s["iq"].dtype, "| 源数:", s["num_sources"])
    for src in s["sources"][1:]:
        print(f"  {src['emitter']:14s} intent={src['intent']:11s} "
              f"freq={src['freq_offset_normalized']:+.3f} doa={src['doa_degree']:+.1f} "
              f"inr={src['inr_db']:.1f} cat={src['v2_category']}")
