"""调制识别工具上限验证（v3：特征集变体对比 + K 种子模板去噪）。

三级测试，用于区分"工具缺陷"与"模型综合缺陷"，并验证特征升级收益：

1. 纯波形上限：干净单源波形（施加频偏后走完整 Hann 切片+解旋路径），
   测试各特征变体 + 模板策略本身的判别力；
2. 数据集切片上限：对真实混合样本（目标+干扰+噪声），按 GT 频偏/带宽
   做 Hann 切片后分类 —— 即"假设 Agent 完美选定切片位置时工具能达到的上限"。

特征变体：
- baseline  : 现 tools_v2._feature_vector 的 7 维（OBW/平坦度/漂移/峰均比/峰度/PAPR/占空比）
- cumulants : baseline + 归一化高阶累积量 |C40|/P² 与 C42/P²
              （BPSK |C40|≈2 vs QPSK ≈1 vs OFDM ≈0 —— 当前 7 维对 BPSK/QPSK 物理盲）
- full      : cumulants + 包络 M-L 比 gamma_max（恒包络 GFSK/NFM vs 波动包络）

模板策略：single（单种子，现状）vs kseed（K 个随机种子取均值，降低模板抽样噪声）。

运行：python eval_mod_upperbound.py [--dataset dataset] [--repeats 3]
      [--variants baseline,cumulants,full] [--kseeds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_SIM_DIR = _ROOT / "simulation"
for _p in (_ROOT, _SIM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tools_v2
from em_signal_simulator.baseband import generate_baseband, measure_obw99
from em_signal_simulator.channel import apply_freq_offset
from em_signal_simulator.jamming import generate_interferer_waveform

MODS = ["BPSK", "QPSK", "GFSK", "LFM", "OFDM"]
BWS = [0.06, 0.15, 0.25, 0.35, 0.45, 0.55]


# ---------------- 特征变体 ----------------

def _cumulant_features(x):
    """归一化高阶累积量 (|C40|/P², C42/P²) + 包络 M-L 比 gamma_max。"""
    x = x - np.mean(x)
    p2 = float(np.mean(np.abs(x) ** 2))
    if p2 <= 0:
        return 0.0, 0.0, 0.0
    m20 = complex(np.mean(x ** 2))
    m40 = complex(np.mean(x ** 4))
    m42 = float(np.mean(np.abs(x) ** 4))
    c40 = abs(m40 - 3.0 * m20 ** 2) / p2 ** 2
    c42 = (m42 - abs(m20) ** 2 - 2.0 * p2 ** 2) / p2 ** 2
    z = np.abs(x) ** 2
    zd = z - np.mean(z)
    var_z = float(np.mean(zd ** 2)) + 1e-30
    dft = np.fft.fft(zd)
    gamma_max = float(np.max(np.abs(dft) ** 2) / (len(z) ** 2 * var_z ** 2))
    return float(c40), float(c42), gamma_max


def feature_variant(name: str):
    """返回 x -> np.ndarray 的特征函数。"""
    base = tools_v2._feature_vector

    def v_cumulants(x):
        return np.concatenate([base(x), np.array(_cumulant_features(x)[:2])])

    def v_full(x):
        return np.concatenate([base(x), np.array(_cumulant_features(x))])

    return {"baseline": base, "cumulants": v_cumulants, "full": v_full}[name]


# ---------------- 模板（带宽自适应 + 可选 K 种子均值） ----------------

_TPL_CACHE: dict = {}


def template_features(variant, name, bw, kseeds=1, length=1024):
    bw_q = round(float(np.clip(bw, 0.02, 0.95)) * 50.0) / 50.0
    key = (variant, name, bw_q, kseeds, length)
    if key in _TPL_CACHE:
        return _TPL_CACHE[key]
    fvec = feature_variant(variant)
    feats = []
    for s in range(max(int(kseeds), 1)):
        rng = np.random.default_rng(100 + s)
        from em_signal_simulator.baseband import MODULATIONS
        if name in MODULATIONS:
            w = generate_baseband(name, length, rng=rng, bandwidth_normalized=bw_q)
        else:
            w = generate_interferer_waveform(
                name, length, rng=rng,
                **tools_v2._waveform_kw_for_bandwidth(name, bw_q))
        feats.append(fvec(w))
    fv = np.mean(feats, axis=0)
    _TPL_CACHE[key] = fv
    return fv


def classify(x_bb, bw, candidates, variant="baseline", kseeds=1):
    fvec = feature_variant(variant)
    tpl = np.stack([template_features(variant, c, bw, kseeds) for c in candidates])
    lo, hi = tpl.min(axis=0), tpl.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    fv = (fvec(x_bb) - lo) / span
    tpl_n = (tpl - lo) / span
    dist = np.sqrt(((tpl_n - fv) ** 2).sum(axis=1))
    order = np.argsort(dist)
    return candidates[int(order[0])], dist


def _slice_freq_band(x, f_center, bw, pad=0.02):
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    half = bw / 2.0 + pad
    i_lo = max(int(np.searchsorted(freq, f_center - half)), 0)
    i_hi = min(int(np.searchsorted(freq, f_center + half)), n - 1)
    return tools_v2._band_slice(x, freq, psd, i_lo, i_hi)


def run_clean(repeats, candidates, variant, kseeds):
    pairs = []
    for m in MODS:
        for bw in BWS:
            for seed in range(repeats):
                rng = np.random.default_rng(1000 + seed)
                wave = generate_baseband(m, 1024, rng=rng, bandwidth_normalized=bw)
                f_off = 0.2
                wave_off = apply_freq_offset(wave, f_off)
                x_bb, bw_meas = _slice_freq_band(wave_off, f_off, bw)
                if x_bb is None:
                    pairs.append((m, f"{m}@{bw}", "SLICE_FAIL"))
                    continue
                pred, _ = classify(x_bb, bw_meas, candidates, variant, kseeds)
                pairs.append((m, f"{m}@{bw}", pred))
    return pairs


def run_dataset(dataset_dir, candidates, variant, kseeds):
    root = Path(dataset_dir)
    gt_data = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    pairs = []
    for rec in gt_data:
        iq = np.load(root / rec["file"])
        x = tools_v2._combine_channels(iq)
        for s in rec["ground_truth"].get("sources", [])[1:]:
            if s.get("waveform_type") is not None:
                continue
            f0 = s["freq_offset_normalized"]
            bw_gt = s["bandwidth_normalized"]
            x_bb, bw_meas = _slice_freq_band(x, f0, bw_gt)
            if x_bb is None:
                pairs.append((s["modulation"], rec["file"], "SLICE_FAIL"))
                continue
            pred, _ = classify(x_bb, bw_meas, candidates, variant, kseeds)
            pairs.append((s["modulation"], rec["file"], pred))
    return pairs


def report(title, pairs, top_n=1):
    total = len(pairs)
    hits = sum(1 for g, _, p in pairs if g.upper() == p.upper())
    print(f"\n=== {title}: top-{top_n} 命中 {hits}/{total} = {hits / max(total,1):.3f} ===")
    conf = Counter([(g, p) for g, _, p in pairs])
    print("混淆 (GT -> Pred):")
    for (g, p), c in sorted(conf.items(), key=lambda kv: (kv[0][0], -kv[1])):
        mark = "OK" if g.upper() == p.upper() else "  "
        print(f"  {g:6s} -> {p:12s} x{c} {mark}")
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", default="baseline,cumulants,full")
    ap.add_argument("--kseeds", type=int, default=5,
                    help="模板 K 种子均值（1 = 现状单种子）")
    args = ap.parse_args()

    candidates = tools_v2._CANDIDATES
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"候选集: {list(candidates)}\n")

    results = {}
    for variant in variants:
        acc_clean = report(f"[{variant}] 第 1 级：干净波形上限",
                           run_clean(args.repeats, candidates, variant, args.kseeds))
        results[variant] = {"clean": acc_clean}

    print("\n" + "=" * 60)
    for v in variants:
        print(f"  {v:10s} clean={results[v]['clean']:.3f}")

    # 数据集切片上限只跑最优与 baseline 对照（省时）
    best = max(results, key=lambda v: results[v]["clean"])
    for variant in dict.fromkeys(["baseline", best]):
        acc_ds = report(f"[{variant}] 第 2 级：数据集 GT 切片上限 (kseeds={args.kseeds})",
                        run_dataset(args.dataset, candidates, variant, args.kseeds))
        results[variant]["dataset"] = acc_ds
    print("\n结论：在线调制准确率应介于 clean 上限与切片上限之间；")
    print("若 upgraded 变体 clean 显著高于 baseline 且 >= 0.80，才值得接入 tools_v2 工具层。")


if __name__ == "__main__":
    main()
