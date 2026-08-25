"""调制识别工具上限验证（v2.5 改进验证）。

两级测试，用于区分"工具缺陷"与"模型综合缺陷"：

1. 纯波形上限：干净单源波形（施加频偏后走完整 Hann 切片+解旋路径），
   测试 特征向量 + 带宽自适应模板 本身的判别力；
2. 数据集切片上限：对真实混合样本（目标+干扰+噪声），按 GT 频偏/带宽
   做 Hann 切片后分类 —— 即"假设 Agent 完美选定切片位置时工具能达到的上限"。

运行：python eval_mod_upperbound.py [--dataset dataset] [--repeats 3]
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
from em_signal_simulator.baseband import generate_baseband
from em_signal_simulator.channel import apply_freq_offset

MODS = ["BPSK", "QPSK", "GFSK", "LFM", "OFDM"]
BWS = [0.06, 0.15, 0.25, 0.35, 0.45, 0.55]


def classify(x_bb, bw, candidates):
    """带宽自适应模板距离分类（与 estimate_modulation_features 同口径）。"""
    tpl = np.stack([tools_v2._template_features(c, bw) for c in candidates])
    lo, hi = tpl.min(axis=0), tpl.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    fv = (tools_v2._feature_vector(x_bb) - lo) / span
    tpl_n = (tpl - lo) / span
    dist = np.sqrt(((tpl_n - fv) ** 2).sum(axis=1))
    return candidates[int(np.argmin(dist))], dist


def _slice_freq_band(x, f_center, bw, pad=0.02):
    """在 f_center ± (bw/2 + pad) 频带做 Hann 切片并解旋到基带（复用工具实现）。"""
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    half = bw / 2.0 + pad
    i_lo = int(np.searchsorted(freq, f_center - half))
    i_hi = int(np.searchsorted(freq, f_center + half))
    i_lo = max(i_lo, 0)
    i_hi = min(i_hi, n - 1)
    return tools_v2._band_slice(x, freq, psd, i_lo, i_hi)


def run_clean(repeats, candidates):
    """第 1 级：干净波形上限。"""
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
                pred, _ = classify(x_bb, bw_meas, candidates)
                pairs.append((m, f"{m}@{bw}", pred))
    return pairs


def run_dataset(dataset_dir, candidates):
    """第 2 级：真实混合样本按 GT 切片的上限。"""
    root = Path(dataset_dir)
    gt_data = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    pairs = []
    for rec in gt_data:
        iq = np.load(root / rec["file"])
        x = tools_v2._combine_channels(iq)
        for s in rec["ground_truth"].get("sources", [])[1:]:
            if s.get("waveform_type") is not None:
                continue  # 只评估真实调制类（与评估器口径一致）
            f0 = s["freq_offset_normalized"]
            bw_gt = s["bandwidth_normalized"]
            x_bb, bw_meas = _slice_freq_band(x, f0, bw_gt)
            if x_bb is None:
                pairs.append((s["modulation"], rec["file"], "SLICE_FAIL"))
                continue
            pred, _ = classify(x_bb, bw_meas, candidates)
            pairs.append((s["modulation"], rec["file"], pred))
    return pairs


def report(title, pairs):
    total = len(pairs)
    hits = sum(1 for g, _, p in pairs if g.upper() == p.upper())
    print(f"\n=== {title}: top-1 命中 {hits}/{total} = {hits / total:.3f} ===")
    conf = Counter([(g, p) for g, _, p in pairs])
    print("混淆 (GT -> Pred):")
    for (g, p), c in sorted(conf.items(), key=lambda kv: (kv[0][0], -kv[1])):
        mark = "OK" if g.upper() == p.upper() else "  "
        print(f"  {g:6s} -> {p:12s} x{c} {mark}")
    return hits / total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    candidates = tools_v2._CANDIDATES
    print(f"候选集（收窄后）: {list(candidates)}")

    pairs_clean = run_clean(args.repeats, candidates)
    acc_clean = report("第 1 级：干净波形上限", pairs_clean)

    pairs_ds = run_dataset(args.dataset, candidates)
    acc_ds = report("第 2 级：数据集 GT 切片上限", pairs_ds)

    print(f"\n结论：工具上限 干净={acc_clean:.3f} / 混合切片={acc_ds:.3f}")
    print("（在线评估调制准确率应介于两者之间；显著低于切片上限 → 模型综合问题）")


if __name__ == "__main__":
    main()
