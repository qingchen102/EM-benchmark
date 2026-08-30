"""两级 oracle 验收（门槛开跑前已注册，勿改）：

- 第 1 级 干净波形上限（独立种子 700000+，无噪声）≥ 0.85
- 第 2 级 冻结 500 的 GT 干扰切片（全部干扰，10 类任务）≥ 0.50

另输出：按 INR 分解、调制类子集（与旧上限 0.282/0.331 可比的口径）、混淆矩阵。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools_v2  # noqa: E402

from data_gen import CLASSES, CLS_IDX, make_wave, augment, to_input, slice_band  # noqa: E402
from model import ModCNN  # noqa: E402

GATE_CLEAN = 0.85
GATE_SLICE = 0.50


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ModCNN()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["classes"], ck.get("val_acc")


def predict(model, waves, snr_db=None):
    rng = np.random.default_rng(4242)
    out = []
    for w in waves:
        x = augment(w, rng, snr_db=snr_db, apply_shift=False) if snr_db is not None else w
        out.append(to_input(x))
    with torch.no_grad():
        logits = model(torch.from_numpy(np.stack(out)))
    return logits.argmax(1).numpy(), logits.softmax(1).numpy()


def level1_clean(model, per_class=30):
    """独立种子 700000+ 干净波形（无噪声），真值频偏已知位置切片——与旧上限同口径。"""
    from em_signal_simulator.channel import apply_freq_offset
    rng = np.random.default_rng(700_000)
    pairs = []
    for name in CLASSES:
        got = 0
        while got < per_class:
            bw = float(rng.uniform(0.05, 0.60))       # 覆盖考卷带宽范围
            w = make_wave(name, rng, bw=bw)
            f_off = float(rng.uniform(-0.40, 0.40))
            w = apply_freq_offset(w, f_off)
            x_bb = slice_band(w, f_off, bw)
            if x_bb is None:
                continue
            pred, _ = predict(model, [x_bb])
            pairs.append((name, CLASSES[pred[0]]))
            got += 1
    return pairs


def level2_slices(model):
    """冻结 500：按 GT 位置切出每条干扰，10 类分类。"""
    sys_path_setup()
    import tools_v2
    gt_data = json.loads((ROOT / "dataset" / "ground_truth.json").read_text(encoding="utf-8"))
    pairs, inr_cells = [], {}
    for rec in gt_data:
        iq = np.load(ROOT / "dataset" / rec["file"])
        x = tools_v2._combine_channels(iq)
        for s in rec["ground_truth"]["sources"][1:]:
            f0, bw = s["freq_offset_normalized"], s["bandwidth_normalized"]
            label = s.get("modulation") or s.get("waveform_type")
            if label not in CLS_IDX:
                continue
            x_bb = slice_band(x, f0, bw)
            if x_bb is None:
                continue
            pred, _ = predict(model, [x_bb])
            pairs.append((label, CLASSES[pred[0]], s.get("inr_db", 99)))
    return pairs, inr_cells


def slice_band(x, f_center, bw, pad=0.02):
    """与 eval_mod_upperbound._slice_freq_band 同口径（Hann 掩膜 + 解旋到基带）。"""
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    psd = np.abs(spec) ** 2 / n
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    half = bw / 2.0 + pad
    i_lo = max(int(np.searchsorted(freq, f_center - half)), 0)
    i_hi = min(int(np.searchsorted(freq, f_center + half)), n - 1)
    res = tools_v2._band_slice(x, freq, psd, i_lo, i_hi)
    if res is None or res[0] is None:
        return None
    return res[0]


def sys_path_setup():
    import sys
    for p in (str(ROOT), str(ROOT / "simulation")):
        if p not in sys.path:
            sys.path.insert(0, p)


def report(title, pairs):
    tot = len(pairs)
    hits = sum(1 for g, p, *_ in pairs if g == p)
    print(f"=== {title}: {hits}/{tot} = {hits / max(tot,1):.3f} ===")
    conf = Counter((g, p) for g, p, *_ in pairs)
    for (g, p), c in sorted(conf.items(), key=lambda kv: (kv[0][0], -kv[1])):
        if c >= 3:
            print(f"  {g:11s} -> {p:11s} x{c} {'OK' if g == p else ''}")
    return hits / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--per-class", type=int, default=30)
    args = ap.parse_args()

    model, classes, vacc = load_model(HERE / args.checkpoint)
    print(f"载入 {args.checkpoint}（训练期 val_acc={vacc}）")

    p1 = level1_clean(model, args.per_class)
    a1 = report("第 1 级 干净波形上限（独立种子，无噪声）", p1)

    p2, _ = level2_slices(model)
    a2 = report("第 2 级 冻结集 GT 切片（10 类全干扰）", p2)

    # 调制类子集：与旧上限 0.282/0.331 同口径
    mods = {"BPSK", "QPSK", "GFSK", "LFM", "OFDM"}
    sub = [(g, p) for g, p, *_ in p2 if g in mods and p in mods]
    asub = report("第 2 级子集：仅调制类（旧口径对比）", sub)

    # 按 INR 分解
    for lo, hi in ((0, 6), (6, 15), (15, 999)):
        sel = [(g, p) for g, p, inr in p2 if lo <= inr < hi]
        if sel:
            h = sum(1 for g, p in sel if g == p)
            print(f"  INR {lo}~{hi}dB: {h}/{len(sel)} = {h/len(sel):.3f}")

    verdict = ("PASS" if (a1 >= GATE_CLEAN and a2 >= GATE_SLICE) else "FAIL")
    print(f"\n门槛: clean>={GATE_CLEAN}, slice>={GATE_SLICE} -> 验收: {verdict}")
    (HERE / "oracle_result.json").write_text(json.dumps({
        "clean_acc": round(a1, 4), "slice_acc": round(a2, 4),
        "mod_only_slice_acc": round(asub, 4), "gate": {"clean": GATE_CLEAN, "slice": GATE_SLICE},
        "verdict": verdict}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
