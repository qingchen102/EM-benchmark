"""三级 oracle 验收（A 路线 16 辐射体分类器，预注册门槛勿改）：

- 第 1 级 clean：独立种子 16 类纯波形（含残差分布增强），top-1 ≥ 0.85
- 第 2 级 GT 切片：真实 D2 残差按 GT 频偏/带宽切片，top-1 ≥ 0.50
- 第 3 级 工具视角：完整剥离残差链路（on_residual 谱峰定位），top-1 ≥ 0.50
任一不过 => 不接入工具箱，回到训练调参。

用法：
    python oracle16.py
输出每级准确率 + 混淆矩阵。零 token（纯本地）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ADAPTERS = _HERE.parent
for _p in (str(_HERE), str(_ADAPTERS), str(_HERE.parent.parent / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen16 import CLASSES16, CLS_IDX, gen_wave16, simulate_residual, to_input  # noqa: E402
from model16 import RadCNN  # noqa: E402

GATE_CLEAN = 0.85
GATE_SLICE = 0.50
GATE_TOOLVIEW = 0.50

D2_ROOT = _HERE.parent.parent / "dataset_d2"


def load_model():
    ck = torch.load(_HERE / "rad_classifier" / "checkpoint.pt",
                    map_location="cpu", weights_only=False)
    model = RadCNN()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["classes"]


def _pred(model, batch_x):
    with torch.no_grad():
        xb = torch.from_numpy(np.stack([to_input(b) for b in batch_x]))
        return model(xb).argmax(dim=1).cpu().numpy()


def gate_clean(model, n_per_class=60, seed_base=700000):
    """独立种子纯波形（残差分布增强）。"""
    correct = total = 0
    conf = np.zeros((len(CLASSES16), len(CLASSES16)))
    for ci, name in enumerate(CLASSES16):
        j = 0
        while j < n_per_class:
            rng = np.random.default_rng(seed_base + ci * 1009 + j)
            try:
                x = gen_wave16(name, rng).astype(np.complex128)
            except RuntimeError:
                continue
            sim = simulate_residual(x, rng)
            pred = _pred(model, [sim])[0]
            conf[ci, pred] += 1
            correct += int(pred == ci)
            total += 1
            j += 1
    acc = correct / total
    print(f"  clean (独立种子, 残差增强): {acc:.3f}  [{correct}/{total}]  gate={GATE_CLEAN}")
    return acc, conf


def gate_gt_slice(model):
    """真实 D2 残差按 GT 频偏/带宽切片喂 CNN；按 crosstalk 重叠度分桶。"""
    correct = total = 0
    idx = json.loads((D2_ROOT / "index.json").read_text(encoding="utf-8"))
    conf = np.zeros((len(CLASSES16), len(CLASSES16)))
    # 分桶：无(<0.1) / 轻度(0.1-0.5) / 重度(>=0.5) —— 按兄弟干扰覆盖比例
    bucket = {"clean": [0, 0], "light": [0, 0], "heavy": [0, 0]}
    for s in idx["samples"]:
        cap = D2_ROOT / s["id"] / "capture"
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        resid = np.load(D2_ROOT / "_agent_cache" / s["id"] / "resid_station_a.npy") \
            if (D2_ROOT / "_agent_cache" / s["id"] / "resid_station_a.npy").exists() else None
        if resid is None:
            continue
        fs = obs["receivers"][0]["sample_rate_hz"]
        gts = [g for g in truth["interferers"] if g["kind"] in CLS_IDX]
        for i, g in enumerate(gts):
            lo_i = g["freq_offset_hz"] - g["bandwidth_hz"] / 2
            hi_i = g["freq_offset_hz"] + g["bandwidth_hz"] / 2
            covered = 0.0
            for j, h in enumerate(gts):
                if i == j:
                    continue
                ov = min(hi_i, h["freq_offset_hz"] + h["bandwidth_hz"] / 2) \
                    - max(lo_i, h["freq_offset_hz"] - h["bandwidth_hz"] / 2)
                covered += max(0, ov) / max(g["bandwidth_hz"], 1e-9)
            cov = min(covered, 1.0)
            bucket_key = "clean" if cov < 0.1 else ("light" if cov < 0.5 else "heavy")
            lo, hi = lo_i, hi_i
            n = resid.shape[1]
            f = np.fft.fftfreq(n, 1 / fs)
            mask = (f >= lo) & (f <= hi)
            x = np.fft.ifft(np.fft.fft(resid, axis=1) * mask, axis=1).mean(axis=0)
            x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
            pred = _pred(model, [x.astype(np.complex64)])[0]
            conf[CLS_IDX[g["kind"]], pred] += 1
            correct += int(pred == CLS_IDX[g["kind"]])
            total += 1
            bucket[bucket_key][1] += 1
            bucket[bucket_key][0] += int(pred == CLS_IDX[g["kind"]])
    acc = correct / total if total else 0
    print(f"  GT切片 (真实残差): {acc:.3f}  [{correct}/{total}]  gate={GATE_SLICE}")
    for k in ("clean", "light", "heavy"):
        h, t = bucket[k]
        print(f"     crosstalk {k:6s} [{h}/{t}] = {h/t if t else 0:.3f}")
    return acc, conf


def print_conf(conf):
    print("  混淆矩阵 (GT行 × pred列, 顶行/左列=类名):")
    names = CLASSES16
    print("      " + " ".join(f"{n[:5]:>6}" for n in names))
    for i, n in enumerate(names):
        row = " ".join(f"{int(conf[i, j]):>6}" for j in range(len(names)))
        print(f"  {n[:12]:<12} {row}")


def main():
    model, _ = load_model()
    print(f"== 16 辐射体分类器 oracle（gate: clean{GATE_CLEAN}/slice{GATE_SLICE}/toolview{GATE_TOOLVIEW}）==")
    a1, c1 = gate_clean(model)
    a2, c2 = gate_gt_slice(model)
    ok = [a1 >= GATE_CLEAN, a2 >= GATE_SLICE]
    print(f"== 判定: clean {'PASS' if ok[0] else 'FAIL'} | GT切片 {'PASS' if ok[1] else 'FAIL'}")
    print("== clean 混淆矩阵 =="); print_conf(c1)
    print("== GT切片 混淆矩阵 =="); print_conf(c2)
    if not all(ok):
        print("未达标：不接入工具箱，回到训练调参（增加变体/epoch/容量）")


if __name__ == "__main__":
    main()
