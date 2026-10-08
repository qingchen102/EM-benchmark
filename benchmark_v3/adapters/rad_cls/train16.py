"""训练 RadCNN（16 辐射体分类器）——v8 训练循环平移，在线增强=残差分布模拟。

用法：
    python train16.py --epochs 25 --n-per-class 300
输出：rad_classifier/checkpoint.pt {'model','classes','val_acc','version'}
CPU 可跑；GPU 存在时自动使用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from data_gen16 import CLASSES16, CLS_IDX, LENGTH, SEED_BASE, gen_wave16, simulate_beamformed_train, to_input  # noqa: E402
from model16 import RadCNN  # noqa: E402

OUT = _HERE / "rad_classifier"


def gen_pool(n_per_class: int, seed_offset: int):
    """干净波形池（独立种子段；增强在线做）。"""
    xs, ys = [], []
    t0 = time.time()
    for ci, name in enumerate(CLASSES16):
        j = 0
        while j < n_per_class:
            rng = np.random.default_rng(SEED_BASE + seed_offset + ci * 100003 + j)
            try:
                xs.append(gen_wave16(name, rng).astype(np.complex64))
                ys.append(ci)
                j += 1
            except RuntimeError:
                continue
        print(f"  [{seed_offset}] {name}: {n_per_class}  ({time.time()-t0:.0f}s)", flush=True)
    return np.array(xs), np.array(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    print("== 生成干净波形池 ==", flush=True)
    xs, ys = gen_pool(args.n_per_class, seed_offset=1000)
    vx, vy = gen_pool(max(args.n_per_class // 5, 40), seed_offset=2000)

    model = RadCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    best_val = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(xs))
        losses = []
        for i in range(0, len(perm), args.batch):
            batch_x, batch_y = [], []
            for j in perm[i:i + args.batch]:
                rng = np.random.default_rng(SEED_BASE + 3000 + ep * 100000 + j)
                sim = simulate_beamformed_train(xs[j].astype(np.complex128), ys[j],
                                                xs, ys, rng)
                batch_x.append(to_input(sim))
                batch_y.append(ys[j])
            xb = torch.from_numpy(np.stack(batch_x)).to(device)
            yb = torch.tensor(batch_y, dtype=torch.long).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        # 验证（固定增强一轮）
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(vx), args.batch):
                rng = np.random.default_rng(SEED_BASE + 4000 + i)
                vb = [to_input(simulate_beamformed_train(vx[j].astype(np.complex128), vy[j],
                                                         vx, vy, rng))
                      for j in range(i, min(i + args.batch, len(vx)))]
                xb = torch.from_numpy(np.stack(vb)).to(device)
                pred = model(xb).argmax(dim=1).cpu().numpy()
                correct += int(np.sum(pred == vy[i:i + args.batch]))
        val = correct / len(vx)
        print(f"[epoch {ep+1}/{args.epochs}] loss={np.mean(losses):.3f} val_acc={val:.3f}",
              flush=True)
        if val > best_val:
            best_val = val
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": CLASSES16,
                        "val_acc": val, "version": "rad16-v1"},
                       OUT / "checkpoint.pt")
    print(f"best val_acc={best_val:.3f} -> {OUT/'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
