"""训练 ModCNN（v2 迭代）。

与 v1 的关键差异：**训练视角 = 评测视角**。每个样本在线执行
"频偏 → 加噪 → Hann 切片（与 eval 同口径）→ 归一化"，
消除 v1 的域差距（v1 全带训练、切片评测，切片级只有 0.213）。
容量 4 倍（0.04M → 0.15M 参数）、epoch 加长 + 余弦退火。
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
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from em_signal_simulator.channel import apply_freq_offset  # noqa: E402

from data_gen import CLASSES, CLS_IDX, LENGTH, SEED_BASE, make_wave, augment, to_input, slice_band
from model import ModCNN

HERE = Path(__file__).resolve().parent


def gen_split(n_per_class: int, seed_offset: int, desc: str):
    """生成 n_per_class × 10 条干净波形，返回 (waves, labels, bws)。"""
    xs, ys, bws = [], [], []
    t0 = time.time()
    for ci, name in enumerate(CLASSES):
        for j in range(n_per_class):
            rng = np.random.default_rng(SEED_BASE + seed_offset + ci * 100003 + j)
            bw = float(rng.uniform(0.05, 0.60))
            xs.append(make_wave(name, rng, bw=bw).astype(np.complex64))
            ys.append(ci)
            bws.append(bw)
        print(f"  [{desc}] {name}: {n_per_class}  ({time.time()-t0:.0f}s)", flush=True)
    return np.array(xs), np.array(ys), np.array(bws, dtype=np.float32)


def build_batch(waves, bws, ys, idx, rng, device):
    """在线增强：频偏 → 加噪 → 切片 → 归一化（与 oracle 第 2 级完全同视角）。"""
    xs = []
    for i in idx:
        f_off = float(rng.uniform(-0.42, 0.42))
        p = float(np.mean(np.abs(waves[i]) ** 2)) + 1e-30
        snr = float(rng.uniform(-10.0, 20.0))
        amp = np.sqrt(p * 10.0 ** (-snr / 10.0) / 2.0)
        noise = amp * (rng.standard_normal(LENGTH) + 1j * rng.standard_normal(LENGTH))
        x = apply_freq_offset(waves[i], f_off) + noise
        x_bb = slice_band(x, f_off, float(bws[i]))
        if x_bb is None:
            x_bb = x          # 切片失败（极窄带）时退化为全带含噪波形
        xs.append(to_input(x_bb))
    xb = torch.from_numpy(np.stack(xs)).to(device)
    yb = torch.from_numpy(np.asarray(ys)[idx]).to(device)
    return xb, yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-per-class", type=int, default=4000)
    ap.add_argument("--val-per-class", type=int, default=400)
    ap.add_argument("--out", default="checkpoint.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    print("生成训练干净波形…", flush=True)
    xtr, ytr, btr = gen_split(args.train_per_class, 0, "train")
    print("生成验证干净波形（独立种子段）…", flush=True)
    xva, yva, bva = gen_split(args.val_per_class, 900_000, "val")

    model = ModCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    # 固定验证批（一次性增强，保证各 epoch 可比）
    vrng = np.random.default_rng(777)
    all_idx = np.arange(len(xva))
    vxt, vy = build_batch(xva, bva, yva, all_idx, vrng, device)

    history, best = [], 0.0
    rng = np.random.default_rng(123)
    n = len(xtr)
    for ep in range(args.epochs):
        model.train()
        t0, order = time.time(), rng.permutation(n)
        tot_loss, tot_hits, tot_n = 0.0, 0, 0
        for step in range(args.steps_per_epoch):
            start = (step * args.batch) % n
            idx = order[start:start + args.batch]
            if len(idx) < args.batch:
                idx = np.concatenate([idx, order[:args.batch - len(idx)]])
            xb, yb = build_batch(xtr, btr, ytr, idx, rng, device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_hits += int((out.argmax(1) == yb).sum())
            tot_n += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vout = model(vxt)
            vacc = float((vout.argmax(1) == vy).float().mean())
        entry = {"epoch": ep + 1, "train_loss": round(tot_loss / max(tot_n, 1), 4),
                 "train_acc": round(tot_hits / max(tot_n, 1), 4),
                 "val_acc": round(vacc, 4), "lr": round(sched.get_last_lr()[0], 6),
                 "sec": round(time.time() - t0)}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if vacc > best:
            best = vacc
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "val_acc": vacc, "version": "v2-sliced"}, HERE / args.out)

    (HERE / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"完成。最佳 val_acc={best:.4f}，checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
