"""_probe_len_sensitivity.py — 冻结 CNN 对"生成长度"的敏感度（零 API）

背景（user m13429「你现在切出来的东西，跟你拿去 CNN 训练的东西有什么区别？」）：
    `_probe_train_vs_real.py` 实测同一个发射体在 length=4096 与 length=40000 下
    占空比/峰度完全不同（Zigbee 24.2%/ku 8.1 → 2.2%/ku 86.6；WiFi 21.1% → 2.4%），
    而训练池（`data_gen16.LENGTH = 4096`）用 4096、数据集波形用 40000。
本探针量"这是不是因果"——同一个 checkpoint，只换输入的生成长度。

四个臂（每类 n 次）：
    aug@4096   = simulate_beamformed_train(gen_wave16(len=4096))  ← 训练域，应≈0.88
                 （与 oracle16.gate_clean 同链路，仅种子不同）
    aug@40000  = simulate_beamformed_train(gen_wave16(len=40000)) ← 同链路、换评测长度
    raw@4096   = gen_wave16(len=4096) 原样            ← 记录"直接喂干净波形"这个坑
    raw@40000  = gen_wave16(len=40000) 原样             （gate_clean docstring 记过：0.32）
若 aug@40000 相对 aug@4096 大幅下跌 ⇒ 训练/评测的"长度=占空比=峰度"错配是 D2 体制轴
失效的直接原因（与门控实验的结论一致：门控对了但冻结分类器不动）。

用法： python -X utf8 _probe_len_sensitivity.py [--n 10] [--pool 8] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_gen16 as dg            # noqa: E402
import oracle16 as o16             # noqa: E402

LEN_TRAIN = 4096
LEN_EVAL = 40000


def make_pool(length: int, per: int, seed: int = 7000):
    xs, ys = [], []
    for ci, nm in enumerate(dg.CLASSES16):
        for j in range(per):
            r = np.random.default_rng(seed + ci * 1009 + j)
            try:
                xs.append(dg.gen_wave16(nm, r, length=length).astype(np.complex128))
                ys.append(ci)
            except RuntimeError:
                pass
    return np.stack(xs), np.array(ys)


def run_arm(model, length: int, n: int, aug: bool, tag: int, pool) -> dict:
    per = {}
    ok = tot = 0
    for ci, nm in enumerate(dg.CLASSES16):
        c_ok = c_n = 0
        for j in range(n):
            r = np.random.default_rng(9000 + tag * 500000 + ci * 1009 + j)
            try:
                x = dg.gen_wave16(nm, r, length=length).astype(np.complex128)
            except RuntimeError:
                continue
            if aug:
                r2 = np.random.default_rng(9000 + tag * 500000 + 50000 + ci * 1009 + j)
                x = dg.simulate_beamformed_train(x, ci, pool[0], pool[1], r2)
            p = int(o16._pred(model, [x])[0])
            c_n += 1
            c_ok += int(p == ci)
        per[nm] = (c_ok, c_n)
        ok += c_ok
        tot += c_n
    return {"ok": ok, "tot": tot, "per": per}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv[1:])
    model, _ = o16.load_model()
    print(f"构建兄弟源池：4096 与 40000 各 {args.pool}/类 …", flush=True)
    pool_tr = make_pool(LEN_TRAIN, args.pool)
    pool_ev = make_pool(LEN_EVAL, args.pool)
    arms = [("aug@4096", LEN_TRAIN, True, 0, pool_tr),
            ("aug@40000", LEN_EVAL, True, 1, pool_ev),
            ("raw@4096", LEN_TRAIN, False, 2, None),
            ("raw@40000", LEN_EVAL, False, 3, None)]
    res = {}
    for nm, ln, aug, tag, pool in arms:
        print(f"[arm] {nm} …", flush=True)
        res[nm] = run_arm(model, ln, args.n, aug, tag, pool)
    lines = [f"同一 checkpoint：只换输入生成长度（每类 n={args.n}，aug = "
             f"simulate_beamformed_train 与 gate_clean 同链路）", ""]
    lines.append(f"{'类':<16}" + "".join(f"{nm:>14}" for nm, *_ in arms))
    lines.append("-" * 72)
    for nm in dg.CLASSES16:
        seg = []
        for a, *_ in arms:
            ok, tot = res[a]["per"].get(nm, (0, 0))
            seg.append(f"{ok / tot:.2f}" if tot else "—")
        lines.append(f"{nm:<16}" + "".join(f"{s:>14}" for s in seg))
    lines.append("-" * 72)
    lines.append(f"{'合计':<16}" + "".join(
        f"{res[a]['ok'] / res[a]['tot']:.3f} [{res[a]['ok']}/{res[a]['tot']}]"
        for a, *_ in arms))
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.out) if args.out else (HERE.parent.parent / "evidence" /
                                           "_len_sensitivity.txt")
    out.write_text(text, encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv)
