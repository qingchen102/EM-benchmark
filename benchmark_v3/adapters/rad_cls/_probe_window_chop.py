"""_probe_window_chop.py — 真实切片「切窗投票」vs「整窗一次判」（零 API）

动机（承 `_probe_len_sensitivity.py` 的决定性结果）：
    同一个 checkpoint、同一条增强链，只把生成长度从 4096 换成 40000，
    Zigbee 1.00→0.00、WiFi_burst 1.00→0.00、总体 0.868→0.747。
    真实评测切片是 40000 样点（10 ms），训练域是 4096（1.024 ms）。
本探针问一句花钱最少的话：**如果不重训，只把 40000 样点按 4096 切窗、逐窗判、投票**，
    能不能把 Zigbee/WiFi 从 0 拉回来？（能 ⇒ 只改提取工具的输出长度即可，无需重训）

臂：
    whole    = 整段 40000 一次判（= 现行工具口径 baseline）
    chop     = 按 4096 不重叠切 9 窗，逐窗判，**多数票**
    chop-max = 逐窗判，取该窗口内预测为"窗内能量最大者"…（不做，留接口）
    chop-mean = 逐窗判，按窗能量加权投票
    trainlen = 只取前 4096 样点一次判（训练长度口径，看"是不是长度本身"）

用法： python -X utf8 _probe_window_chop.py [deck] [--n 40] [--win 4096]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADP = HERE.parent
for _p in (str(ADP), str(HERE), str(ADP / "diag")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oracle16 as o16             # noqa: E402
import _probe_burst_gate as bg     # noqa: E402

FS = 4e6


def extract(root: Path, s: dict, station: str = "station_a"):
    """返回 [(kind, w)]，w = 全带 GT 切片提取波形（40000 样点）。"""
    cap = root / s["id"] / "capture"
    obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
    truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
    ns = obs["receivers"][0]["shape"][1]
    try:
        refs = bg.rca._build_target_reference(cap / "private", ns)
    except Exception:                                        # noqa: BLE001
        refs = None
    gts = [(i + 1, g) for i, g in enumerate(truth["interferers"])
           if g["kind"] in o16.CLS_IDX]
    if not gts:
        return []
    rec = next((x for x in obs["receivers"] if x["receiver_id"] == station), None)
    if rec is None:
        return []
    try:
        r = bg._resid_for(cap, obs, s["id"], station, refs)
    except Exception:                                        # noqa: BLE001
        return []
    if r is None or np.asarray(r).ndim != 2:
        return []
    A = bg._steering(rec, [g["position_m"] for _gi, g in gts])
    try:
        if np.linalg.cond(A) > 1e6:
            return []
        S = np.linalg.pinv(A) @ r
    except np.linalg.LinAlgError:
        return []
    t = np.arange(r.shape[1])
    out = []
    for k, (gi, g) in enumerate(gts):
        w = S[k] * np.exp(-1j * 2 * np.pi * g["freq_offset_hz"] * t / FS)
        out.append((g["kind"], w))
    return out


def rms(x):
    return x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", nargs="?", default="d2")
    ap.add_argument("--n", type=int, default=40, help="每类最多取多少个源")
    ap.add_argument("--win", type=int, default=4096)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv[1:])
    model, _ = o16.load_model()
    root = o16.D2_ROOT
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    per = {}                       # kind -> {arm: [ok, n]}
    for s in idx["samples"]:
        got = extract(root, s)
        if not got:
            continue
        for kind, w in got:
            d = per.setdefault(kind, {a: [0, 0] for a in
                                      ("whole", "chop", "chop_e", "trainlen")})
            if d["whole"][1] >= args.n:
                continue
            ci = o16.CLS_IDX[kind]
            w = rms(w)
            # whole
            d["whole"][0] += int(int(o16._pred(model, [w])[0]) == ci)
            d["whole"][1] += 1
            # chop：不重叠 4096 窗，多数票
            wins = [w[i:i + args.win] for i in range(0, len(w) - args.win + 1, args.win)]
            preds, ews = [], []
            for x in wins:
                preds.append(int(o16._pred(model, [x])[0]))
                ews.append(float(np.mean(np.abs(x) ** 2)))
            if preds:
                d["chop"][0] += int(Counter(preds).most_common(1)[0][0] == ci)
                d["chop"][1] += 1
                ew = np.array(ews)
                votes = {}
                for p, e in zip(preds, ew):
                    votes[p] = votes.get(p, 0.0) + float(e)
                d["chop_e"][0] += int(max(votes, key=votes.get) == ci)
                d["chop_e"][1] += 1
            # trainlen：前 4096 样点（真·训练长度口径）
            d["trainlen"][0] += int(int(o16._pred(model, [w[:args.win]])[0]) == ci)
            d["trainlen"][1] += 1
    arms = ["whole", "chop", "chop_e", "trainlen"]
    lines = [f"{root.name}：真实切片「整窗一次判 vs 4096 切窗投票」（窗={args.win}，每类 ≤{args.n} 源）",
             "", f"{'类':<16}" + "".join(f"{a:>12}" for a in arms), "-" * 64]
    tot = {a: [0, 0] for a in arms}
    for kind in o16.CLASSES16:
        if kind not in per:
            continue
        seg = []
        for a in arms:
            ok, n = per[kind][a]
            seg.append(f"{ok / n:.2f}" if n else "—")
            tot[a][0] += ok
            tot[a][1] += n
        lines.append(f"{kind:<16}" + "".join(f"{s:>12}" for s in seg))
    lines.append("-" * 64)
    lines.append(f"{'合计':<16}" + "".join(
        f"{tot[a][0] / tot[a][1]:.3f} [{tot[a][0]}/{tot[a][1]}]" if tot[a][1] else "—"
        for a in arms))
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.out) if args.out else (ADP.parent / "evidence" /
                                           f"_window_chop_{args.deck}.txt")
    out.write_text(text, encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv)
