"""_probe_train_vs_real.py — 训练输入 vs 真实切片：逐特征对照（只读，零 API）

动机（user m13429「你有没有尝试比对过，你现在切出来的东西，跟你拿去 CNN 训练的东西有什么区别？」）：
    train16.py 的训练样本不是干净波形，而是 `data_gen16.simulate_beamformed_train`
    造出来的"模拟提取产物"：长度 **4096**（1.024 ms）、平坦复信道、0–2 个兄弟源、
    过接收机 firwin129、已知方向的 LS 提取、**50% 概率再切一刀以主源带宽 ×1.0–1.4
    为中心窄带**、RMS 归一化。
    真实评测输入是：长度 **40000**（10 ms）、RT 多径（每链路 1–6 条径、真实时延）、
    共道伙伴（Δu 可以很小）、平台前端（含 IMD3/噪声开关）、按聚类带（或完全不带限）
    提取、RMS 归一化。
本探针把两边拉到同一张表上：占空比 / 突发段数 / 99% 占带 / 复样本峰度 / PAPR / 带内能量比。

用法： python -X utf8 _probe_train_vs_real.py [deck] [--n 20] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # benchmark_v3/adapters/rad_cls
ADP = HERE.parent                               # benchmark_v3/adapters
for _p in (str(ADP), str(HERE), str(ADP / "diag")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_gen16 as dg            # noqa: E402
import oracle16 as o16             # noqa: E402
import _probe_burst_gate as bg     # noqa: E402  复用残差构建/导向矩阵/块统计

FS = 4e6
BLK = 64


def feats(x: np.ndarray) -> dict:
    """占空比 / 突发段数 / 99% 占带 / 复样本峰度 / PAPR。"""
    x = np.asarray(x, np.complex128)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    e, ku_all, nb = bg._block_stats(x)
    act = bg._clean_blocks(e > 0.2 * e.max())
    duty = float(act.mean())
    # 突发段数
    bursts, i = 0, 0
    while i < act.size:
        if act[i]:
            bursts += 1
            while i < act.size and act[i]:
                i += 1
        else:
            i += 1
    # 99% 占带：功率对频率的 0.5%/99.5% 分位之差（复信号双侧谱）
    sp = np.abs(np.fft.fft(x)) ** 2
    f = np.fft.fftfreq(len(x), 1 / FS)
    order = np.argsort(f)
    fs_, sps = f[order], sp[order]
    cw = np.cumsum(sps) / (sps.sum() + 1e-30)
    lo = float(fs_[min(int(np.searchsorted(cw, 0.005)), fs_.size - 1)])
    hi = float(fs_[min(int(np.searchsorted(cw, 0.995)), fs_.size - 1)])
    obw = hi - lo
    p2 = float(np.mean(np.abs(x) ** 2))
    ku = float(np.mean(np.abs(x) ** 4) / (p2 ** 2 + 1e-60))
    papr = float(np.max(np.abs(x) ** 2) / (p2 + 1e-30))
    # 带内（±400 kHz）能量占比
    inband = float(sp[np.abs(f) <= 400e3].sum() / (sp.sum() + 1e-30))
    return {"duty": duty, "bursts": bursts, "obw_khz": obw / 1e3, "ku": ku,
            "papr": papr, "in400k": inband, "n": len(x)}




def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        out[k] = float(np.median(vals))
    return out


def train_side(class_names: list[str], n: int, seed: int = 5000):
    """训练侧：干净波形（4096）与 simulate_beamformed_train 产物（4096）。"""
    pool_x, pool_y = [], []
    for ci, nm in enumerate(dg.CLASSES16):
        for j in range(8):
            r = np.random.default_rng(seed + ci * 100003 + j)
            try:
                pool_x.append(dg.gen_wave16(nm, r).astype(np.complex128))
                pool_y.append(ci)
            except RuntimeError:
                continue
    pool_x = np.array(pool_x)
    pool_y = np.array(pool_y)
    out = {}
    for nm in class_names:
        ci = dg.CLS_IDX[nm]
        raw, raw40, aug = [], [], []
        for j in range(n):
            r = np.random.default_rng(seed + 777 + ci * 50021 + j)
            try:
                w = dg.gen_wave16(nm, r)
            except RuntimeError:
                continue
            raw.append(feats(w))
            try:
                r3 = np.random.default_rng(seed + 555 + ci * 30011 + j)
                raw40.append(feats(dg.gen_wave16(nm, r3, length=40000)))
            except RuntimeError:
                pass
            r2 = np.random.default_rng(seed + 999 + ci * 70001 + j)
            aug.append(feats(dg.simulate_beamformed_train(
                w.astype(np.complex128), ci, pool_x, pool_y, r2)))
        out[nm] = {"train_raw": _agg(raw), "train_raw40": _agg(raw40),
                   "train_aug": _agg(aug)}
    return out


def real_side(root: Path, class_names: list[str], n: int):
    """真实侧：GT 位置 + 全可见源 LS 提取（oracle16 的 GT 切片口径，station_a）。"""
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    acc = {nm: {"real_full": [], "real_band": []} for nm in class_names}
    for s in idx["samples"]:
        if all(len(v["real_full"]) >= n for v in acc.values()):
            break
        cap = root / s["id"] / "capture"
        try:
            obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
            truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        except OSError:
            continue
        ns = obs["receivers"][0]["shape"][1]
        try:
            refs = bg.rca._build_target_reference(cap / "private", ns)
        except Exception:                                    # noqa: BLE001
            refs = None
        gts = [(i + 1, g) for i, g in enumerate(truth["interferers"])
               if g["kind"] in o16.CLS_IDX]
        if not gts:
            continue
        st = "station_a"
        rec = next((x for x in obs["receivers"] if x["receiver_id"] == st), None)
        if rec is None:
            continue
        try:
            r = bg._resid_for(cap, obs, s["id"], st, refs)
        except Exception:                                    # noqa: BLE001
            continue
        if r is None or np.asarray(r).ndim != 2:
            continue
        A = bg._steering(rec, [g["position_m"] for _gi, g in gts])
        if not np.isfinite(np.linalg.cond(A)) or np.linalg.cond(A) > 1e6:
            continue
        S = np.linalg.pinv(A) @ r
        t = np.arange(r.shape[1])
        for k, (gi, g) in enumerate(gts):
            nm = g["kind"]
            if nm not in acc or len(acc[nm]["real_full"]) >= n:
                continue
            w = S[k] * np.exp(-1j * 2 * np.pi * g["freq_offset_hz"] * t / FS)
            acc[nm]["real_full"].append(feats(w))
            # 带限版：以该源真值为中心、±(bw/2 + 5 kHz) 的 brick（工具聚焦口径）
            bw = float(g["bandwidth_hz"])
            f = np.fft.fftfreq(w.size, 1 / FS)
            m = np.abs(f - float(g["freq_offset_hz"])) <= (bw / 2 + 5e3)
            wb = np.fft.ifft(np.fft.fft(w) * m)
            acc[nm]["real_band"].append(feats(wb))
    return {nm: {"real_full": _agg(v["real_full"]), "real_band": _agg(v["real_band"])}
            for nm, v in acc.items()}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", nargs="?", default="d2")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv[1:])
    classes = ["Zigbee", "WiFi_burst", "Welder", "DECT", "Microwave", "Corona",
               "VFD_noise", "Bluetooth_FHSS"]
    root = o16.D2_ROOT
    print(f"训练侧：干净 4096 + simulate_beamformed_train 4096；真实侧：{root} GT 切片 station_a",
          flush=True)
    tr = train_side(classes, args.n)
    re = real_side(root, classes, args.n)

    hdr = (f"{'类':<14}{'生成@4096':>26}{'生成@40000':>26}{'训练增强@4096':>26}"
           f"{'真实切片全带':>26}{'真实切片带限':>26}\n"
           f"{'':14}" + "".join(f"{'占空比/突发数/占带kHz':>26}" for _ in range(5)))
    rows = [hdr, "-" * len(hdr)]
    fmt = lambda d: (f"{d['duty']:.3f}/{d['bursts']:.0f}/{d['obw_khz']:.0f}" if d else "—")
    for nm in classes:
        rows.append(f"{nm:<14}"
                    + "".join(f"{fmt(d):>26}" for d in (
                        tr[nm]["train_raw"], tr[nm].get("train_raw40"),
                        tr[nm]["train_aug"],
                        re[nm]["real_full"], re[nm]["real_band"])))
    rows.append("")
    rows.append("峰度/PAPR/带内±400k 能量占比（中位）")
    for nm in classes:
        seg = []
        for d in (tr[nm]["train_raw"], tr[nm].get("train_raw40"),
                  tr[nm]["train_aug"],
                  re[nm]["real_full"], re[nm]["real_band"]):
            seg.append(f"{d['ku']:.2f}/{d['papr']:.1f}/{d['in400k']:.2f}" if d else "—")
        rows.append(f"{nm:<14}" + "".join(f"{s:>26}" for s in seg))
    text = "\n".join(rows) + "\n"
    print(text)
    out = Path(args.out) if args.out else (ADP.parent / "evidence" /
                                           f"_train_vs_real_{args.deck}.txt")
    out.write_text(text, encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv)
