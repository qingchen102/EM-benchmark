"""_probe_burst_gate.py — 低占空比源的时间维门控原型（只读，零 API）

物理（为什么要时间门控）：
    整窗 RMS 归一化 + 全带 LS 提取之后，占空比只有 ~1% 的源（Zigbee 1.7% /
    WiFi_burst 0.9%）在其"空闲"时段被**同频连续源的泄漏**填满，分类器于是把它
    读成 Welder/Corona。改成"先判哪些时间块真有信号、只留这些块"，泄漏随之消失。

判据（借鉴，不自己设计）：
    · CA-CFAR 功率判据 —— Sensors 2026, 26(14), 4468：门限由邻域参考单元给出、
      虚警率恒定；该文的核心动作就是"只把含信号的时间区间转发下去"。
    · 复样本峰度（spectral-kurtosis 同族）判据 —— RAS Techniques and Instruments
      2024 (10.1093/rasti/rzae035)：占空比源让复电压分布偏离高斯（恒模→肥尾），
      而热噪/失配底是复高斯（峰度≈2）。该文结论：时变/频变干扰以 SK/SW 检出最好。

用法：
    python -X utf8 diag/_probe_burst_gate.py [deck] [--n N] [--samples "d2_00001,d2_00002"]
        [--mode zero|concat|both] [--kurt] [--nostation]

输出：证据 txt（默认 benchmark_v3/evidence/_burst_gate_<deck>.txt）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # benchmark_v3/adapters/diag
ADP = HERE.parent                               # benchmark_v3/adapters
RAD = ADP / "rad_cls"
for _p in (str(ADP), str(RAD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools as ct          # noqa: E402
import oracle16 as o16              # noqa: E402
import run_capture_agent as rca     # noqa: E402

FS = 4e6
C = 299792458.0
FC = 2.4e9

# ---- 门控参数（方案 §2 的四个新参数）--------------------------------------
BLK = 64                    # 块长 16 µs @4 MHz
PFA = 1e-3                  # CA-CFAR 设计虚警率
NREF = 32                   # 参考块数（CFAR 损失按此定标）
ALPHA = NREF * (PFA ** (-1.0 / NREF) - 1.0)   # ≈ 7.7 倍本地块均值
GUARD = 2                   # 保护块
KURT_MAX = 1.6              # 复样本峰度上限（复高斯 ≈ 2.0，恒模 → 1.0）
DILATE = 1                  # 掩膜两端各扩 1 块
GAP_FILL = 2                # 填 ≤2 块的内部空隙
K_STATIONS = 2              # 跨站一致：≥2 站同时判到才算该源的块
CONT_FRAC = 0.90            # 掩膜占比 > 该值视为连续源，不做门控

_BG_CACHE = o16.D2_ROOT / "_bg_cache"
MODES = ["base", "zero", "concat", "crop", "seg"]   # crop=最长活跃段；seg=逐突发段分类后多数票

def _runs_impl(act: np.ndarray, min_blocks: int = 1):
    """块掩膜里所有连续 True 段，返回 [(i0, i1), …]（左闭右开）。"""
    out = []
    i = 0
    nb = act.size
    while i < nb:
        if act[i]:
            j = i
            while j < nb and act[j]:
                j += 1
            if (j - i) >= min_blocks:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _longest_run(act: np.ndarray):
    """块掩膜里最长的连续 True 段，返回 (i0, i1) 块索引（左闭右开）。"""
    best = None
    i = 0
    nb = act.size
    while i < nb:
        if act[i]:
            j = i
            while j < nb and act[j]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best

def _block_stats(w: np.ndarray, blk: int = BLK):
    """返回 (每块平均功率 e, 每块复样本峰度 ku, 块数)。"""
    nb = w.shape[0] // blk
    seg = w[: nb * blk].reshape(nb, blk)
    p2 = (np.abs(seg) ** 2).mean(axis=1)
    p4 = (np.abs(seg) ** 4).mean(axis=1)
    e = p2 + 1e-30
    return e, p4 / (p2 ** 2 + 1e-60), nb


def _cfar_blocks(e: np.ndarray) -> np.ndarray:
    """滑窗 CA-CFAR（时间轴，块粒度）：门限 = 参考块均值的 ALPHA 倍。"""
    nb = e.size
    act = np.zeros(nb, bool)
    for i in range(nb):
        lo = max(0, i - GUARD - NREF // 2)
        hi = min(nb, i + GUARD + NREF // 2 + 1)
        idx = np.r_[lo: max(lo, i - GUARD), min(nb, i + GUARD + 1): hi]
        if idx.size < 8:
            continue
        act[i] = e[i] > ALPHA * e[idx].mean()
    return act


def _quiet_blocks(e: np.ndarray) -> np.ndarray:
    """全局静默参考版 CFAR：底噪取块功率的 20 分位（占空比源的空闲时段即底噪）。

    对连续源，20 分位≈源功率 ⇒ 几乎判不出"活跃"，因此配合 CONT_FRAC 兜底
    （掩膜占比过高时视为连续源，不做门控）。
    """
    floor = np.percentile(e, 20)
    return e > ALPHA * floor


def _clean_blocks(act: np.ndarray) -> np.ndarray:
    """膨胀 + 填小空隙：突发边缘与内部短谷都算信号。"""
    nb = act.size
    out = act.copy()
    if DILATE > 0:
        for d in range(1, DILATE + 1):
            out |= np.r_[np.zeros(d, bool), act[: nb - d]]
            out |= np.r_[act[d:], np.zeros(d, bool)]
    # 填空隙
    i = 0
    while i < nb:
        if not out[i]:
            j = i
            while j < nb and not out[j]:
                j += 1
            if 0 < i and j < nb and (j - i) <= GAP_FILL:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _mask_from_blocks(act: np.ndarray, n: int, blk: int = BLK) -> np.ndarray:
    nb = act.size
    m = np.repeat(act, blk)
    if m.size < n:
        m = np.r_[m, np.zeros(n - m.size, bool)]
    else:
        m = m[:n]
    return m


def _steering(rec: dict, positions):
    off = np.asarray(rec["element_offsets_world_m"], float)
    rp = np.asarray(rec["receiver_position_m"], float)
    A = np.zeros((off.shape[0], len(positions)), complex)
    for k, p in enumerate(positions):
        u = np.asarray(p, float) - rp
        u = u / (np.linalg.norm(u) + 1e-12)
        A[:, k] = np.exp(2j * np.pi * FC / C * (off @ u))
    return A


def _resid_for(cap: Path, obs: dict, sid: str, station: str, refs):
    """站点残差（目标剥离后），带独立缓存（_bg_cache，不碰 gate 的残差缓存）。"""
    p = _BG_CACHE / sid / f"resid_{station}.npy"
    if p.exists():
        r = np.load(p)
        if np.asarray(r).ndim == 2:
            return np.asarray(r, np.complex128)
    rec0 = obs["receivers"][0]
    ctx = ct.ToolContext(
        capture_dir=cap / "public",
        cache_dir=_BG_CACHE / sid,
        allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
        fs=rec0["sample_rate_hz"], fc=rec0["center_frequency_hz"],
        target_bw_normalized=0.25, aoi_center=[0, 0], target_reference=refs)
    ct.set_context(ctx)
    r = np.asarray(ct._strip_target_at_station(station), np.complex128)
    if r.ndim == 2:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, r.astype(np.complex64))
    return r


def run(deck: str, n_max: int, only: list[str], mode: str, use_kurt: bool,
        use_station: bool, verbose: bool):
    root = o16.D2_ROOT
    model, _classes = o16.load_model()
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    samples = [s for s in idx["samples"] if (not only or s["id"] in only)][:n_max]

    per_cls = {}          # kind -> [base_ok, base_n, gate_ok, gate_n, concat_ok, concat_n]
    duty = {}             # kind -> [掩膜占比和, 样本数]
    n_used = n_skip = 0
    t0 = time.time()

    for s in samples:
        cap = root / s["id"] / "capture"
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        full = json.loads((cap / "private" / "truth.json").read_text(encoding="utf-8"))
        gts = [(i + 1, g) for i, g in enumerate(truth["interferers"])
               if g["kind"] in o16.CLS_IDX]
        if not gts:
            continue
        rec0 = obs["receivers"][0]
        ns = rec0["shape"][1]
        stations = [r["receiver_id"] for r in obs["receivers"]] if use_station else ["station_a"]
        try:
            refs = rca._build_target_reference(cap / "private", ns)
        except Exception:                                    # noqa: BLE001
            refs = None

        # ---- 逐站残差与逐源波形 ----
        W = {}          # (station, k) -> 整窗波形（未归一化）
        vis = {}        # (station, k) -> bool 是否可见（INR≥3 dB）
        for st in stations:
            try:
                r = _resid_for(cap, obs, s["id"], st, refs)
            except Exception:                                # noqa: BLE001
                continue
            if r is None or np.asarray(r).ndim != 2:
                continue
            rec = next(x for x in obs["receivers"] if x["receiver_id"] == st)
            stf = next((x for x in full["stations"] if x["receiver_id"] == st), None)
            rx = stf["receiver"] if stf else None
            noise_dbm = None
            pw = {}
            if rx is not None:
                try:
                    noise_dbm = 10 * np.log10(rx.get("effective_output_noise_w")
                                              or rx["expected_output_noise_w"]) + 30
                    pw = {x["emitter_id"]: 10 * np.log10(float(np.mean(x["mean_port_power_w"])) + 1e-30) + 30
                          for x in stf["sources"]}
                except Exception:                            # noqa: BLE001
                    noise_dbm = None
            A = _steering(rec, [g["position_m"] for _gi, g in gts])
            if not np.isfinite(np.linalg.cond(A)) or np.linalg.cond(A) > 1e6:
                continue
            S = np.linalg.pinv(A) @ r
            t = np.arange(r.shape[1])
            for k, (gi, g) in enumerate(gts):
                eid = f"interferer_{gi}"
                if noise_dbm is not None and pw:
                    vis[(st, k)] = (pw.get(eid, -999) - noise_dbm) >= 3.0
                else:
                    vis[(st, k)] = (st == "station_a")
                W[(st, k)] = S[k] * np.exp(-1j * 2 * np.pi * g["freq_offset_hz"] * t / fs_of(rec0))

        if not W:
            n_skip += 1
            continue
        n_used += 1

        for k, (gi, g) in enumerate(gts):
            sts = [st for st in stations if (st, k) in W and vis.get((st, k), False)]
            if not sts:
                continue
            # ---- 逐站掩膜 ----
            acts, e_ref = {}, None
            for st in sts:
                w = W[(st, k)]
                e, ku, nb = _block_stats(w)
                a_cfar = _cfar_blocks(e)
                a_quiet = _quiet_blocks(e)
                a = a_quiet | a_cfar
                if use_kurt:
                    a = a & (ku < KURT_MAX)
                a = _clean_blocks(a)
                acts[st] = a
                if e_ref is None:
                    e_ref = e
            # ---- 跨站一致 ----
            A_stack = np.stack([acts[st] for st in sts], axis=0)
            need = min(K_STATIONS, len(sts))
            act = A_stack.sum(axis=0) >= need
            act = _clean_blocks(act)
            frac = float(act.mean())
            w0 = W[(sts[0], k)]
            n = w0.shape[0]
            if frac > CONT_FRAC or frac == 0.0:
                mask = np.ones(n, bool)          # 连续源/判不出 → 不门控
            else:
                mask = _mask_from_blocks(act, n)

            # ---- 四种输入：整窗（基线）/ 门控置零 / 门控拼接 / 突发裁剪 ----
            base = w0 / (np.sqrt(np.mean(np.abs(w0) ** 2)) + 1e-12)
            zg = np.where(mask, w0, 0).astype(np.complex128)
            zg = zg / (np.sqrt(np.mean(np.abs(zg) ** 2)) + 1e-12)
            cc = w0[mask]
            cc = cc / (np.sqrt(np.mean(np.abs(cc) ** 2)) + 1e-12)
            cr = _longest_run(act)
            cp = zg
            if cr is not None:
                i0 = max(0, cr[0] - 1) * BLK
                i1 = min(n, (cr[1] + 1) * BLK)
                seg = w0[i0:i1]
                if seg.size >= 256:
                    cp = seg / (np.sqrt(np.mean(np.abs(seg) ** 2)) + 1e-12)

            ci = o16.CLS_IDX[g["kind"]]
            cin = {"base": base, "zero": zg, "concat": cc, "crop": cp}
            preds = {}
            for mn, wv in cin.items():
                w_in = np.asarray(wv, np.complex64)
                preds[mn] = int(o16._pred(model, [w_in])[0]) if w_in.size >= 64 else -1
            # 逐突发段分类 + 多数票（小目标检测思路：不依赖整窗平均）
            pv = []
            for (a_, b_) in _runs_impl(act)[:8]:
                j0 = max(0, a_ - 1) * BLK
                j1 = min(n, (b_ + 1) * BLK)
                if j1 - j0 >= 512:
                    s_ = w0[j0:j1]
                    s_ = s_ / (np.sqrt(np.mean(np.abs(s_) ** 2)) + 1e-12)
                    pv.append(int(o16._pred(model, [np.asarray(s_, np.complex64)])[0]))
            preds["seg"] = max(set(pv), key=pv.count) if pv else preds["base"]   # 无段 → 同基线
            row = per_cls.setdefault(g["kind"], [0, 0] * len(MODES))
            marks = []
            for mi, mn in enumerate(MODES):
                p = preds.get(mn, -1)
                row[2 * mi] += int(p == ci)
                row[2 * mi + 1] += 1
                marks.append(f"{mn}={'✓' if p == ci else '✗'}")
            pn = lambda q: (_classes[q] if 0 <= q < len(_classes) else "?")   # noqa: E731
            dd = duty.setdefault(g["kind"], [0.0, 0, 0.0])
            dd[0] += frac
            dd[1] += 1
            if e_ref is not None and e_ref.size == act.size:
                dd[2] += float(e_ref[act].sum() / (e_ref.sum() + 1e-30))
            if verbose:
                print(f"  {s['id']} {g['kind']:<14} duty={frac:.3f} " + " ".join(marks)
                      + f"  base→{pn(preds['base'])} crop→{pn(preds['crop'])}"
                      + f" seg→{pn(preds['seg'])}")

    tot_ok = [0] * len(MODES)
    tot_n = [0] * len(MODES)
    lines = []
    for kind, r in sorted(per_cls.items(), key=lambda kv: -kv[1][1]):
        segs = []
        for mi, mn in enumerate(MODES):
            ok, cnt = r[2 * mi], r[2 * mi + 1]
            tot_ok[mi] += ok
            tot_n[mi] += cnt
            segs.append(f"{mn} {ok}/{cnt}={ok / max(cnt, 1):.3f}")
        dd = duty[kind]
        lines.append(f"  {kind:<14} n={r[1]:>3}  " + "  ".join(segs)
                     + f"   占空比均≈{dd[0] / max(dd[1], 1):.3f}"
                     + f" 能量集中度均≈{dd[2] / max(dd[1], 1):.3f}")
    hdr = (f"== 突发时间门控原型（{deck}）==\n"
           f"参数：块长 {BLK} 样点 | CA-CFAR Pfa={PFA:.0e}（参考块 {NREF}、保护 {GUARD}、"
           f"α={ALPHA:.2f}）| 峰度门限 {'<' + str(KURT_MAX) if use_kurt else '关'} | "
           f"跨站一致 K={K_STATIONS if use_station else 1} | 膨胀 {DILATE} | 填空隙 {GAP_FILL}\n"
           f"样本 {n_used}（跳过 {n_skip}） | 用时 {time.time() - t0:.1f}s\n")
    tot = "  合计            " + "  ".join(
        f"{mn} {tot_ok[mi]}/{tot_n[mi]}={tot_ok[mi] / max(tot_n[mi], 1):.3f}"
        for mi, mn in enumerate(MODES))
    text = hdr + "\n".join(lines) + "\n" + tot + "\n"
    return text


def fs_of(rec0) -> float:
    return float(rec0["sample_rate_hz"])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", nargs="?", default="d2")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", default="")
    ap.add_argument("--mode", default="both", choices=["zero", "concat", "both"])
    ap.add_argument("--kurt", action="store_true", help="叠加复样本峰度判据")
    ap.add_argument("--nostation", action="store_true", help="只用 station_a（不跨站）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv[1:])

    if o16.D2_ROOT.name != f"dataset_{args.deck}":
        print(f"!! DSH_D2_ROOT={o16.D2_ROOT} 与 deck={args.deck} 不一致，"
              f"请设置 $env:DSH_D2_ROOT", flush=True)
    only = [x for x in args.samples.split(",") if x]
    text = run(args.deck, args.n, only, args.mode, args.kurt, not args.nostation, True)
    print(text)
    out = Path(args.out) if args.out else (ADP.parent / "evidence" / f"_burst_gate_{args.deck}.txt")
    out.write_text(text, encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv)
