"""长窗题目探针（只读数据集 + 写临时输出目录，不改任何数据集）。

目的（用户 m02455 提问："每个题目里题型能不能多一点点，比如加入几个长窗题"）：
  1) 用 frozen channel 重放把采集窗从 10 ms 拉长 —— 验证"不重跑 RT 就能做长窗"
     并做一次 bit-exact 自检（0.01 s 重放是否与原样本逐位相同）。
  2) 量出长窗的代价（墙钟时间 / 磁盘）。
  3) 量出跳频源的时频结构 —— 关键问题：**把窗拉长会不会让跳频变物理？**
     预期答案：不会。emitters.py 的时序参数是绝对采样点（按 length=1024 校准），
     窗变长只会得到更多非物理跳（等效跳速恒定在 ~数十 khops/s，真实蓝牙 1600 hops/s）。

复用 benchmark_v3/城市低空电磁信号仿真平台/src/cityrf/replay.py 的链路重建逻辑。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

try:  # Windows 控制台默认 GBK，中文/符号会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_PLATFORM = _V3 / "城市低空电磁信号仿真平台"
for _p in (str(_PLATFORM / "src"), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2" / "simulation"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cityrf.channels import LinkChannel  # noqa: E402
from cityrf.config import Scenario  # noqa: E402
from cityrf.runner import run_scenario  # noqa: E402

OUTROOT = _HERE / "_probe_longwin_out"
DURS = [0.01, 0.10, 0.50]
N_SAMPLES = 2
HOPSHAPES = {"Bluetooth_FHSS", "UAV_RC_hopping"}
TIMESHAPES = HOPSHAPES | {"WiFi_burst", "Zigbee", "Microwave", "DECT",
                          "Radar_pulsed_LFM", "Welder", "VFD_noise", "Corona"}

REAL_HOP_HZ = {"Bluetooth_FHSS": 1600.0}   # 真实蓝牙：1600 hops/s（驻留 625 µs）

LOG: list[str] = []


def log(s: str = "") -> None:
    LOG.append(s)
    print(s, flush=True)


# ---------------- 时频指纹（复刻 capture_tools.analyze_time_frequency_station 的口径） ----------------


def bandlimited_mean(iq: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    n = iq.shape[1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = (f >= lo) & (f <= hi)
    return np.fft.ifft(np.fft.fft(iq, axis=1) * m, axis=1).mean(axis=0)


def tf_fingerprint(x: np.ndarray, fs: float, win: int = 64, hop: int = 32) -> dict:
    """复刻工具口径：win/hop=64/32（16 µs 帧），活动检测用 p10 噪声参考。"""
    n = x.size
    nf = max((n - win) // hop + 1, 1)
    w = np.hanning(win)
    frames = np.stack([x[i * hop:i * hop + win] * w for i in range(nf)])
    pfr = np.abs(np.fft.fft(frames, axis=1)) ** 2
    fp = pfr.sum(axis=1)
    ref = np.percentile(fp, 10) + 1e-30
    active = fp > ref * 10 ** (6 / 10)
    fgrid = np.fft.fftfreq(win, 1 / fs)
    peaks = fgrid[np.argmax(pfr, axis=1)]
    hist, _ = np.histogram(peaks, bins=64, range=(-2e6, 2e6))
    return {"duty": round(float(active.mean()), 3),
            "frame_power_cv": round(float(np.std(fp) / (np.mean(fp) + 1e-30)), 2),
            "hop_channel_count": int(np.sum(hist > max(5, 0.05 * len(peaks)))),
            "frame_us": round(win / fs * 1e6, 1)}


def hop_stats(x: np.ndarray, fs: float, win: int = 32, hop: int = 8) -> dict:
    """估跳速/驻留：细帧峰频序列 → 落到被占用信道 → 数跳变次数。

    win=32 (8 µs) / hop=8 (2 µs) —— 帧长短于标称驻留（6–15 µs）才能分辨单次跳。
    """
    n = x.size
    nf = max((n - win) // hop + 1, 1)
    w = np.hanning(win)
    frames = np.stack([x[i * hop:i * hop + win] * w for i in range(nf)])
    pfr = np.abs(np.fft.fft(frames, axis=1)) ** 2
    fgrid = np.fft.fftfreq(win, 1 / fs)
    peaks = fgrid[np.argmax(pfr, axis=1)]
    hist, edges = np.histogram(peaks, bins=32, range=(-2e6, 2e6))
    occ = np.nonzero(hist > 0.02 * nf)[0]
    if occ.size < 2:
        return {"n_channels": int(occ.size), "transitions": 0, "hop_rate_hz": 0.0,
                "dwell_us": None, "frame_us": round(win / fs * 1e6, 2)}
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.array([int(np.argmin(np.abs(centers[occ] - p))) for p in peaks])
    # 去抖：只在连续 ≥3 帧保持新信道时才认一次跳
    seq = [idx[0]]
    run, cur = 1, idx[0]
    for v in idx[1:]:
        if v == cur:
            run += 1
        else:
            if run >= 3:
                seq.append(v)
            cur, run = v, 1
    trans = max(len(seq) - 1, 0)
    dur_s = n / fs
    return {"n_channels": int(occ.size), "transitions": int(trans),
            "hop_rate_hz": round(trans / dur_s, 1),
            "dwell_us": round(dur_s / trans * 1e6, 2) if trans else None,
            "frame_us": round(win / fs * 1e6, 2)}


# ---------------- 样本扫描 / 重放 ----------------


def scan_candidates() -> list[dict]:
    out = []
    for ds in ("dataset_d2", "dataset_d3", "dataset_d4"):
        root = _V3 / ds
        if not root.is_dir():
            continue
        for cap in sorted(root.glob("d*_*/capture")):
            tp = cap / "private" / "d1_truth.json"
            if not tp.exists():
                continue
            t = json.loads(tp.read_text(encoding="utf-8"))
            kinds = [iw.get("kind") for iw in (t.get("interferers") or [])]
            score = sum(2 if k in HOPSHAPES else 1 for k in kinds if k in TIMESHAPES)
            if score:
                out.append({"cap": cap, "ds": ds, "kinds": kinds, "score": score})
    out.sort(key=lambda r: -r["score"])
    return out


def build_links(cap: Path, rt_truth: dict) -> list[LinkChannel]:
    links = []
    for station in rt_truth["stations"]:
        for source in station["sources"]:
            tx, rx = source["emitter_id"], station["receiver_id"]
            d = np.load(cap / "private" / f"channel_{tx}_{rx}.npz", allow_pickle=False)
            links.append(LinkChannel(tx, rx, d["coefficients"], d["delays_s"],
                                     source["geometry"], source["channel_provenance"]))
    return links


def free_gb() -> float:
    return shutil.disk_usage(str(_V3)).free / 2 ** 30


def main() -> None:
    cands = scan_candidates()
    log(f"候选样本（含时域结构源）：{len(cands)} 个")
    for c in cands[:6]:
        log(f"  {c['ds']}/{c['cap'].parent.name}  score={c['score']}  kinds={c['kinds']}")

    picked = cands[:N_SAMPLES]
    results = []
    OUTROOT.mkdir(exist_ok=True)

    for c in picked:
        cap = c["cap"]
        sid = cap.parent.name
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        rt = json.loads((cap / "private" / "truth.json").read_text(encoding="utf-8"))
        cfg0 = Scenario.model_validate_json((cap / "private" / "scenario.json").read_text(encoding="utf-8"))
        fs = float(getattr(cfg0, "sample_rate_hz", 4e6))
        links = build_links(cap, rt)
        log("")
        log("=" * 78)
        log(f"样本 {c['ds']}/{sid}  源={c['kinds']}  fs={fs/1e6:.2f} MHz  "
            f"原始窗={cfg0.duration_s*1e3:.1f} ms  链路={len(links)}")

        for dur in DURS:
            cfg = cfg0.model_copy(update={"duration_s": dur})
            n = cfg.num_samples
            outdir = OUTROOT / f"{sid}_{int(dur*1000)}ms"
            est_mb = n * sum(r.antenna.elements for r in cfg.receivers) * 8 / 2 ** 20
            free0 = free_gb()
            t0 = time.perf_counter()
            status, err = "ok", None
            try:
                run_scenario(cfg, outdir, frozen_links=links)
            except MemoryError as e:
                status, err = "MemoryError", str(e)
            except Exception as e:  # noqa: BLE001
                status, err = type(e).__name__, str(e)[:300]
            dt = time.perf_counter() - t0
            free1 = free_gb()
            log(f"  -- dur={dur*1e3:5.1f} ms  n={n:>9,}  预估 IQ={est_mb:8.1f} MB  "
                f"重放 {dt:6.2f} s  磁盘 {free0-free1:7.1f} MB  [{status}]"
                + (f"  ERR={err}" if err else ""))
            if status != "ok":
                continue

            rec = {"sample": sid, "ds": c["ds"], "dur_s": dur, "n": int(n),
                   "replay_s": round(dt, 3), "kinds": c["kinds"]}
            # bit-exact 自检（只有 0.01 s 才可能与原样本一致）
            if abs(dur - cfg0.duration_s) < 1e-12:
                try:
                    a_new = np.load(outdir / "public" / "iq_station_a.npy")
                    a_old = np.load(cap / "public" / "iq_station_a.npy")
                    if a_new.shape == a_old.shape:
                        dmax = float(np.max(np.abs(a_new - a_old)))
                        rec["bitexact_max_abs_diff"] = dmax
                        log(f"     bit-exact 自检：shape={a_new.shape}  max|Δ|={dmax:.3e}  "
                            f"{'逐位相同 ✅' if dmax == 0 else '≠ 不一致'}")
                    else:
                        rec["bitexact_max_abs_diff"] = None
                        log(f"     bit-exact 自检：形状不同 {a_new.shape} vs {a_old.shape}")
                except Exception as e:  # noqa: BLE001
                    log(f"     bit-exact 自检失败：{type(e).__name__}: {str(e)[:120]}")

            # 时频分析：对每个跳频/时域结构源
            iq = np.load(outdir / "public" / "iq_station_a.npy")
            rec["iq_shape"] = list(iq.shape)
            tfs = []
            for iw in (truth.get("interferers") or []):
                k = iw.get("kind")
                if k not in TIMESHAPES:
                    continue
                f0 = float(iw.get("freq_offset_hz", 0.0))
                bw = float(iw.get("bandwidth_hz", 200e3))
                lo, hi = max(f0 - bw / 2, -fs * 0.45), min(f0 + bw / 2, fs * 0.45)
                if hi - lo < fs / 64 * 2:
                    lo, hi = f0 - max(bw, 200e3), f0 + max(bw, 200e3)
                x = bandlimited_mean(iq, fs, lo, hi)
                fp = tf_fingerprint(x, fs)
                hs = hop_stats(x, fs)
                entry = {"kind": k, "gt_f0_hz": round(f0), "gt_bw_hz": round(bw),
                         "tool_fingerprint(win64/hop32)": fp,
                         "hop_stats(win32/hop8)": hs}
                if k in REAL_HOP_HZ and hs.get("hop_rate_hz"):
                    entry["real_hop_rate_hz"] = REAL_HOP_HZ[k]
                    entry["hop_rate_ratio_vs_real"] = round(hs["hop_rate_hz"] / REAL_HOP_HZ[k], 1)
                tfs.append(entry)
                log(f"     [{k:15s}] f0={f0/1e3:8.1f}kHz bw={bw/1e3:6.1f}kHz  "
                    f"工具指纹 duty={fp['duty']:.3f} cv={fp['frame_power_cv']:.2f} hop_ch={fp['hop_channel_count']:2d}"
                    f" | 细帧 {hs['n_channels']} 信道 {hs['transitions']:6d} 跳  "
                    f"跳速={hs['hop_rate_hz']:>10.0f} Hz  驻留={hs['dwell_us']} µs"
                    + (f"  (真实 {REAL_HOP_HZ[k]:.0f} Hz → ×{entry['hop_rate_ratio_vs_real']})"
                       if k in REAL_HOP_HZ else ""))
            rec["time_frequency"] = tfs
            results.append(rec)

    # 汇总
    log("")
    log("=" * 78)
    log("汇总：跳速是否随窗长变化？（若恒定 → 拉长窗长不会让跳频变物理）")
    for rec in results:
        for e in rec["time_frequency"]:
            if e["kind"] in REAL_HOP_HZ:
                log(f"  {rec['sample']}  dur={rec['dur_s']*1e3:5.1f} ms  {e['kind']:15s}  "
                    f"跳速={e['hop_stats(win32/hop8)']['hop_rate_hz']:>10.0f} Hz  "
                    f"驻留={e['hop_stats(win32/hop8)']['dwell_us']} µs  "
                    f"hop_ch={e['tool_fingerprint(win64/hop32)']['hop_channel_count']}")

    (_HERE / "_probe_longwin.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (_HERE / "_probe_longwin.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("")
    log(f"落盘：{_HERE / '_probe_longwin.json'} / _probe_longwin.txt")


if __name__ == "__main__":
    main()
