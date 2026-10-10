"""只读核查：v2 阶段那几条物理约束，在 v3 现行数据里是否仍然成立。

参照 benchmark_v2 的口径（benchmark_v2/simulation/README.md §4.4.6、benchmark_v2/README.md:24/222）：
  A 奈奎斯特防折叠：干扰「完整频带」[f0-BW/2, f0+BW/2] 不得越过归一化 ±0.5
  B 多源可分离  ：|Δfreq| ≥ 0.15 或 ΔDOA ≥ 15°
  C 可检测性    ：至少一站 INR ≥ 3 dB（v2 曾要求全站）
  D 阵元失配    ：v2 每样本抽阵元增益/相位失配
  E 非线性/阻塞 ：v2 有 LNA 饱和触发；v3 是否开
本脚本只读数据集，不写任何东西。

用法: python _an_v3_constraints.py [d1 d2 d3 d4]
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAT = ROOT / "城市低空电磁信号仿真平台" / "src"
if str(PLAT) not in sys.path:
    sys.path.insert(0, str(PLAT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import gen_dataset as gd  # noqa: E402  （取 COARSE 粗类映射）
FS = 4.0e6
PASS_HZ = 0.405 * FS     # 接收机通带半宽（RX_FRAC = 0.45*0.9 → ±810 kHz）
NYQ_HZ = 0.5 * FS        # 奈奎斯特半宽 ±2 MHz
STATIONS = ["s0", "s1", "s2", "s3"]


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def src_list(t):
    """返回 [(name, dict)]，含 target 与全部干扰。"""
    out = []
    tgt = t.get("target") or {}
    if tgt:
        out.append(("target", tgt))
    for i, s in enumerate(t.get("interferers") or []):
        out.append((f"ifw{i + 1}", s))
    return out


def doa_u(s, st):
    ps = (s.get("per_station") or {}).get(st) or {}
    d = ps.get("doa_deg")
    if d is None:
        return None
    return float(np.sin(np.deg2rad(float(d))))


def inr_of(s, st):
    ps = (s.get("per_station") or {}).get(st) or {}
    v = ps.get("inr_db")
    return None if v is None else float(v)


def main(diffs):
    for d in diffs:
        root = ROOT / f"dataset_{d}"
        idx = jload(root / "index.json")
        sids = [s["id"] if isinstance(s, dict) else s for s in idx["samples"]]

        head, clip, fold = [], [], []
        n_src = 0
        pairs = []            # (d_freq_norm, max_ddoa_deg, min_du, overlap_frac, both_vis)
        invis = 0
        n_src_inr = 0
        obs_keys = None
        ant_info = None
        for sid in sids:
            cap = root / sid / "capture"
            t = jload(cap / "private" / "d1_truth.json")
            srcs = src_list(t)
            recs = t.get("receivers") or []
            for name, s in srcs:
                f0 = float(s.get("freq_offset_hz") or 0.0)
                bw = float(s.get("bandwidth_hz") or 0.0)
                h = abs(f0) + bw / 2.0
                head.append(h / FS)
                n_src += 1
                if h > PASS_HZ:
                    clip.append((sid, name, round(h / FS, 4)))
                if h > NYQ_HZ:
                    fold.append((sid, name, round(h / FS, 4)))
                if name != "target":
                    mx = max([v for v in (inr_of(s, st) for st in STATIONS) if v is not None] or [-99])
                    n_src_inr += 1
                    if mx < 3.0:
                        invis += 1
            # 两两分离性
            for i in range(len(srcs)):
                for j in range(i + 1, len(srcs)):
                    (n1, s1), (n2, s2) = srcs[i], srcs[j]
                    f1, b1 = float(s1.get("freq_offset_hz") or 0.0), float(s1.get("bandwidth_hz") or 0.0)
                    f2, b2 = float(s2.get("freq_offset_hz") or 0.0), float(s2.get("bandwidth_hz") or 0.0)
                    df = abs(f1 - f2) / FS
                    lo = max(f1 - b1 / 2, f2 - b2 / 2)
                    hi = min(f1 + b1 / 2, f2 + b2 / 2)
                    ov = max(0.0, hi - lo) / max(min(b1, b2), 1.0)
                    dd, du, both = [], [], False
                    for st in STATIONS:
                        i1, i2 = inr_of(s1, st), inr_of(s2, st)
                        if i1 is None or i2 is None or i1 < 3.0 or i2 < 3.0:
                            continue
                        both = True
                        u1, u2 = doa_u(s1, st), doa_u(s2, st)
                        if u1 is not None and u2 is not None:
                            du.append(abs(u1 - u2))
                            dd.append(abs(np.rad2deg(np.arcsin(u1) - np.arcsin(u2))))
                    if both:
                        pairs.append((df, max(dd) if dd else None, min(du) if du else None, ov, sid, n1, n2,
                                      gd.COARSE.get(s1.get("kind")) == gd.COARSE.get(s2.get("kind"))))
            # 观测口径（取第一个样本）
            if obs_keys is None:
                obs = jload(cap / "public" / "observations.json")
                r0 = (obs.get("receivers") or [{}])[0]
                obs_keys = sorted(obs.keys())
                ant = r0.get("antenna") or {}
                ant_info = {
                    "calibrated": r0.get("calibrated"),
                    "num_cols": ant.get("num_cols"),
                    "spacing_wl": ant.get("horizontal_spacing_wavelengths"),
                    "gain_error_std_db": ant.get("gain_error_std_db"),
                    "phase_error_std_deg": ant.get("phase_error_std_deg"),
                    "tuning_offset_hz": r0.get("tuning_offset_hz"),
                }

        h = np.asarray(head, dtype=float)
        print(f"\n================ dataset_{d}  ({len(sids)} 样本 / {n_src} 源) ================")
        print(f"A 防折叠   带宽半宽 |f0|+BW/2 归一化: max {h.max():.4f}  p90 {np.percentile(h, 90):.4f}  中位 {np.median(h):.4f}")
        print(f"           越过接收通带(±0.405): {len(clip)} 个源 ' {clip[:6]}")
        print(f"           越过奈奎斯特(±0.5)  : {len(fold)} 个源 ' {fold[:6]}")
        print(f"C 可检测性 干扰中至少一站 INR<3dB 的源: {invis}/{n_src_inr}（目标不计入）")
        # 分离性（只在两侧都可见的站上算）
        same_ov = [p for p in pairs if p[3] > 0.5 and p[7]]
        cross_ov = [p for p in pairs if p[3] > 0.5 and not p[7]]
        qa_viol = [p for p in same_ov if p[2] is not None and p[2] < 0.25]
        v2_viol = [p for p in pairs if p[0] < 0.15 and (p[1] is None or p[1] < 15.0)]
        print(f"B 分离性   双侧可见源对 {len(pairs)}")
        print(f"           重叠>50%: 同粗类 {len(same_ov)} / 跨粗类 {len(cross_ov)}（跨粗类不要求可分）")
        print(f"           QA 约束 2 违例(同粗类 且 min|Δu|<0.25): {len(qa_viol)}"
              f"  {[(p[4], p[5], p[6], round(p[2], 3)) for p in qa_viol[:5]]}")
        print(f"           v2 口径违例(|Δf|<0.15 且 ΔDOA<15°): {len(v2_viol)}")
        for p in v2_viol[:6]:
            df, dd, du, ov, sid, n1, n2, sc = p
            print(f"             {sid} {n1}/{n2}  Δf {df:.3f}  ΔDOA {dd if dd is None else round(dd, 1)}°"
                  f"  min|Δu| {du if du is None else round(du, 3)}  重叠 {ov:.2f}  同粗类 {sc}")
        if same_ov:
            dmin = np.array([p[2] for p in same_ov if p[2] is not None], dtype=float)
            if dmin.size:
                print(f"           同粗类重叠对 min|Δu|: 中位 {np.median(dmin):.3f}  min {dmin.min():.3f}  p10 {np.percentile(dmin, 10):.3f}")
        print(f"D 阵列     观测顶层键 {obs_keys}")
        print(f"           {ant_info}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    out = None
    ds = []
    for a in argv:
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
        else:
            ds.append(a)
    ds = ds or ["d1", "d2", "d3", "d4"]
    if out:
        buf = open(out, "w", encoding="utf-8")
        old, sys.stdout = sys.stdout, buf
        try:
            main(ds)
        finally:
            sys.stdout = old
            buf.close()
        print(f"[written] {out}")
    else:
        main(ds)
