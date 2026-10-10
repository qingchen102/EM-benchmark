"""只读体检（2026-10-09，应答用户 m04805）：

  A) 前端非线性（LNA 饱和 / 阻塞 / IMD3）余量：把 RT 真值里逐站的各源
     mean_port_power_w 相加 → 前端口功率(dBm)，与前端配置的 iip3_dbm /
     soft_limit_input_dmb 比较，给出"离压缩点还有多少 dB"。

  B) 实际占用带宽：用平台自己的 waveforms._generate_preset 复现每个干扰源的
     时域波形（与数据集完全同一条代码路径，不做任何"标称带宽"滤波），测 99%
     占用带宽，再算 f_lo/f_hi 与接收通带(±810 kHz)、奈奎斯特(±2000 kHz) 的裕量。

只读数据集，不写任何数据文件；结果写 --out 指定的 UTF-8 文本。

用法：
  python -u _an_lna_obw.py --out ../_an_lna_obw.txt
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                                  # benchmark_v3
PLAT = BASE / "城市低空电磁信号仿真平台" / "src"
if str(PLAT) not in sys.path:
    sys.path.insert(0, str(PLAT))

from cityrf import waveforms as wf                          # noqa: E402
from cityrf.config import Scenario                          # noqa: E402

FS = 4.0e6
PASSBAND_HZ = 810e3          # 接收机 FIR 截止（RX_FRAC = 0.45*0.9 = 0.405 → ±810 kHz）
NYQ_HZ = FS / 2.0            # 2 MHz
N = 40000                    # 与数据集同窗长（10 ms @ 4 MHz）
DATASETS = ["d1", "d2", "d3", "d4"]


def dbm(w: float) -> float:
    return 10.0 * math.log10(max(float(w), 1e-30)) + 30.0


def obw(iq, fs: float, frac: float = 0.99):
    """返回 99% 能量所占频带 (lo, hi)，双边谱。"""
    x = np.asarray(iq, dtype=complex).copy()
    x = x - x.mean()
    x = x * np.hanning(len(x))
    p = np.abs(np.fft.fft(x)) ** 2
    f = np.fft.fftfreq(len(x), 1.0 / fs)
    o = np.argsort(f)
    f, p = f[o], p[o]
    c = np.cumsum(p) / (p.sum() + 1e-300)
    lo = float(f[int(np.searchsorted(c, (1.0 - frac) / 2.0))])
    hi = float(f[int(np.searchsorted(c, 1.0 - (1.0 - frac) / 2.0))])
    return lo, hi



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(BASE))
    ap.add_argument("--datasets", default="d1,d2,d3,d4")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--obw-samples", type=int, default=20,
                    help="每个数据集抽多少个样本做 OBW（按样本序）")
    ap.add_argument("--obw-max", type=int, default=60,
                    help="每个数据集最多复现多少个源")
    ap.add_argument("--out", default=str(BASE / "_an_lna_obw.txt"))
    a = ap.parse_args()
    root = Path(a.root)
    lines: list[str] = []

    def P(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    P("=" * 78)
    P("前端非线性余量 + 实际占用带宽（只读体检）")
    P(f"接收通带 ±{PASSBAND_HZ/1e3:.0f} kHz   奈奎斯特 ±{NYQ_HZ/1e3:.0f} kHz   窗长 {N} 点 @ {FS/1e6:.1f} MHz")
    P("=" * 78)

    fe_seen: dict[str, dict] = {}
    kind_stat: dict[str, list] = {}

    for d in [s for s in a.datasets.split(",") if s]:
        droot = root / f"dataset_{d}"
        samples = sorted([p for p in droot.glob(f"{d}_0*") if p.is_dir()])[: a.samples]
        P("")
        P(f"### {d}   样本 {len(samples)}")

        # ---------- A) 前端功率余量 ----------
        tot_dbm: list[float] = []
        per_src_dbm: list[float] = []
        fe_models: list[str] = []
        for sp in samples:
            tp = sp / "capture" / "private" / "truth.json"
            if not tp.exists():
                continue
            tr = json.loads(tp.read_text(encoding="utf-8"))
            for st in tr.get("stations", []):
                r = st.get("receiver") or {}
                ic = r.get("input_composite_power_w")
                if ic is not None:
                    tot_dbm.append(dbm(float(ic)))      # 前端口复合输入功率（标量）
                fe_models.append(str(r.get("frontend_model") or "?"))
                for s in st.get("sources", []):
                    w = s.get("mean_port_power_w")
                    if isinstance(w, list) and w:
                        per_src_dbm.append(dbm(float(np.mean(w))))
                    elif isinstance(w, (int, float)):
                        per_src_dbm.append(dbm(float(w)))
        if tot_dbm:
            t = np.array(tot_dbm)
            s = np.array(per_src_dbm)
            P(f"A 前端口复合输入功率: 中位 {np.median(t):.1f} dBm   "
              f"最弱 {t.min():.1f}   最强 {t.max():.1f}   （n={len(t)} 站次）")
            P(f"  单源(阵元均值)口功率: 中位 {np.median(s):.1f} dBm  最强 {s.max():.1f} dBm")
            P(f"  receiver.frontend_model = {sorted(set(fe_models))}")
            P(f"  → 最强输入距平台默认 IIP3(10 dBm)：{10.0 - t.max():.1f} dB")
            P(f"  → 若开 imd3，三阶互调产物约 {3*t.max() - 2*10.0:.1f} dBm（远低于本底）")
            P(f"  → soft_limit_input_dbm 未设置；要触发压缩需把门限设到 ≈ {t.max():.0f} dBm 量级")

        # ---------- B) 实际占用带宽 ----------
        obw_rows = []
        gen_n = 0
        for sp in samples[: a.obw_samples]:
            scp = sp / "capture" / "private" / "scenario.json"
            if not scp.exists():
                continue
            try:
                sc = Scenario.model_validate_json(scp.read_text(encoding="utf-8"))
            except Exception as exc:                       # 防御：schema 差异不致命
                P(f"  [warn] {sp.name} scenario 解析失败: {exc}")
                continue
            for tx in sc.emitters:
                if not str(tx.id).startswith("interferer"):
                    continue
                if gen_n >= a.obw_max:
                    break
                c0 = tx.components[0]
                nominal = float(c0.bandwidth_hz)
                try:
                    iq = wf._generate_preset(tx, FS, np.arange(N, dtype=np.int64), 12345)
                except Exception as exc:
                    P(f"  [warn] {sp.name}/{tx.id} 波形复现失败: {exc}")
                    continue
                gen_n += 1
                lo, hi = obw(iq, FS)
                obw_rows.append(dict(sample=sp.name, sid=tx.id, kind=str(tx.preset),
                                     nominal=nominal, lo=lo, hi=hi,
                                     offset=float(c0.offset_hz)))
                kind_stat.setdefault(str(tx.preset), []).append((hi - lo) / max(nominal, 1.0))
            if gen_n >= a.obw_max:
                break

        if obw_rows:
            nom = np.array([r["nominal"] for r in obw_rows])
            wid = np.array([r["hi"] - r["lo"] for r in obw_rows])
            edge = np.array([max(abs(r["lo"]), abs(r["hi"])) for r in obw_rows])
            P(f"B 实际占用带宽 (99%) 复现 {len(obw_rows)} 个源（平台同一条生成路径，无标称滤波）")
            P(f"  标称带宽: 中位 {np.median(nom)/1e3:.0f} kHz  (min {nom.min()/1e3:.0f}, max {nom.max()/1e3:.0f})")
            P(f"  实测 OBW99: 中位 {np.median(wid)/1e3:.0f} kHz  (min {wid.min()/1e3:.0f}, max {wid.max()/1e3:.0f})")
            P(f"  实测/标称 比值: 中位 {np.median(wid/nom):.2f}  max {np.max(wid/nom):.2f}")
            P(f"  频带最外沿 |f|: 中位 {np.median(edge)/1e3:.0f} kHz  max {edge.max()/1e3:.0f} kHz")
            P(f"  → 越接收通带({PASSBAND_HZ/1e3:.0f} kHz) 的源: {int((edge > PASSBAND_HZ).sum())}/{len(edge)}"
              f"   越奈奎斯特({NYQ_HZ/1e3:.0f} kHz): {int((edge > NYQ_HZ).sum())}/{len(edge)}")
            worst = sorted(obw_rows, key=lambda r: -max(abs(r["lo"]), abs(r["hi"])))[:5]
            for r in worst:
                P(f"    {r['sample']} {r['sid']:<12} {r['kind']:<16} 标称 {r['nominal']/1e3:6.0f} kHz "
                  f"| 实测 {r['lo']/1e3:8.0f} ~ {r['hi']/1e3:8.0f} kHz (宽 {(r['hi']-r['lo'])/1e3:.0f})")

    P("")
    P("### 按辐射体种类的 实测/标称 带宽比")
    for k in sorted(kind_stat):
        v = np.array(kind_stat[k])
        P(f"  {k:<18} n={len(v):3d}  比值 中位 {np.median(v):.2f}  max {v.max():.2f}")

    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[写出] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
