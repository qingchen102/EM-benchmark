"""一次性：统计"方向分辨带"是否真的把源自己的载频框住了（对照旧的组带）。

只读 _diag_d2_off.json 的 freq_rows。
"""
import json

import numpy as np

d = json.load(open("_diag_d2_off.json", encoding="utf-8"))
rows = [r for r in d["freq_rows"] if r.get("f0") is not None]
n = len(rows)
print(f"freq_rows n = {n}")

g_cov = sum(1 for r in rows if r["lo"] <= r["f0"] <= r["hi"])
d_cov = sum(1 for r in rows if r.get("db_lo") is not None
            and r["db_lo"] <= r["f0"] <= r["db_hi"])
print(f"组带框住真载频      : {g_cov}/{n} = {g_cov / n:.1%}")
print(f"方向带框住真载频    : {d_cov}/{n} = {d_cov / n:.1%}")

gw = np.array([r["hi"] - r["lo"] for r in rows]) / 1e3
dw = np.array([r["db_hi"] - r["db_lo"] for r in rows
               if r.get("db_lo") is not None]) / 1e3
print(f"组带宽中位 {np.median(gw):8.1f} kHz   (min {gw.min():.1f} / max {gw.max():.1f})")
print(f"方向带宽中位 {np.median(dw):8.1f} kHz   (min {dw.min():.1f} / max {dw.max():.1f})")

print()
print(f"{'载频来源':22s} {'误差中位(kHz)':>14s} {'p90':>10s} {'n':>6s}")
for key, lab in (("cand_c", "组带中心(旧)"), ("db_c", "方向带重心"), ("bg", "GT方向带重心"),
                 ("b50", "方向带放宽50%重心"), ("cent", "提取波形谱重心")):
    e = [abs(r[key] - r["f0"]) / 1e3 for r in rows if r.get(key) is not None]
    if not e:
        continue
    e = np.array(e)
    print(f"{lab:22s} {np.median(e):14.1f} {np.percentile(e, 90):10.1f} {len(e):6d}")

