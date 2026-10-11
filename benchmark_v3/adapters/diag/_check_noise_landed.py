"""只读审计：新「按站抽签」噪声档位是否真的落进数据集。

逐样本核对三件事：
  ① 每站配方 == `noise_profiles.plan_sample(difficulty, 站序, 样本序号)`
     —— 抽签是纯函数，所以任何一份数据都能逐位复现它的配方；
  ② 实测各站底与配方应有的抬升吻合（thermal / correlated 应为 0；
     colored / impulse 用 `analytic_floor_rise_db` 作对照，容差 --tol dB）；
  ③ `noise.noise_floor_dbm` == **纯热噪参照底**（跨档/跨版本可比），
     且真值里的 `inr_db` / `snr_db` 是用**本站在效底**重算的。

用法：
    python _check_noise_landed.py                       # 默认查 d2,d3,d4
    python _check_noise_landed.py --diff d1,d2,d3,d4 --tol 0.5
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
import argparse
import json
import pathlib
import statistics as st
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import noise_profiles as nprof


def _lift_guess(name: str, difficulty: str) -> float:
    """该配方应有的带内抬底（解析对照，只用来判"量级对不对"）。"""
    if name in ("thermal", "correlated"):
        return 0.0
    return float(nprof.analytic_floor_rise_db(nprof.archetype_fields(name, difficulty))["total_rise_guess_db"])


def audit(root: pathlib.Path, diff: str, tol: float) -> int:
    ds = root / f"dataset_{diff}"
    if not ds.is_dir():
        print(f"  [跳过] {ds} 不存在")
        return 0
    idx = json.loads((ds / "index.json").read_text(encoding="utf-8"))
    ids = [s["id"] for s in idx["samples"]]
    if not ids:
        print(f"  [跳过] {ds}/index.json 没有 samples")
        return 0

    problems = 0
    cnt: dict[str, int] = {}
    per_st: dict[str, dict] = {}
    floors: dict[str, list] = {}
    plan_bad: list = []
    no_menu: list = []
    ref_set: list = []
    inr_bad = 0
    inr_n = 0
    snr_bad = 0
    snr_n = 0

    for si, sid in enumerate(ids):
        tp = ds / sid / "capture/private/d1_truth.json"
        if not tp.is_file():
            no_menu.append(sid)
            continue
        tk = json.loads(tp.read_text(encoding="utf-8"))
        nz = tk.get("noise") or {}
        if "stations" not in nz:
            no_menu.append(sid)
            continue
        recv = [r["id"] for r in tk["receivers"]]
        want = nprof.plan_sample(diff, recv, si)
        got = {rid: v.get("archetype") for rid, v in nz["stations"].items()}
        if got != want:
            plan_bad.append((sid, got, want))
        for rid in recv:
            ar = got.get(rid)
            cnt[ar] = cnt.get(ar, 0) + 1
            per_st.setdefault(rid, {})
            per_st[rid][ar] = per_st[rid].get(ar, 0) + 1
            f = float(nz["per_station"][rid]["effective_output_noise_dbm"])
            floors.setdefault(ar, []).append(f)
        ref_set.append(round(float(nz.get("noise_floor_dbm", float("nan"))), 2))
        # --- 派生字段：必须用本站在效底重算 ---
        for i, rid in enumerate(recv):
            d = float(nz["per_station"][rid]["effective_output_noise_dbm"])
            for g in tk.get("interferers") or []:
                ps = (g.get("per_station") or {}).get(f"s{i}") or {}
                if "pr_dbm" in ps and "inr_db" in ps:
                    inr_n += 1
                    if abs(round(float(ps["pr_dbm"]) - d, 2) - float(ps["inr_db"])) > 0.011:
                        inr_bad += 1
            ps = ((tk.get("target") or {}).get("per_station") or {}).get(f"s{i}") or {}
            if "pr_dbm" in ps and "snr_db" in ps:
                snr_n += 1
                if abs(round(float(ps["pr_dbm"]) - d, 2) - float(ps["snr_db"])) > 0.011:
                    snr_bad += 1

    total = sum(cnt.values())
    print(f"== dataset_{diff}  {len(ids)} 样本 / {total} 站  ({ds})")
    if no_menu:
        print(f"  !! {len(no_menu)} 个样本没有 noise.stations（未落档位？）: {no_menu[:5]}")
        problems += len(no_menu)
    print(f"  ① 配方逐位复现: {'全部一致 ✓' if not plan_bad else f'{len(plan_bad)} 处不一致 ✗'}")
    for sid, got, want in plan_bad[:3]:
        print(f"      {sid}: 实际 {got} != 期望 {want}")
    problems += len(plan_bad)

    ref = st.median(floors.get("thermal") or [float("nan")])
    shares = " / ".join(f"{k} {v / total:.3f}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]))
    print(f"  ② 配方分布: {shares}   （槽位权重应为 "
          f"{' / '.join(f'{a} {w / nprof.B_SLOTS:.2f}' for a, w in nprof.MENUS[nprof._key(diff)] if w)})")
    for rid, c in per_st.items():
        print(f"      {rid}: {c}")
    print(f"  ③ 实测底（按配方分组，参照底 {ref:.2f} dBm）:")
    for ar, vals in sorted(floors.items()):
        lo, hi = min(vals), max(vals)
        lift = st.median(vals) - ref
        guess = _lift_guess(ar, diff)
        ok = "✓" if abs(lift - guess) <= tol else "✗"
        print(f"      {ar:10s} n={len(vals):3d}  {st.median(vals):9.3f} dBm [{lo:9.3f}, {hi:9.3f}]"
              f"  抬 {lift:+5.2f} dB (解析 {guess:+5.2f}) {ok}")
        if ok == "✗":
            problems += 1
    vals_ref = [v for v in ref_set if v == v]          # 去掉 NaN
    spread = (max(vals_ref) - min(vals_ref)) if vals_ref else 0.0
    if len(vals_ref) != len(ref_set):
        print(f"  !! {len(ref_set) - len(vals_ref)} 个样本缺 noise_floor_dbm")
        problems += 1
    elif spread > 0.15:
        print(f"  !! noise_floor_dbm 离散过大: 跨度 {spread:.3f} dB  {sorted(set(vals_ref))[:6]}")
        problems += 1
    else:
        print(f"  ④ noise_floor_dbm（纯热噪参照底）: 中位 {st.median(vals_ref):.2f} dBm"
              f"  跨度 {spread:.3f} dB（每样本各一次噪声 realization，故有微弱抖动）✓")
    if "noise_profiles" in idx and "noise_menus" in idx:
        print(f"  ⑤ index: noise_profiles={idx['noise_profiles'].get(diff)}  "
              f"noise_menus={[(m['archetype'], m['weight']) for m in idx['noise_menus'].get(diff, [])]}")
    else:
        print("  !! index.json 缺 noise_profiles / noise_menus")
        problems += 1
    print(f"  ⑥ 派生字段重算: INR {inr_n - inr_bad}/{inr_n} 一致，SNR {snr_n - snr_bad}/{snr_n} 一致"
          f"  {'✓' if not (inr_bad or snr_bad) else '✗'}")
    problems += inr_bad + snr_bad
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..", help="数据集根目录（默认 ..，即 benchmark_v3/）")
    ap.add_argument("--diff", default="d2,d3,d4", help="要查的档位，逗号分隔")
    ap.add_argument("--tol", type=float, default=0.5, help="抬升与解析对照的容差 dB")
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()
    bad = 0
    for diff in [s.strip() for s in a.diff.split(",") if s.strip()]:
        bad += audit(root, diff, a.tol)
    print(f"\n== 总计: {'全部通过 ✓' if bad == 0 else f'{bad} 处问题 ✗'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
