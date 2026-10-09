"""把"背景噪声"真正加进数据集 —— frozen-channel 重放 + 实测噪声口径 + 真值回写。

设计（不重跑 RT、不改平台源码）：

  1. 读 `capture/private/scenario.json` 与 `truth.json`，用已落盘的 `channel_*.npz` 重建 frozen links；
  2. 用 `noise_profiles.py` 的档位覆盖每个 receiver 的 `frontend` 噪声开关；
  3. `run_scenario(cfg, out_capture, frozen_links=links)` → 产出"含噪声"的新 IQ；
  4. **零输入噪声实测**：用同一 seed 调 `receiver.receive()`（`composite=0`，镜像 runner 的
     `start_index=-rx_guard` 与切片），得到与数据集里那一份噪声**同一实现**的输出噪声功率。
     噪声是加性的、且种子只由 `(seed, rx.id, 阵元号)` 决定，所以这一步不需要重放信号链。
  5. 回写 `private/truth.json`（每站 `receiver.effective_output_noise_w/_dbm`）与
     `private/d1_truth.json`（`noise` 块 + 由噪声派生的 `inr_db` / `snr_db` / `env_over_noise_db` / `impact_class`）；
  6. 重算 `run_summary.json` 的 `hashes`（**否则 `cityrf.audit.audit_run` 会拒绝该样本**，replay_scenario 随之失效）。

口径说明（回答"INR/SNR 能否把所有噪声叠加考虑"）：
  * 输入端可解析叠加，但**不能按带宽比折算有色**，也**不能只乘滤波增益**；
  * 因此 `noise.noise_floor_dbm` 改为**实测**值 = `10log10(实测输出噪声功率) + 30`，
    D1（纯白热噪）下它与旧解析值 `-174 + 10log10(1.62e6) + 5 = -106.9` dBm 逐位一致（自检点）；
  * 非线性（D4 的 IMD3 / 软限幅）下零输入实测是 **SNR 上界**，平台自己标了
    `noise_expectation_scope: "linear receiver only"`，故落盘时如实标注。

用法：
  # 只测不写（秒级，用于档位扫描）
  python regen_noise.py --src ../dataset_d3 --measure-only --limit 5
  # 完整重放（产出新数据集）
  python regen_noise.py --src ../dataset_d3 --out ../_regen_noise/dataset_d3 --difficulty d3
  # 单样本 + 仪器口径审计（工具峰 vs GT）
  python regen_noise.py --src ../dataset_d3 --out ../_regen_noise/dataset_d3 --samples d3_00000 --audit
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3 = HERE.parent
for _p in (
    str(HERE),
    str(_V3 / "simulation"),
    str(_V3.parent / "benchmark_v2"),
    str(_V3.parent / "benchmark_v2" / "simulation"),
    str(_V3 / "城市低空电磁信号仿真平台" / "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from cityrf.channels import LinkChannel       # noqa: E402
from cityrf.config import Scenario            # noqa: E402
from cityrf.export import digest              # noqa: E402
from cityrf.receiver import receive           # noqa: E402
from cityrf.runner import run_scenario        # noqa: E402

import noise_profiles as nprof                # noqa: E402


# ---------------------------------------------------------------------------
# 读样本 / 重建链路
# ---------------------------------------------------------------------------

def load_sample(src: Path, sid: str):
    cap = src / sid / "capture"
    priv = cap / "private"
    truth_rt = json.loads((priv / "truth.json").read_text(encoding="utf-8"))
    truth_key = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    cfg = Scenario.model_validate_json((priv / "scenario.json").read_text(encoding="utf-8"))
    return cap, priv, truth_rt, truth_key, cfg


def build_links(priv: Path, truth_rt: dict, cfg: Scenario):
    """按 replay.py 的口径重建 frozen links（每 emitter×receiver 一对，缺一不可）。"""
    links = []
    for st in truth_rt["stations"]:
        rx = st["receiver_id"]
        for s in st["sources"]:
            tx = s["emitter_id"]
            d = np.load(priv / f"channel_{tx}_{rx}.npz")
            links.append(
                LinkChannel(tx, rx, d["coefficients"], d["delays_s"],
                            s["geometry"], s["channel_provenance"])
            )
    want = {(e.id, r.id) for e in cfg.emitters for r in cfg.receivers}
    got = {(lk.tx_id, lk.rx_id) for lk in links}
    if want - got:
        raise RuntimeError(f"{cfg.seed}: 缺链路 {sorted(want - got)}")
    return links


# ---------------------------------------------------------------------------
# 噪声实测（零输入，不需要重放信号链）
# ---------------------------------------------------------------------------

def rx_guard_of(cfg: Scenario) -> int:
    return max(r.frontend.filter_taps // 2 for r in cfg.receivers)


def measure_noise_only(cfg: Scenario, guard: int) -> dict:
    """零输入 `receive()` → 每站实测输出噪声功率（与数据集里那一份同实现）。

    注意：噪声种子只由 `(seed, rx.id, 阵元号)` 决定，与 `composite` 无关；
    但 `start_index` 会进时间基，故此处严格镜像 runner.py 的调用方式。
    """
    out: dict[str, dict] = {}
    n_tot = cfg.num_samples + 2 * guard
    for rx in cfg.receivers:
        z = np.zeros((rx.antenna.elements, n_tot), dtype=np.complex128)
        iq, meta = receive(z, rx, cfg.sample_rate_hz, cfg.seed, start_index=-guard)
        iq = iq[:, guard:guard + cfg.num_samples]
        per_el = np.mean(np.abs(iq) ** 2, axis=1)
        w = float(np.mean(per_el))
        out[rx.id] = {
            "effective_output_noise_w": w,
            "effective_output_noise_dbm": float(10.0 * np.log10(w) + 30.0),
            "per_element_w": [float(v) for v in per_el],
            "platform_expected_output_noise_w": float(meta.get("expected_output_noise_w") or 0.0),
            "noise_expectation_scope": meta.get("noise_expectation_scope"),
        }
    return out


def _median_dbm(meas: dict) -> float:
    return float(np.median([m["effective_output_noise_dbm"] for m in meas.values()]))


def measure_thermal_only(cfg: Scenario, guard: int) -> dict:
    """零输入 + 关掉全部**加性**噪声项（保留 phase / imd3）→ 该档的“背景参照底”。

    用途：档位是按站抽签的，`noise_floor_dbm` 若取各站中位就会随档位构成漂移
    （d4 只有 20% 的站是纯热噪，中位很可能落在带噪声的站上）。故真值的
    `noise.noise_floor_dbm` 统一取**纯热噪**实测值 —— 它与加噪前的 -106.9 dBm
    逐位可比，也让下游只读该字段的脚本（eval_dataset 等）语义不变。
    """
    cfg2 = cfg.model_copy(deep=True)
    for rx in cfg2.receivers:
        rx.frontend.noise_enabled = True
        rx.frontend.colored_noise_enabled = False
        rx.frontend.impulse_noise_enabled = False
        rx.frontend.correlated_noise_enabled = False
    return measure_noise_only(cfg2, guard)


# ---------------------------------------------------------------------------
# 真值回写
# ---------------------------------------------------------------------------

def refresh_achieved(cap: Path, key: dict) -> int:
    """按平台新 truth.json 重算答案键的 `achieved`（RT 实测可见性回填）。

    与 `gen_dataset.py` 主循环里的 achieved 回填逐字同口径。重放模式下答案键是从源样本
    拷来的，其中的 achieved 属**旧波形**；带宽修好（软带限）后接收功率的采样实现变了，
    尖峰型源可差 1~5 dB，会让 `≥3 站可见` 的记录与实际数据不一致。故重放后重算。

    返回刷新的（源×站）条数。
    """
    tp = cap / "private" / "truth.json"
    if not tp.exists():
        return 0
    trt = json.loads(tp.read_text(encoding="utf-8"))
    kmap = {r["id"]: f"s{i}" for i, r in enumerate(key.get("receivers") or [])}
    ach: dict[str, dict] = {}
    n = 0
    for st in trt.get("stations") or []:
        sk = kmap.get(st["receiver_id"])
        if sk is None:
            continue
        for src in st.get("sources") or []:
            pw = float(np.mean(src["mean_port_power_w"]))
            ach.setdefault(src["emitter_id"], {})[sk] = {
                "pr_dbm": round(10 * np.log10(pw + 1e-30) + 30, 2),
                "path_count": src["path_count"],
                "path_azimuths_deg": [round(np.degrees(a), 1)
                                     for a in src["geometry"]["arrival_azimuth_rad"][:6]],
            }
            n += 1
    if ach:
        key["achieved"] = ach
    return n


def patch_truth(cap: Path, src_priv: Path, meas: dict, difficulty: str,
                applied: dict, plan: dict | None = None,
                thermal: dict | None = None) -> dict:
    """回写 RT truth 的接收机元数据 + 答案键的 noise 块与派生字段。返回摘要。

    注意：`run_scenario` 只写平台自己的产物（`private/scenario.json` / `truth.json` /
    `channel_*.npz` / `public/*`）；答案键 `private/d1_truth.json` 是基准层
    （`gen_dataset.py`）写的，重放不会重现它 —— 故先从源样本拷贝，再打补丁。

    `thermal` = 纯热噪实测（见 `measure_thermal_only`），写进 `noise.noise_floor_dbm`
    作为跨档可比的**背景参照底**；各站**实际生效**底放在 `per_station`。
    """
    out_priv = cap / "private"
    out_priv.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_priv.iterdir()):
        if p.is_file() and not (out_priv / p.name).exists():
            shutil.copy2(p, out_priv / p.name)

    nz = _median_dbm(meas)                            # 各站实际生效底的中位
    nz_th = _median_dbm(thermal) if thermal else nz    # 纯热噪参照底（写进 noise_floor_dbm）
    # --- RT truth：每站 receiver 补实测噪声 ---
    tp = out_priv / "truth.json"
    trt = json.loads(tp.read_text(encoding="utf-8"))
    for st in trt["stations"]:
        m = meas.get(st["receiver_id"])
        if not m:
            continue
        rec = st["receiver"]
        rec["effective_output_noise_w"] = m["effective_output_noise_w"]
        rec["effective_output_noise_dbm"] = m["effective_output_noise_dbm"]
        rec["noise_only_per_element_w"] = m["per_element_w"]
        rec["noise_measurement"] = "zero-input receive() with identical seed/start_index"
    tp.write_text(json.dumps(trt, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 答案键：noise 块 + INR/SNR/受害者层 ---
    kp = out_priv / "d1_truth.json"
    tk = json.loads(kp.read_text(encoding="utf-8"))
    recv_order = [r["id"] for r in tk["receivers"]]
    nz_by_station = {rid: meas[rid]["effective_output_noise_dbm"] for rid in recv_order if rid in meas}
    old_nz = float((tk.get("noise") or {}).get("noise_floor_dbm", -106.9))
    n_ach = refresh_achieved(cap, tk)

    tk["noise"] = {
        "model": nprof.model_label(difficulty),
        "nf_db": float((tk.get("noise") or {}).get("nf_db", 5.0)),
        "noise_floor_dbm": round(nz_th, 2),
        "noise_floor_definition": "thermal-only stations (zero-input receive); 跨档/跨版本可比的背景参照底",
        "station_floor_median_dbm": round(nz, 2),
        # 按站抽签：每站一行"配方 + 实际生效字段"（可逐位复现）
        "stations": {rid: {"archetype": (plan or {}).get(rid), "frontend": applied.get(rid)}
                     for rid in recv_order if rid in applied},
        "uniform_profile": (next(iter(applied.values()))
                            if applied and len({json.dumps(v, sort_keys=True) for v in applied.values()}) == 1
                            else None),
        "effective_output_noise_w": float(np.median([m["effective_output_noise_w"] for m in meas.values()])),
        "per_station": {
            rid: {
                "effective_output_noise_w": meas[rid]["effective_output_noise_w"],
                "effective_output_noise_dbm": meas[rid]["effective_output_noise_dbm"],
            }
            for rid in recv_order if rid in meas
        },
        "measurement": "zero-input receiver.receive() (same seed, same start_index/guard as runner.py)",
        "note": ("含全部已开噪声项（白/有色/脉冲/共模）；已实测，不再用解析式。"
                 "非线性(IMD3/软限幅)配置下该值是 SNR 上界，平台自标 linear receiver only。"),
    }

    for i, rid in enumerate(recv_order):
        sk = f"s{i}"
        d = nz_by_station.get(rid, nz)
        for ifw in tk.get("interferers") or []:
            ps = (ifw.get("per_station") or {}).get(sk)
            if ps and "pr_dbm" in ps:
                ps["inr_db"] = round(float(ps["pr_dbm"]) - d, 2)
        ps = ((tk.get("target") or {}).get("per_station") or {}).get(sk)
        if ps and "pr_dbm" in ps:
            ps["snr_db"] = round(float(ps["pr_dbm"]) - d, 2)

    events = []
    for ev in tk.get("interference_events") or []:
        p = float(ev.get("power_at_victim_dbm", -999.0))
        env = p - nz_th
        ev["env_over_noise_db"] = round(env, 1)
        ev["impact_class"] = "severe" if env >= 20.0 else "moderate" if env >= 10.0 else "negligible"
        events.append({"event_id": ev.get("event_id"), "env_over_noise_db": ev["env_over_noise_db"],
                       "impact_class": ev["impact_class"]})
    kp.write_text(json.dumps(tk, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "noise_floor_dbm": round(nz_th, 2),
        "station_floor_median_dbm": round(nz, 2),
        "old_noise_floor_dbm": old_nz,
        "rise_db": round(nz - old_nz, 2),
        "per_station_dbm": {k: round(v, 2) for k, v in nz_by_station.items()},
        "events": events,
        "achieved_refreshed": n_ach,
    }


def refresh_hashes(root: Path, note: dict) -> None:
    """重算 run_summary.json 的 hashes（模板驱动，保持 runner 的 key 格式）。"""
    sp = root / "run_summary.json"
    if not sp.exists():
        return
    summ = json.loads(sp.read_text(encoding="utf-8"))
    old = summ.get("hashes") or {}
    new = {}
    for k in old:
        p = root / k
        if p.is_file():
            new[k] = digest(p)
    # 目录里出现、但 old 里没有的文件（正常不应发生）
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "run_summary.json":
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel not in new and rel not in old:
                new[rel] = digest(p)
    summ["hashes"] = new
    summ["noise_patch"] = note
    sp.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")


def _range_of(cells, key: str):
    """从 per_station 字典序列里取某字段的 [min, max]（两位小数），无值返回 None。"""
    vals = [c[key] for c in cells if isinstance(c, dict) and c.get(key) is not None]
    return [round(min(vals), 2), round(max(vals), 2)] if vals else None


def patch_index(src: Path, out: Path, per_sample: dict) -> None:
    """复制源 index.json，补上每档档位名，并**按重算后的真值刷新每样本噪声底与 INR/SNR 区间**。

    噪声底一改，`d1_truth.json` 里的 `inr_db` / `snr_db` 就跟着改（regen_noise 里重算了），
    index 里的 `inr_db_range` / `snr_db_range` 若不跟着重算，**索引与真值会打架**。
    """
    ip = src / "index.json"
    if not ip.exists():
        return
    idx = json.loads(ip.read_text(encoding="utf-8"))
    idx["noise_profiles"] = {k: nprof.model_label(k) for k in nprof.DIFFICULTIES}
    idx["noise_menus"] = {k: [{"archetype": n, "weight": w / nprof.B_SLOTS} for n, w in nprof.MENUS[k]]
                          for k in nprof.DIFFICULTIES}
    for s in idx.get("samples") or []:
        m = per_sample.get(s["id"])
        if m:
            s["noise_profile"] = m["profile_name"]
            s["noise_floor_dbm"] = m["noise_floor_dbm"]
        tp = out / s["id"] / "capture" / "private" / "d1_truth.json"
        if not tp.exists():
            continue
        t = json.loads(tp.read_text(encoding="utf-8"))
        nz = t.get("noise") or {}
        s["noise_profile"] = nz.get("model", s.get("noise_profile"))
        s["noise_floor_dbm"] = nz.get("noise_floor_dbm", s.get("noise_floor_dbm"))
        tg = t.get("target") or {}
        stg = s.get("target") or {}
        rng = _range_of((tg.get("per_station") or {}).values(), "snr_db")
        if rng and isinstance(stg.get("snr_db_range"), list):
            stg["snr_db_range"] = rng
        ifw_truth = t.get("interferers") or []
        for j, cell in enumerate(s.get("interferers") or []):
            if j >= len(ifw_truth):
                break
            rng = _range_of((ifw_truth[j].get("per_station") or {}).values(), "inr_db")
            if rng and isinstance(cell.get("inr_db_range"), list):
                cell["inr_db_range"] = rng
    (out / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 仪器口径审计（工具峰 vs GT）
# ---------------------------------------------------------------------------

def audit_capture(cap: Path, sid: str, cache_root: Path) -> dict:
    """跑 `capture_tools` 的残差峰判据，逐源对 GT，数真峰与假簇。"""
    import capture_tools as ct
    import run_capture_agent as rca

    pub = cap / "public"
    priv = cap / "private"
    obs = json.loads((pub / "observations.json").read_text(encoding="utf-8"))
    n = int(obs["receivers"][0]["shape"][1])
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    tk = json.loads((priv / "d1_truth.json").read_text(encoding="utf-8"))
    try:
        refs = rca._build_target_reference(priv, n)
    except Exception:                                     # noqa: BLE001
        refs = {}
    ctx = ct.ToolContext(
        capture_dir=pub,
        cache_dir=cache_root / sid,
        allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
        fs=fs,
        fc=float(obs["receivers"][0]["center_frequency_hz"]),
        target_bw_normalized=0.25,
        aoi_center=[0, 0],
        target_reference=refs,
    )
    ct.set_context(ctx)

    gts = tk.get("interferers") or []
    rows = []
    stations = [r["receiver_id"] for r in obs["receivers"]]
    for st in stations:
        occ = ct.estimate_band_occupancy_by_correlation(st)
        peaks = occ.get("residual_peaks") or []
        hit = [False] * len(gts)
        for p in peaks:
            pf = float(p.get("freq_offset_hz", 0.0))
            pw = max(float(p.get("bandwidth_hz", 0.0)), 60e3)
            best, bd = None, 1e18
            for j, g in enumerate(gts):
                d = abs(pf - float(g.get("freq_offset_hz", 0.0)))
                if d < bd:
                    best, bd = j, d
            true = best is not None and bd <= max(pw / 2.0, 100e3)
            if true:
                hit[best] = True
            rows.append({
                "station": st,
                "peak_hz": round(pf),
                "over_floor_db": round(float(p.get("over_floor_db", 0.0)), 2),
                "bandwidth_hz": round(float(p.get("bandwidth_hz", 0.0)), 0),
                "coherence": round(float(p.get("coherence", float("nan"))), 3),
                "match": (gts[best].get("kind") if true else None),
                "gt_hz": (round(float(gts[best].get("freq_offset_hz", 0.0)), 0) if true else None),
                "df_hz": (round(bd) if true else None),
            })
    return {
        "sample": sid,
        "n_gt": len(gts),
        "gt": [{"kind": g.get("kind"), "freq_offset_hz": round(float(g.get("freq_offset_hz", 0.0)), 0),
                "bandwidth_hz": round(float(g.get("bandwidth_hz", 0.0)), 0)} for g in gts],
        "peaks": rows,
        "n_peaks_true": sum(1 for r in rows if r["match"]),
        "n_peaks_false": sum(1 for r in rows if not r["match"]),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def sample_ids(src: Path, limit: int | None, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    idx = json.loads((src / "index.json").read_text(encoding="utf-8"))
    ids = [s["id"] for s in idx["samples"]]
    return ids[:limit] if limit else ids


def _parse_override(text: str | None) -> dict | None:
    """PowerShell 会把 JSON 的双引号剥掉，故兼容 Python 字面量写法 {'a': 1}。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源数据集目录（如 ../dataset_d3）")
    ap.add_argument("--out", default=None, help="输出数据集目录（重放模式必填）")
    ap.add_argument("--difficulty", default=None, help="档位名 d1/d2/d3/d4；默认取 index.json 的 difficulty")
    ap.add_argument("--samples", default=None, help="逗号分隔的样本 id；默认按 index 顺序")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--override", default=None, help="JSON，逐字段覆盖档位（扫描用）")
    ap.add_argument("--measure-only", action="store_true", help="只做零输入噪声实测，不重放不写盘")
    ap.add_argument("--audit", action="store_true", help="重放后跑 capture_tools 峰判据审计")
    ap.add_argument("--report", default=None, help="把结果 JSON 写到该路径")
    ap.add_argument("--index-only", action="store_true", help="不重放，只按已重写的真值刷新 out/index.json")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve() if args.out else None
    difficulty_arg = (args.difficulty or "").lower() or None
    if difficulty_arg and difficulty_arg not in nprof.DIFFICULTIES:
        raise SystemExit(f"未知难度档 {difficulty_arg!r}（可选 {list(nprof.DIFFICULTIES)}）")
    override = _parse_override(args.override)
    ids = sample_ids(src, args.limit, args.samples.split(",") if args.samples else None)

    print(f"[src] {src}")
    print(f"[override] {override}")
    print(nprof.summary_line())
    if args.index_only:
        if out is None:
            raise SystemExit("--index-only 需要 --out")
        patch_index(src, out, {})
        print(f"[index-only] {out / 'index.json'}")
        return 0
    if out:
        out.mkdir(parents=True, exist_ok=True)

    results = []
    per_sample_index = {}
    t_all = time.time()
    for sidx, sid in enumerate(ids):
        try:
            cap, priv, truth_rt, truth_key, cfg = load_sample(src, sid)
        except Exception as exc:                          # noqa: BLE001
            print(f"  !! {sid}: 读取失败 {exc!r}")
            continue

        diff = difficulty_arg or sid.split("_")[0].lower()
        if diff not in nprof.DIFFICULTIES:
            print(f"  !! {sid}: 未知难度档 {diff!r}")
            continue
        # 按站抽签：--override 时全站同一套（探针路径），否则用档位菜单
        stations = [rx.id for rx in cfg.receivers]
        if override:
            plan = {rid: "override" for rid in stations}
            applied = {}
            for rx in cfg.receivers:
                applied[rx.id] = nprof.apply_noise_profile(rx, diff, override)
        else:
            plan = nprof.plan_sample(diff, stations, sidx)
            applied = {}
            for rx in cfg.receivers:
                applied[rx.id] = nprof.apply_station(rx, diff, plan[rx.id])
        guard = rx_guard_of(cfg)
        meas = measure_noise_only(cfg, guard)
        thermal = measure_thermal_only(cfg, guard)
        nz = _median_dbm(meas)
        nz_th = _median_dbm(thermal)
        old_nz = float((truth_key.get("noise") or {}).get("noise_floor_dbm", -106.9))

        # 有效 INR 核算：每个 GT 干扰在每个可见站还剩多少 dB
        recv_order = [r["id"] for r in truth_key["receivers"]]
        inrs = []
        for i, rid in enumerate(recv_order):
            sk = f"s{i}"
            d = meas.get(rid, {}).get("effective_output_noise_dbm", nz)
            for g in truth_key.get("interferers") or []:
                ps = (g.get("per_station") or {}).get(sk) or {}
                if "pr_dbm" in ps:
                    inrs.append(round(float(ps["pr_dbm"]) - d, 2))

        rec = {
            "id": sid,
            "difficulty": diff,
            "profile_name": nprof.model_label(diff),
            "menu": {rid: plan.get(rid) for rid in recv_order},
            "applied": {rid: applied[rid] for rid in applied},
            "noise_floor_dbm": round(nz_th, 2),
            "station_floor_median_dbm": round(nz, 2),
            "old_noise_floor_dbm": old_nz,
            "rise_db": round(nz - old_nz, 2),
            "platform_expected_dbm": round(
                float(10 * np.log10(np.median([m["platform_expected_output_noise_w"] for m in meas.values()])) + 30), 2),
            "inr_db": sorted([v for v in inrs if v > -200]),
        }
        per_sample_index[sid] = rec

        tag = "MEASURE" if args.measure_only else "REPLAY"
        print(f"  [{tag}] {sid}  实测底 {nz:8.2f} dBm (旧 {old_nz:8.2f}, 抬 {nz - old_nz:+6.2f} dB)"
              f"  平台解析 {rec['platform_expected_dbm']:8.2f}")
        if inrs:
            good = [v for v in inrs if v > -200]
            if good:
                below8 = sum(1 for v in good if v < 8.0)
                print(f"          有效 INR 中位 {float(np.median(good)):6.2f} dB  "
                      f"min {min(good):7.2f}  max {max(good):7.2f}  <8dB 的 {below8}/{len(good)}")

        if args.measure_only:
            results.append(rec)
            continue

        if out is None:
            raise SystemExit("重放模式需要 --out")
        out_cap = out / sid / "capture"
        if out_cap.exists():
            shutil.rmtree(out_cap)
        out_cap.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        run_scenario(cfg, out_cap, frozen_links=build_links(priv, truth_rt, cfg))
        dt = time.time() - t0
        summary = patch_truth(out_cap, priv, meas, diff, applied, plan, thermal)
        rec["write"] = summary
        rec["replay_s"] = round(dt, 2)
        refresh_hashes(out_cap, {"profile": rec["profile_name"], "menu": dict(rec["menu"]),
                                 "noise_floor_dbm": summary["noise_floor_dbm"],
                                 "measured_by": "regen_noise.py"})
        menu_txt = " ".join(f"{rid.split('_')[1]}={plan[rid][:4]}" for rid in sorted(plan))
        print(f"          重放 {dt:5.2f}s  背景底 {summary['noise_floor_dbm']:8.2f}"
              f"  各站中位 {summary['station_floor_median_dbm']:8.2f} (抬 {summary['rise_db']:+5.2f} dB)"
              f"  菜单 {menu_txt}  事件 {summary['events']}")
        if args.audit:
            aud = audit_capture(out_cap, sid, out / "_agent_cache")
            rec["audit"] = aud
            print(f"          审计: GT {aud['n_gt']} 个 | 工具峰 真 {aud['n_peaks_true']} / 假 {aud['n_peaks_false']}")
            for r in aud["peaks"]:
                mark = r["match"] or "FALSE"
                print(f"            {r['station']:>10s} {r['peak_hz']/1e3:9.1f}k "
                      f"over {r['over_floor_db']:6.2f}dB bw {r['bandwidth_hz']/1e3:7.1f}k "
                      f"coh {r['coherence']:.3f}  <- {mark}")
        results.append(rec)

    if out and not args.measure_only and per_sample_index:
        patch_index(src, out, per_sample_index)
        # 平台 replay 会整体重写 public/observations.json（把公开先验冲空），所以重放后
        # 必须重做公开层：受害者层投影 + cooperative_sources 公开先验 + run_summary 摘要。
        # 机理粒度写死 full；之后可用 `expose_evidence.py --mechanism coarse|none` 原地改。
        import expose_evidence as ee
        est = ee.process(out, "full", False, True)
        print(f"[public] 受害者层/公开先验：事件 {est['events']} 条  恢复 {est['coop_restored']} 样本"
              f"  摘要同步 {est['hashes_refreshed']}  隔离问题 {len(est['problems'])} 条")
    print(f"\n[done] {len(results)} 样本  总耗时 {time.time() - t_all:.1f}s")
    if args.report:
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
