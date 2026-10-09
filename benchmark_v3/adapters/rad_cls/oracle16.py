"""三级 oracle 验收（A 路线 16 辐射体分类器，预注册门槛勿改）：

- 第 1 级 clean：独立种子 16 类纯波形（含残差分布增强），top-1 ≥ 0.85
- 第 2 级 GT 切片：真实 D2 残差按 GT 频偏/带宽切片，top-1 ≥ 0.50
- 第 3 级 工具视角：完整剥离残差链路（on_residual 谱峰定位），top-1 ≥ 0.50
任一不过 => 不接入工具箱，回到训练调参。

用法：
    python oracle16.py
输出每级准确率 + 混淆矩阵。零 token（纯本地）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ADAPTERS = _HERE.parent
for _p in (str(_HERE), str(_ADAPTERS), str(_HERE.parent.parent / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen16 import CLASSES16, CLS_IDX, gen_wave16, simulate_residual, to_input  # noqa: E402
from model16 import RadCNN  # noqa: E402

GATE_CLEAN = 0.85
GATE_SLICE = 0.50
GATE_TOOLVIEW = 0.50

D2_ROOT = Path(os.environ.get("DSH_D2_ROOT") or (_HERE.parent.parent / "dataset_d2")).resolve()
# DSH_D2_ROOT 覆写入口（2026-10-10）：用于“加噪前 vs 加噪后”同代码 A/B。
# 不改任何门限，只换数据根目录（其下需有 index.json 与 <sid>/capture/）。


def load_model():
    ck = torch.load(_HERE / "rad_classifier" / "checkpoint.pt",
                    map_location="cpu", weights_only=False)
    model = RadCNN()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["classes"]


def _pred(model, batch_x):
    with torch.no_grad():
        xb = torch.from_numpy(np.stack([to_input(b) for b in batch_x]))
        return model(xb).argmax(dim=1).cpu().numpy()


def gate_clean(model, n_per_class=60, seed_base=700000):
    """独立种子波形 + 训练同链路增强（beamformed+滤波+切片）。

    2026-10-02 口径统一：clean 门限测"模型对训练分布的泛化（非背诵）"，
    增强链路必须与训练完全一致（simulate_beamformed_train），仅种子独立。
    原 simulate_residual（无 LS/兄弟/同滤波口径）与训练链路分布不一致，
    曾把 clean 压到 0.32 造成误报。"""
    from data_gen16 import simulate_beamformed_train
    correct = total = 0
    conf = np.zeros((len(CLASSES16), len(CLASSES16)))
    pool, pool_y = [], []
    for ci, name in enumerate(CLASSES16):
        j = 0
        while j < n_per_class:
            rng = np.random.default_rng(seed_base + ci * 1009 + j)
            try:
                x = gen_wave16(name, rng).astype(np.complex128)
            except RuntimeError:
                continue
            pool.append(x)
            pool_y.append(ci)
            j += 1
    pool = np.stack(pool)
    pool_y = np.array(pool_y)
    for ci in range(len(CLASSES16)):
        idxs = [i for i, y in enumerate(pool_y) if y == ci]
        for i in idxs:
            rng = np.random.default_rng(seed_base + 50000 + i)
            sim = simulate_beamformed_train(pool[i], ci, pool, pool_y, rng)
            pred = _pred(model, [sim])[0]
            conf[ci, pred] += 1
            correct += int(pred == ci)
            total += 1
    acc = correct / total
    print(f"  clean (独立种子, 训练同链路): {acc:.3f}  [{correct}/{total}]  gate={GATE_CLEAN}")
    return acc, conf


def gate_gt_slice(model):
    """真实 D2 残差，按 GT 位置构造全部干扰源导向矩阵 → 多源 LS 提取 → 分类。

    口径 = 评测链路的"提取后波形"（与训练分布对齐）；空间聚类定位误差另计
    （此处用 GT 位置，是提取正确性上限评测）。"""
    correct = total = 0
    idx = json.loads((D2_ROOT / "index.json").read_text(encoding="utf-8"))
    conf = np.zeros((len(CLASSES16), len(CLASSES16)))
    C = 299792458.0
    for s in idx["samples"]:
        cap = D2_ROOT / s["id"] / "capture"
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        resid_p = D2_ROOT / "_agent_cache" / s["id"] / "resid_station_a.npy"
        resid = None
        if resid_p.exists():
            resid = np.load(resid_p).astype(np.complex128)
            if np.asarray(resid).ndim != 2:          # 2026-10-09：历史降级路径写过
                print(                                 # 1-D 文件（d2_00043），读取端
                    f"    [gt-slice] {s['id']} 残差缓存形状异常 "  # 必须自愈而不是崩掉
                    f"{np.asarray(resid).shape}，现场重建")
                resid = None
        if resid is None:
            # 2026-10-08（审查者抓出的可复现缺口）：残差缓存在**清缓存后会消失**，
            # 旧行为直接 `continue` → 全部样本被跳过、分母 0/0、指标 0.000 却仍打印
            # gate=0.5 判 FAIL（静默跳过小样本会虚高，全跳过则变成无意义的 0/0）。
            # 改为**现场重建**：与 gate_tool_view 同口径建 ctx 并直接调目标剥离，
            # 使 gate 自给自足（不再依赖上一次工具运行的产物）。
            # 2026-10-10 修：原先进的是 `estimate_band_occupancy_by_correlation`，它**不写**
            # `resid_station_a.npy`（写盘的是 `_strip_target_at_station`）→ 清缓存后仍报
            # `FileNotFoundError`。改为直接调剥离函数并用它的返回值。
            try:
                import capture_tools as ct
                import run_capture_agent as rca
                refs = rca._build_target_reference(
                    cap / "private", obs["receivers"][0]["shape"][1])
                ctx = ct.ToolContext(
                    capture_dir=cap / "public",
                    cache_dir=D2_ROOT / "_agent_cache" / s["id"],
                    allowed_stations=[r["receiver_id"] for r in obs["receivers"]],
                    fs=obs["receivers"][0]["sample_rate_hz"],
                    fc=obs["receivers"][0]["center_frequency_hz"],
                    target_bw_normalized=0.25,
                    aoi_center=[0, 0], target_reference=refs)
                ct.set_context(ctx)
                resid = np.asarray(ct._strip_target_at_station("station_a"), np.complex128)
            except Exception as exc:                          # noqa: BLE001
                print(f"    [gt-slice] {s['id']} 残差重建失败，跳过：{str(exc)[:60]}")
                continue
        fs = obs["receivers"][0]["sample_rate_hz"]
        rec = next(r for r in obs["receivers"] if r["receiver_id"] == "station_a")
        off = np.asarray(rec["element_offsets_world_m"], float)
        gts = [g for g in truth["interferers"] if g["kind"] in CLS_IDX]
        if len(gts) < 1:
            continue
        # 可见性筛查（与 eval_dataset 的 interferer_visible_at 口径一致）：
        # station_a 收不到（0 径/INR<3dB）的干扰物理上无信息，不计入门限——
        # 用 truth.json 的 runner 实测端口功率判定。
        full = json.loads((cap / "private" / "truth.json").read_text(encoding="utf-8"))
        st_full = next(s2 for s2 in full["stations"] if s2["receiver_id"] == "station_a")
        # 2026-10-10：真值新增"实测噪声"字段 `effective_output_noise_w`（噪声先于 IMD3/软限幅，
        # 平台自标的 `expected_output_noise_w` 只含白热噪）。优先用实测值，旧数据集回退旧字段。
        _rx = st_full["receiver"]
        noise_dbm = 10 * np.log10(_rx.get("effective_output_noise_w") or _rx["expected_output_noise_w"]) + 30
        pw_dbm = {s2["emitter_id"]: 10 * np.log10(float(np.mean(s2["mean_port_power_w"])) + 1e-30) + 30
                  for s2 in st_full["sources"]}
        visible = []
        for k_i, g in enumerate(gts):
            # d1_truth 干扰序号 → truth.json 的 emitter_id（interferer_N 顺序一致）
            eid = f"interferer_{truth['interferers'].index(g) + 1}"
            if pw_dbm.get(eid, -999) - noise_dbm >= 3.0:
                visible.append(g)
        gts = visible
        if len(gts) < 1:
            continue
        if np.asarray(resid).ndim != 2:
            print(f"    [gt-slice] {s['id']} 残差不可用，跳过")
            continue
        n = resid.shape[1]
        # 2026-10-02：残差缓存已带限 ±500k，**不再做二次窄带 brick 切**——
        # 通带外为 0 的切片口径 CNN 从未见过（训练分布通带外恒有噪声），
        # 二次带限曾把 GT-slice 从 0.71 打到 0.11；全带 LS 提取 + 变频即正确口径。
        X = resid
        A = np.zeros((8, len(gts)), complex)
        for k, g in enumerate(gts):
            u_src = np.asarray(g["position_m"], float) - np.asarray(rec["receiver_position_m"], float)
            u_src = u_src / (np.linalg.norm(u_src) + 1e-12)
            A[:, k] = np.exp(2j * np.pi * 2.4e9 / C * (off @ u_src))
        cond = float(np.linalg.cond(A))
        if not np.isfinite(cond) or cond > 1e6:
            continue
        S = np.linalg.pinv(A) @ X                     # (K, n) 每源 LS 提取波形
        t = np.arange(n)
        for k, g in enumerate(gts):
            # 变频到 0 中心（与训练分布一致：CNN 输入以源载频为中心）
            w = S[k] * np.exp(-1j * 2 * np.pi * g["freq_offset_hz"] * t / fs)
            w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)
            pred = _pred(model, [w.astype(np.complex64)])[0]
            conf[CLS_IDX[g["kind"]], pred] += 1
            correct += int(pred == CLS_IDX[g["kind"]])
            total += 1
    if total == 0:
        # 2026-10-08：分母为 0 时不再静默返回 0.000（旧行为会让“残差缓存缺失/
        # 可见性筛查把所有样本滤掉”伪装成“gate FAIL”——0/0 不是指标）。
        print("  !! GT切片 gate 分母为 0：无任何样本进入评测（残差重建失败或可见性"
              "筛查全滤）。这不是 0.000 的准确率，是无效运行——请先查残差链路。")
        return 0.0, conf
    acc = correct / total
    print(f"  GT切片 (GT位置LS提取后分类): {acc:.3f}  [{correct}/{total}]  gate={GATE_SLICE}")
    return acc, conf


def print_conf(conf):
    print("  混淆矩阵 (GT行 × pred列, 顶行/左列=类名):")
    names = CLASSES16
    print("      " + " ".join(f"{n[:5]:>6}" for n in names))
    for i, n in enumerate(names):
        row = " ".join(f"{int(conf[i, j]):>6}" for j in range(len(names)))
        print(f"  {n[:12]:<12} {row}")


def gate_tool_view(model, n_samples=50):
    """第 3 级 tool-view：完整链路（目标剥离→空间聚类→波束成形提取→CNN 分类）。

    2026-10-03 补齐（此前 GATE_TOOLVIEW 只是常量、无 harness——不可复现）。
    口径与 _diag_metrics 一致，固定双报：
      - per-source 召回   = 被认领的真源数 / 任一 allowed 站可见干扰总数（INR≥3dB，
        与 eval_dataset::interferer_visible_at 同口径——**不是只看 station_a**，
        只看单站会把指标高估 ~9pp）
      - 假簇率            = 未命中真源的簇 / 输出簇总数
      - 条件分类准确率    = 认领真源的簇中 CNN top-1 正确的比例（**读工具自己
        落盘的 bfw_{station_a}_{k}.npy 缓存**，不在 gate 里另建 LS 提取——
        那样会绕开工具侧的 cond 筛查/水平化导向/能量口径）
      - 端到端 per-interferer = 召回 × 条件准确率
    召回分子对 GT 去重：每个真源至多被认领一次（50m 内簇按距离贪心最近匹配），
    否则同源多簇会重复计数、召回可 >1。
    判定：条件准确率与召回均 ≥ GATE_TOOLVIEW（端到端 ≥ 0.25）。
    无 station_a 目标参考的样本仍计入分母（标记 skip——静默踢掉难样本会虚高）。
    """
    import capture_tools as ct
    import run_capture_agent as rca

    D2 = D2_ROOT          # 2026-10-10 修：原先这里另写了一份硬编码 dataset_d2 路径，
    # 导致模块级 D2_ROOT / DSH_D2_ROOT 覆写只对 gate_gt_slice 生效、
    # 三级 gate 可能跑在不同数据集上（且 A/B 无法隔离 tool-view）。
    # 三级 gate 必须共用唯一数据根。
    idx = json.loads((D2 / "index.json").read_text(encoding="utf-8"))
    C_LIGHT = 299792458.0
    recall_n = recall_d = 0
    ans_n = ans_d = 0
    n_clusters = 0
    cond_c = cond_t = 0
    n_cand = 0
    n_src = 0
    src_recall_n = 0
    per_rows = []
    fs_s = None
    for s in idx["samples"][:n_samples]:
        sid = s["id"]
        cap = D2 / sid / "capture"
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        private = cap / "private"
        ns = obs["receivers"][0]["shape"][1]
        refs = rca._build_target_reference(private, ns)
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ctx = ct.ToolContext(capture_dir=cap / "public",
                             cache_dir=D2 / "_agent_cache" / sid,
                             allowed_stations=allowed,
                             fs=obs["receivers"][0]["sample_rate_hz"],
                             fc=obs["receivers"][0]["center_frequency_hz"],
                             target_bw_normalized=0.25, aoi_center=aoi,
                             target_reference=refs)
        ct.set_context(ctx)
        fs_s = obs["receivers"][0]["sample_rate_hz"]
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        full = json.loads((cap / "private" / "truth.json").read_text(encoding="utf-8"))
        # 可见性：任一 allowed 站可见（INR≥3dB）即计入分母——与
        # eval_dataset::interferer_visible_at 同口径（遍历 stations）。
        vis_count = np.zeros(len(truth["interferers"]), int)
        for x in full["stations"]:
            if x["receiver_id"] not in allowed:
                continue
            _rx = x["receiver"]
            nz = 10 * np.log10(_rx.get("effective_output_noise_w") or _rx["expected_output_noise_w"]) + 30
            for i_g in range(len(truth["interferers"])):
                eid = "interferer_%d" % (i_g + 1)
                src_x = next((y for y in x["sources"] if y["emitter_id"] == eid), None)
                if src_x is None:
                    continue
                pr = 10 * np.log10(float(np.mean(src_x["mean_port_power_w"])) + 1e-30) + 30
                if pr - nz >= 3.0:
                    vis_count[i_g] += 1
        # vis_gts 元素 = (干扰, 可见站数)——answerable 子集（≥3 站可见）是
        # trio 类方法的硬上限（2026-10-03 天花板分解：≥3 站可见仅 47%）
        vis_gts = [(g, int(vis_count[i_g]))
                   for i_g, g in enumerate(truth["interferers"])
                   if vis_count[i_g] >= 1 and g["kind"] in CLS_IDX]
        # 2026-10-03 P0：主指标测 **candidates**（工具原始输出=稳定的能力刻度，
        # 不绑手调阈值）。先直接调聚类拿双栏；extract_source_waveforms 二次
        # 调用时聚类走 _result_cache，不重复计算。
        try:
            clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
            out = ct.extract_source_waveforms()
        except Exception as exc:
            per_rows.append((sid, "EXC:" + str(exc)[:30], 0, len(vis_gts),
                             sum(1 for g, ns in vis_gts if ns >= 3)))
            continue
        if "error" in clu or "error" in out:
            per_rows.append((sid, "无簇", 0, len(vis_gts),
                             sum(1 for g, ns in vis_gts if ns >= 3)))
            continue
        # 分类读工具自己落盘的 bfw_{station_a}_{k}.npy（真"完整链路"口径）。
        cand_list = clu.get("candidates") or []
        n_cand += len(cand_list)
        rec = next(r for r in obs["receivers"] if r["receiver_id"] == "station_a")
        claimed = {}                       # gt_idx -> (簇idx, 距离) 贪心最近认领
        pair = []
        for k, sp in enumerate(cand_list):
            p2 = np.asarray(sp["position_xy_m"])
            for i_g, (g, _ns) in enumerate(vis_gts):
                d = float(np.linalg.norm(p2 - np.asarray(g["position_m"][:2])))
                if d < 50:
                    pair.append((d, k, i_g))
        pair.sort()
        used_k, used_g = set(), set()
        for d, k, i_g in pair:
            if k in used_k or i_g in used_g:
                continue
            claimed[i_g] = (k, d)
            used_k.add(k); used_g.add(i_g)
            if vis_gts[i_g][1] >= 3:
                ans_n += 1
        n_clusters += len(cand_list)
        recall_n += len(claimed)
        # 次要栏：sources（过支持度门槛）的召回/精度
        pair_g = []
        for k, sp in enumerate(out["sources"]):
            p2 = np.asarray(sp["position_xy_m"])
            for i_g, (g, _ns) in enumerate(vis_gts):
                d = float(np.linalg.norm(p2 - np.asarray(g["position_m"][:2])))
                if d < 50:
                    pair_g.append((d, k, i_g))
        pair_g.sort()
        used_kg, used_gg = set(), set()
        n_src += len(out["sources"])
        for d, k, i_g in pair_g:
            if k in used_kg or i_g in used_gg:
                continue
            used_kg.add(k); used_gg.add(i_g)
            src_recall_n += 1
        for i_g, (k, _d) in claimed.items():
            wf = D2 / "_agent_cache" / sid / ("bfw_station_a_%d.npy" % k)
            if not wf.exists():
                continue
            # bfw 缓存已由工具落盘为 CNN-ready 口径（变频到候选带中心+RMS 归一）
            wk = np.load(wf).astype(np.complex64)
            pred = _pred(model, [wk])[0]
            cond_c += int(pred == CLS_IDX[vis_gts[i_g][0]["kind"]])
            cond_t += 1
        # review checklist（2026-10-03 第 4 次同形状 bug 后立规）：任何新增计数/
        # 分母一律写进 per_rows、循环外求和——不允许在带 continue 的主循环里累加。
        ans_d_sample = sum(1 for g, ns in vis_gts if ns >= 3)
        per_rows.append((sid, "簇%d 认领%d" % (len(cand_list), len(claimed)),
                         len(claimed), len(vis_gts), ans_d_sample))
    # 统计在循环外做（主循环只收集 per_rows）——"带 continue 的主循环里累加
    # 计数"这一类 bug（连续几轮踩坑：分母 166 vs 124）在结构上不再可能。
    recall_d = sum(r[3] for r in per_rows)
    recall_n = sum(r[2] for r in per_rows)
    ans_d = sum(r[4] for r in per_rows)
    # EXC 拦截（2026-10-03）：工具级异常被 per-sample except 吞成"无簇"时，
    # 会长得像一个模型结果——EXC 率超 5% 直接 abort，不给虚假基线。
    n_exc = sum(1 for r in per_rows if str(r[1]).startswith(("EXC:", "ERR:")))
    if n_exc > 0.05 * len(per_rows):
        print(f"  !! 工具异常样本 {n_exc}/{len(per_rows)} 超过 5% —— gate 中止，"
              "先修工具再谈指标")
        return False, {"recall": None, "far": None, "cond_acc": None,
                       "e2e": None, "per_rows": per_rows, "aborted": True}
    recall = recall_n / max(recall_d, 1)
    far = 1 - recall_n / max(n_clusters, 1)
    cond = cond_c / max(cond_t, 1)
    e2e = recall * cond
    print(f"  tool-view (完整链路, 前{n_samples}样本, 主指标=candidates):")
    print(f"    per-source 召回 : {recall:.3f}  [{recall_n}/{recall_d}]")
    print(f"    假簇率          : {far:.3f}  [{n_clusters - recall_n}/{n_clusters} 簇]")
    print(f"    条件分类准确率  : {cond:.3f}  [{cond_c}/{cond_t}]")
    print(f"    端到端/干扰     : {e2e:.3f}  (gate: 召回≥{GATE_TOOLVIEW} 且 条件≥{GATE_TOOLVIEW})")
    ans_recall = ans_n / max(ans_d, 1)
    print(f"    [可解子集召回   : {ans_recall:.3f}  [{ans_n}/{ans_d}] —— 分母=≥3站可见"
          f"（trio 硬上限），与全可见口径并列报")
    src_recall = src_recall_n / max(recall_d, 1)
    print(f"    [次要] sources 门槛栏: 召回 {src_recall:.3f}  簇数 {n_src}")
    ok = recall >= GATE_TOOLVIEW and cond >= GATE_TOOLVIEW
    return ok, {"recall": recall, "far": far, "cond_acc": cond, "e2e": e2e,
                "src_recall": src_recall, "per_rows": per_rows}


def main():
    model, _ = load_model()
    print(f"== 16 辐射体分类器 oracle（gate: clean{GATE_CLEAN}/slice{GATE_SLICE}/toolview{GATE_TOOLVIEW}）==")
    a1, c1 = gate_clean(model)
    a2, c2 = gate_gt_slice(model)
    ok12 = [a1 >= GATE_CLEAN, a2 >= GATE_SLICE]
    print(f"== 判定: clean {'PASS' if ok12[0] else 'FAIL'} | GT切片 {'PASS' if ok12[1] else 'FAIL'}")
    print("== clean 混淆矩阵 =="); print_conf(c1)
    print("== GT切片 混淆矩阵 =="); print_conf(c2)
    ok3, tv = gate_tool_view(model)
    print(f"== 判定: tool-view {'PASS' if ok3 else 'FAIL'}")
    if not all(ok12) or not ok3:
        print("未达标：不接入工具箱，回到训练调参（增加变体/epoch/容量）")


if __name__ == "__main__":
    main()
