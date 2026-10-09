"""D1–D4 数据集评分器：难度 × 观测条件(S0/S1/S2) 准确率矩阵。

任务与口径（沿 v2/v3 约定；任务层 T1 感知 / T2 认知 / T3 决策）：
    检测   有无干扰（该 S 级子集内 INR≥3 dB 视为可检；GT 侧同口径）—— T1
           注意：D3/D4 全部样本都含干扰（无空场景负类），检测只考数量匹配
    体制   干扰体制分类（D1: tone/noise 体制级；D2–D4: 16 辐射体名 = 溯源）—— T2
    频偏   |Δ|≤0.02（归一化）命中率 + MAE —— T1
    DOA    仅 S0（单站视角）：|Δ|≤10° 命中率 + MAE —— T1
    定位   仅 S1/S2（多站）：50 m 门限命中率 + MAE —— T1
    意图   敌我意图 malicious/incidental/benign（贪心匹配对上）—— T3
    欺骗   is_spoof 专项：GT 侧 = (kind=="GNSS_spoof")，仅 D4 有正分母 —— T3

意图 GT 口径：真值 `intent` 字段优先；缺省时按 kind 查 INTENT 映射（16 辐射体
确定性映射）。D1 free 模式的 tone/noise 不在映射内 → 不计入意图轴分母。
匹配规则与容差为已注册口径（贪心频偏匹配、FREQ_TOL=0.02 等），不随数据集变化。

答案文件（每样本一个 JSON，放 --answers 目录；字段全给，评分端按 S 级取用）：
    {"num_interferers": int,
     "interferers": [{"freq_offset_hz": f, "doa_deg": d,      # S0 评 DOA
                      "position_m": [x, y],                    # S1/S2 评定位
                      "kind": "...",                           # T2 体制/溯源
                      "intent": "malicious|incidental|benign", # T3 意图（可选）
                      "is_spoof": bool}, ...]}                 # T3 欺骗（可选）

可选元字段：abstained(bool) —— 模型写了负哨兵（= 拒答，prompt 明令禁止）时由
runner 置真。评测端额外报 abstain_n / abstain_rate / detection_hit_excl_abstain，
但 **detection_hit 仍是含拒答的注册口径**；旧答案目录没有该字段时，靠同目录
_run_log.json 的 raw_output 反推（见 _load_abstain_ids）。

缺省 --answers：从 GT 构造"该 S 级可见性下的理想答案"回放，验证评分管道（应全 1.0，
含意图/欺骗轴）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_V3SIM = Path(__file__).resolve().parent.parent / "simulation"
if str(_V3SIM) not in sys.path:
    sys.path.insert(0, str(_V3SIM))

from em_signal_simulator_v3.emitters import EMITTERS, INTENT  # noqa: E402

FREQ_TOL = 0.02          # 归一化容差（v2 口径）
DOA_TOL_DEG = 10.0
POS_GATE_M = 50.0        # 平台定位门限
INR_VIS_DB = 3.0         # 可检测性门限（v1-5 口径）

SPOOF_KIND = "GNSS_spoof"
VALID_KINDS = set(EMITTERS) | {"tone", "noise"}   # GT kind 白名单（校验告警用）


def load_truth(root: Path, sid: str) -> dict:
    return json.loads((root / sid / "capture" / "private" / "d1_truth.json")
                      .read_text(encoding="utf-8"))


def station_key_map(truth: dict) -> dict:
    """station_id -> per_station 键（'s0'..'s3'，按接收机顺序）。"""
    return {r["id"]: f"s{i}" for i, r in enumerate(truth["receivers"])}


def interferer_visible_at(ifw: dict, truth: dict, stations: list[str],
                          eid: str | None = None) -> bool:
    """可见性优先用 RT 实测（achieved，0 径=被楼挡）；无实测时退回公式 INR。

    注意 achieved 的子键是 sK（station_key_map），不是站名——2026-09-28 修复：
    旧实现用站名查键永远 miss，RT 实测可见性从未生效过（D2–D4 答案键一直
    含"该站物理收不到"的幽灵干扰）。
    """
    kmap = station_key_map(truth)
    ach = (truth.get("achieved") or {}).get(eid or "", {})
    for st in stations:
        sk = kmap[st]
        if eid and sk in ach:
            if ach[sk]["pr_dbm"] - truth["noise"]["noise_floor_dbm"] >= INR_VIS_DB:
                return True
        elif ifw["per_station"][sk]["inr_db"] >= INR_VIS_DB:
            return True
    return False


def gt_at_slevel(truth: dict, stations: list[str]) -> list[dict]:
    """该 S 级子集下可检测的干扰 GT 列表（RT 不可见 = 不计入）。"""
    out = []
    for j, ifw in enumerate(truth.get("interferers", [])):
        if interferer_visible_at(ifw, truth, stations, eid=f"interferer_{j + 1}"):
            out.append(ifw)
    return out


def gt_intent(ifw: dict) -> str | None:
    """敌我意图 GT：真值 intent 字段优先，缺省按 kind 查 INTENT 映射。"""
    return ifw.get("intent") or INTENT.get(ifw["kind"])


def offline_answer(truth: dict, stations: list[str]) -> dict:
    """理想答案：只含该 S 级可见的干扰；字段按该 S 级任务给出（含 T2/T3 轴）。"""
    gts = gt_at_slevel(truth, stations)
    items = []
    for gt in gts:
        s0 = gt["per_station"]["s0"]
        a = {"freq_offset_hz": gt["freq_offset_hz"], "kind": gt["kind"],
             "position_m": [gt["position_m"][0], gt["position_m"][1]],
             "is_spoof": gt["kind"] == SPOOF_KIND}
        intent = gt_intent(gt)
        if intent:
            a["intent"] = intent
        if "station_a" in stations:
            a["doa_deg"] = s0["doa_deg"]           # S0 任务字段
        items.append(a)
    return {"num_interferers": len(items), "interferers": items}


def score_answer(ans: dict, gt_list: list[dict], slevel: str) -> dict:
    r = {"detection_hit": int(ans.get("num_interferers", 0) == len(gt_list))}
    # 检测部分分诊断（口径不变，新增列）：数量差 ±1 内记半对 + 绝对误差
    n_pred = int(ans.get("num_interferers", 0) or 0)
    abs_err = abs(n_pred - len(gt_list))
    r["detection_within1"] = [0.5 if abs_err == 1 else (1.0 if abs_err == 0 else 0.0)]
    r["detect_abs_err"] = [abs_err]
    pred = [p for p in (ans.get("interferers") or [])
            if slevel != "S0" or "doa_deg" in p]          # S0 答案须含 doa 字段
    pairs = match(pred, gt_list)
    r["kind_hits"] = [int(p.get("kind") == g["kind"]) for p, g in pairs]
    by_gt: dict[str, list] = {}
    for p, g in pairs:
        df = abs(p.get("freq_offset_hz", 0) - g["freq_offset_hz"]) / 2e6
        r["freq_hits"] = r.get("freq_hits", []) + [int(df <= FREQ_TOL)]
        r["freq_aes"] = r.get("freq_aes", []) + [df]
        if slevel == "S0" and "doa_deg" in p:
            da = abs(p["doa_deg"] - g["per_station"]["s0"]["doa_deg"])
            r["doa_hits"] = r.get("doa_hits", []) + [int(da <= DOA_TOL_DEG)]
            r["doa_aes"] = r.get("doa_aes", []) + [da]
        if slevel in ("S1", "S2") and "position_m" in p:
            pe = float(np.hypot(p["position_m"][0] - g["position_m"][0],
                                p["position_m"][1] - g["position_m"][1]))
            r["pos_hits"] = r.get("pos_hits", []) + [int(pe <= POS_GATE_M)]
            r["pos_errs"] = r.get("pos_errs", []) + [pe]
        # T3 敌我意图：GT 不在 INTENT 映射内的干扰（D1 free 模式）不进分母
        gi = gt_intent(g)
        if gi:
            hit = int(str(p.get("intent", "")).strip().lower() == gi)
            r["intent_hits"] = r.get("intent_hits", []) + [hit]
            h, t = by_gt.get(gi, (0, 0))
            by_gt[gi] = [h + hit, t + 1]
        # T3 GNSS 欺骗专项：命中/虚警按匹配对统计（报告侧池化成比率）
        gspoof = g["kind"] == SPOOF_KIND
        pspoof = bool(p.get("is_spoof"))
        r["spoof_gt"] = r.get("spoof_gt", []) + [int(gspoof)]
        r["spoof_hits"] = r.get("spoof_hits", []) + [int(gspoof and pspoof)]
        r["spoof_fa"] = r.get("spoof_fa", []) + [int(pspoof and not gspoof)]
        r["spoof_notgt"] = r.get("spoof_notgt", []) + [int(not gspoof)]
    if by_gt:
        r["intent_by_gt"] = by_gt
    return r


def _pos_dist_m(p: dict, g: dict) -> float | None:
    """候选/答案位置与 GT 位置的 2D 距离（m）。任一方缺位置返回 None。"""
    pp = p.get("position_m") or p.get("position_xy_m")
    gp = g.get("position_m")
    if not pp or not gp or len(gp) < 2:
        return None
    try:
        return float(np.hypot(float(pp[0]) - float(gp[0]),
                              float(pp[1]) - float(gp[1])))
    except (TypeError, ValueError):
        return None


def match(pred: list[dict], gt_list: list[dict],
          gate: float | None = 0.45) -> list[tuple]:
    """贪心两段配对（2026-10-03 修正）：**位置优先、频偏兜底**。

    旧版纯频偏最近配对——频偏是结构上最弱且不可提取的轴（三估计器 + oracle
    带切上限 42% 全撞墙），配对错位会污染 kind/intent 计分（u-NMS 后 kind
    0.556→0.444 的假象即此）。修正：位置可信（两侧都有位置）的 GT 先按
    位置距离 ≤50 m 贪心配对（与 pos_hits 的 50 m 口径一致），剩余 GT 再按
    频偏 gate 配对。gate=None（oracle 关联）= 位置优先 + 频偏最近兜底。
    """
    pairs, used_pred = [], set()
    used_gt = set()
    # ---- 第一段：位置优先（双侧都有位置才参与）----
    pos_cand = []
    for i_g, g in enumerate(gt_list):
        for i_p, p in enumerate(pred):
            d = _pos_dist_m(p, g)
            if d is not None and d <= 50.0:
                pos_cand.append((d, i_p, i_g))
    pos_cand.sort()
    for d, i_p, i_g in pos_cand:
        if i_p in used_pred or i_g in used_gt:
            continue
        used_pred.add(i_p); used_gt.add(i_g)
        pairs.append((pred[i_p], gt_list[i_g]))
    # ---- 第二段：频偏兜底（gate 适用于此段；位置段不受 gate 限制——
    #      50 m 位置本身就是强证据）----
    for i_g, g in enumerate(gt_list):
        if i_g in used_gt:
            continue
        best, bi = None, None
        for i, p in enumerate(pred):
            if i in used_pred:
                continue
            d = abs(p.get("freq_offset_hz", 0) - g["freq_offset_hz"]) / 2e6
            if best is None or d < best:
                best, bi = d, i
        if bi is not None and best is not None and (gate is None or best <= gate):
            used_pred.add(bi)
            used_gt.add(i_g)
            pairs.append((pred[bi], g))
    return pairs


def assoc_answer(ans: dict, all_gt: list[dict]) -> dict:
    """oracle-association：考生每个干扰与**全部** GT 干扰频偏最近关联（无门槛），
    只评"识别质量"（kind/intent/频偏/欺骗），与检测能力（数量/漏报）解耦——
    分源失败的样本也能单独考出 T2/T3 识别水平。"""
    pred = [p for p in (ans.get("interferers") or [])]
    pairs = match(pred, all_gt, gate=None)
    r: dict = {"assoc_pairs": [float(len(pairs))]}
    kind, intent, fh, fa = [], [], [], []
    spoof_gt = spoof_hit = spoof_fa = spoof_notgt = 0
    for p, g in pairs:
        kind.append(int(str(p.get("kind", "")).strip() == g["kind"]))
        gi = gt_intent(g)
        if gi:
            intent.append(int(str(p.get("intent", "")).strip().lower() == gi))
        df = abs(p.get("freq_offset_hz", 0) - g["freq_offset_hz"]) / 2e6
        fh.append(int(df <= FREQ_TOL))
        fa.append(df)
        gsp = g["kind"] == SPOOF_KIND
        psp = bool(p.get("is_spoof"))
        spoof_gt += int(gsp)
        spoof_hit += int(gsp and psp)
        spoof_fa += int(psp and not gsp)
        spoof_notgt += int(not gsp)
    r["assoc_kind_hits"] = kind
    r["assoc_intent_hits"] = intent
    r["assoc_freq_hits"] = fh
    r["assoc_freq_aes"] = fa
    r["assoc_spoof"] = {"gt": spoof_gt, "hit": spoof_hit,
                        "fa": spoof_fa, "notgt": spoof_notgt}
    return r


_SENT_RE = re.compile(r'"num_interferers"\s*:\s*(-?\d+)')


def _load_abstain_ids(answers_dir: Path | None) -> set[str]:
    """从答案目录旁的 `_run_log.json` 里找出“模型写了负哨兵”的样本 id。

    背景：capture prompt 明写 "NEVER use a negative num_interferers (e.g. -1) or
    any sentinel"，而 run_capture_agent._sanitize 会把任意负数清成 0 —— 于是一次
    拒答在 detection 上与“判定无干扰”不可区分，且必然判错。本函数只做统计口径的
    拆分（答案文件里若已带 abstained 字段则优先用它），不改变 detection_hit。
    续跑过的目录 `_run_log.json` 可能只含末轮记录 → 覆盖面可能 < n。
    """
    if answers_dir is None:
        return set()
    p = answers_dir / "_run_log.json"
    if not p.exists():
        return set()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return set()
    recs = d if isinstance(d, list) else d.get("runs", [])
    out: set[str] = set()
    for r in recs if isinstance(recs, list) else []:
        if not isinstance(r, dict):
            continue
        sid = r.get("sample_id")
        if not sid:
            continue
        if r.get("abstained") is True:
            out.add(str(sid))
            continue
        m = _SENT_RE.search(r.get("raw_output") or "")
        if m and int(m.group(1)) < 0:
            out.add(str(sid))
    return out


def evaluate(root: str | Path, answers_dir: str | Path | None) -> dict:
    root = Path(root)
    answers_dir = Path(answers_dir) if answers_dir else None
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    subsets = idx["observation_subsets"]
    abstain_ids = _load_abstain_ids(answers_dir)
    matrix: dict[str, dict] = {s: {} for s in subsets}
    unknown_kinds: set[str] = set()
    assoc_pool: dict = {}

    for s in idx["samples"]:
        truth = load_truth(root, s["id"])
        unknown_kinds |= {g["kind"] for g in truth.get("interferers", [])
                          if g["kind"] not in VALID_KINDS}
        ans_obj = None
        if answers_dir:
            ans_p = answers_dir / f"{s['id']}.json"
            if not ans_p.exists():
                continue                       # 部分运行（早停/断点）缺答案的样本跳过
            ans_obj = json.loads(ans_p.read_text(encoding="utf-8"))
        abstained = isinstance(ans_obj, dict) and (
            ans_obj.get("abstained") is True or s["id"] in abstain_ids)
        if ans_obj is not None:
            # oracle-association 池化：与 S 级无关（全 GT 列表），每样本一次
            for k, v in assoc_answer(ans_obj, truth.get("interferers", [])).items():
                if isinstance(v, list):
                    assoc_pool.setdefault(k, []).extend(v)
                elif isinstance(v, dict):
                    d0 = assoc_pool.setdefault(k, {})
                    for kk, vv in v.items():
                        d0[kk] = d0.get(kk, 0) + vv
                else:
                    assoc_pool.setdefault(k, []).append(v)
        for slevel, stations in subsets.items():
            gt_list = gt_at_slevel(truth, stations)
            ans = ans_obj if ans_obj is not None else offline_answer(truth, stations)
            r = score_answer(ans, gt_list, slevel)
            m = matrix[slevel]
            m.setdefault("abstain", []).append(1 if abstained else 0)
            for k, v in r.items():
                if isinstance(v, list):        # 逐干扰对的 hits/aes：展平聚合
                    m.setdefault(k, []).extend(v)
                elif isinstance(v, dict):      # intent_by_gt：{intent: [hits, total]}
                    d = m.setdefault(k, {})
                    for kk, vv in v.items():
                        cur = d.get(kk, [0, 0])
                        d[kk] = [cur[0] + vv[0], cur[1] + vv[1]]
                else:
                    m.setdefault(k, []).append(v)

    if unknown_kinds:
        print(f"[warn] GT kind 超出白名单（16 辐射体 + tone/noise）: {sorted(unknown_kinds)}",
              file=sys.stderr)

    report = {}
    for slevel, m in matrix.items():
        row = {"n": len(next(iter(m.values()))) if m else 0}
        # T3 池化指标：先取出原始计数，报告里写成比率（分母为 0 的级别不出现）
        det_hits = m.pop("detection_hit", [])
        abst_flags = m.pop("abstain", [])
        intent_hits = m.pop("intent_hits", [])
        intent_by_gt = m.pop("intent_by_gt", None)
        s_gt = m.pop("spoof_gt", [])
        s_hit = m.pop("spoof_hits", [])
        s_fa = m.pop("spoof_fa", [])
        s_notgt = m.pop("spoof_notgt", [])
        for k, vals in m.items():
            row[k] = (round(float(np.mean(vals)), 3) if k.endswith("_hit")
                      else round(float(np.mean(vals)), 4))
        if det_hits:
            row["detection_hit"] = round(float(np.mean(det_hits)), 3)      # 注册口径：含拒答
        if abst_flags and any(abst_flags):
            keep = [d for d, a in zip(det_hits, abst_flags) if not a]
            row["abstain_n"] = int(sum(abst_flags))
            row["abstain_rate"] = round(float(np.mean(abst_flags)), 3)
            if keep:
                row["detection_hit_excl_abstain"] = round(float(np.mean(keep)), 3)
                row["detection_denom_excl_abstain"] = len(keep)
        if intent_hits:
            row["intent_accuracy"] = round(float(np.mean(intent_hits)), 3)
            row["intent_pairs"] = len(intent_hits)
        if intent_by_gt:
            row["intent_by_gt"] = intent_by_gt
        if sum(s_gt):
            row["spoof_hit_rate"] = round(sum(s_hit) / sum(s_gt), 3)
            row["spoof_gt_total"] = int(sum(s_gt))
        if sum(s_notgt):
            row["spoof_fa_rate"] = round(sum(s_fa) / sum(s_notgt), 3)
        report[slevel] = row

    # oracle-association 段：识别质量与检测解耦（P(kind|assoc) 等）
    if answers_dir and assoc_pool:
        oa: dict = {"assoc_pairs": int(sum(assoc_pool.get("assoc_pairs", [])))}
        if assoc_pool.get("assoc_kind_hits"):
            oa["P_kind_assoc"] = round(float(np.mean(assoc_pool["assoc_kind_hits"])), 3)
        if assoc_pool.get("assoc_intent_hits"):
            oa["P_intent_assoc"] = round(float(np.mean(assoc_pool["assoc_intent_hits"])), 3)
        if assoc_pool.get("assoc_freq_hits"):
            oa["freq_hits_oracle"] = round(float(np.mean(assoc_pool["assoc_freq_hits"])), 3)
            oa["freq_ae_oracle"] = round(float(np.mean(assoc_pool["assoc_freq_aes"])), 4)
        sp = assoc_pool.get("assoc_spoof", {})
        if sp.get("gt"):
            oa["spoof_hit_oracle"] = round(sp["hit"] / sp["gt"], 3)
        if sp.get("notgt"):
            oa["spoof_fa_oracle"] = round(sp["fa"] / sp["notgt"], 3)
        report["oracle_assoc"] = oa
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--answers", default=None, help="答案目录；缺省=GT 回放（应满分）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = evaluate(args.root, args.answers)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    out = Path(args.out) if args.out else Path(args.root) / "eval_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
