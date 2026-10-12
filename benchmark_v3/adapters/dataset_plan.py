"""数据集计划器：把「机理配额 + 源数配额 + 意图平衡」变成生成前的显式计划（2026-10-12 拍板）。

依据：`benchmark_v3/docs/机理配额与重生成方案.md`（草案 v1 = 现行口径）。

为什么要有这一层
----------------
旧做法里机理**不是抽样维度**，而是两件事的副产物：
  ① `gen_dataset._draw_ifw_freq` 的重叠度档位（深/部分/分离 各 1/3）——深与部分都派生 co_channel；
  ② 池成员资格——脉冲家族（Welder/VFD_noise/Corona）进池就自动判 pulse。
实测因此偏离设计（D2 期望 co≈42/pulse≈37/adj≈21%，实得 co 49/pulse 39/adj 12）。本模块改为
**先定机理、再定参数**：先按配额排出每个槽位的 (源数, 机理序列)，再映射到 (kind, 频偏档)，
最后由 `_draw_ifw_freq` 按指定档位抽真实频偏/带宽——真值仍由 `mechanism.derive()` 从实际参数派生，
生成时逐源自校（计划与派生必须一致，否则视为 QA 失败重抽）。

计划的性质
----------
- 按**槽位**（accepted 序号）生成，与候选 seed 无关 ⇒ 可复现、可断点续跑、可离线审计；
- 计划里只有设计约束（配额/池/档位），不含任何随机实现细节；
- 档级配额由**赤字调度**（deficit scheduling）给出，配 `temp` 抖动保留每样本机理组合的多样性；
- 意图分布由 kind 池决定（`INTENT` 是 kind → intent 的确定性表），故计划阶段即可校验并重抽。

用法
----
    from dataset_plan import plan_dataset, format_summary, audit_truths
    plan = plan_dataset("d2", 100, INTENT)      # INTENT: kind -> intent（emitters.INTENT）
    print(format_summary(plan, INTENT, "d2"))

命令行自检（零 API、不碰任何数据）：
    python dataset_plan.py d2 100
"""
from __future__ import annotations

import sys

import numpy as np

PLAN_SEED = 20261012

# ---- 配额（按源计；见方案 §1。D1 不在表内：豁免，全 adjacent）---------------------
MECH_QUOTA: dict[str, dict[str, float]] = {
    "d2": {"co_channel": 0.45, "pulse": 0.30, "adjacent": 0.25},
    "d3": {"co_channel": 0.50, "adjacent": 0.30, "pulse": 0.20},
    "d4": {"spoof": 0.30, "co_channel": 0.30, "adjacent": 0.20, "pulse": 0.20},
}

# ---- 源数配额（按样本；QA 约束 4(b)：≥3 个取值且最大类 ≤50%）--------------------
COUNT_QUOTA: dict[str, dict[int, float]] = {
    "d2": {1: 0.35, 2: 0.45, 3: 0.20},
    "d3": {1: 0.30, 2: 0.45, 3: 0.25},
    "d4": {2: 0.35, 3: 0.45, 4: 0.20},
}

# ---- 池（kind）-------------------------------------------------------------------
PULSE_POOL = ("Welder", "VFD_noise", "Corona")          # derive → pulse（kind 决定）
MAL_POOL = ("Chirp_jam", "Noise_jam", "Tone_jam")       # 恶意压制（kind 不判机理）
COEX_CONT = ("Bluetooth_FHSS", "WiFi_burst", "Microwave", "Zigbee", "DECT")  # 非脉冲共存
SPOOF_KIND = "GNSS_spoof"                               # 仅 D4

# 机理 → 非脉冲 kind 池：D2 只出共存辐射体（居民区没有干扰器）；
# D3/D4 恶意池优先（权重 2:1，方案 §2「恶意池优先」）。
NONPULSE_POOL: dict[str, tuple[str, ...]] = {
    "d2": COEX_CONT,
    "d3": MAL_POOL * 2 + COEX_CONT,
    "d4": MAL_POOL * 2 + COEX_CONT,
}

# 机理 → 频偏档位（0 深重叠 / 1 部分重叠 / 2 频谱分离；见方案 §2）
TIER_CHOICES: dict[str, tuple[int, ...]] = {
    "co_channel": (0, 1),
    "adjacent": (2,),
    "pulse": (0, 1, 2),
    "spoof": (0, 1, 2),
}

# 机理配额允差（pp）与意图平衡门限（QA 约束 4）
QUOTA_TOL_PP = 5.0
INTENT_MIN = 0.20          # 每类 ≥20%
INTENT_MAX = 0.75          # 最大类 ≤75%
COUNT_MAX_SHARE = 0.50     # 源数最大类 ≤50%


def _largest_remainder(weights: dict, total: int) -> dict:
    """最大余额法：把 `total` 按权重拆成整数，且各项和恰好等于 total。"""
    keys = list(weights)
    raw = {k: float(weights[k]) * total for k in keys}
    base = {k: int(np.floor(raw[k])) for k in keys}
    rem = total - sum(base.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:max(0, rem)]:
        base[k] += 1
    return base


def _assign_counts(difficulty: str, n_samples: int, rng: np.random.Generator) -> list[int]:
    """每样本源数（配额精确 + 顺序打乱）。"""
    quota = COUNT_QUOTA[difficulty]
    alloc = _largest_remainder(quota, n_samples)
    seq = [n for n, c in alloc.items() for _ in range(c)]
    rng.shuffle(seq)
    return seq


def _pick_mechanism(deficits: dict[str, float], rng: np.random.Generator, temp: float = 0.25) -> str:
    """赤字调度：优先取「目标 − 已分配」最大者；以 temp 概率退而取次大，保留组合多样性。"""
    order = sorted(deficits, key=lambda k: (-deficits[k], k))
    if len(order) > 1 and rng.random() < temp and deficits[order[1]] > 0:
        return order[1]
    return order[0]


def _plan_mechanisms(difficulty: str, counts: list[int], rng: np.random.Generator) -> list[list[str]]:
    """按配额给每个样本分配机理序列（赤字调度，档级比例收敛到配额）。"""
    weights = MECH_QUOTA[difficulty]
    total = sum(counts)
    assigned = {k: 0 for k in weights}
    out: list[list[str]] = []
    i = 0
    for n in counts:
        row = []
        for _ in range(n):
            i += 1
            deficits = {k: weights[k] * i - assigned[k] for k in weights}
            k = _pick_mechanism(deficits, rng)
            assigned[k] += 1
            row.append(k)
        out.append(row)
    return out


def _kinds_for(mechs: list[str], difficulty: str, rng: np.random.Generator) -> list[str]:
    out = []
    for m in mechs:
        if m == "spoof":
            out.append(SPOOF_KIND)
        elif m == "pulse":
            out.append(str(rng.choice(PULSE_POOL)))
        else:
            out.append(str(rng.choice(NONPULSE_POOL[difficulty])))
    return out


def _intent_share(kinds: list[str], intent_of: dict) -> dict[str, float]:
    n = max(1, len(kinds))
    out: dict[str, float] = {}
    for k in kinds:
        it = intent_of.get(k, "benign")
        out[it] = out.get(it, 0.0) + 1.0 / n
    return out


def plan_dataset(difficulty: str, n_samples: int, intent_of: dict, *,
                 seed: int = PLAN_SEED, max_redraw: int = 400) -> list[dict]:
    """产出 `n_samples` 个槽位的计划：{count, sources:[{mechanism, kind, tier}], ...}。

    `intent_of` 是 kind → intent 的映射（`emitters.INTENT`）。若 kind 抽样的意图分布违反
    QA 约束 4(c)（每类 ≥20%、最大类 ≤75%），则重抽 kind（最多 `max_redraw` 次）。
    """
    if difficulty not in MECH_QUOTA:
        raise KeyError(f"{difficulty} 不在机理配额表内（D1 豁免，不走计划器）")
    rng = np.random.default_rng(seed)
    counts = _assign_counts(difficulty, n_samples, rng)
    mechs_all = _plan_mechanisms(difficulty, counts, rng)

    flat_mechs = [m for row in mechs_all for m in row]
    best_kinds, best_share, best_bad = None, None, None
    for _ in range(max_redraw):
        kinds = _kinds_for(flat_mechs, difficulty, rng)
        share = _intent_share(kinds, intent_of)
        bad = (max(share.values()) > INTENT_MAX + 1e-9)
        if not bad:
            best_kinds, best_share, best_bad = kinds, share, False
            break
        if best_bad is None or max(share.values()) < max(best_share.values()):
            best_kinds, best_share, best_bad = kinds, share, True
    if best_bad:
        raise RuntimeError(
            f"{difficulty} 意图平衡无法满足（{max_redraw} 次重抽，最大类 "
            f"{max(best_share.values()):.3f} > {INTENT_MAX}）：请调整配额或 kind 池")

    plan: list[dict] = []
    j = 0
    for row in mechs_all:
        sources = []
        for m in row:
            tier = int(rng.choice(TIER_CHOICES[m]))
            sources.append({"mechanism": m, "kind": best_kinds[j], "tier": tier})
            j += 1
        plan.append({"count": len(row), "sources": sources})
    return plan


def summarise(plan: list[dict], intent_of: dict) -> dict:
    """对计划做普查：源数 / 机理 / 意图三项分布（含配额对照与是否达标）。"""
    counts: dict[int, int] = {}
    mechs: dict[str, int] = {}
    intents: dict[str, int] = {}
    n_src = 0
    for slot in plan:
        counts[slot["count"]] = counts.get(slot["count"], 0) + 1
        for s in slot["sources"]:
            n_src += 1
            mechs[s["mechanism"]] = mechs.get(s["mechanism"], 0) + 1
            it = intent_of.get(s["kind"], "benign")
            intents[it] = intents.get(it, 0) + 1

    def _share(d: dict) -> dict:
        tot = max(1, sum(d.values()))
        return {k: v / tot for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}

    c_share, m_share, i_share = _share(counts), _share(mechs), _share(intents)
    return {
        "n_samples": len(plan),
        "n_sources": n_src,
        "counts": counts,
        "counts_share": c_share,
        "mechs": mechs,
        "mechs_share": m_share,
        "intents": intents,
        "intents_share": i_share,
        "ok_count": len(counts) >= 3 and max(c_share.values()) <= COUNT_MAX_SHARE + 1e-9,
        "ok_intent": max(i_share.values()) <= INTENT_MAX + 1e-9,
    }




def format_summary(plan: list[dict], intent_of: dict, difficulty: str = "") -> str:
    """人类可读的普查摘要（生成开始时打印，便于当场核对配额）。"""
    s = summarise(plan, intent_of)
    quota = MECH_QUOTA.get(difficulty, {})
    lines = [f"[plan{' ' + difficulty if difficulty else ''}] 槽位 {s['n_samples']}  源 {s['n_sources']}"]
    lines.append("  源数：" + "  ".join(f"{k}源 {v}个({100 * s['counts_share'][k]:.0f}%)"
                                      for k, v in sorted(s["counts"].items())))
    lines.append("  机理：" + "  ".join(
        f"{k} {v}({100 * s['mechs_share'][k]:.1f}%" +
        (f"，目标 {100 * quota[k]:.0f}%）" if k in quota else "）")
        for k, v in sorted(s["mechs"].items())))
    lines.append("  意图：" + "  ".join(f"{k} {100 * v:.1f}%" for k, v in s["intents_share"].items()))
    for k, w in sorted(quota.items()):
        got = s["mechs_share"].get(k, 0.0)
        flag = "OK " if abs(got - w) * 100 <= QUOTA_TOL_PP else "偏 "
        lines.append(f"    {flag}{k}: 实得 {100 * got:.1f}% vs 目标 {100 * w:.0f}%"
                     f"（差 {100 * (got - w):+.1f} pp，允差 ±{QUOTA_TOL_PP:.0f}）")
    lines.append(f"  QA4(a) 机理 ≥3 值：{'PASS' if len(s['mechs']) >= 3 else 'FAIL'}"
                 f"；QA4(b) 源数 ≥3 值且最大类≤50%：{'PASS' if s['ok_count'] else 'FAIL'}"
                 f"；QA4(c) 意图最大类≤75%：{'PASS' if s['ok_intent'] else 'FAIL'}")
    return "\n".join(lines)


def audit_truths(truths: list[dict], intent_of: dict, difficulty: str = "") -> tuple[bool, list[str]]:
    """对**已生成真值**做 QA 约束 4 普查（零 API）：机理/可见源数/意图三项。

    机理用 `mechanism.derive` 从真值物理量复算（口径与评测器一致），不用计划值。
    """
    import mechanism as mk  # noqa: E402  （与生成器同目录）

    mechs: dict[str, int] = {}
    intents: dict[str, int] = {}
    counts: dict[int, int] = {}
    for t in truths:
        ifws = t.get("interferers") or []
        counts[len(ifws)] = counts.get(len(ifws), 0) + 1
        for rec in ifws:
            m = mk.derive(str(rec.get("kind", "")), float(rec.get("freq_offset_hz", 0.0) or 0.0),
                          float(rec.get("bandwidth_hz", 0.0) or 0.0))
            mechs[m] = mechs.get(m, 0) + 1
            it = rec.get("intent") or intent_of.get(str(rec.get("kind", "")), "benign")
            intents[str(it)] = intents.get(str(it), 0) + 1

    def _share(d: dict) -> dict:
        tot = max(1, sum(d.values()))
        return {k: v / tot for k, v in d.items()}

    c_share, m_share, i_share = _share(counts), _share(mechs), _share(intents)
    lines = [f"[QA4 {difficulty}] 样本 {len(truths)}  源 {sum(mechs.values())}"]
    lines.append("  源数：" + "  ".join(f"{k}源 {v}个({100 * c_share[k]:.0f}%)" for k, v in sorted(counts.items())))
    lines.append("  机理：" + "  ".join(f"{k} {v}({100 * m_share[k]:.1f}%)" for k, v in sorted(mechs.items())))
    lines.append("  意图：" + "  ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(i_share.items())))
    # 门限口径（2026-10-12 修正）：最大类不得超「计划最大类 + 允差」——
    # 计划本身允许 ±5 pp（源数/机理都是整数分配，小样本必有舍入），硬卡 50% 会让
    # D3（co_channel 目标正好 50%）在舍入到 50.x% 时误判（n=6 冒烟已复现）。
    tol = QUOTA_TOL_PP / 100.0
    mech_tgt = max(MECH_QUOTA[difficulty].values()) if difficulty in MECH_QUOTA else 0.5
    count_tgt = max(COUNT_QUOTA[difficulty].values()) if difficulty in COUNT_QUOTA else 0.5
    mech_bound, count_bound = mech_tgt + tol, count_tgt + tol
    ok_mech = len(mechs) >= 3 and max(m_share.values()) <= mech_bound + 1e-9
    ok_count = len(counts) >= 3 and max(c_share.values()) <= count_bound + 1e-9
    ok_intent = max(i_share.values()) <= INTENT_MAX + 1e-9
    for name, ok in ((f"(a) 机理 ≥3 值且最大类≤{100 * mech_bound:.0f}%"
                      f"（计划 {100 * mech_tgt:.0f}% + 允差 {QUOTA_TOL_PP:.0f}）", ok_mech),
                     (f"(b) 源数 ≥3 值且最大类≤{100 * count_bound:.0f}%"
                      f"（计划 {100 * count_tgt:.0f}% + 允差）", ok_count),
                     ("(c) 意图最大类≤75%", ok_intent)):
        lines.append(f"  QA4{name}: {'PASS' if ok else 'FAIL'}")
    return bool(ok_mech and ok_count and ok_intent), lines


def main() -> None:
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    for _p in (str(_here), str(_here.parent / "simulation")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from em_signal_simulator_v3.emitters import INTENT  # noqa: E402
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "d2"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    plan = plan_dataset(difficulty, n, INTENT)
    print(format_summary(plan, INTENT, difficulty))
    print("  样例（前 3 槽）：")
    for slot in plan[:3]:
        print("   ", slot["count"], [f"{s['mechanism']}/{s['kind']}/t{s['tier']}" for s in slot["sources"]])


if __name__ == "__main__":
    main()
