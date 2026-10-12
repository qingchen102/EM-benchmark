"""逐轴「平凡预测器」体检（零 API，只读）。

为什么需要它：`_count_axis_baseline.py` 只查了检测计数轴，结果发现 D2 的真值可见源数
48/50 都是 2 个 —— 常数预测器“永远答 2”就能拿 0.96，比所有真实臂都高。同一个陷阱
在其它轴上同样存在：如果某档真值的体制/意图/机理高度集中在某一类，“永远答那一类”
就能拿到接近 1.0 的分数。**任何跨臂、跨模型的轴间比较，都必须先减掉常数基线。**

两种模式：
  1) 默认（不给臂）：只算真值侧的常数基线（每档 × 每个注册 S 级）。
  2) 给 `--arm <答案目录名>`：再加一段“臂 vs 同分母常数基线”的对照表。

**同分母**很重要：体制/载频/空间/意图/机理这些轴只在**配对成功的对**上计分
（`match()` 的输出），配对失败的 GT 不进入分母。如果只拿“全 GT 的多数类占比”做基线，
就会拿臂的分母和基线的分母做不对等的比较。因此 `--arm` 模式下的基线一律在该臂
**实际配对上的那批 GT** 上重算。

用法：
    python -X utf8 diag/_axis_baselines.py                     # 四档基线
    python -X utf8 diag/_axis_baselines.py d2                  # 只跑 D2
    python -X utf8 diag/_axis_baselines.py d2 --arm answers_S2_agent_fix2_mech
    python -X utf8 diag/_axis_baselines.py --arm answers_S2_agent_fix2_mech   # 四档都找这个目录名
    python -X utf8 diag/_axis_baselines.py d1 --root ../_pre_quota_20261012      # 指定数据集根（目录下含 dataset_dX/）

输出：benchmark_v3/evidence/axis_baselines.txt（给了 --arm 时另写 axis_baselines_arm_<name>.txt）
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ADAPTERS = HERE.parent
if str(_ADAPTERS) not in sys.path:
    sys.path.insert(0, str(_ADAPTERS))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eval_dataset as ed  # noqa: E402

_V3 = _ADAPTERS.parent
DEFAULTS = ("d1", "d2", "d3", "d4")
FREQ_TOL_HZ = ed.FREQ_TOL * 2e6      # 0.02 × 2 MHz = 40 kHz（口径见 score_answer）
POS_GATE_M = ed.POS_GATE_M
DOA_TOL_DEG = ed.DOA_TOL_DEG


def entropy(counter: collections.Counter) -> float:
    n = sum(counter.values())
    h = 0.0
    for v in counter.values():
        p = v / n
        if p:
            h -= p * math.log2(p)
    return h


def top_detail(counter: collections.Counter, k: int = 3) -> str:
    top = ", ".join(f"{name}:{v}" for name, v in counter.most_common(k))
    return f"top{k} {top} / 共 {len(counter)} 类"


def best_within1(counts: list[int]) -> tuple[int, float]:
    """within1 口径下最优常数报数（|报数 − 真值| ≤ 1 记半对）。"""
    best_c, best = 0, -1.0
    for c in range(0, 7):
        r = sum(1.0 if abs(g - c) == 0 else (0.5 if abs(g - c) == 1 else 0.0)
                for g in counts) / len(counts)
        if r > best:
            best_c, best = c, r
    return best_c, best


def analyse(root: Path, sids: list[str], slevel: str, stations: list[str]) -> dict:
    counts: list[int] = []
    kinds: collections.Counter = collections.Counter()
    intents: collections.Counter = collections.Counter()
    mechs: collections.Counter = collections.Counter()
    freq_hit = freq_n = 0
    pos_hit = pos_n = 0
    doa_tgt_hit = doa_zero_hit = doa_n = 0
    for sid in sids:
        truth = ed.load_truth(root, sid)
        gt = ed.gt_at_slevel(truth, stations)
        counts.append(len(gt))
        kmap = ed.station_key_map(truth)
        sk = kmap[stations[0]]
        tgt = truth.get("target") or {}
        tgt_pos = tgt.get("position_m")
        tgt_doa = (tgt.get("per_station") or {}).get(sk, {}).get("doa_deg")
        for g in gt:
            kinds[g["kind"]] += 1
            gi = ed.gt_intent(g)
            if gi:
                intents[gi] += 1
            mechs[ed.gt_mechanism(g)] += 1
            freq_n += 1
            freq_hit += int(abs(g["freq_offset_hz"]) <= FREQ_TOL_HZ)
            if slevel in ("S1", "S2"):
                pos_n += 1
                if tgt_pos:
                    d = math.hypot(g["position_m"][0] - tgt_pos[0],
                                   g["position_m"][1] - tgt_pos[1])
                    pos_hit += int(d <= POS_GATE_M)
            if slevel == "S0":
                ps = (g.get("per_station") or {}).get(sk, {})
                if "doa_deg" in ps:
                    doa_n += 1
                    if tgt_doa is not None:
                        doa_tgt_hit += int(abs(ps["doa_deg"] - tgt_doa) <= DOA_TOL_DEG)
                    doa_zero_hit += int(abs(ps["doa_deg"]) <= DOA_TOL_DEG)
    cnt = collections.Counter(counts)
    return {
        "n_samples": len(sids), "n_gt": sum(counts), "counts": cnt,
        "kinds": kinds, "intents": intents, "mechs": mechs,
        "freq": (freq_hit, freq_n), "pos": (pos_hit, pos_n),
        "doa_tgt": (doa_tgt_hit, doa_n), "doa_zero": (doa_zero_hit, doa_n),
    }


def render(deck: str, slevel: str, stations: list[str], a: dict) -> list[str]:
    n, cnt = a["n_samples"], a["counts"]
    maj_k = max(cnt, key=cnt.get)
    maj = cnt[maj_k] / n
    w1_c, w1 = best_within1([c for c, k in cnt.items() for _ in range(k)])
    out = [f"[{deck.upper()}/{slevel}] 用站 {stations}   可见 GT 干扰 {a['n_gt']} 个"
           f"（{n} 样本，均值 {a['n_gt'] / n:.2f}）",
           f"  {'轴':<6}{'平凡预测器（常数）':<30}{'基线':<9}备注",
           "  " + "-" * 74]
    out.append(f"  {'计数':<6}{'永远答 ' + str(maj_k):<30}{maj:<9.3f}"
               f"真值分布 {dict(sorted(cnt.items()))}   熵 {entropy(cnt):.2f} bit")
    out.append(f"  {'':<6}{'永远答 ' + str(w1_c) + '（within1 口径）':<30}{w1:<9.3f}"
               f"|报数−真值|≤1 最优常数")
    if a["kinds"]:
        k = a["kinds"]
        out.append(f"  {'体制':<6}{'永远答 ' + k.most_common(1)[0][0]:<30}"
                   f"{k.most_common(1)[0][1] / sum(k.values()):<9.3f}"
                   f"{top_detail(k)}   熵 {entropy(k):.2f} bit")
    fh, fn = a["freq"]
    if fn:
        out.append(f"  {'载频':<6}{'永远报 0 Hz（无频偏）':<30}{fh / fn:<9.3f}"
                   f"|f0| ≤ 40 kHz 的占比（n={fn}）")
    if slevel in ("S1", "S2"):
        ph, pn = a["pos"]
        out.append(f"  {'空间':<6}{'永远报目标位置（≤50 m）':<30}{ph / pn:<9.3f}"
                   f"目标与干扰同址的占比（n={pn}）")
    else:
        th, tn = a["doa_tgt"]
        zh, _ = a["doa_zero"]
        if tn:
            out.append(f"  {'空间':<6}{'永远报目标方位（±10°）':<30}{th / tn:<9.3f}"
                       f"干扰方位与目标方位同向占比（n={tn}）")
            out.append(f"  {'':<6}{'永远报 0°（±10°）':<30}{zh / tn:<9.3f}"
                       f"阵面法向命中率")
    if a["mechs"]:
        m = a["mechs"]
        out.append(f"  {'机理':<6}{'永远答 ' + m.most_common(1)[0][0]:<30}"
                   f"{m.most_common(1)[0][1] / sum(m.values()):<9.3f}"
                   f"{top_detail(m)}   熵 {entropy(m):.2f} bit")
    if a["intents"]:
        i = a["intents"]
        out.append(f"  {'意图':<6}{'永远答 ' + i.most_common(1)[0][0]:<30}"
                   f"{i.most_common(1)[0][1] / sum(i.values()):<9.3f}"
                   f"{top_detail(i)}   熵 {entropy(i):.2f} bit")
    out.append("")
    return out


# --------------------------------------------------------------------------- #
# 模式 2：臂 vs 同分母常数基线
# --------------------------------------------------------------------------- #

def _ans_count(ans: dict) -> int:
    if "interferers" in ans:
        return len(ans.get("interferers") or [])
    return int(ans.get("num_interferers", 0) or 0)


def arm_vs_baseline(root: Path, sids: list[str], slevel: str, stations: list[str],
                    arm_dir: Path) -> list[str]:
    """臂的各轴得分 vs **在同一批配对 GT 上**重算的常数基线。"""
    counts: list[int] = []
    det_hit = det_n = 0
    kinfo: dict[str, object] = {
        "kind": [collections.Counter(), 0, 0],
        "intent": [collections.Counter(), 0, 0],
        "mech": [collections.Counter(), 0, 0],
        "freq": [collections.Counter(), 0, 0],
        "space": [collections.Counter(), 0, 0],
    }
    for sid in sids:
        p = arm_dir / f"{sid}.json"
        if not p.exists():
            continue
        ans = json.loads(p.read_text(encoding="utf-8"))
        truth = ed.load_truth(root, sid)
        gt_list = ed.gt_at_slevel(truth, stations)
        counts.append(len(gt_list))
        det_hit += int(_ans_count(ans) == len(gt_list))
        det_n += 1
        kmap = ed.station_key_map(truth)
        sk = kmap[stations[0]]
        tgt = truth.get("target") or {}
        tgt_pos = tgt.get("position_m")
        tgt_doa = (tgt.get("per_station") or {}).get(sk, {}).get("doa_deg")
        pred = [x for x in (ans.get("interferers") or [])
                if slevel != "S0" or "doa_deg" in x]
        for pp, g in ed.match(pred, gt_list):
            # 体制
            c, hit, tot = kinfo["kind"]
            c[g["kind"]] += 1
            kinfo["kind"] = [c, hit + int(pp.get("kind") == g["kind"]), tot + 1]
            # 意图（GT 不在 INTENT 映射内的不进分母）
            gi = ed.gt_intent(g)
            if gi:
                c, hit, tot = kinfo["intent"]
                c[gi] += 1
                kinfo["intent"] = [c, hit + int(str(pp.get("intent", "")).strip().lower() == gi), tot + 1]
            # 机理
            gm = ed.gt_mechanism(g)
            c, hit, tot = kinfo["mech"]
            c[gm] += 1
            kinfo["mech"] = [c, hit + int(str(pp.get("mechanism", "")).strip().lower() == gm), tot + 1]
            # 载频：基线 = 该批配对 GT 里 |f0| ≤ 40 kHz 的占比
            c, hit, tot = kinfo["freq"]
            df = abs(pp.get("freq_offset_hz", 0) - g["freq_offset_hz"]) / 2e6
            c[int(abs(g["freq_offset_hz"]) <= FREQ_TOL_HZ)] += 1
            kinfo["freq"] = [c, hit + int(df <= ed.FREQ_TOL), tot + 1]
            # 空间
            c, hit, tot = kinfo["space"]
            if slevel in ("S1", "S2"):
                if "position_m" in pp and tgt_pos:
                    pe = math.hypot(pp["position_m"][0] - g["position_m"][0],
                                    pp["position_m"][1] - g["position_m"][1])
                    h = int(pe <= POS_GATE_M)
                else:
                    h = 0
                if tgt_pos:
                    dg = math.hypot(g["position_m"][0] - tgt_pos[0],
                                    g["position_m"][1] - tgt_pos[1])
                    c[int(dg <= POS_GATE_M)] += 1
            else:
                ps = (g.get("per_station") or {}).get(sk, {})
                if "doa_deg" in pp and "doa_deg" in ps:
                    h = int(abs(pp["doa_deg"] - ps["doa_deg"]) <= DOA_TOL_DEG)
                else:
                    h = 0
                if tgt_doa is not None:
                    c[int(abs(ps.get("doa_deg", 1e9) - tgt_doa) <= DOA_TOL_DEG)] += 1
            kinfo["space"] = [c, hit + h, tot + 1]

    n_arm = det_n
    out = [f"  [臂 {arm_dir.name}]  已评样本 {n_arm}/{len(sids)}"]
    if not n_arm:
        return out + ["    （该臂没有答案文件）", ""]
    out += [f"    {'轴':<6}{'臂得分':<9}{'同分母常数基线':<15}{'差值':<10}备注",
            "    " + "-" * 70]
    cmaj = collections.Counter(counts)
    cbase = max(cmaj.values()) / n_arm
    out.append(f"    {'计数':<6}{det_hit / n_arm:<9.3f}{cbase:<15.3f}"
               f"{det_hit / n_arm - cbase:<+10.3f}样本级；常数=永远答 {max(cmaj, key=cmaj.get)}"
               f"（真值分布 {dict(sorted(cmaj.items()))}）")
    names = {"kind": "体制", "freq": "载频", "space": "空间",
             "intent": "意图", "mech": "机理"}
    # 分类轴的基线 = 多数类占比；载频/空间轴的基线 = “永远报 0 / 永远报目标位置”的命中率
    CONST_HIT = {"freq": "永远报 0 Hz", "space": "永远报目标位置/目标方位"}
    for key in ("kind", "freq", "space", "intent", "mech"):
        c, hit, tot = kinfo[key]
        if not tot:
            continue
        if key in CONST_HIT:
            base = c.get(1, 0) / tot
            label = f"常数 = {CONST_HIT[key]}   分布 {{命中: {c.get(1, 0)}, 未命中: {c.get(0, 0)}}}"
        else:
            base = max(c.values()) / tot
            label = (f"基线类 = {max(c, key=c.get)}"
                     + (f"，分布 {dict(c.most_common(4))}" if len(c) > 1 else "（单类，无信息）"))
        out.append(f"    {names[key]:<6}{hit / tot:<9.3f}{base:<15.3f}"
                   f"{hit / tot - base:<+10.3f}配对上重算（分母 n={tot}），{label}")
    out.append("")
    return out


def main(argv: list[str]) -> int:
    arm = None
    root_arg = None
    args = []
    it = iter(argv)
    for a in it:
        if a == "--arm":
            arm = next(it, None)
        elif a == "--root":
            root_arg = next(it, None)
        else:
            args.append(a)
    base_dir = Path(root_arg).expanduser() if root_arg else None
    if base_dir is not None and not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    decks = [a for a in args if not a.startswith("-")] or list(DEFAULTS)
    out: list[str] = [
        "逐轴「平凡预测器（常数基线）」体检 —— 零 API，只读（diag/_axis_baselines.py）",
        "读法：任何臂的该轴得分若不明显高于这里的基线，就说明该轴在本档没有判别力。",
        f"口径：载频 ±{FREQ_TOL_HZ / 1e3:.0f} kHz（FREQ_TOL={ed.FREQ_TOL} × 2 MHz）、"
        f"定位 ≤{POS_GATE_M:.0f} m、DOA ±{DOA_TOL_DEG:.0f}°；可见性 = ed.gt_at_slevel（RT achieved）",
        "注：机理轴这里的基线按**全部可见 GT** 算；diag/_mech_axis_quality.py 按配对成功的对算，"
        "两者分母不同（差 1–3 pp）。--arm 模式下的基线已统一为同分母。",
    ]
    for deck in decks:
        root = (base_dir / f"dataset_{deck}") if base_dir is not None else (_V3 / f"dataset_{deck}")
        if not (root / "index.json").is_file():
            out += ["", "=" * 78, f"（跳过）{root} 无 index.json", "=" * 78, ""]
            continue
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        sids = [s["id"] for s in idx["samples"]]
        subsets = idx["observation_subsets"]
        out += ["", "=" * 78,
                f"dataset_{deck}   样本 {len(sids)}   S 级子集 {list(subsets)}",
                "=" * 78, ""]
        for slevel, stations in subsets.items():
            out += render(deck, slevel, stations, analyse(root, sids, slevel, stations))
            if arm:
                adir = root / arm
                if not adir.is_dir() and "_S2_" in arm:
                    adir = root / arm.replace("_S2_", "_S0_")   # D1 用 S0 目录名
                if adir.is_dir():
                    out += arm_vs_baseline(root, sids, slevel, stations, adir)
    text = "\n".join(out) + "\n"
    print(text, end="")
    name = "axis_baselines.txt" if not arm else f"axis_baselines_arm_{arm}.txt"
    if base_dir is not None:
        name = name.replace(".txt", f"_{base_dir.name}.txt")
    dst = _V3 / "evidence" / name
    dst.write_text(text, encoding="utf-8")
    print(f"[已写入] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
