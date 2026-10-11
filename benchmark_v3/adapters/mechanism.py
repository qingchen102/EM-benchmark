"""干扰机理（mechanism）派生——把"到底怎么干扰的"变成真值里的一等字段。

依据（benchmark_v3/docs/场景设计规范.md，v1.0-frozen）
- §3 干扰机理十分类（与申报书"同频/邻道/阻塞/脉冲"对齐）；
- §7 受害者层 schema 的取值域：`co_channel | adjacent | blocking | imd | pulse | spoof | jam | harmonic`；
- §2 洞察 1：「T3 意图 = 从瞄准关系 + 机理推理，**非查表**」——机理是意图的前置条件。
- §5 决策 D-5：受害者层 schema 按 §7 确认。

现状（2026-10-11 核查）
- `gen_dataset.py:451` 只产出 `jam` / `spoof` 两个值，且只为 D3/D4 的 preset 恶意源建事件；
- 评分器（`eval_dataset.py`）没有机理轴；公开层只把 `jam`/`spoof` 当证据暴露。
→ 本模块给出**只用真值物理量即可复算**的派生规则，作为加机理轴的前置。

派生规则（按优先级取第一个命中）
  1. `spoof`       结构欺骗（kind 属于 SPOOF_KINDS）——与耦合方式无关，单列
  2. `pulse`       时域脉冲结构为主（kind 属于 PULSE_KINDS）
  3. `co_channel`  干扰带与目标占用带（±TARGET_BW_HZ/2，目标 freq_offset=0）**重叠**
  4. `adjacent`    不重叠、但仍落在接收通带（±RX_HALF_HZ）内 → 邻道泄漏
  5. `jam`         兜底（带内、频带信息缺失或类别未知）

**不参与标注的三类**：`blocking` / `imd` / `harmonic` 需要平台"窗外强源前端注入"
（规范 §9 第 7 项、§11 远期）：当前几何下前端口复合功率只有 −79 dBm 量级、比 IIP3 低 60+ dB，
物理上不会发生 → 无正样本，标了就是伪造。

另附 `suppression` 布尔（规范 §3 第 10 类"压制"）：它描述的是**战术/意图**维度，与耦合方式正交
（一个瞄准式压制源同时也是 co_channel）。塞进单值 mechanism 会与 1–4 冲突，故单列。
"""
from __future__ import annotations

from typing import Iterable

# ---- 取值域（与 §7 schema 一致）---------------------------------------------
MECHANISMS = ("co_channel", "adjacent", "blocking", "imd", "pulse", "spoof", "jam", "harmonic")
DERIVABLE = ("spoof", "pulse", "co_channel", "adjacent", "jam")     # 当前可派生
NOT_COVERED = ("blocking", "imd", "harmonic")                      # 需平台扩展，零正样本

# ---- 物理常量（与 gen_dataset.py 对齐）--------------------------------------
TARGET_BW_HZ = 1e6          # gen_dataset.TGT_BANDWIDTH_HZ：目标组件带宽
RX_HALF_HZ = 0.45 * 0.9 * 4e6   # gen_dataset.RX_FRAC * FS_HZ = ±810 kHz 接收通带

# ---- 类别集合 ---------------------------------------------------------------
SPOOF_KINDS = ("GNSS_spoof",)
PULSE_KINDS = ("Welder", "VFD_noise", "Corona", "Radar_pulsed_LFM")
# §3 第 10 类"压制"对应的辐射体（瞄准式、以压死链路为目的）
SUPPRESS_KINDS = ("Chirp_jam", "Noise_jam", "Tone_jam", "GNSS_jam", "GNSS_spoof")


def band_overlap_hz(f0: float, bw: float, target_bw_hz: float = TARGET_BW_HZ) -> float:
    """干扰带 [f0±bw/2] 与目标带 [∓target_bw/2] 的重叠宽度（Hz，非负）。"""
    lo, hi = f0 - bw / 2.0, f0 + bw / 2.0
    tlo, thi = -target_bw_hz / 2.0, target_bw_hz / 2.0
    return max(0.0, min(hi, thi) - max(lo, tlo))


def inside_passband(f0: float, bw: float, rx_half_hz: float = RX_HALF_HZ) -> bool:
    """干扰带的中心是否落在接收通带内（用于区分"邻道泄漏"与"窗外"）。"""
    return abs(f0) <= rx_half_hz


def derive(kind: str, freq_offset_hz: float, bandwidth_hz: float, *,
           target_bw_hz: float = TARGET_BW_HZ, rx_half_hz: float = RX_HALF_HZ) -> str:
    """按优先级派生单个干扰源的机理（纯函数，只用真值物理量）。"""
    if kind in SPOOF_KINDS:
        return "spoof"
    if kind in PULSE_KINDS:
        return "pulse"
    if band_overlap_hz(freq_offset_hz, bandwidth_hz, target_bw_hz) > 0.0:
        return "co_channel"
    if inside_passband(freq_offset_hz, bandwidth_hz, rx_half_hz):
        return "adjacent"
    return "jam"


def suppression(kind: str, mode: str | None = None) -> bool:
    """是否属于 §3 第 10 类"压制"（战术维度，与机理正交）。"""
    return kind in SUPPRESS_KINDS or str(mode or "").lower() in ("jam", "spoof")


def annotate(truth: dict, *, overwrite: bool = False) -> dict:
    """给真值的每个干扰源补 `mechanism`（+ `suppression`），返回计数。

    只动真值字典，不写盘；`overwrite=False` 时保留已有 mechanism（兼容未来生成器直出）。
    """
    counts: dict[str, int] = {}
    for rec in (truth.get("interferers") or []):
        kind = str(rec.get("kind", ""))
        f0 = float(rec.get("freq_offset_hz", 0.0) or 0.0)
        bw = float(rec.get("bandwidth_hz", 0.0) or 0.0)
        mk = derive(kind, f0, bw)
        if overwrite or not rec.get("mechanism"):
            rec["mechanism"] = mk
        rec.setdefault("suppression", suppression(kind, rec.get("mode")))
        counts[rec["mechanism"]] = counts.get(rec["mechanism"], 0) + 1
    # 事件表同步（若存在）
    by_src = {f"interferer_{i + 1}": r for i, r in enumerate(truth.get("interferers") or [])}
    for ev in (truth.get("interference_events") or []):
        src = by_src.get(str(ev.get("source_id")))
        if src and src.get("mechanism"):
            if overwrite or not ev.get("mechanism"):
                ev["mechanism"] = src["mechanism"]
            ev.setdefault("suppression", src.get("suppression", False))
    return counts


def census(truths: Iterable[dict]) -> dict[str, int]:
    """对一批真值做机理普查（不改动入参）。"""
    out: dict[str, int] = {}
    for t in truths:
        for rec in (t.get("interferers") or []):
            mk = derive(str(rec.get("kind", "")), float(rec.get("freq_offset_hz", 0.0) or 0.0),
                        float(rec.get("bandwidth_hz", 0.0) or 0.0))
            out[mk] = out.get(mk, 0) + 1
    return out
