"""受害者层公开视图（场景设计规范 v1.0 §7）——把真值 interference_events 投影成
Agent 可见的"合作链路事件日志"，供 observations.json 与运行器提示使用。

规范 §7 的可见性规定（`benchmark_v3/docs/场景设计规范.md` 第 213 行）：
    Agent 可见 interference_events 的**机理与行为证据**，**不可见** GT 的
    aimed / victim 字段——从行为推断意图。

本模块把这条规定落成一个**单点、可审计**的投影，避免"哪份脚本各自决定暴露什么"：

    公开字段  PUBLIC_EVENT_FIELDS   = event_id / source_id / mechanism / impact_class
    屏蔽字段  WITHHELD_EVENT_FIELDS = aimed / victim_id / power_at_victim_dbm
                                      env_over_noise_db

设计取舍（2026-10-09 用户批准"按冻结 §7 schema 暴露机理/影响等级、隐藏 aimed/victim"）：
  * `aimed` 是**硬编码标签**（`gen_dataset.py` 对 preset 恶意类目直接写 True），
    当前没有天线/朝向模型在仿真它，所以它既不是行为证据、暴露它等于直接给意图答案
    → 屏蔽。
  * `power_at_victim_dbm` / `env_over_noise_db` 是目标接收机前端的链路预算量，属私有
    真值（公开侧连目标功率都没有）→ 屏蔽；影响用离散化的 `impact_class` 表达，与
    observations.json 的 `cooperative_sources.link_quality_per_station`（合作站自测；
    由 `gen_dataset.cooperative_sources_prior` 生成、`expose_evidence.py --coop` 恢复到
    public）同源同粒度。
  * `source_id`（interferer_{j+1}）保留：它只是生成器的内部序号，公开侧没有干扰源清单，
    Agent 无法把它映射到自己定位到的源，因此不构成答案泄漏。

**无事件时不写 key**（而不是写空列表）：D1/D2 真值里根本没有 events 字段，写空列表会被
读成"监测网报告零干扰"，那是比现在更强的提示。不写 key 还能让 D1/D2 的
observations.json 与历史基线逐字节一致（它们的答案目录/评分保持可比）。
"""

from __future__ import annotations

PUBLIC_EVENT_FIELDS: tuple[str, ...] = (
    "event_id", "source_id", "mechanism", "impact_class")
WITHHELD_EVENT_FIELDS: tuple[str, ...] = (
    "aimed", "victim_id", "power_at_victim_dbm", "env_over_noise_db")

# 机理暴露粒度（题面可调，切换只需重跑 expose_evidence.py，不必重生成数据）：
#   full   —— 按 §7 schema 暴露 mechanism（jam / spoof / …）。
#   coarse —— mechanism 统一写 "interference"：只告诉"有源在压合作链路"，不区分
#             压制/欺骗，于是 is_spoof 与逐源 intent 不被直接送出。
#   none   —— 不带 mechanism 键，事件只剩"存在 + 影响等级"。
MECHANISM_MODES: tuple[str, ...] = ("full", "coarse", "none")

# 注意：这段说明会进入 observations.json，`audit_dataset.py` 的 v2-c 信息隔离检查会
# 对 JSON 全文做子串黑名单（emitter / intent / is_jamming / freq_offset_hz），
# 所以这里的措辞必须避开这些词。
NOTE = ("cooperative-link event log: each entry is an interference event reported by the "
        "cooperative ground station's own link-quality monitor (public evidence). An entry "
        "proves that a radiator is degrading the cooperative link, and its mechanism is "
        "direct behavioural evidence. This log is a PARTIAL view: it only sees radiators "
        "that reach the cooperative link, so it is NOT a census of the scene. Aiming and "
        "target-identity fields are withheld.")



def public_interference_events(truth: dict, mechanism: str = "full") -> list[dict]:
    """真值 -> Agent 可见事件列表（只保留 PUBLIC_EVENT_FIELDS 中真值里存在的键）。"""
    if mechanism not in MECHANISM_MODES:
        raise ValueError(f"mechanism must be one of {MECHANISM_MODES}")
    out: list[dict] = []
    for ev in (truth.get("interference_events") or []):
        if not isinstance(ev, dict):
            continue
        item = {k: ev[k] for k in PUBLIC_EVENT_FIELDS if k in ev}
        if "mechanism" in item:
            if mechanism == "coarse":
                item["mechanism"] = "interference"
            elif mechanism == "none":
                item.pop("mechanism")
        out.append(item)
    return out


def attach_public_events(obs: dict, truth: dict, mechanism: str = "full") -> int:
    """就地把公开事件写进 observations.json 的 dict；返回写入的事件数。

    无事件时不写 key（保持 D1/D2 与历史数据逐字节一致）。切换粒度时先删旧键再写，
    避免残留上一档的内容。
    """
    events = public_interference_events(truth, mechanism)
    obs.pop("interference_events", None)
    obs.pop("interference_events_note", None)
    if events:
        obs["interference_events"] = events
        obs["interference_events_note"] = NOTE
    return len(events)
