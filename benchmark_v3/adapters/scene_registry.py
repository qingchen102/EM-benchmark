"""新增场景（卡 3/7/10/11）的生成参数登记表 —— 2026-10-14 试用批次。

为什么单独一个文件：`gen_dataset.py`（生成器）与 `noise_profiles.py`（噪声档位）都要读它，
放在任何一边都会形成环。这里只放**生成参数与预设难度**，不含真值逻辑（真值仍由
`mechanism.derive()` 从实际参数派生）。

设计原则
--------
四个新档**一律复用现有信号模型与平台能力**：不新增机理、不换载频（D-2 不动）、不改 10 ms 窗长
（D-4 不动）。因此它们是既有能力的**参数重排**，不是新物理；这也意味着它们能立刻生成、立刻评测。

  d5 = 卡 11 建筑工地/城管    脉冲家族（Welder/VFD/Corona）为主 + 城市 RT + 楼后 NLoS
  d6 = 卡 7  电力线巡检       电晕/变频器簇（Corona/VFD/Welder）+ 少量单音
  d7 = 卡 3  蜂群/灯光秀      同质 UAV_video_OFDM 多实例（**分选题**，源数 3–5）
  d8 = 卡 10 滨水/江面        开阔近水面（低多径）+ ISM 共存源

`tier` 是**预设难度等级**（1 最易 … 4 最难，与 D1–D4 对齐），属主观预设，跑分后按实测调整。
`note` 记录该档与场景卡原文的差距（哪些能力还没实现、用什么替代）。

字段
----
card        场景卡编号与名称
tier        预设难度等级（1–4）
pool        干扰源 kind 池（必须都在 `em_signal_simulator_v3.emitters.EMITTERS` 内）
n_src       每样本源数范围 (lo, hi]，闭区间，均匀抽
target      合作目标 preset（考生要分析的"受害者"）
nlos        True = 目标取楼后 NLoS 位置、站点固定环布（同 D3/D4）；False = 开阔视距
tgt_power   目标发射功率范围 dBm
noise       噪声档位（复用 `noise_profiles` 的 d2/d3 菜单）
force_tier  强制频偏档位（0 深重叠 / None 由 `_draw_ifw_freq` 随机）；d7 同频复用必须 0
note        与场景卡原文的差距说明
"""

SCENES: dict[str, dict] = {
    "d5": dict(
        card="卡 11 建筑工地/城管",
        tier=3,
        pool=("Welder", "VFD_noise", "Corona", "Microwave", "DECT", "WiFi_burst"),
        n_src=(2, 4),
        target="UAV_video_OFDM",
        nlos=True,
        tgt_power=(0, 10),
        noise="d3",
        force_tier=None,
        note="脉冲家族为主 + 城市 RT；瞄准层（aim_at）未实现，先用脉冲簇替代",
    ),
    "d6": dict(
        card="卡 7 电力线巡检",
        tier=2,
        pool=("Corona", "VFD_noise", "Welder", "Tone_jam"),
        n_src=(1, 3),
        target="UAV_video_OFDM",
        nlos=True,
        tgt_power=(0, 10),
        noise="d3",
        force_tier=None,
        note="电晕/变频器脉冲簇 + 走廊低密度；脉冲簇时序（§9-3/4）未实现",
    ),
    "d7": dict(
        card="卡 3 蜂群/灯光秀",
        tier=4,
        pool=("UAV_video_OFDM",),
        n_src=(3, 5),
        target="UAV_video_OFDM",
        nlos=False,
        tgt_power=(10, 20),
        noise="d2",
        force_tier=0,
        note="同质多实例（分选）；源数 3–5 属有意压力测试（仪器 max_sources 可能卡上限）",
    ),
    "d8": dict(
        card="卡 10 滨水/江面",
        tier=2,
        pool=("WiFi_burst", "Bluetooth_FHSS", "Microwave", "DECT"),
        n_src=(1, 3),
        target="UAV_video_OFDM",
        nlos=False,
        tgt_power=(10, 20),
        noise="d2",
        force_tier=None,
        note="开阔近水面（低多径）；水面色片未入 RT（D-16 待定），暂以地面镜面近似",
    ),
}

# 档位 → 场景 id（便于脚本遍历；顺序 = 建议生成顺序）
SCENE_IDS: tuple[str, ...] = ("d5", "d6", "d7", "d8")


def tier_label(scene_id: str) -> str:
    """把预设难度等级写成人读串（跑分后按实测调整）。"""
    return f"T{SCENES[scene_id]['tier']}"


def describe(scene_id: str) -> str:
    s = SCENES[scene_id]
    return (f"[{scene_id}] {s['card']}  预设难度={tier_label(scene_id)}  "
            f"池={len(s['pool'])} 类  源数={s['n_src'][0]}–{s['n_src'][1]}  "
            f"目标={s['target']}  {'楼后NLoS' if s['nlos'] else '开阔视距'}  "
            f"噪声档={s['noise']}  备注：{s['note']}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for sid in SCENE_IDS:
        print(describe(sid))
