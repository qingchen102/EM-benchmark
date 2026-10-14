"""四档"背景噪声"档位表 —— 平台 Frontend 噪声开关的唯一真实来源。

【2026-10-08/09 夜 大修】从“每档一个固定档位”改成「配方菜单 + 按站均衡抽签」：
  * 热噪（thermal）四档永开、不可协商 —— 它是接收机的物理下限，也是唯一的"参照系"；
  * 其余三个旋钮（colored / impulse / correlated）按 **站**（不是按样本）抽签，
    每档留出"只有热噪"的配额，让"有背景噪声的站"与"纯热噪的站"出现在同一份数据里，
    从而把"背景噪声的影响"与"数据集的难度"解耦；
  * 抽签是**确定性**的（见 `plan_sample`），不需要随机种子，重放可逐位复现；
  * 相位噪声（phase）是 d2–d4 的固有项（d1 无），不参与抽签。

【为什么按站而不是按样本】一个样本里的 4 个站就是 4 台不同位置的接收机。
按样本随机会把"整份数据都有某种噪声"与"某台接收机有某种噪声"混成一个概念，
而 S0/S1/S2 曲线只在站数上变化 —— 必须保证**每个站位置**的配方分布一致，
否则 station_a 系统性偏干净会直接污染 S0→S2 信息增益曲线。

【平台噪声机制审计（2026-10-08/09 夜，源码级；user m03195 要求“不要完全认为平台没有问题”）】
  * `receiver.py:9 _colored(indices, fs, bandwidth_hz, seed)`：白噪过
    `firwin(65, min(bw/2, 0.45·fs))` 后**归一到时域单位平均功率**（`y/rms`），
    不是归一到谱密度。所以 `colored_noise_db` 的语义是
    "有色分量的**时域总功率** / 热噪的**时域总功率**"，而它集中在 ±100 kHz、
    热噪摊在 ±2 MHz —— 故 `colored_noise_db = 0` 就已把**带内**底抬高 **+5.45 dB**。
    实测映射 `rise_dB ≈ 10·log10(1 + 2.47·10^(db/10))`（0→+5.45、5→+9.52、10→+14.17，
    斜率 ~0.87 dB/dB）。→ **不是 bug，但语义必须写清楚**，否则档位刻度会系统性偏大 ~5 dB。
  * `receiver.py:19 _impulses(...)`：占空比 duty = rate·width/fs，
    等效抬底 = 10·log10(1 + duty·10^(pdb/10))；pdb = 25 → 只有 **+1.08 dB**。
    行为与文档一致；它的价值在**时域结构**（`analyze_time_frequency_station` 的
    duty / CV / burst 指纹），不在 PSD 峰判据。
  * `receiver.py:46 if f.phase_noise_enabled and f.phase_noise_deg > 0`：
    **`phase_noise_deg` 是每个采样点的相位步长 σ，不是总相位**；
    `common = exp(1j·cumsum(step))` 是随机游走。10 ms / 4 MHz = 40000 点，
    0.01° 步长 → 积分相噪 0.01·√40000 = **2° RMS**（合理的廉价本振）；
    1.0° 步长 → 200° RMS、瞬时频偏随机游走达 ±MHz，会把源频谱抹开
    —— 这是更早几轮“相噪伤害最大”的表象来源之一。
  * `receiver.py:57 n0 = ... if f.noise_enabled else 0.`、`:60 if f.colored_noise_enabled`、
    `:69 if f.impulse_noise_enabled`：**colored / impulse / phase 三个开关不受
    `noise_enabled` 总开关管辖**（共模白噪在 `if n0:` 内，故 n0 = 0 时自动关）。
    → 造"无噪声参考"时必须逐个关；只关 `noise_enabled` 会漏掉三项。
    （2026-10-08/09 夜已修 `run_capture_agent._build_target_reference`，见该函数注释里的
      A/C 两模型对比；结论是**只关加性项、保留 phase**，因为数据与参考走同一台
      接收机、同一时刻，相位误差是共享的乘性项，关掉它反而在残差里留下目标残影。）

【D1 保持恒等】D1 是唯一被证明"能精准测能力"的档（detection 0.083 → 1.00、虚警 0），
四档一起加噪就没有参照系。D1 的字段集必须是**空**。

【档位为什么不取“强”】2026-10-08 夜的扫描把底抬 +5.4 / +9.5 / +14.1 / +19 dB，
`capture_tools` 的残差峰输出几乎不变（d2_00001 真 3 / 假 15 四档完全一致）。
原因是残差底由**参考直减的失配能量（~-88 dBm）**主导，比热噪底高 ~19 dB。
所以噪声是**真实性 / 时域结构**旋钮，**不是难度旋钮**；真正的难度瓶颈是
**残差提取链路的泄漏底**。把底抬 +18 dB 只会白白改写真值 INR / impact_class 的语义。
（注：该结论的早期口径只看了 station_a 的峰计数；加性噪声对**参考减法**的影响
  在 2026-10-08/09 夜后半段用“模型 C”重测，见 run_capture_agent 里的注释。）

基准量：未滤波白噪 `n0·fs = -102.95 dBm`；经 1.62 MHz 等效带通后 `-106.88 dBm`
（= 真值的 `noise_floor_dbm: -106.9`，即 `-174 + 10log10(1.62e6) + 5`）。
"""

from __future__ import annotations

import copy
import math

# ---------------------------------------------------------------------------
# 刻度常数（唯一调节处 —— 改档位就是改这里）
# ---------------------------------------------------------------------------

B_SLOTS = 10        # 每档的抽签槽位数
SLOT_STRIDE = 7     # 站间步长；必须与 B_SLOTS 互素（7 与 10 互素 → 同一样本内 4 站取到 4 个不同槽）
SAMPLE_STRIDE = 3   # 样本间步长；也须与 B_SLOTS 互素（3 与 10 互素）
                    # → ① 每个站位置在 50 样本上遍历全部 10 个槽各 5 次（配方占比 == 菜单权重）
                    #    ② 每样本用到的 4 个槽是 {0,1,4,7}+t 的平移、t 随样本变化
                    #       → 各样本的背景噪声配方构成不同，不是一个构成复制 50 遍

COLORED_BW_HZ = 200e3
# 刻度按"实测抬底"反推：rise = 10log10(1 + 2.47·10^(db/10)) → 目标是 d2 ≈ +1.4 dB、d3/d4 ≈ +2.5 dB
COLORED_DB = {"d2": -8.0, "d3": -5.0, "d4": -5.0}
IMPULSE_RATE_HZ = 200.0
IMPULSE_WIDTH_US = 5.0
IMPULSE_POWER_DB = 25.0     # duty = 1e-3 → 只抬 +1.08 dB；价值在时域结构
CORR_RHO = 0.05             # 参考侧 n0=0 时共模支路被跳过 → 它的伤害是真实的信号通路效应，故取小
PHASE_DEG = {"d2": 0.01, "d3": 0.01, "d4": 0.01}   # 每采样点步长 → 10 ms 积分 ~2° RMS

ARCHETYPE_NAMES = ("thermal", "colored", "impulse", "correlated")

# 每档的配方菜单：[(archetype, 权重)]，权重和必须 == B_SLOTS
MENUS: dict[str, tuple] = {
    "d1": (("thermal", 10),),
    "d2": (("thermal", 6), ("colored", 4)),
    "d3": (("thermal", 4), ("colored", 3), ("impulse", 3)),
    "d4": (("thermal", 2), ("colored", 2), ("impulse", 3), ("correlated", 3)),
}

DIFFICULTIES = ("d1", "d2", "d3", "d4")

# 新增场景（卡 3/7/10/11）→ 复用哪个现有档位的噪声菜单（2026-10-14 试用批次）。
# 只做别名，不新增配方：噪声菜单是“按站随机”的，场景差异不等于噪声差异。
SCENE_ALIAS: dict[str, str] = {}
try:  # 登记表缺席也不影响四档（生成器一侧会先报错）
    from scene_registry import SCENES as _SC
    SCENE_ALIAS = {sid: str(s["noise"]) for sid, s in _SC.items()}
except Exception:  # pragma: no cover
    SCENE_ALIAS = {}


def _key(difficulty: str) -> str:
    k = str(difficulty).strip().lower()
    k = SCENE_ALIAS.get(k, k)
    if k not in MENUS:
        raise KeyError(f"未知难度档 {difficulty!r}，可选 {sorted(MENUS)} + 场景 {sorted(SCENE_ALIAS)}")
    return k


# ---------------------------------------------------------------------------
# 配方（archetype）→ Frontend 字段
# ---------------------------------------------------------------------------

def archetype_fields(name: str, difficulty: str) -> dict:
    """一个配方在某档下的完整 Frontend 字段集。

    **会显式写出三个加性开关的 True/False** —— 这是刻意的：D4 的源数据在
    `gen_dataset.py` 里自带 `correlated_noise_enabled=True / ρ=0.3`，若不显式
    置 False，抽到 colored/impulse 的站会同时带共模，档位表就失去意义。
    相噪不在这里（它是档的固有项，见 `base_fields`）。
    """
    k = _key(difficulty)
    if name not in ARCHETYPE_NAMES:
        raise KeyError(f"未知配方 {name!r}，可选 {list(ARCHETYPE_NAMES)}")
    f = {
        "noise_enabled": True,
        "colored_noise_enabled": False,
        "impulse_noise_enabled": False,
        "correlated_noise_enabled": False,
    }
    if name == "colored":
        f.update(
            colored_noise_enabled=True,
            colored_noise_db=COLORED_DB.get(k, -8.0),
            colored_bandwidth_hz=COLORED_BW_HZ,
        )
    elif name == "impulse":
        f.update(
            impulse_noise_enabled=True,
            impulse_rate_hz=IMPULSE_RATE_HZ,
            impulse_power_db=IMPULSE_POWER_DB,
            impulse_width_us=IMPULSE_WIDTH_US,
        )
    elif name == "correlated":
        f.update(correlated_noise_enabled=True, noise_correlation=CORR_RHO)
    return f


def base_fields(difficulty: str) -> dict:
    """档的固有项（不参与抽签）。d1 必须是空 —— 它是恒等参照系。"""
    k = _key(difficulty)
    if k == "d1":
        return {}
    return {"phase_noise_enabled": True, "phase_noise_deg": PHASE_DEG.get(k, 0.01)}


# ---------------------------------------------------------------------------
# 确定性抽签
# ---------------------------------------------------------------------------

def _slot_pattern(difficulty: str) -> tuple:
    """把菜单铺到 B_SLOTS 个槽位上（尽量均匀铺开，确定性）。

    做法：每个配方按自己的等间距 `(i+0.5)·B/n` 撒点，撞位就往后找空槽。
    d2 → [T,C,T,C,T,T,C,T,C,T]；d3 → [I,T,C,T,I,C,T,I,T,C]；
    d4 → [corr,I,T,C,corr,I,corr,T,C,I]
    """
    k = _key(difficulty)
    menu = MENUS[k]
    total = sum(w for _, w in menu)
    if total != B_SLOTS:
        raise ValueError(f"{k} 菜单权重和 {total} ≠ B_SLOTS={B_SLOTS}")
    slots: list = [None] * B_SLOTS
    for name, cnt in menu:
        for i in range(int(cnt)):
            pos = int((i + 0.5) * B_SLOTS / cnt) % B_SLOTS
            while slots[pos] is not None:
                pos = (pos + 1) % B_SLOTS
            slots[pos] = name
    if any(s is None for s in slots):
        raise RuntimeError(f"{k} 槽位未铺满：{slots}")
    return tuple(slots)


def plan_sample(difficulty: str, station_ids, sample_index: int) -> dict:
    """给一个样本的若干站分配配方（确定性、无需种子）。

    `slot = (SAMPLE_STRIDE·sample_index + SLOT_STRIDE·k) mod B_SLOTS`，k = 站在该样本里的序号。
    两个步长都与 B_SLOTS 互素，于是：① 同一样本内各站取到**不同**槽；
    ② 每个站位置在 50 个样本上遍历全部 10 个槽各 5 次
       → 每个站位置的配方分布 == 菜单权重（`self_check()` 会验）；
    ③ 每样本的配方构成随样本变化（避免把"背景噪声"与"某批样本"混同）。
    """
    pat = _slot_pattern(difficulty)
    return {sid: pat[(SAMPLE_STRIDE * int(sample_index) + SLOT_STRIDE * k) % B_SLOTS]
            for k, sid in enumerate(station_ids)}


def plan_distribution(difficulty: str, station_ids, n_samples: int) -> dict:
    """核对用：每个站位置的配方计数（应当逐站相同且等于菜单权重·n/B）。"""
    counts = {sid: {} for sid in station_ids}
    for i in range(int(n_samples)):
        for sid, arch in plan_sample(difficulty, station_ids, i).items():
            counts[sid][arch] = counts[sid].get(arch, 0) + 1
    return counts


def control_samples(difficulty: str, station_ids, n_samples: int) -> int:
    """四站全热噪的样本数（档内"对照组"）。"""
    n = 0
    for i in range(int(n_samples)):
        if set(plan_sample(difficulty, station_ids, i).values()) == {"thermal"}:
            n += 1
    return n


# ---------------------------------------------------------------------------
# 写进接收机
# ---------------------------------------------------------------------------

def _set_fields(receiver, fields: dict) -> dict:
    frontend = getattr(receiver, "frontend", None)
    target = frontend if frontend is not None else receiver
    for name, value in fields.items():
        if not hasattr(target, name):
            raise AttributeError(f"Frontend 没有字段 {name!r}（档位表与平台 config.py 脱同步）")
        setattr(target, name, value)
    return fields


def apply_station(receiver, difficulty: str, archetype: str, override: dict | None = None) -> dict:
    """把某站抽到的配方写进该接收机的 `frontend`，返回实际生效字段（落盘留证）。"""
    fields = archetype_fields(archetype, difficulty)
    fields.update(base_fields(difficulty))
    if override:
        fields.update(override)
    return _set_fields(receiver, fields)


def apply_noise_profile(receiver, difficulty: str, override: dict | None = None) -> dict:
    """**探针专用**接口：`override` 全权接管（只写 override 里给出的字段）。

    与 `apply_station` 的区别：这里不碰 override 未提到的字段 —— 所以变体扫描
    不会被档位菜单干扰。（regen_noise.py 的主路径用 `apply_station`。）
    """
    if not override:
        raise ValueError("apply_noise_profile 是探针接口，必须给 override；"
                         "常规重放请用 apply_station(difficulty, archetype)")
    return _set_fields(receiver, dict(override))


def model_label(difficulty: str) -> str:
    """真值 `noise.model` 字段用的一行标签。"""
    k = _key(difficulty)
    parts = ["thermal"]
    for name, w in MENUS[k]:
        if name != "thermal":
            parts.append(name)
    if k != "d1" and base_fields(k).get("phase_noise_enabled"):
        parts.append("phase")
    return "+".join(parts)


def analytic_floor_rise_db(fields: dict, fs: float = 4e6, rx_bw: float = 1.62e6) -> dict:
    """解析预估抬底（**只作对照，不作判据**）。

    有色那一项用实测闭式 `10log10(1 + 2.47·10^(db/10))`（2.47 = 实测拟合常数，
    来自"`_colored` 归一到时域单位功率"这一平台事实，解析式给不出该常数）。
    """
    duty = 0.0
    if fields.get("impulse_noise_enabled"):
        # 占空比 = 每秒脉冲数 × 脉宽(秒)；**不要再除 fs**（fs 已在宽度里体现）
        duty = float(fields.get("impulse_rate_hz", 0.0)) * float(fields.get("impulse_width_us", 0.0)) * 1e-6
        duty = min(duty, 1.0)
    imp_ratio = duty * 10 ** (float(fields.get("impulse_power_db", 0.0)) / 10.0)
    col_db = float(fields.get("colored_noise_db", 0.0)) if fields.get("colored_noise_enabled") else None
    col_rise = 10.0 * math.log10(1.0 + 2.47 * 10 ** (col_db / 10.0)) if col_db is not None else 0.0
    imp_rise = 10.0 * math.log10(1.0 + imp_ratio)
    return {
        "duty": duty,
        "impulse_ratio": imp_ratio,
        "impulse_rise_db": imp_rise,
        "colored_noise_db": col_db,
        "colored_rise_db": col_rise,
        "total_rise_guess_db": col_rise + imp_rise,
        "note": "只作对照；真实抬底必须用 zero-input receive() 实测（噪声先于 IMD3/软限幅）",
    }


def describe() -> str:
    """人类可读的档位表（写日志 / 文档用）。"""
    lines = []
    for diff in DIFFICULTIES:
        menu = ", ".join(f"{n}×{int(w)}/{B_SLOTS}" for n, w in MENUS[diff])
        lines.append(f"[{diff}] model={model_label(diff)}  菜单: {menu}")
        lines.append(f"    槽位: {list(_slot_pattern(diff))}")
        if diff == "d1":
            lines.append("    （保持 Frontend 默认：纯白热噪，恒等参照系）")
            continue
        lines.append(f"    固有项: {base_fields(diff)}")
        for name in ("colored", "impulse", "correlated"):
            if any(n == name for n, _ in MENUS[diff]):
                f = archetype_fields(name, diff)
                lines.append(f"    {name}: " + ", ".join(f"{k}={v}" for k, v in f.items()
                                                         if k.endswith(("_db", "_hz", "_us", "correlation"))
                                                         and v is not None and not isinstance(v, bool)))
    return "\n".join(lines)


def summary_line() -> str:
    """一行档位摘要（日志用）。"""
    parts = []
    for diff in DIFFICULTIES:
        menu = " + ".join(f"{n} {int(w*100/B_SLOTS)}%" for n, w in MENUS[diff] if n != "thermal")
        parts.append(f"{diff}: " + ("纯热噪(参照系)" if not menu else f"热噪 + {menu}"))
    return "[档位] " + " | ".join(parts)


def self_check(n_samples: int = 50, n_stations: int = 4) -> None:
    """机器化纪律：① 菜单权重和 == B_SLOTS；② 站间分布一致；③ D1 恒等。"""
    stations = [f"station_{c}" for c in "abcd"[:n_stations]]
    for diff in DIFFICULTIES:
        dist = plan_distribution(diff, stations, n_samples)
        first = dist[stations[0]]
        for sid, c in dist.items():
            if c != first:
                raise AssertionError(f"{diff} 站间配方分布不一致：{sid}={c} vs {stations[0]}={first}")
        for name, w in MENUS[diff]:
            want = int(round(n_samples * w / B_SLOTS))
            if first.get(name, 0) != want:
                raise AssertionError(f"{diff}/{name} 计数 {first.get(name, 0)} ≠ 期望 {want}")
        # 同一样本内各站必须取到不同槽
        for i in range(n_samples):
            plan = plan_sample(diff, stations, i)
            if len(set(plan.values())) == 1 and len(MENUS[diff]) > 1:
                raise AssertionError(f"{diff} 样本 {i} 四站同配方：{plan}")
        if diff == "d1" and base_fields("d1") != {}:
            raise AssertionError("D1 必须保持干净（唯一参照系）")
    assert base_fields("d1") == {} and archetype_fields("thermal", "d1").get("colored_noise_enabled") is False


if __name__ == "__main__":
    print(describe())
    print()
    print(summary_line())
    print()
    self_check()
    print("self_check ✅ 菜单权重、站间均衡、D1 恒等 全部通过")
    print()
    for d in DIFFICULTIES:
        st = ["station_a", "station_b", "station_c", "station_d"]
        print(f"[{d}] 前 8 个样本的分配:")
        for i in range(8):
            print(f"    sample {i:2d}: {plan_sample(d, st, i)}")
