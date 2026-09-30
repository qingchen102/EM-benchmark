"""v3 信号源 → 师兄平台信道的融合适配器（不改平台源码）。

用途：
    把 benchmark_v3 的辐射体输出当作"发射源"，送进「城市低空电磁信号仿真平台」
    （已移入 benchmark_v3/城市低空电磁信号仿真平台/）的传播信道做环境渲染。

分工：
    你的 v3 = 信号源（决定发射什么）：16 辐射体，基带 IQ 按 power_dbm 标定。
    他的平台 = 环境/信道（信号怎么传）：Backbone 解链路（free_space/tapped/sionna_rt），
                apply_channel 卷积，receive 接收，run_scenario 导出 public/private。

做法（不改平台源码）：
    run_scenario 内部用 `runner.generate(tx, fs, indices, seed)` 生成每个发射源的波形
    （见 城市低空电磁信号仿真平台/src/cityrf/runner.py）。
    本模块在运行时把 `cityrf.runner.generate` 替换成 `v3_generate`，再调用原样的
    `run_scenario`——信道求解、卷积、接收、导出全部复用平台代码，只换了"源"。
"""
from __future__ import annotations

import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent                                  # benchmark_v3/
_PLATFORM_SRC = _V3 / "城市低空电磁信号仿真平台" / "src"
_V3SIM = _V3 / "simulation"
_V2SIM = _V3.parent / "benchmark_v2" / "simulation"
for _p in (str(_PLATFORM_SRC), str(_V3SIM), str(_V2SIM), str(_V2SIM.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cityrf.config import (Antenna, Component, Emitter, Frontend,  # noqa: E402
                           Propagation, Receiver, Scenario, Task)
from cityrf import runner  # noqa: E402

from em_signal_simulator_v3.emitters import EMITTERS, INTENT, generate_emitter  # noqa: E402
from em_signal_simulator.baseband import generate_baseband  # noqa: E402


# --------------------------------------------------------------------------- #
#  1. 你的信号源：把发射源 id 解析成 v3 辐射体，产出按 power_dbm 标定的 IQ
# --------------------------------------------------------------------------- #
def v3_generate(tx, fs: float, indices, seed: int) -> np.ndarray:
    """替代 runner.generate 的源：tx.id 是 v3 辐射体名（或特殊 target_qpsk）。

    返回在场景采样率 fs 上的复基带 IQ，长度 = len(indices)，功率 = tx.power_dbm 瓦。
    """
    n = np.asarray(indices, dtype=np.int64)
    # 确定性随机源：由平台 seed + 发射源 id 派生，保证可复现
    rng = np.random.default_rng((seed ^ 0x9E3779B9 ^ zlib.crc32(tx.id.encode())) & 0xFFFFFFFF)
    if tx.id in EMITTERS:
        iq = generate_emitter(tx.id, rng, length=len(n))["iq"]
    elif tx.id == "target_qpsk":
        iq = generate_baseband("QPSK", len(n), rng=rng, bandwidth_normalized=0.5)
    else:
        raise ValueError(f"未知 v3 信号源 id: {tx.id}")
    iq = np.asarray(iq, dtype=complex)
    # 标定到发射功率（平台约定：active mean power = 10^((power_dbm-30)/10) 瓦）
    p_w = 10.0 ** ((tx.power_dbm - 30.0) / 10.0)
    rms = float(np.mean(np.abs(iq) ** 2) + 1e-30)
    return (iq * np.sqrt(p_w / rms)).astype(np.complex128)


# --------------------------------------------------------------------------- #
#  2. 用 v3 源清单构造平台的 Scenario
# --------------------------------------------------------------------------- #
def source_at(emitter: str, doa_deg: float, radius: float, height: float,
              power_dbm: float, role: str = "interference") -> dict:
    """把 v3 DOA/距离/高度转成平台的世界坐标（free_space 下几何到达角 = doa 摆角）。"""
    import math
    az = math.radians(doa_deg)
    return {
        "id": emitter,
        "emitter": emitter,
        "role": role,
        "position_m": (radius * math.cos(az), radius * math.sin(az), height),
        "power_dbm": power_dbm,
    }


def build_fusion_scenario(sources: list[dict], receivers: list[dict] | None = None, *,
                          receiver_position=(0.0, 0.0, 5.0),
                          fs: float = 2e6, duration: float = 0.05,
                          carrier: float = 2.4e9, backend: str = "free_space",
                          seed: int = 42, task: Task | None = None,
                          antenna: Antenna | None = None) -> Scenario:
    """构造平台 Scenario（不生成波形）。sources 每个含 id/role/position_m/power_dbm。

    receivers: 每项含 id/position_m；缺省时用单站 station_a（向后兼容单接收机）。
    task: 可选 Task（如把多个接收站链接到目标源），用于 cooperative_sources / public_tasks。
    antenna: 各接收站统一的阵列规格（为将来 sionna_rt 迁移保持同规格）；缺省 cross 8+8。
    """
    if receivers is None:
        receivers = [{"id": "station_a", "position_m": receiver_position}]
    emitters = []
    for s in sources:
        emitters.append(Emitter(
            id=s["id"], role=s.get("role", "interference"),
            position_m=tuple(s["position_m"]), orientation_rad=(0.0, 0.0, 0.0),
            power_dbm=s["power_dbm"],
            # 占位成分：只满足 schema 校验，真正波形由 v3_generate 生成
            components=[Component(kind="noise", bandwidth_hz=200000)],
        ))
    rxs = [Receiver(id=r["id"], position_m=tuple(r["position_m"]),
                    bandwidth_hz=fs * 0.5 * 0.9, frontend=Frontend(),
                    antenna=antenna or Antenna())
           for r in receivers]
    propagation = Propagation(backend=backend)
    return Scenario(name="fusion_multi", seed=seed, duration_s=duration,
                    sample_rate_hz=fs, carrier_frequency_hz=carrier,
                    max_working_mb=4096, emitters=emitters, receivers=rxs,
                    propagation=propagation, tasks=[task] if task else [])


# --------------------------------------------------------------------------- #
#  3. 运行融合：monkey-patch 源 + 复用平台 run_scenario + 写 v3 答案轴边车
# --------------------------------------------------------------------------- #
def run_fusion(sources: list[dict], out_dir: str | Path, *,
               overwrite: bool = False, **build_kw) -> tuple:
    """sources 进 v3 源 + 平台信道渲染，产出 capture/。返回 (summary, scenario, sources)。"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在且未开 overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    scenario = build_fusion_scenario(sources, **build_kw)

    orig = runner.generate
    runner.generate = v3_generate          # 运行时只换"源"，不改平台源码
    try:
        summary = runner.run_scenario(scenario, out_dir)
    finally:
        runner.generate = orig

    # v3 答案轴边车：辐射体名/敌我意图/角色/DOA 摆角，存进私有侧供后续评分
    truth = {
        "schema_version": "v3-fusion-0.1",
        "sources": [{
            "id": s["id"], "emitter": s.get("emitter", s["id"]),
            "role": s.get("role", "interference"),
            "intent": s.get("intent") or INTENT.get(s["emitter"], "benign")
                     if s.get("emitter") in INTENT else s.get("intent", "benign"),
            "power_dbm": s["power_dbm"],
            "position_m": list(s["position_m"]),
        } for s in sources],
    }
    (out_dir / "private" / "fusion_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary, scenario, truth


def _polar(az_deg: float, radius: float, height: float) -> tuple:
    """世界坐标（水平方位摆角 → x/y）。"""
    a = np.deg2rad(az_deg)
    return (float(radius * np.cos(a)), float(radius * np.sin(a)), float(height))


def generate_fusion_sample(rng=None, *, backend: str = "free_space",
                           n_interferers=(2, 5), n_receivers=(2, 4),
                           target: str = "target_qpsk", target_power_dbm: float = 0.0,
                           source_radius=(25.0, 120.0), source_height=(10.0, 60.0),
                           power_range_dbm=(0.0, 20.0),
                           receiver_radius=(60.0, 200.0), seed: int = 42) -> tuple:
    """随机造一个「多源 + 多接收机」高难度样例。

    返回 (sources, receivers, difficulty)：
      - sources : 1 目标 + 若干干扰（16 辐射体池抽样、敌我混合），几何错开、功率/距离随机；
      - receivers: 若干监测站（地面高度、围绕源区形成测向阵网——多站定位成立）；
      - difficulty: 难度清单（源数/站数/backend/辐射体/功率范围），供 A/B 与记录。
    计数区间为闭区间（n_interferers=(2,5) 即 2~5 个干扰）。
    """
    rng = rng or np.random.default_rng(seed)
    n_int = int(rng.integers(n_interferers[0], n_interferers[1] + 1))
    n_rx = int(rng.integers(n_receivers[0], n_receivers[1] + 1))

    pool = list(EMITTERS.keys())
    # 目标：相对近/高，作为协作受保护源
    sources = [source_at(target, float(rng.uniform(-180, 180)),
                         float(rng.uniform(20, source_radius[0] + 20)),
                         float(rng.uniform(30, 60)),
                         target_power_dbm, role="communication")]
    radiators = [str(x) for x in rng.choice(pool, size=n_int, replace=False)]
    for name in radiators:
        sources.append(source_at(name, float(rng.uniform(-180, 180)),
                                 float(rng.uniform(*source_radius)),
                                 float(rng.uniform(*source_height)),
                                 float(rng.uniform(*power_range_dbm)),
                                 role="interference"))

    receivers = [{"id": f"station_{chr(ord('a') + i)}",
                  "position_m": _polar(float(rng.uniform(-180, 180)),
                                       float(rng.uniform(*receiver_radius)), 5.0)}
                 for i in range(n_rx)]

    difficulty = {
        "backend": backend, "seed": seed, "target": target,
        "num_sources": len(sources), "num_interferers": n_int,
        "num_receivers": n_rx, "radiators": radiators,
        "source_radius": list(source_radius), "power_range_dbm": list(power_range_dbm),
    }
    return sources, receivers, difficulty


def run_fusion_multi(sources: list[dict], receivers: list[dict], out_dir: str | Path, *,
                     overwrite: bool = False, difficulty: dict | None = None,
                     **build_kw) -> tuple:
    """多源 + 多接收机融合：构造 Scenario → 复用平台 run_scenario → 写 truth + difficulty。

    自动加一个 Task：把全部接收站链接到"协作目标"（role=communication 的第一个源），
    使 public_tasks / cooperative_sources 如实反映"协作目标 + 多站监测网"。
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在且未开 overwrite: {out_dir}")
        shutil.rmtree(out_dir)

    target_id = next((s["id"] for s in sources if s.get("role") == "communication"),
                     sources[0]["id"])
    task = Task(id="monitor",
                receiver_ids=[r["id"] for r in receivers],
                desired_emitter_ids=[target_id],
                instruction="Use all monitoring stations' IQ to analyze interference "
                            "and localize the sources.")
    scenario = build_fusion_scenario(sources, receivers, task=task, **build_kw)

    orig = runner.generate
    runner.generate = v3_generate          # 运行时只换"源"，不改平台源码
    try:
        summary = runner.run_scenario(scenario, out_dir)
    finally:
        runner.generate = orig

    truth = {
        "schema_version": "v3-fusion-multi-0.1",
        "sources": [{
            "id": s["id"], "emitter": s.get("emitter", s["id"]),
            "role": s.get("role", "interference"),
            "intent": INTENT.get(s["emitter"], "benign")
                     if s.get("emitter") in INTENT else s.get("intent", "benign"),
            "power_dbm": s["power_dbm"],
            "position_m": list(s["position_m"]),
        } for s in sources],
        "receivers": [{"id": r["id"], "position_m": list(r["position_m"])} for r in receivers],
    }
    if difficulty:
        truth["difficulty"] = difficulty
    (out_dir / "private" / "fusion_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    if difficulty:
        (out_dir / "private" / "difficulty.json").write_text(
            json.dumps(difficulty, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary, scenario, truth


def report_multi(out_dir: str | Path) -> None:
    """多源多站结果简述：每站看到哪些源、各几条路径。"""
    out_dir = Path(out_dir)
    t = json.loads((out_dir / "private" / "truth.json").read_text(encoding="utf-8"))
    print(f"  物理源数={len(t['sources'])}  站数={len(t['stations'])}")
    for st in t["stations"]:
        per = [f"{s['emitter_id']}({s['path_count']}径)" for s in st["sources"]]
        print(f"  站 {st['receiver_id']:10s}: " + "  ".join(per))


def report_links(out_dir: str | Path) -> None:
    """打印每链路路径数/时延（验证平台信道作用在 v3 源上）。"""
    out_dir = Path(out_dir)
    for npz in sorted((out_dir / "private").glob("channel_*.npz")):
        with np.load(npz, allow_pickle=False) as d:
            coef = d["coefficients"]
            delays = d["delays_s"]
        print(f"  {npz.name:28s} paths={coef.shape[1]}  "
              f"delay(max)={float(np.max(delays))*1e9:6.1f} ns")


# --------------------------------------------------------------------------- #
#  演示：1 目标(QPSK) + 2 干扰(GNSS_jam, Microwave)，free_space
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # `simple` → 单站回归(自由空间)；`rt` → Sionna RT 真实多径(多源×多站)；默认 → 自由空间多源多站
    if "simple" in sys.argv[1:]:
        sources = [
            source_at("target_qpsk", doa_deg=0.0,  radius=30.0, height=45.0,
                      power_dbm=0, role="communication"),
            source_at("GNSS_jam",   doa_deg=-40., radius=30.0, height=25.0, power_dbm=20),
            source_at("Microwave",  doa_deg=30.0, radius=28.0, height=10.0, power_dbm=15),
        ]
        out = _V3 / "_fusion_demo_simple"
        print("=== 简单融合（单接收机，回归）===")
        summary, scenario, truth = run_fusion(sources, out, overwrite=True,
                                              backend="free_space", fs=2e6,
                                              duration=0.05, seed=42)
        print(f"  backend={summary['backend']}  源数={summary['emitter_count']}  "
              f"接收站={summary['receiver_count']}  path_count={summary['path_count']}")
        report_links(out)
        print("  输出目录:", out)
    elif "rt" in sys.argv[1:] or "sionna_rt" in sys.argv[1:]:
        # 用已在慕尼黑场景通过"天顶可见"校验的真实坐标（来自 example_case01），
        # 端点在室外、看得见天空，Sionna RT 才能求解。
        _RT_TX = [(-150.0, -100.0, 85.0), (150.0, -80.0, 55.0), (-80.0, 40.0, 35.0)]
        _RT_RX = [(80.0, 60.0, 5.0), (0.0, 100.0, 5.0), (100.0, 0.0, 65.0)]
        names = ["target_qpsk", "GNSS_jam", "Chirp_jam"]
        roles = ["communication", "interference", "interference"]
        sources = [{"id": n, "emitter": n, "role": r, "position_m": p, "power_dbm": 20.0}
                   for n, r, p in zip(names, roles, _RT_TX)]
        receivers = [{"id": f"station_{chr(ord('a') + i)}", "position_m": p}
                     for i, p in enumerate(_RT_RX)]
        out = _V3 / "_fusion_rt"
        print("=== Sionna RT 融合（真实多径，多源×多站）===")
        for s in sources:
            print(f"  {s['id']:14s} role={s['role']:13s} intent={INTENT.get(s['emitter'],'-'):11s} "
                  f"pos={tuple(s['position_m'])}")
        summary, scenario, truth = run_fusion_multi(
            sources, receivers, out, overwrite=True,
            backend="sionna_rt", fs=2e6, duration=0.05, seed=1, antenna=Antenna())
        print("\n=== Sionna RT 多站信道（每条链路多条真实反射径）===")
        print(f"  backend={summary['backend']}  源数={summary['emitter_count']}  "
              f"接收站={summary['receiver_count']}  path_count(加权)={summary['path_count']}")
        report_multi(out)
        print("  输出目录:", out)
        print("  (运行方式：需在 WSL 内用 .venv-rt 的 python 跑，并带 LD_LIBRARY_PATH)")
    else:
        rng = np.random.default_rng(7)
        sources, receivers, difficulty = generate_fusion_sample(
            rng, backend="free_space",
            n_interferers=(4, 6), n_receivers=(3, 5),   # 闭区间 → 4~5 干扰、3~4 站
            seed=7)
        out = _V3 / "_fusion_multi"
        print("=== 高难度融合（多源 + 多接收机）===")
        for s in sources:
            print(f"  {s['id']:14s} role={s['role']:13s} intent={INTENT.get(s['emitter'],'-'):11s} "
                  f"pos={tuple(round(v,1) for v in s['position_m'])}  power={s['power_dbm']:4.0f}dBm")
        for r in receivers:
            print(f"  {r['id']:14s} (接收站)  pos={tuple(round(v,1) for v in r['position_m'])}")

        summary, scenario, truth = run_fusion_multi(
            sources, receivers, out, overwrite=True, difficulty=difficulty,
            backend="free_space", fs=2e6, duration=0.05, seed=7,
            antenna=Antenna(geometry_type="rectangular", num_rows=1, num_cols=8))
        print("\n=== 平台多站信道作用结果 ===")
        print(f"  backend={summary['backend']}  源数={summary['emitter_count']}  "
              f"接收站={summary['receiver_count']}  path_count(加权)={summary['path_count']}")
        report_multi(out)
        print("\n  difficulty:", json.dumps(difficulty, ensure_ascii=False))
        print("  输出目录:", out)