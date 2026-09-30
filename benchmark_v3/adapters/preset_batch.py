"""预设/自由批量生成：把一批场景写进平台 `runs/`（网页自动枚举）。

用法：
    # free_space：本机任意 python（含 Windows python）
    python preset_batch.py --backend free_space
    # sionna_rt：需在 WSL 的 .venv-rt 里跑并带 LLVM 库路径
    #   wsl -d Ubuntu -e bash -lc 'export LD_LIBRARY_PATH=/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH; \
    #     <platform>/.venv-rt/bin/python preset_batch.py --backend sionna_rt''

每个场景 = 若干发射源（预设选 v3 辐射体 / 自由选 kind+参数）+ 接收站。产出 runs/<name>/capture。
RT 后端必须用慕尼黑室外、天顶可见的坐标（示例用已验证的 example_case01 坐标）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent                                   # benchmark_v3
_PLATFORM = _V3 / "城市低空电磁信号仿真平台"
for _p in (str(_PLATFORM / "src"), str(_V3 / "simulation"),
           str(_V3.parent / "benchmark_v2" / "simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cityrf.config import (Antenna, Component, Emitter, Frontend,  # noqa: E402
                           Propagation, Receiver, Scenario)
from cityrf.runner import run_scenario  # noqa: E402

# 慕尼黑已验证"天顶可见/室外"的坐标（example_case01），RT 与 free_space 均可用
RT_TX = [(-150.0, -100.0, 85.0), (150.0, -80.0, 55.0), (-80.0, 40.0, 35.0)]
RT_RX = [(80.0, 60.0, 5.0), (0.0, 100.0, 5.0), (100.0, 0.0, 65.0)]


def _emitter(preset=None, kind=None, *, eid=None, role="interference", power_dbm=20,
             pos=None, **kw):
    """构造一个发射源 dict。preset=辐射体名 → 预设模式；kind=调制名 → 自由模式。"""
    pos = pos or RT_TX[0]
    return dict(eid=eid or (preset or kind or "tx"), role=role, power_dbm=power_dbm,
                preset=preset, kind=kind, pos=tuple(pos), **kw)


def build_scenario(sc, *, backend="free_space", fs=2e6, duration=0.05, seed=42):
    """sc: {name, n_rx, emitters:[_emitter dict]} → 平台 Scenario。"""
    ems = []
    for e in sc["emitters"]:
        preset = e.get("preset")
        if preset:
            comps = [Component(kind="noise", bandwidth_hz=200000)]
        else:
            comps = [Component(kind=e.get("kind", "qpsk"),
                               bandwidth_hz=e.get("bandwidth_hz", 240e3),
                               symbol_rate_hz=e.get("symbol_rate_hz", 25e3))]
        ems.append(Emitter(id=e["eid"], role=e.get("role", "interference"),
                           position_m=tuple(e["pos"]), orientation_rad=(0.0, 0.0, 0.0),
                           power_dbm=e.get("power_dbm", 20), preset=preset, components=comps))
    n_rx = sc.get("n_rx", 1)
    rxs = [Receiver(id=f"station_{chr(ord('a') + i)}", position_m=tuple(RT_RX[i]),
                    bandwidth_hz=fs * 0.5 * 0.9, frontend=Frontend(),
                    antenna=Antenna(geometry_type="rectangular", num_rows=1, num_cols=8))
           for i in range(n_rx)]
    prop = Propagation(backend=backend)
    return Scenario(name=sc["name"], seed=seed, duration_s=duration, sample_rate_hz=fs,
                    carrier_frequency_hz=2.4e9, max_working_mb=4096,
                    emitters=ems, receivers=rxs, propagation=prop)


def run_batch(scenarios, backend="free_space", runs_root=None):
    """逐场景 run_scenario → 写 <runs_root>/<name>/capture。返回已写场景。"""
    runs_root = Path(runs_root) if runs_root else (_PLATFORM / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    done = []
    for sc in scenarios:
        scenario = build_scenario(sc, backend=backend)
        out = runs_root / sc["name"] / "capture"
        if out.exists():
            import shutil
            shutil.rmtree(out)
        t0 = time.time()
        summary = run_scenario(scenario, out)
        ems = ", ".join(e["eid"] for e in sc["emitters"])
        print(f"  {sc['name']:22s} backend={backend}  源=[{ems}]  "
              f"路径={summary['path_count']}  {time.time()-t0:.1f}s")
        done.append(sc["name"])
    return done


# 演示：预设(蓝牙/雷达/GNSS压制/微波炉) + 自由(BPSK/16QAM/FHSS)
DEFAULT_SCENARIOS = [
    dict(name="preset_bluetooth", n_rx=1, emitters=[
        _emitter(preset="Bluetooth_FHSS", eid="bluetooth", role="interference",
                 power_dbm=20, pos=RT_TX[0])]),
    dict(name="preset_radar", n_rx=1, emitters=[
        _emitter(preset="Radar_pulsed_LFM", eid="radar", role="interference",
                 power_dbm=25, pos=RT_TX[0])]),
    dict(name="preset_gnss_jam", n_rx=1, emitters=[
        _emitter(preset="GNSS_jam", eid="gnss_jam", role="interference",
                 power_dbm=20, pos=RT_TX[0])]),
    dict(name="preset_microwave", n_rx=1, emitters=[
        _emitter(preset="Microwave", eid="microwave", role="interference",
                 power_dbm=15, pos=RT_TX[0])]),
    dict(name="free_bpsk", n_rx=1, emitters=[
        _emitter(kind="bpsk", eid="bpsk_tx", role="communication", power_dbm=0,
                 bandwidth_hz=400e3, symbol_rate_hz=50e3, pos=RT_TX[0])]),
    dict(name="free_16qam", n_rx=1, emitters=[
        _emitter(kind="16qam", eid="qam16_tx", role="communication", power_dbm=0,
                 bandwidth_hz=400e3, pos=RT_TX[0])]),
    dict(name="free_fhss", n_rx=1, emitters=[
        _emitter(kind="fhss", eid="fhss_tx", role="interference", power_dbm=20,
                 bandwidth_hz=600e3, pos=RT_TX[0])]),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["free_space", "sionna_rt"], default="free_space")
    ap.add_argument("--runs-root", default=None, help="写 runs 的根目录，默认平台 runs/")
    ap.add_argument("--only", default=None, help="逗号分隔的场景名，默认全部")
    args = ap.parse_args()

    scs = DEFAULT_SCENARIOS
    if args.only:
        keep = set(x.strip() for x in args.only.split(","))
        scs = [s for s in scs if s["name"] in keep]

    print(f"=== 批量生成 {len(scs)} 个场景（backend={args.backend}）===")
    done = run_batch(scs, backend=args.backend, runs_root=args.runs_root)
    print("完成，共", len(done), "个运行；网页端会自动枚举这些 runs。")