"""v3 冒烟测试：辐射体生成 → 样本构建 → v2 工具兼容性。

不依赖网络与 API；通过标准：
1. 6 种辐射体全部可生成、功率有限；
2. factory 可产出 (4,1024) complex128 样本与 ground_truth；
3. v2 的 tools_v2.analyze_spectrum / estimate_doa 可直接消费 v3 样本。
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "simulation"))
sys.path.insert(0, str(HERE.parent / "benchmark_v2"))

from em_signal_simulator_v3 import emitters  # noqa: E402
from factory_v3 import build_sample  # noqa: E402

import tools_v2  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    print("== 1) 辐射体生成 ==")
    for name in emitters.EMITTERS:
        out = emitters.generate_emitter(name, rng)
        p = float(np.mean(np.abs(out["iq"]) ** 2) + 1e-30)
        assert np.all(np.isfinite(out["iq"])), name
        print(f"  {name:18s} power={p:8.3f}  mod={out['modulation']:5s}")

    print("== 2) 样本构建（蓝牙 + 雷达 双干扰）==")
    s = build_sample(["Bluetooth_FHSS", "Radar_pulsed_LFM"], snr_db=10.0, rng=rng)
    iq = s["iq"]
    assert iq.shape == (4, 1024) and iq.dtype == np.complex128
    assert np.all(np.isfinite(iq))
    gt = s["ground_truth"]
    assert gt["num_sources"] == 3
    emitters_of_gt = [i["emitter"] for i in gt["interferers"]]
    print(f"  iq={iq.shape} {iq.dtype}  干扰={emitters_of_gt}")

    print("== 3) v2 工具兼容性 ==")
    np.save("_smoke_v3.npy", iq)
    spec = tools_v2.analyze_spectrum("_smoke_v3.npy", target_bandwidth_normalized=gt["target"]["bandwidth_normalized"])
    print(f"  analyze_spectrum: 候选源={len(spec['sources_candidates'])} "
          f"路径={spec.get('candidate_source')}")
    doa = tools_v2.estimate_doa("_smoke_v3.npy", num_sources=3)
    print(f"  estimate_doa: {doa['doa_estimates_deg']}")
    Path("_smoke_v3.npy").unlink()
    print("\n冒烟通过 ✓")


if __name__ == "__main__":
    main()
