"""D2 加噪前/加噪后：Agent 可见证据（门控残差峰 / 聚类用宽带 legacy 峰）对照。

只读（只读数据集与信道，缓存写到系统临时目录）。
动机：同样的 50 个样本，agent 报源数从 43/50 掉到 24/50（欠报 24、17 个样本报 0），
而仪器侧 gate 只变 −5pp。要判断「证据变少了」还是「agent 判断变保守了」，
就必须直接看两边喂给 agent 的那份峰清单。
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # adapters（迁移后 HERE 下沉一级）
_PLAT = HERE.parent / "城市低空电磁信号仿真平台" / "src"
if str(_PLAT) not in sys.path:
    sys.path.insert(0, str(_PLAT))

import capture_tools as ct          # noqa: E402
import run_capture_agent as rca     # noqa: E402

ROOTS = {
    "OLD(加噪前)": (HERE.parent / "_pre_noise_20261010" / "dataset_d2").resolve(),
    "NEW(加噪后)": (HERE.parent / "dataset_d2").resolve(),
}
SIDS = sys.argv[1:] or ["d2_00001", "d2_00009", "d2_00020", "d2_00002"]

tmp = Path(tempfile.mkdtemp(prefix="d2evidence_"))

for sid in SIDS:
    print("=" * 78)
    print(sid)
    for tag, root in ROOTS.items():
        cap = root / sid / "capture"
        obs = json.loads((cap / "public" / "observations.json").read_text(encoding="utf-8"))
        truth = json.loads((cap / "private" / "d1_truth.json").read_text(encoding="utf-8"))
        ns = obs["receivers"][0]["shape"][1]
        refs = rca._build_target_reference(cap / "private", ns)
        allowed = [r["receiver_id"] for r in obs["receivers"]]
        aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()
        ctx = ct.ToolContext(capture_dir=cap / "public",
                             cache_dir=tmp / tag / sid,
                             allowed_stations=allowed,
                             fs=obs["receivers"][0]["sample_rate_hz"],
                             fc=obs["receivers"][0]["center_frequency_hz"],
                             target_bw_normalized=0.25, aoi_center=aoi,
                             target_reference=refs)
        ct.set_context(ctx)
        occ = ct.estimate_band_occupancy_by_correlation("station_a")
        gated = occ.get("residual_peaks") or []
        legacy = occ.get("residual_peaks_legacy") or []
        gt = truth["interferers"]
        print(f"  {tag}  可见干扰 {len(gt)} 个:")
        for g in gt:
            print(f"      GT kind={g['kind']:<16s} f0={g['freq_offset_hz']/1e3:9.1f} kHz  bw={g['bandwidth_hz']/1e3:7.1f} kHz")
        print(f"    门控峰（Agent 可见）{len(gated)} 个:")
        for p in gated:
            print(f"      f={p['freq_offset_hz']/1e3:9.1f} kHz  over={p['over_floor_db']:6.2f} dB  "
                  f"coh={p.get('coherence', float('nan')):.3f}  bw={p['bandwidth_hz']/1e3:7.1f} kHz")
        print(f"    legacy 宽带峰（聚类定带用）{len(legacy)} 个:")
        for p in legacy:
            print(f"      f={p['freq_offset_hz']/1e3:9.1f} kHz  over={p['over_floor_db']:6.2f} dB  "
                  f"bw={p['bandwidth_hz']/1e3:7.1f} kHz")
print(f"\n临时缓存目录（可删）：{tmp}")
