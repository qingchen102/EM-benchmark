"""工具适配层零 token 冒烟：7 件工具逐一点名 + 与 GT 对照（不调 LLM）。

用法：
    python smoke_capture_tools.py ../dataset_d1 d1_00000   # 简单场景
    python smoke_capture_tools.py ../dataset_d2 d2_00000   # 频谱重叠难例
站点白名单用 S2（全站），以便同时验证 estimate_position。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import capture_tools as ct


def main() -> None:
    root = Path(sys.argv[1])
    sid = sys.argv[2] if len(sys.argv) > 2 else "d1_00000"
    capture = root / sid / "capture" / "public"
    obs = json.loads((capture / "observations.json").read_text(encoding="utf-8"))
    stations = [r["receiver_id"] for r in obs["receivers"]]
    fs = float(obs["receivers"][0]["sample_rate_hz"])
    fc = float(obs["receivers"][0]["center_frequency_hz"])
    aoi = np.mean([r["receiver_position_m"] for r in obs["receivers"]], axis=0).tolist()

    truth = json.loads((root / sid / "capture" / "private" / "d1_truth.json")
                       .read_text(encoding="utf-8"))
    print(f"== {sid} ==  target: {truth['target']['emitter']}")
    for g in truth["interferers"]:
        print(f"  GT interferer: {g['kind']:12s} f0={g['freq_offset_hz']:+9.0f} Hz "
              f"bw={g['bandwidth_hz']/1e3:6.0f} kHz "
              f"doa_s0={g['per_station']['s0']['doa_deg']:+7.2f} deg "
              f"pos=({g['position_m'][0]:.0f}, {g['position_m'][1]:.0f}) "
              f"inr_s0={g['per_station']['s0']['inr_db']:.1f} dB")
    if not truth["interferers"]:
        print("  GT interferer: (none)")

    ctx = ct.ToolContext(capture_dir=capture, cache_dir=root / "_agent_cache" / sid,
                         allowed_stations=stations, fs=fs, fc=fc,
                         target_bw_normalized=0.3, aoi_center=aoi)
    ct.set_context(ctx)

    def t(name, fn, *a, **kw):
        t0 = time.time()
        try:
            res = fn(*a, **kw)
        except Exception as exc:                      # 冒烟脚本允许工具报错并继续
            print(f"[{name}] ERROR ({time.time() - t0:.1f}s): {exc}")
            return None
        print(f"[{name}] {time.time() - t0:.1f}s")
        return res

    cat = t("catalog", ct.list_radiator_catalog)
    if cat:
        print(f"  radiators: {len(cat['radiators'])}")

    spec = t("spectrum(a)", ct.analyze_spectrum_station, "station_a")
    if spec:
        print(f"  obw99={spec['obw99_hz']/1e3:.0f} kHz  candidate_source={spec['candidate_source']}")
        for c in spec["sources_candidates"][:5]:
            print(f"   cand f={c.get('freq_offset_hz', 'nan'):+9.0f} Hz "
                  f"bw={c.get('bandwidth_hz', float('nan'))/1e3:6.0f} kHz "
                  f"angle={c.get('angle_deg')}")

    ns = t("num_sources(a)", ct.estimate_num_sources_station, "station_a")
    if ns:
        print(f"  spatial_components={ns.get('num_spatial_components')} "
              f"u_peaks={ns.get('u_peaks')}")

    if truth["interferers"]:
        f0 = truth["interferers"][0]["freq_offset_hz"]
        bw0 = truth["interferers"][0]["bandwidth_hz"]
        doa = t("doa(a,band0)", ct.estimate_doa_station, "station_a", f0, bw0)
        if doa:
            print(f"  u_peaks={doa['direction_cosine_u_peaks']}")
            print(f"  az_candidates={doa['azimuth_candidates_deg']} "
                  f"(GT s0={truth['interferers'][0]['per_station']['s0']['doa_deg']:+.1f}, "
                  f"aoi_bearing={doa.get('aoi_bearing_deg')})")
        pos = t("position(band0)", ct.estimate_position, f0, bw0)
        if pos:
            for h in pos["hypotheses"][:3]:
                print(f"   hyp xy=({h['position_xy_m'][0]:.0f}, {h['position_xy_m'][1]:.0f}) "
                      f"h={h['height_m']:.0f} u_rms={h['u_residual_rms']:.3f}")
            gtp = truth["interferers"][0]["position_m"]
            print(f"  GT pos=({gtp[0]:.0f}, {gtp[1]:.0f})")

    td = t("time_domain(a)", ct.detect_time_domain_station, "station_a")
    if td:
        print(f"  papr={td['papr_db']} duty={td['pulse_duty_cycle']} "
              f"clip={td['clipping_fraction']}")

    mod = t("mod_features(a)", ct.estimate_modulation_features_station, "station_a")
    if mod:
        print(f"  backend={mod['backend']} slices={len(mod['per_peak_template_distances'])}")
        for s in mod["per_peak_template_distances"][:3]:
            top = s["template_distances"][0] if s["template_distances"] else None
            print(f"   peak={s['peak_freq']:+.3f} obw={s['slice_obw99']:.2f} "
                  f"top={top['template'] if top else None}({top['feature_distance'] if top else '-'})")

    # 站点白名单：S0 上下文下访问其它站必须报错
    ct.set_context(ct.ToolContext(capture_dir=capture, cache_dir=root / "_agent_cache" / sid,
                                  allowed_stations=["station_a"], fs=fs, fc=fc,
                                  target_bw_normalized=0.3, aoi_center=aoi))
    bad = ct.estimate_doa_station("station_b")
    print(f"[whitelist] S0 下访问 station_b -> {'BLOCKED ok' if 'error' in bad else 'LEAK!!'}")
    pos0 = ct.estimate_position(f0 if truth["interferers"] else 100e3)
    print(f"[whitelist] S0 下 estimate_position -> {'BLOCKED ok' if 'error' in pos0 else 'LEAK!!'}")


if __name__ == "__main__":
    main()
