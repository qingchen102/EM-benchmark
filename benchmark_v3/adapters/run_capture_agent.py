"""D1–D4 capture 数据集的 Agent 考生运行器（LLM + 站点感知工具）。

复用 benchmark_v2 冻结代码（monkey-patch 模式，同 eval_v3.py，零改 v2 文件）：
- OpenAICompatibleAgent.diagnose：10 轮工具循环 / repair 轮 / 指数退避重试原样继承；
- 替换模块级 SYSTEM_PROMPT / REPAIR_PROMPT / _format_observation /
  TOOL_FUNCTIONS_V2 / TOOL_SCHEMAS_V2 为 capture 域版本（capture_tools.py）。

先验范围（与 v2 的 target_modulation 先验同等合法，不暴露任何干扰信息）：
站列表/坐标/fs/阵列几何、本 S 级可用站、难度名、目标辐射体型号与名义 OBW99
（读自 private/scenario.json 的 target id——由考题构建方构造先验，非 Agent 可见 GT）。

用法：
    python run_capture_agent.py ../dataset_d1 --slevel S0 --max-samples 3
    python run_capture_agent.py ../dataset_d1 --slevel S0          # 50 样本全量
    python run_capture_agent.py ../dataset_d1 --slevel S0 --resume # 断点续跑
答案写 <root>/answers_S0/<sample_id>.json（eval_dataset.py --answers 直接可评），
跑完自动调用扩展版 eval_dataset.evaluate 打印该 S 级矩阵并存
<root>/eval_report_S0.json（不覆盖离线回放的 eval_report.json）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_V2 = _V3.parent / "benchmark_v2"
_PLATFORM_SRC = _V3 / "城市低空电磁信号仿真平台" / "src"
for _p in (str(_V2), str(_V2 / "simulation"), str(_V3 / "simulation"),
           str(_PLATFORM_SRC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_tools  # noqa: E402
import evaluator_v2 as ev2  # noqa: E402
import eval_dataset  # noqa: E402
from em_signal_simulator_v3.emitters import EMITTERS, generate_emitter  # noqa: E402
from run_bailian_deepseek import NoThinkDeepSeekAgent  # noqa: E402
from cityrf.config import Scenario  # noqa: E402
from cityrf.channels import LinkChannel, apply_channel, DELAY_HALF  # noqa: E402
from cityrf.waveforms import generate  # noqa: E402
from cityrf.receiver import receive  # noqa: E402

# ---- 名义带宽先验（公开规格先验，非逐样本 GT）----
# 目标在捕获里按 scenario 组件带宽 kw 生成（仅接受 bw 参数的生成器），先验必须
# 与部署信号一致：接受 bw 的辐射体存标记、运行时用 组件带宽/fs；否则用默认
# 生成器实测 OBW99。表值 (接受bw?, 默认实测OBW99)。


def _preset_bw_table() -> dict[str, tuple[bool, float]]:
    import inspect
    rng = np.random.default_rng(0)
    table: dict[str, tuple[bool, float]] = {}
    for name in EMITTERS:
        gen = EMITTERS[name][0]
        accepts_bw = "bw" in inspect.signature(gen).parameters
        try:
            out = generate_emitter(name, rng, length=8192)
            meas = float(out["bandwidth_normalized"])
        except Exception:
            meas = 0.1
        table[name] = (accepts_bw, meas)
    return table


def _target_bw_normalized(preset: str, component_bw_hz: float | None, fs: float,
                          bw_table: dict[str, tuple[bool, float]]) -> float:
    accepts_bw, meas = bw_table.get(preset, (False, 0.1))
    if accepts_bw and component_bw_hz:
        return min(max(component_bw_hz / fs, 1e-3), 1.0)
    return meas


def _build_target_reference(private: Path, num_samples: int) -> dict:
    """构造"无噪声、无干扰"的每站目标接收参考（复用平台 _inband_oracle 回放范式）。

    读 private/scenario（seed + 目标 Emitter + 各站 Frontend）与 private/channel_target_*.npz，
    与平台 runner 完全一致：用**全局**最大多径时延算 history，**一次**再生目标基带
    （同 seed），再逐站 apply_channel + 关噪声 receive，裁 [rx_guard:rx_guard+num_samples]。
    必须用全局 history（而非逐站）——否则 OFDM 符号窗和捕获错位，相关剥离失效。
    参考由运行器注入 ToolContext（工具内部取用、不暴露给 Agent）。
    """
    sc = Scenario.model_validate_json((private / "scenario.json").read_text(encoding="utf-8"))
    tgt_em = next(t for t in sc.emitters if t.id == "target")
    rx_guard = max(r.frontend.filter_taps // 2 for r in sc.receivers)
    rx_by_id = {r.id: r for r in sc.receivers}
    # 读全部链路：目标链路用于回放，但 history 必须与 runner 完全一致——
    # runner 用**所有源所有站**的最大延迟算 history；若只用目标链路，某干扰
    # 延迟更大时 indices 长度不同，_generate_preset 的伪随机波形时间轴错位，
    # 参考与真实接收完全不相关（d2_00001/00003 的 corr≈0.016 总根因）。
    links: dict[str, LinkChannel] = {}
    global_max = 0.0
    for npz in private.glob("channel_*.npz"):
        with np.load(npz, allow_pickle=False) as d:
            if d["delays_s"].size:
                global_max = max(global_max, float(np.max(d["delays_s"])))
    for sid, rx in rx_by_id.items():
        npz = private / f"channel_{tgt_em.id}_{sid}.npz"
        if not npz.is_file():
            continue
        with np.load(npz, allow_pickle=False) as d:
            coeff = np.array(d["coefficients"])
            delays = np.array(d["delays_s"])
            if delays.size == 0:
                continue                       # 该站目标链路 0 径（RT NLoS 完全遮挡）
            links[sid] = LinkChannel(tgt_em.id, sid, coeff, delays, {}, {"backend": "rt"})
    if not links:
        return {}
    history = int(np.ceil(global_max * sc.sample_rate_hz)) + DELAY_HALF + rx_guard + 2
    end = num_samples + rx_guard + DELAY_HALF + 2
    indices = np.arange(-history, end, dtype=np.int64)
    wave = generate(tgt_em, sc.sample_rate_hz, indices, sc.seed)
    refs: dict[str, np.ndarray] = {}
    for sid, link in links.items():
        arrival = apply_channel(wave, indices[0], link, sc.sample_rate_hz,
                                num_samples + 2 * rx_guard, output_start=-rx_guard)
        clean = rx_by_id[sid].model_copy(deep=True)
        # 关掉**加性**噪声：`noise_enabled`（热噪 + 共模白噪）、`colored_noise_enabled`、
        # `impulse_noise_enabled`、`correlated_noise_enabled`。后三者**不受
        # `noise_enabled` 管辖**（`receiver.py:60 / :69`），只置 master 会让它们漏进
        # 这份所谓"无噪声"的参考里（2026-10-10 发现）。
        # **相位噪声 `phase_noise_enabled` 故意保留**：它是本振的乘性相位误差，与数据
        # 走的是同一台接收机、同一时刻、同一 realization（种子只由
        # `(seed, rx.id, indices)` 决定）→ 目标项在直减里精确抵消，只有干扰项保留
        # 同一份相位调制。关掉它反而让被照射的目标在残差里留下残影：
        # 实测 D2 tool-view 召回 0.367→0.173、假簇 0.623→0.879（作废的 A 模型）。
        # `imd3_enabled` 同样保留：参考与数据应走同一条非线性链。
        for _fname in ("noise_enabled", "colored_noise_enabled",
                       "impulse_noise_enabled", "correlated_noise_enabled"):
            setattr(clean.frontend, _fname, False)
        y, _ = receive(arrival, clean, sc.sample_rate_hz, sc.seed, start_index=-rx_guard)
        refs[sid] = np.ascontiguousarray(y[:, rx_guard:rx_guard + num_samples]).astype(complex)
    return refs


SYSTEM_PROMPT_CAPTURE = """You are an RF spectrum monitoring analyst for a synchronized \
multi-station low-altitude monitoring network (each station: 8-element ULA, half-wavelength \
spacing). Each capture contains ONE cooperative target signal (a benign UAV data link, near \
0 Hz) plus 0~3 interferers (other urban radiators or jammers); at difficulty D2 the
scene holds exactly 2 co-channel radiators.

Available measurement tools (per station):
- analyze_spectrum_station(station, use_spatial): PSD peaks, per-source candidates
  {freq, bandwidth (normalized), freq_offset_hz, bandwidth_hz, power ratio}, obw99_hz
- estimate_num_sources_station(station, max_sources): number of spatial components
  in the full receiver band (MDL rank; each component = one source or a strong
  multipath echo, so RT scenes may exceed the true emitter count)
- estimate_doa_station(station, freq_offset_hz, bandwidth_hz): DOA for that band in
  direction-cosine u space (no degeneracy): u peaks -> azimuth candidate PAIRS
  [asin(u), 180-asin(u)]. A horizontal ULA cannot distinguish az from its mirror
  180-az (identical response) — a single station cannot resolve the pair;
  multi-station estimate_position resolves it.
- estimate_position(freq_offset_hz, bandwidth_hz): multi-station cone-intersection
  least squares -> ranked hypotheses {position_xy_m [x,y], height_m, u_residual_rms};
  smaller residual = more consistent bearings. Needs >= 2 stations
- detect_time_domain_station(station): PAPR, pulse duty cycle, pulse edge count,
  clipping fraction
- estimate_modulation_features_station(station): per-peak band slices + modulation
  template distances (CNN backend when available)
- list_radiator_catalog(): public spec sheet — 16 urban low-altitude radiators +
  tone/noise, with modulation class and waveform description
- estimate_band_occupancy_by_correlation(station): cooperative-target correlation
  strip — subtracts the target reference (provided by the monitoring network) and
  caches the target-stripped residual; reports residual_peaks and
  band_occupied_beyond_target. Each residual_peak is {freq_offset_hz,
  over_floor_db, bandwidth_hz, band_hz, coherence}: it has ALREADY passed two
  gates (>= 8 dB over a LOCAL median floor AND spatial coherence
  lambda1/sum >= 0.6 in a +-20 kHz window), so peaks in this list are real
  emitters — spatially-white mismatch energy is filtered out. coherence ~1 means
  point source; ~0.13 means noise-like (keep it only as a last resort).
- analyze_time_frequency_station(station, freq_offset_hz, bandwidth_hz): (u,f)
  time-frequency ridge map of the band — per-ridge {u, f_lo, f_hi, span, strength}
  plus per-ridge modulation features. Key instrument for telling co-channel
  radiators apart when their spectra fuse into one blob.
- estimate_sources_by_spatial_clustering(on_residual=true): MULTI-STATION spatial
  separation (needs >= 3 stations; S1/S2). Returns TWO lists: candidates = the full
  scored candidate board {position_xy_m, band_hz, n_explained_stations, max_du,
  perp_rms_m, deflation_db, confidence} ranked by (perp_rms asc, deflation desc,
  triads desc) — do NOT use len(candidates) as the interferer count — and sources
  = candidates passing the support gate; K = len(sources) IS the spatial
  source-count evidence. Height unobservable (parallel horizontal arrays):
  position_xy_m is 2D.
- extract_source_waveforms(): beamform-extract each candidate's INDEPENDENT
  waveform (own-projection LS; co-channel crosstalk left to the classifier which
  is trained with siblings). Waveforms cached CNN-ready as bfw_{station}_{k}.npy
  (shifted to candidate band center + RMS normalized) — feed directly to radiator
  identification. Needs >= 3 stations.
- classify_extracted_waveform(station="station_a"): classify each candidate's
  extracted waveform (CNN over 19 radiators) and return READY-TO-COPY answer
  fields per candidate: {freq_offset_hz, kind, intent, is_spoof, position_xy_m,
  confidence}. Call this AFTER extract_source_waveforms, then write the final
  JSON answer directly from these items — no further measurement needed.

Measurement-to-answer mapping:
- freq_offset_hz  <- analyze_spectrum_station sources_candidates freq_offset_hz (Hz,
  relative to receiver center; the target sits near 0 Hz — it is NOT an interferer)
- kind            <- identify each interferer from measured features (slices in
  estimate_modulation_features_station, bandwidth/power in analyze_spectrum_station,
  time-domain evidence) against list_radiator_catalog. Answer EXACTLY one catalog
  name. At difficulty D1 only "tone" and "noise" are possible; at D2-D4 any of the
  16 catalog radiators is possible.
- doa_deg         <- estimate_doa_station on station_a for that interferer's band
  (world azimuth). Mind the mirror ambiguity: pick the branch most consistent with
  the multi-station position hypothesis when available.
- position_m      <- estimate_position on that interferer's band; top hypothesis
  position_xy_m. Co-band contamination: when the interferer band overlaps the wide
  target signal, hypotheses may include the TARGET's position — first run
  estimate_position on a narrow band at freq_offset_hz=0 to identify the target's
  own position, then pick the interferer hypothesis that is NOT the target.
  With a single station, position is barely observable — best effort only.
- num_interferers <- **estimate_sources_by_spatial_clustering: K = len(sources)**
  (the support-gated spatial source count). Cross-check against spectral
  candidates; use len(candidates) only as an UPPER bound with the per-candidate
  scoring (perp_rms / deflation / triads) to judge how many are real. At D2 the
  scene holds exactly 2 co-channel radiators. The target itself is a wideband
  signal at ~0 Hz and is NOT an interferer — never report one of its spectral
  lobes or the filter edges as an interferer.
- intent / is_spoof <- judged from the identified kind and its measured behaviour

Intent classification (answer for EACH interferer):
- "malicious": deliberately hostile emitter (jamming / spoofing / ECM: sweep or
  chirp jammers, noise or tone jammers, GNSS spoofers)
- "incidental": unintentional coexistence interference (e.g. microwave oven, DECT,
  Zigbee)
- "benign": normal legal signal (communications / broadcast / navigation / video
  link / radar / RC link)
- is_spoof: true ONLY for a GNSS spoofing emitter (legitimate GNSS-like structure
  with wrong parameters); otherwise false

Output ONLY valid JSON with schema:
{"num_interferers": <0..3>,
 "interferers": [{"freq_offset_hz": <Hz>, "kind": "<exact catalog name>",
                  "intent": "malicious|incidental|benign", "is_spoof": <bool>,
                  "doa_deg": <world azimuth, deg>, "position_m": [<x_m>, <y_m>]}, ...]}
Rules:
- num_interferers must equal the number of entries in interferers.
- num_interferers MUST be a NON-NEGATIVE integer. It is parsed literally: a negative
  value (e.g. -1) is read as 0 and scored as a WRONG answer; there is no re-ask.
- ALWAYS include every key per interferer (doa_deg / position_m best effort even
  when only one station is available).
- HARD TOOL BUDGET — you have at most 10 rounds and you MUST follow this recipe.
  Tools already fuse multi-station data internally: calling a tool "per station"
  is WASTED and FORBIDDEN. Never repeat a call with the same arguments.
    Round 1: analyze_spectrum_station (any ONE station) -> band candidates
    Round 2: estimate_band_occupancy_by_correlation (same station) -> strip target
    Round 3: estimate_sources_by_spatial_clustering(on_residual=true) -> the
             candidate board + K = len(sources) (spatial source count; needs a
             station subset with >= 3 stations — S1/S2. On S0 fall back to the
             per-band DOA/feature route below.)
    Round 4: extract_source_waveforms() -> per-candidate CNN-ready waveforms
    Round 5: classify_extracted_waveform() -> READY-TO-COPY answer fields for
             EVERY candidate {freq_offset_hz, kind, intent, is_spoof,
             position_xy_m}. Copy them into the final JSON answer verbatim.
    Round 6: (only if needed) analyze_time_frequency_station /
             estimate_modulation_features_station for ambiguous candidates
    S0 FALLBACK (single station, no clustering/extract/classify): after Round 2
             use the residual evidence —
        3a. estimate_band_occupancy_by_correlation reports residual_peaks
            [{freq_offset_hz, over_floor_db, bandwidth_hz, band_hz, coherence}].
            This list is ALREADY a gated measurement: every entry passed ">= 8 dB
            over a LOCAL median floor" AND "spatial coherence >= 0.6" in a +-20 kHz
            window, so spatially-white residual energy (which used to appear as
            spurious 46-49 dB "peaks") is excluded. An EMPTY list means no
            interferer — report num_interferers = 0; do NOT invent one.
        3b. MERGE BEFORE COUNTING: one physical emitter can split into several
            adjacent entries (a noise-like source yields 2-3 sub-bands). Merge
            entries whose band_hz ranges touch or whose freq_offset_hz differ by
            less than ~150 kHz into ONE interferer, reporting the strongest
            (highest over_floor_db) sub-band's frequency. Only bands separated by
            more than ~150 kHz are separate interferers.
        3c. estimate_modulation_features_station(on_residual=true) + read
            global_template_distances: single_tone distance < 0.2 -> "tone";
            OFDM distance < 0.3 with spectral_flatness ~ 0 -> "noise" (band-limited
            noise reads as OFDM-like flat spectrum on a single station);
            other flat profiles -> "noise" is still the best D1 hypothesis.
            Add estimate_doa_station(on_residual=true) for the peak band.
        3d. Copy each merged peak into the final answer: freq_offset_hz = the
            residual peak frequency, kind from the template match, doa from the
            DOA scan.
    Round 7+: NO MORE TOOL CALLS. Write the final JSON answer NOW — the
              classify items already contain every field the answer schema
              needs. If evidence is partial, still answer with your best
              estimates for the bands you measured. Before emitting, check that
              num_interferers is a non-negative integer (0 is a valid answer).
- REPORT ONLY MEASURED BANDS: report exactly the merged coherent residual bands
  you measured (or the classifier candidates), nothing else. Do not invent a band
  from a spectral lobe of the wideband target (the target is at ~0 Hz and is
  NEVER an interferer), do not add a "safety" extra entry, and do not report the
  same emitter twice under two nearby frequencies. If the measurement shows no
  coherent band, num_interferers = 0 is the correct answer.
- NEVER use a negative num_interferers (e.g. -1) or any sentinel to signal
  "not sure": it is parsed as 0 and recorded as a WRONG answer (there is no
  clarification round). If the evidence is partial, commit to the bands you
  measured; if unsure of the kind, use the closest catalog match. Reporting too
  few interferers is still a real attempt; -1 is a guaranteed miss.
- CO-CHANNEL RESOLUTION: when the wideband target overlaps the interferers (spectrum
  is one fused blob around 0 Hz), FIRST call estimate_band_occupancy_by_correlation
  on a station. If band_occupied_beyond_target is true, re-run
  analyze_spectrum_station / estimate_num_sources_station / estimate_doa_station /
  estimate_position with on_residual=true — the stripped band reveals the
  co-channel interferers the fused spectrum hides. This is the correct path for
  D2-D4 scenes; do not guess source counts from a fused spectrum.
- After gathering enough measurements, ALWAYS output the final JSON answer. Do not
  keep calling tools without producing the answer.
- The final message must contain ONLY the JSON answer object and nothing else.
"""

# —— 公开层（2026-10-10）：事件日志段落。刻意与主 prompt 分离：只在样本真的带
# interference_events 时注入，否则 D1/D2 的 system prompt 与公开层之前逐字相同
# （历史基线可直接比较）。
EVENT_LOG_PROMPT_CAPTURE = (
    "- Cooperative-link event log: the observation may list interference events reported\n"
    "  by the cooperative ground station's own link monitor. Each entry is public evidence\n"
    "  that a radiator is degrading the cooperative link, and its mechanism (jam / spoof /\n"
    "  ...) is direct behavioural evidence for the intent axis — a reported jamming or\n"
    "  spoofing event means a deliberate emitter is present in the scene. The log is a\n"
    "  PARTIAL view (it only sees radiators that reach the cooperative link), so NEVER use\n"
    "  it as the source census: still measure the residual/spatial source count. Do not\n"
    "  copy event ids into the answer; answer with the interferers you measured.\n")
_EVENT_LOG_ANCHOR = "  with wrong parameters); otherwise false\n"


def system_prompt_for(events) -> str:
    """带事件日志段落的 system prompt；无事件时返回基础 prompt（逐字等于公开层之前）。"""
    if not events:
        return SYSTEM_PROMPT_CAPTURE
    out = SYSTEM_PROMPT_CAPTURE.replace(
        _EVENT_LOG_ANCHOR, _EVENT_LOG_ANCHOR + EVENT_LOG_PROMPT_CAPTURE, 1)
    assert out != SYSTEM_PROMPT_CAPTURE, "event-log anchor missing in SYSTEM_PROMPT_CAPTURE"
    return out


REPAIR_PROMPT_CAPTURE = (
    "Your previous reply did not contain a valid final JSON answer. Reply NOW with "
    "ONLY the final JSON object matching this schema — no prose, no markdown, no "
    "reasoning, no tool calls:\n"
    '{"num_interferers": 1, "interferers": [{"freq_offset_hz": 150000.0, '
    '"kind": "Zigbee", "intent": "incidental", "is_spoof": false, '
    '"doa_deg": 35.0, "position_m": [-650.0, -1410.0]}]}'
)

INTENTS = ("malicious", "incidental", "benign")


def _format_observation_capture(obs: dict) -> str:
    """把 capture 域公开先验转成人类可读描述（不含任何干扰真值）。"""
    lines = [
        f"Monitoring network: {len(obs['stations'])} synchronized stations "
        f"(8-element ULA, 0.5-wavelength spacing):",
    ]
    for st in obs["stations"]:
        x, y = st["position_m"][0], st["position_m"][1]
        lines.append(f"  {st['id']}: position ({x:.1f}, {y:.1f}) m")
    lines += [
        f"Sampling rate: {obs['fs'] / 1e6:.1f} MHz; receiver center: "
        f"{obs['fc'] / 1e9:.2f} GHz; capture window: {obs['num_samples'] / obs['fs'] * 1e3:.0f} ms "
        f"({obs['num_samples']} samples/station)",
        f"AOI center (deployment target zone): ({obs['aoi_center'][0]:.1f}, "
        f"{obs['aoi_center'][1]:.1f}) m",
        f"Available stations at this observation level: "
        f"{', '.join(obs['allowed_stations'])} ({obs['slevel']} = "
        f"{'single station' if len(obs['allowed_stations']) == 1 else 'multi-station'})",
        f"Difficulty: {obs['difficulty']} — "
        + ("interferer kinds are limited to: tone / noise" if obs["difficulty"] == "D1"
           else "interferer kind can be any of the 16 catalog radiators"),
        f"Target prior: cooperative UAV link, preset {obs['target_emitter']}, "
        f"nominal OBW99 = {obs['target_bw_hz'] / 1e3:.0f} kHz "
        f"({obs['target_bw_normalized']:.2f} normalized), near 0 Hz, benign",
        "Conventions: freq_offset_hz in Hz relative to receiver center; doa_deg in "
        "world azimuth (atan2(y,x)) degrees; position_m in world meters [x, y].",
    ]
    events = obs.get("events") or []
    if events:
        lines.append(
            f"Cooperative-link event log: {len(events)} event(s) reported by the "
            "cooperative ground station's own link monitor (public evidence):")
        for e in events:
            bits = [f"{e.get('event_id')}: source {e.get('source_id')}"]
            if e.get("mechanism"):
                bits.append(f"mechanism {e['mechanism']}")
            if e.get("impact_class"):
                bits.append(f"impact {e['impact_class']}")
            lines.append("  " + ", ".join(bits))
        if obs.get("events_note"):
            lines.append("  note: " + str(obs["events_note"]))
    return "\n".join(lines)


def _to_f(value, default=None):
    v = ev2._to_float(value, float("nan")) if not isinstance(value, (int, float)) else float(value)
    return float(v) if np.isfinite(v) else default


def _sanitize(pred: dict) -> dict:
    """把模型答案清洗成 eval_dataset 认可的 schema（缺字段不伪造，评分端按口径处理）。

    `abstained`：模型写了负数（= prompt 明令禁止的哨兵 -1）时为 True。负数会被清成
    0，于是在 detection 上与“判定无干扰”不可区分且必然判错——所以把这次拒答单列
    出来，评测端可同时报“含拒答 / 排除拒答”两个 detection
    （见 eval_dataset._load_abstain_ids 与报告里的 abstain_n）。
    """
    n_raw = _to_f(pred.get("num_interferers"), 0.0) if isinstance(pred, dict) else 0.0
    abstained = bool(n_raw is not None and n_raw < 0)
    n = max(0, int(n_raw or 0))
    items_raw = pred.get("interferers") if isinstance(pred, dict) else None
    items_raw = items_raw if isinstance(items_raw, list) else []
    items = []
    for p in items_raw[:n]:
        if not isinstance(p, dict):
            continue
        it: dict = {"freq_offset_hz": _to_f(p.get("freq_offset_hz"), 0.0) or 0.0,
                    "kind": str(p.get("kind", "")).strip()}
        intent = str(p.get("intent", "")).strip().lower()
        if intent in INTENTS:
            it["intent"] = intent
        it["is_spoof"] = bool(p.get("is_spoof"))
        doa = _to_f(p.get("doa_deg"))
        if doa is not None:
            it["doa_deg"] = doa
        pos = p.get("position_m")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = _to_f(pos[0]), _to_f(pos[1])
            if x is not None and y is not None:
                it["position_m"] = [x, y]
        items.append(it)
    return {"num_interferers": n, "interferers": items, "abstained": abstained}


# ---- monkey-patch：capture 域 prompt / 观测格式 / 工具（零改 v2 冻结文件） ----
ev2.SYSTEM_PROMPT = SYSTEM_PROMPT_CAPTURE
ev2.REPAIR_PROMPT = REPAIR_PROMPT_CAPTURE
ev2._format_observation = _format_observation_capture
ev2.TOOL_FUNCTIONS_V2 = capture_tools.TOOL_FUNCTIONS
ev2.TOOL_SCHEMAS_V2 = capture_tools.TOOL_SCHEMAS


def run(dataset_root: Path, slevel: str, answers_dir: Path, agent,
        max_samples: int | None, resume: bool, verbose: bool,
        early_stop: float = 0.10, early_stop_min: int = 15) -> None:
    """early_stop：预注册门槛（滚动 detection_hit 低于此值且样本数≥early_stop_min
    时停止；0=关闭）。门槛在开跑前定死，符合"预注册验收"纪律——不是跑一半看心情。"""
    idx = json.loads((dataset_root / "index.json").read_text(encoding="utf-8"))
    subsets = idx["observation_subsets"]
    if slevel not in subsets:
        raise SystemExit(f"--slevel must be one of {list(subsets)}")
    allowed = list(subsets[slevel])
    answers_dir.mkdir(parents=True, exist_ok=True)

    bw_table = _preset_bw_table()
    samples = idx["samples"][:max_samples] if max_samples else idx["samples"]
    log_path = answers_dir / "_run_log.json"
    run_log: list[dict] = []
    det_scores: list[float] = []
    early_stopped: dict | None = None

    for k, rec in enumerate(samples, 1):
        sid = rec["id"]
        ans_path = answers_dir / f"{sid}.json"
        if resume and ans_path.exists():
            continue
        capture = dataset_root / sid / "capture" / "public"
        private = dataset_root / sid / "capture" / "private"
        obs_json = json.loads((capture / "observations.json").read_text(encoding="utf-8"))
        stations = [{"id": r["receiver_id"], "position_m": r["receiver_position_m"]}
                    for r in obs_json["receivers"]]
        aoi = np.mean([s["position_m"] for s in stations], axis=0).tolist()
        fs = float(obs_json["receivers"][0]["sample_rate_hz"])
        fc = float(obs_json["receivers"][0]["center_frequency_hz"])
        num_samples = int(obs_json["receivers"][0]["shape"][1])
        # 目标先验：优先公开的 cooperative_sources（考卷上印的合法先验）；
        # 旧数据集（D2–D4，字段为空）回退读 private scenario 的 target id。
        coop = (obs_json.get("cooperative_sources") or [{}])[0]
        if coop.get("preset"):
            tgt_preset = coop["preset"]
            nominal_bw = coop.get("nominal_bandwidth_hz")
        else:
            scenario = json.loads((private / "scenario.json").read_text(encoding="utf-8"))
            target = next(e for e in scenario["emitters"] if e["id"] == "target")
            tgt_preset = target["preset"]
            nominal_bw = (target.get("components") or [{}])[0].get("bandwidth_hz")
        tgt_bw_norm = _target_bw_normalized(tgt_preset, nominal_bw, fs, bw_table)
        # 合作目标参考（无噪声无干扰、含多径）——注入给相关剥离工具；失败则留空
        # （工具会明确报 target reference not available，不阻断其余分析）。
        try:
            target_ref = _build_target_reference(private, num_samples)
        except Exception as exc:
            target_ref = {}
            print(f"    [warn] target reference build failed for {sid}: {str(exc)[:80]}",
                  flush=True)

        ctx = capture_tools.ToolContext(
            capture_dir=capture,
            cache_dir=dataset_root / "_agent_cache" / sid,
            allowed_stations=allowed, fs=fs, fc=fc,
            target_bw_normalized=tgt_bw_norm, aoi_center=aoi,
            target_reference=target_ref)
        capture_tools.set_context(ctx)

        prior = {"stations": stations, "allowed_stations": allowed, "slevel": slevel,
                 "difficulty": (str(idx.get("difficulty", "")).split("(")[0].strip().upper()
                                or sid.split("_")[0].upper()),
                 "target_emitter": tgt_preset,
                 "target_bw_hz": tgt_bw_norm * fs, "target_bw_normalized": tgt_bw_norm,
                 "events": (obs_json.get("interference_events") or []),
                 "events_note": obs_json.get("interference_events_note"),
                 "aoi_center": aoi, "fs": fs, "fc": fc, "num_samples": num_samples}
        # 公开层：事件日志段落只在样本真有事件时注入（D1/D2 无事件 → prompt 与公开层
        # 之前逐字相同，历史基线可直接比较）。
        ev2.SYSTEM_PROMPT = system_prompt_for(prior["events"])

        t0 = time.time()
        pred_raw, calls, raw, round_log = agent.diagnose(capture, prior)
        pred = _sanitize(pred_raw)
        ans_path.write_text(json.dumps(pred, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        # —— 滚动监测（预注册门槛）：detection 对"报零/报错数量"类失败最敏感。
        # 每样本就地打分，达到最少样本数后滚动均值低于门槛即停，不烧完整批。
        det = None
        try:
            truth_d = json.loads((private / "d1_truth.json").read_text(encoding="utf-8"))
            gt_list = eval_dataset.gt_at_slevel(truth_d, allowed)
            det = eval_dataset.score_answer(pred, gt_list, slevel)["detection_hit"]
        except Exception:
            pass
        det_scores.append(det) if det is not None else None
        run_log.append({"sample_id": sid, "tool_calls": int(calls),
                        "elapsed_s": round(time.time() - t0, 1),
                        "round_log": round_log, "raw_output": str(raw)[:300],
                        "prediction": pred, "detection_hit": det})
        log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        det_s = f" det={det}" if det is not None else ""
        marker = f"[{k}/{len(samples)}] {sid}: tools={calls} " \
                 f"({time.time() - t0:.0f}s) n_ifw={pred['num_interferers']}{det_s}"
        print(marker, flush=True)
        if verbose:
            print("  " + json.dumps(pred, ensure_ascii=False)[:300], flush=True)
        if early_stop > 0 and len(det_scores) >= early_stop_min:
            rolling = float(np.mean(det_scores))
            if rolling < early_stop:
                early_stopped = {"samples_run": len(det_scores),
                                 "rolling_detection_hit": round(rolling, 3),
                                 "threshold": early_stop}
                print(f"[EARLY-STOP] 跑了 {len(det_scores)} 个样本，滚动 detection_hit="
                      f"{early_stopped['rolling_detection_hit']} < 预注册门槛 {early_stop}"
                      f" → 提前停止（已生成答案保留；确认要继续可用 --resume 绕过）",
                      flush=True)
                break

    if early_stopped:
        run_log.append({"early_stop": early_stopped})
        log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    # 有答案就出矩阵（评分器跳过缺答案样本）——早停/断点场景也能看到部分成绩
    n_have = sum(1 for rec in idx["samples"]
                 if (answers_dir / f"{rec['id']}.json").exists())
    if n_have >= 1:
        report = eval_dataset.evaluate(str(dataset_root), str(answers_dir))
        out = dataset_root / f"eval_report_{slevel}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"report -> {out}  (评了 {n_have}/{len(idx['samples'])} 样本"
              + ("；早停" if early_stopped else "") + ")")
    else:
        print("[partial] 无答案，跳过自动评分")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root")
    ap.add_argument("--slevel", required=True, help="S0 | S1 | S2")
    ap.add_argument("--answers", default=None,
                    help="答案目录（默认 <root>/answers_<slevel>）")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-v3.2"))
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="跳过已有答案文件的样本（断点续跑）")
    ap.add_argument("--early-stop-threshold", type=float, default=0.10,
                    help="预注册门槛：滚动 detection_hit 低于此值即停止（0=关闭）。默认 0.10")
    ap.add_argument("--early-stop-min", type=int, default=15,
                    help="开始判定前最少完成的样本数。默认 15")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base_url = args.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    agent_cls = NoThinkDeepSeekAgent if "dashscope" in base_url else ev2.OpenAICompatibleAgent
    agent = agent_cls(model=args.model, base_url=base_url, api_key=args.api_key)

    run(Path(args.dataset_root), args.slevel,
        Path(args.answers) if args.answers else Path(args.dataset_root) / f"answers_{args.slevel}",
        agent, args.max_samples, args.resume, args.verbose,
        early_stop=args.early_stop_threshold, early_stop_min=args.early_stop_min)


if __name__ == "__main__":
    main()
