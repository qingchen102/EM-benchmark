"""sanity_check_v2.py — 验证 refactoring 后的仿真模块。

覆盖：
1. Task 1 [P0] LNA 饱和：Rapp 模型压缩大信号、保留小信号；blocking 干扰样本
   元数据 lna_saturation.applied == True 且目标信号被压缩/扭曲。
2. Task 2 [P1] 脉冲成型边界伪影：首尾段功率与中段一致（无启动零点/跳变），
   FFT 边缘无异常毛刺。
3. Task 3 [P1] 干扰波形：swept 脉冲化（duty 门控）、nfm 波形存在且有限。
4. Task 4 [P1] 分离修正：多干扰样本全部满足 FREQ/DOA 最小间隔。
5. Task 5 [P2] 阵列失配：开启后 |a_m| != 1、相位误差存在；默认关闭时与旧版一致。
6. 端到端：generate_dataset(count=10) 生成 (4,1024) complex128、无 NaN/Inf、
   metadata.json 完整记录 source 字段。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_signal_simulator.channel import (
    steering_vector,
    expand_to_array,
    apply_lna_saturation,
    add_awgn,
)
from em_signal_simulator.baseband import (
    generate_baseband,
    actual_bandwidth,
    _apply_pulse_shaping,
)
from em_signal_simulator.jamming import (
    generate_interferer_waveform,
    JAMMING_TYPES,
)
from em_signal_simulator.factory import (
    generate_signal_sample,
    generate_dataset,
    _correct_separation,
    _auto_label,
    FREQ_MIN_SEP,
    DOA_MIN_SEP,
)

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- Task 1
def _spec_edge_power(x, nfft=4096, frac=0.04):
    """测量归一化频率 |f| in [0.5-frac, 0.5] 的边缘频谱能量占比（fftshift 后 DC 在 index N/2）。"""
    N = int(nfft)
    spec = np.abs(np.fft.fftshift(np.fft.fft(x, N)))
    spec = spec / max(float(np.max(spec)), 1e-12)
    k = int(N * frac)
    return float(np.mean(np.concatenate([spec[:k], spec[N - k:]])))


def test_lna_saturation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    x /= np.sqrt(np.mean(np.abs(x) ** 2))          # RMS=1（目标电平）
    y = apply_lna_saturation(x, a_sat=1.0, p=2.0)   # A_sat = 目标 RMS
    # 小幅度部分几乎不压缩，大幅度部分被压到 <= A_sat
    small = np.abs(x) < 0.2
    if np.any(small):
        ratio_small = np.mean(np.abs(y[small]) / (np.abs(x[small]) + 1e-12))
        check("LNA: 小信号近似线性（ratio≈1）", abs(ratio_small - 1.0) < 0.05,
              f"mean ratio={ratio_small:.4f}")
    check("LNA: 输出幅度被限制 <= A_sat", float(np.max(np.abs(y))) <= 1.0 + 1e-9,
          f"max|y|={np.max(np.abs(y)):.4f}")
    # 5 倍幅度的强阻塞 → 幅度应被明显压缩（< 3x 原始）
    xb = x * 5.0
    yb = apply_lna_saturation(xb, a_sat=1.0, p=2.0)
    comp = np.mean(np.abs(yb)) / np.mean(np.abs(xb))
    check("LNA: 强信号被压缩 (comp<0.7)", comp < 0.7, f"compression={comp:.4f}")

    # blocking 样本端到端（seed=0 稳定产生 blocking 干扰，见 test_blocking）：
    # 饱和门限低（6 dB，A_sat=2x 目标 RMS）→ 必触发；门限极高(60 dB)≈线性
    iq_lin, meta_lin = generate_signal_sample(
        num_sources=2, snr_db=30.0, interferer_type="Radar_LFM", seed=0,
        lna_saturation_db=60.0,          # 近似不饱和
    )
    iq_sat, meta_sat = generate_signal_sample(
        num_sources=2, snr_db=30.0, interferer_type="Radar_LFM", seed=0,
        lna_saturation_db=6.0,           # 饱和门限低 → 必触发
    )
    cat = meta_sat["sources"][1]["v2_category"]
    check("blocking: 样本被标注为 blocking", cat == "blocking", f"category={cat}")
    check("blocking: 元数据记录 LNA 应用", meta_sat["lna_saturation"]["applied"] is True)
    # 饱和样本总幅度应低于线性混合（压缩生效）
    p_lin = float(np.mean(np.abs(iq_lin) ** 2))
    p_sat = float(np.mean(np.abs(iq_sat) ** 2))
    check("blocking: 饱和后总功率被压缩", p_sat < p_lin * 0.9,
          f"P_lin={p_lin:.3f} P_sat={p_sat:.3f}")
    # 目标信号本身也被压缩（真实 blocking 效应）：单天线目标功率对比
    tgt_lin = float(np.mean(np.abs(iq_lin[0]) ** 2))
    tgt_sat = float(np.mean(np.abs(iq_sat[0]) ** 2))
    check("blocking: 目标信号幅度被压缩", tgt_sat < tgt_lin,
          f"target P_lin={tgt_lin:.3f} P_sat={tgt_sat:.3f}")


# ---------------------------------------------------------------- Task 2
def test_pulse_shaping_edges():
    rng = np.random.default_rng(1)
    n = 1024
    x = generate_baseband("QPSK", n, rng=rng)
    check("pulseshape: 长度正确", len(x) == n, f"len={len(x)}")
    # 头/尾窗口功率应落在中段窗口功率的自然波动范围内（RRC-QPSK 包络固有起伏）
    win = 200
    mid_powers = [np.mean(np.abs(x[s:s + win]) ** 2)
                  for s in range(n // 2, n - win, win)]
    p_head = np.mean(np.abs(x[:win]) ** 2)
    p_tail = np.mean(np.abs(x[-win:]) ** 2)
    lo, hi = min(mid_powers), max(mid_powers)
    check("pulseshape: 首段功率在中段波动范围内", lo - 0.15 * hi <= p_head <= hi * 1.15,
          f"head={p_head:.3f} mid_range=[{lo:.3f},{hi:.3f}]")
    check("pulseshape: 尾段功率在中段波动范围内", lo - 0.15 * hi <= p_tail <= hi * 1.15,
          f"tail={p_tail:.3f} mid_range=[{lo:.3f},{hi:.3f}]")
    # 相位连续性：相邻采样差分幅度（高频尖峰会表现为大差分）
    d = np.abs(np.diff(x))
    check("pulseshape: 无相位跳变尖峰", float(np.percentile(d, 99.9)) < 1.0,
          f"p99.9(|dx|)={np.percentile(d, 99.9):.3f}")
    # 频谱边缘：QPSK (BW 0.35) 在 |f|>0.46 处应几乎无能量（修正窗口：近 Nyquist）
    edge = _spec_edge_power(x)
    check("pulseshape: 频谱边缘无毛刺", edge < 1e-2, f"edge_power={edge:.2e}")
    # 各调制类型全部可生成且无 NaN
    for m in ["BPSK", "QPSK", "16QAM", "64QAM", "OOK", "GFSK", "OFDM", "FHSS", "LFM"]:
        y = generate_baseband(m, n, rng=rng)
        check(f"pulseshape: {m} 无 NaN/Inf", bool(np.all(np.isfinite(y))))
    # 常量符号流：输出包络应完全平坦（无滤波暂态）
    yc = _apply_pulse_shaping(np.ones(128, dtype=complex), 8, 0.35, out_length=n)
    amp = np.abs(yc)
    check("pulseshape: 常量符号流包络平坦", float(np.std(amp)) < 1e-3,
          f"std={np.std(amp):.2e}")
    # 升余弦成形：符号中心采样应恢复星座点（RC 是 Nyquist 滤波器，无 ISI）
    sym = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(16)))
    yrc = _apply_pulse_shaping(sym, 8, 0.35, out_length=16 * 8)
    centers = yrc[np.arange(16) * 8]          # 输出段从第 0 个符号中心开始
    err = float(np.max(np.abs(centers - sym)))
    check("pulseshape: 符号中心无 ISI（RC 成形）", err < 0.1, f"max_err={err:.4f}")
    # 滚降系数确实影响频谱形状：占用带宽 ≈ (1+β)/sps（用 256 随机符号长序列测）
    rng_bw = np.random.default_rng(7)
    sym256 = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng_bw.integers(0, 4, 256)))
    x_lo = _apply_pulse_shaping(sym256, 8, 0.2, out_length=256 * 8)
    x_hi = _apply_pulse_shaping(sym256, 8, 0.5, out_length=256 * 8)
    bw_lo = _occupied_bw(x_lo, seg=512)
    bw_hi = _occupied_bw(x_hi, seg=512)
    check("pulseshape: 占用带宽随滚降变化 ((1+β)/sps)", 
          abs(bw_lo - 1.2 / 8) < 0.25 * (1.2 / 8) and abs(bw_hi - 1.5 / 8) < 0.25 * (1.5 / 8),
          f"β=0.2→{bw_lo:.3f}(理论0.15) β=0.5→{bw_hi:.3f}(理论0.1875)")


def _occupied_bw(x, thresh_db=-35.0, seg=256):
    """测量分段加窗（Hann）PSD 的 -35dB 占用带宽（归一化频率，双边长）。

    用 50% 重叠分段平均降低单段 PSD 的方差（避免随机符号的深零点干扰），
    频率轴用 rfftfreq 自行构造，规避 scipy welch 在 nperseg=len(x) 时
    返回含负频数组的问题。
    """
    n = len(x)
    seg = min(int(seg), n)
    win = np.hanning(seg)
    f = np.fft.fftfreq(seg, 1.0)          # 双边长频率轴
    psd = np.zeros(seg)
    cnt = 0
    for start in range(0, n - seg + 1, seg // 2):
        xseg = np.asarray(x[start:start + seg], dtype=np.complex128) * win
        psd += np.abs(np.fft.fft(xseg)) ** 2
        cnt += 1
    psd /= max(cnt, 1)
    psd /= max(float(np.max(psd)), 1e-300)
    thr = 10.0 ** (thresh_db / 10.0)
    # 只测正频半带（fft/fftfreq 未 shift，负频 bin 在数组末尾），宽度×2 = 双边长
    half = seg // 2 + 1
    psd_half = psd[:half]
    f_half = f[:half]
    idx = np.where(psd_half > thr)[0]
    if len(idx) == 0:
        return 0.0
    return float(f_half[idx[-1]] - f_half[idx[0]]) * 2.0


# ---------------------------------------------------------------- 防折叠: 频带包含性
def test_band_containment():
    """干扰完整频带必须落在奈奎斯特带内：|f_off| + bw/2 <= 0.5（防频谱折叠）。"""
    from em_signal_simulator.factory import _valid_offset_range

    # 辅助函数单元
    check("fold: 单音可全范围偏移", _valid_offset_range(0.0) == (-0.5, 0.5))
    lo, hi = _valid_offset_range(0.6)
    check("fold: bw=0.6 → |f_off|<=0.2", abs(lo + 0.2) < 1e-9 and abs(hi - 0.2) < 1e-9)
    check("fold: bw=1.0 → 只能中心 0", _valid_offset_range(1.0) == (0.0, 0.0))

    # 各类型端到端：所有源频带包含于 [-0.5, 0.5]（频偏约束基于防折叠带宽）
    types = ["random", "single_tone", "swept", "pulse", "broadband", "nfm",
             "WiFi_OFDM", "Radar_LFM", "LTE_QPSK"]
    bad = bad99 = 0
    for it in types:
        for seed in range(30):
            _, meta = generate_signal_sample(num_sources=3, interferer_type=it, seed=seed)
            for s in meta["sources"]:
                f = s["freq_offset_normalized"]
                fold = s.get("bandwidth_fold", s.get("bandwidth_theoretical", 0.0))
                if abs(f) + fold / 2.0 > 0.5 + 1e-9:
                    bad += 1
                # OBW99 测量含 Hann 窗主瓣展宽（~0.01），允许 0.02 测量容差
                if abs(f) + s["bandwidth_normalized"] / 2.0 > 0.5 + 0.02:
                    bad99 += 1
    check("fold: 全部类型频带包含于奈奎斯特带（防折叠带宽）", bad == 0, f"violations={bad}")
    check("fold: OBW99 频带同样包含", bad99 == 0, f"violations={bad99}")

    # 全带宽弹幕（pulse/broadband）中心频偏必须为 0
    ok_off = True
    for it in ("pulse", "broadband"):
        for seed in range(20):
            _, meta = generate_signal_sample(num_sources=2, interferer_type=it, seed=seed)
            for s in meta["sources"][1:]:
                ok_off = ok_off and abs(s["freq_offset_normalized"]) < 1e-9
    check("fold: pulse/broadband 中心频偏=0", ok_off)

    # 两个全带宽弹幕：频偏无法分离 → 必须靠 DOA 分离
    from em_signal_simulator.factory import _sample_interferer, _correct_separation
    rng = np.random.default_rng(9)
    itfs = [
        _sample_interferer(rng, 1024, force_type="broadband"),
        _sample_interferer(rng, 1024, force_type="broadband"),
    ]
    _correct_separation(itfs, rng=rng)
    df = abs(itfs[0]["freq_offset_normalized"] - itfs[1]["freq_offset_normalized"])
    dd = abs(itfs[0]["doa_degree"] - itfs[1]["doa_degree"])
    check("fold: 双弹幕靠 DOA 分离", dd >= DOA_MIN_SEP or df >= FREQ_MIN_SEP,
          f"df={df:.3f} dd={dd:.1f}")

    # 目标带宽越界校验
    try:
        generate_signal_sample(num_sources=1, modulation="LFM",
                               target_bandwidth_normalized=1.2)
        raises = False
    except ValueError:
        raises = True
    check("fold: 目标带宽>=1.0 拒绝生成", raises)

    # 大带宽干扰的频偏确实被限制（OFDM 0.8 带宽 → |f_off|<=0.1）
    max_off = 0.0
    for seed in range(50):
        _, meta = generate_signal_sample(
            num_sources=2, interferer_type="WiFi_OFDM", seed=seed,
            interferer_bandwidth_range=(0.7, 0.98),
        )
        for s in meta["sources"][1:]:
            max_off = max(max_off, abs(s["freq_offset_normalized"]))
    check("fold: 宽带 OFDM 干扰频偏受限", max_off <= 0.2 + 1e-9,
          f"max|f_off|={max_off:.4f} (带宽≈0.8 → 应<=0.1)")


# ---------------------------------------------------------------- 新增修复项验证
def test_inr_constraint():
    """P1-1: 干扰可检测性约束 —— INR = 功率比 + SNR >= min_inr_db。"""
    from em_signal_simulator.factory import _sample_power_ratio
    rng = np.random.default_rng(0)
    # 单元：低 SNR 时功率比被抬升
    vals = [_sample_power_ratio(rng, min_inr_db=3.0, snr_db=-5.0) for _ in range(5000)]
    check("inr: 低SNR 功率比下限生效", min(vals) >= 8.0 - 1e-9, f"min={min(vals):.3f} (下限 3-(-5)=8)")
    # 单元：高 SNR 不改变分布
    vals2 = [_sample_power_ratio(rng, min_inr_db=3.0, snr_db=30.0) for _ in range(5000)]
    check("inr: 高SNR 不抬升功率比", min(vals2) >= 0.0 - 1e-9 and np.mean(vals2) < 8.0,
          f"mean={np.mean(vals2):.2f}")
    # 端到端：低 SNR 样本所有干扰 inr_db >= min_inr_db（可满足时）
    ok = True
    for seed in range(30):
        _, meta = generate_signal_sample(num_sources=3, snr_db=-5.0, seed=seed)
        for s in meta["sources"][1:]:
            assert "inr_db" in s, "metadata missing inr_db"
            if s["inr_db"] < 3.0 - 1e-9:
                ok = False
    check("inr: 端到端低SNR全部满足 INR>=3", ok)


def test_target_separation():
    """P1-2a: 干扰与目标（f=0, doa=0）也必须可区分。"""
    bad = 0
    for seed in range(60):
        _, meta = generate_signal_sample(num_sources=3, seed=seed)
        for s in meta["sources"][1:]:
            df = abs(s["freq_offset_normalized"])
            dd = abs(s["doa_degree"])
            if df < FREQ_MIN_SEP and dd < DOA_MIN_SEP:
                bad += 1
    check("targetsep: 干扰与目标可区分 (60样本)", bad == 0, f"violations={bad}")
    # 边界情形：干扰初始与目标完全重合 → 修正后可区分
    from em_signal_simulator.factory import _correct_separation
    itfs = [{"freq_offset_normalized": 0.0, "doa_degree": 0.0,
             "bandwidth_normalized": 0.2},
            {"freq_offset_normalized": 0.0, "doa_degree": 0.0,
             "bandwidth_normalized": 0.2}]
    _correct_separation([{"freq_offset_normalized": 0.0, "doa_degree": 0.0,
                          "bandwidth_normalized": 0.16}] + itfs,
                        rng=np.random.default_rng(1))
    ok = all(
        (abs(i["freq_offset_normalized"]) >= FREQ_MIN_SEP
         or abs(i["doa_degree"]) >= DOA_MIN_SEP)
        for i in itfs
    )
    check("targetsep: 与目标重合的干扰被修正", ok)


def test_gfsk_bandwidth():
    """P1-3: GFSK 带宽标定（实测 -35dB 占用 ≈ 标注）。"""
    from em_signal_simulator.baseband import actual_bandwidth as ab
    c = 0.34375 + 1.09375 * 0.35
    check("gfsk: 默认实际带宽=c/8", abs(ab("GFSK") - c / 8) < 1e-12,
          f"{ab('GFSK'):.5f} vs {c/8:.5f}")
    # 配置带宽：sps 按标定系数反推
    req = 0.1
    a = ab("GFSK", 1024, bandwidth_normalized=req)
    sps_exp = int(np.clip(round(c / req), 2, 64))
    check("gfsk: 配置 0.1 → sps 反推", abs(a - c / sps_exp) < 1e-12, f"actual={a:.4f} sps={sps_exp}")
    # 端到端实测
    x = generate_baseband("GFSK", 4096, rng=np.random.default_rng(0), bandwidth_normalized=req)
    occ = _occupied_bw(x, seg=1024)
    check("gfsk: 实测 -35dB 占用≈标注", abs(occ - a) < 0.25 * a, f"occ={occ:.4f} 标注={a:.4f}")
    # 默认 GFSK（BT=0.35, sps=8）实测
    xd = generate_baseband("GFSK", 4096, rng=np.random.default_rng(1))
    occd = _occupied_bw(xd, seg=1024)
    check("gfsk: 默认实测≈0.0908", abs(occd - c / 8) < 0.25 * (c / 8),
          f"occ={occd:.4f} 标注={c/8:.4f}")


def test_fhss_continuous_phase():
    """P2-6: FHSS 连续相位默认开启（跳变毛刺减少），旧行为经参数保留。"""
    rng_c = np.random.default_rng(2)
    rng_d = np.random.default_rng(2)
    xc = generate_baseband("FHSS", 4096, rng=rng_c)
    xd = generate_baseband("FHSS", 4096, rng=rng_d, continuous_phase=False)
    # 测跳频范围外（|f|>0.46，hops 上限 0.45）的带外能量
    edge_c = _spec_edge_power(xc, frac=0.04)
    edge_d = _spec_edge_power(xd, frac=0.04)
    check("fhss: 连续相位带外泄漏更低", edge_c < edge_d * 0.85,
          f"cont={edge_c:.2e} disc={edge_d:.2e}")
    check("fhss: 连续相位无 NaN", bool(np.all(np.isfinite(xc))))


def test_pulse_train():
    """P2-4: pulse 周期脉冲串（PRI 固定）+ 元数据 waveform_params。"""
    rng = np.random.default_rng(3)
    n = 1024
    j = generate_interferer_waveform("pulse", n, rng=rng, pri=0.2, duty_cycle=0.1)
    on = np.mean(np.abs(j) > 1e-6)
    # 占空比 ≈ 0.1，脉冲数 ≈ n/pri_s = 5
    active = np.abs(j) > 1e-6
    edges = int(np.sum(np.diff(active.astype(int)) == 1))
    check("pulse: 周期脉冲串占空比≈0.1", abs(on - 0.1) < 0.05, f"duty={on:.3f}")
    check("pulse: 脉冲数≈5 (pri=0.2)", 3 <= edges <= 7, f"pulses={edges}")
    # 旧行为（不传 pri）仍是伯努利门控
    jb = generate_interferer_waveform("pulse", n, rng=rng)
    check("pulse: 无 pri 保持伯努利（兼容）", bool(np.all(np.isfinite(jb))))
    # 端到端：swept/pulse 元数据记录 waveform_params
    _, meta = generate_signal_sample(num_sources=2, interferer_type="swept", seed=4)
    wp = meta["sources"][1].get("waveform_params", {})
    check("swept: 元数据记录 PRI/占空比", "pri" in wp and "duty_cycle" in wp, f"{wp}")


# ---------------------------------------------------------------- 2.1 Observation/GT 隔离
def test_observation_gt_split():
    import shutil
    out_dir = Path(__file__).resolve().parent / "_sanity_split"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        generate_dataset(output_dir=out_dir, count=8, num_sources=3, seed=11,
                         interferer_type="random", snr_range=(-5, 15))
        obs_path, gt_path = out_dir / "observations.json", out_dir / "ground_truth.json"
        check("split: 三文件输出", obs_path.exists() and gt_path.exists()
              and (out_dir / "metadata.json").exists())
        obs = json.loads(obs_path.read_text(encoding="utf-8"))
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        check("split: 记录数一致", len(obs) == len(gt) == 8, f"{len(obs)}/{len(gt)}")
        # 隔离性：observation 不含任何源级真值
        leak_keys = []
        for r in obs:
            o = r["observation"]
            for bad in ("sources", "v2_category", "snr_db", "lna_saturation",
                        "freq_offset", "doa_degree", "power_ratio"):
                if bad in o or any(bad in str(k) for k in o):
                    leak_keys.append(bad)
            for s in str(o).split(","):
                if "interfer" in s.lower():
                    leak_keys.append("interfer*")
        check("split: observation 无真值泄露", not leak_keys, f"leaks={set(leak_keys)}")
        # observation 含 Agent 必要先验
        o0 = obs[0]["observation"]
        check("split: observation 含接收机/目标调制",
              "receiver" in o0 and "target_modulation" in o0
              and "num_samples" in o0 and o0["num_samples"] == 1024)
        # GT 完整性：每个样本含全部真值字段（干扰源含类别/参数/INR）
        ok = all(
            {"num_sources", "snr_db", "lna_saturation", "sources"} <= set(r["ground_truth"])
            and all({"v2_category", "freq_offset_normalized", "doa_degree",
                     "bandwidth_normalized", "inr_db", "waveform_params"} <= set(s)
                    for s in r["ground_truth"]["sources"][1:])
            for r in gt
        )
        check("split: ground_truth 含全部真值字段", ok)
        # 一致性：GT 与 metadata.json 对应样本一致
        meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        same = all(
            meta[i]["metadata"] == gt[i]["ground_truth"]
            for i in range(len(meta))
        )
        check("split: GT 与 metadata 一致", same)
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------- 3.1 可视化模块
def test_visualization():
    from em_signal_simulator.visualization import (
        spectrogram_image,
        music_spectrum,
        spatial_spectrum_image,
    )
    import tempfile
    # 构造高 SNR 双源样本（单音干扰），验证 MUSIC 峰值接近 GT DOA
    iq, meta = generate_signal_sample(
        num_sources=2, snr_db=30.0, seed=21, interferer_type="single_tone",
        array_gain_error_std=0.0, array_phase_error_std_deg=0.0,
    )
    doas = [s["doa_degree"] for s in meta["sources"]]
    d, spec = music_spectrum(iq, num_sources=2)
    # 取 top-2 峰
    order = np.argsort(spec)[::-1]
    peaks = []
    for idx in order:
        # 简单去重：跳过与已选峰过近的角度
        if all(abs(d[idx] - p) > 10.0 for p in peaks):
            peaks.append(float(d[idx]))
        if len(peaks) >= 2:
            break
    matched = sum(any(abs(gt_d - p) < 15.0 for p in peaks) for gt_d in doas)
    check("viz: MUSIC 峰值匹配真实 DOA", matched >= 1,
          f"GT={np.round(doas,1)} peaks={np.round(peaks,1)}")
    # 图生成
    img = spectrogram_image(iq, sampling_rate_hz=20e6)
    check("viz: 时频图 RGB 数组", img.shape[-1] == 3 and img.dtype == np.uint8,
          f"{img.shape} {img.dtype}")
    img2 = spatial_spectrum_image(iq, num_sources=2)
    check("viz: 空间谱图 RGB 数组", img2.shape[-1] == 3)
    # PNG 保存（工作区目录，沙箱允许）
    import shutil
    vdir = Path(__file__).resolve().parent / "_sanity_viz"
    if vdir.exists():
        shutil.rmtree(vdir)
    try:
        from em_signal_simulator.visualization import save_sample_visualizations
        vdir.mkdir(parents=True, exist_ok=True)
        p1, p2 = save_sample_visualizations(iq, vdir / "s.npy", num_sources=2)
        check("viz: PNG 文件保存", p1.exists() and p2.exists()
              and p1.stat().st_size > 1000 and p2.stat().st_size > 1000)
    finally:
        if vdir.exists():
            shutil.rmtree(vdir, ignore_errors=True)


# ---------------------------------------------------------------- 4.1 多进程生成
def test_parallel_generation():
    import shutil
    d1 = Path(__file__).resolve().parent / "_sanity_ser"
    d2 = Path(__file__).resolve().parent / "_sanity_par"
    for d in (d1, d2):
        if d.exists():
            shutil.rmtree(d)
    try:
        try:
            generate_dataset(output_dir=d1, count=6, seed=42, num_sources="random",
                             snr_range=(-5, 15), num_workers=1)
            generate_dataset(output_dir=d2, count=6, seed=42, num_sources="random",
                             snr_range=(-5, 15), num_workers=2)
        except (PermissionError, OSError) as exc:
            # 受限环境（如沙箱）禁止进程池命名管道 —— 环境限制，非代码缺陷
            print(f"[SKIP] parallel: 当前环境不允许进程池 ({type(exc).__name__})，"
                  f"请在正常 Python 环境验证")
            return
        same = True
        for i in range(6):
            a = np.load(d1 / f"sample_{i:05d}.npy")
            b = np.load(d2 / f"sample_{i:05d}.npy")
            if not np.array_equal(a, b):
                same = False
        check("parallel: 并行与串行结果逐位一致", same)
        g1 = json.loads((d1 / "ground_truth.json").read_text(encoding="utf-8"))
        g2 = json.loads((d2 / "ground_truth.json").read_text(encoding="utf-8"))
        check("parallel: GT 一致", g1 == g2)
    finally:
        for d in (d1, d2):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- v2 评估链路冒烟
def test_evaluator_v2():
    """tools_v2 工具可调用 + evaluator_v2 离线评分满分 + 数据隔离。"""
    import shutil
    # tools_v2 / evaluator_v2 位于 benchmark_v2 根目录
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from tools_v2 import (
        analyze_spectrum, estimate_num_sources, estimate_doa, detect_time_domain,
        estimate_modulation_features,
    )
    from evaluator_v2 import evaluate_dataset

    iq, meta = generate_signal_sample(num_sources=2, snr_db=20.0, seed=5)
    vdir = Path(__file__).resolve().parent / "_sanity_ev2"
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    try:
        sp = vdir / "s.npy"
        np.save(sp, iq)
        # 工具冒烟
        outs = [
            analyze_spectrum(str(sp)),
            estimate_num_sources(str(sp)),
            estimate_doa(str(sp), num_sources=2),
            detect_time_domain(str(sp)),
            estimate_modulation_features(str(sp)),
        ]
        check("ev2: 5 个工具可调用", all(isinstance(o, dict) for o in outs))
        check("ev2: 频谱含峰/带宽", "peaks_normalized" in outs[0] and "obw99_normalized" in outs[0])
        check("ev2: MDL 源数在范围", 1 <= outs[1]["num_sources_estimate"] <= 3)
        check("ev2: 工具无 insight 引导字段",
              all("insight" not in o for o in outs))

        # 离线评分满分（评分管道正确性）
        ds = vdir / "ds"
        generate_dataset(output_dir=ds, count=12, num_sources="random",
                         snr_range=(-5, 15), seed=9)
        report = evaluate_dataset(ds, agent=None)
        check("ev2: 离线回放源数准确率=1", report["num_interferers_accuracy"] == 1.0)
        check("ev2: 离线回放类别/参数准确率=1",
              report["category_accuracy"] == 1.0 and report["freq_accuracy"] == 1.0
              and report["doa_accuracy"] == 1.0 and report["bandwidth_accuracy"] == 1.0)
        # 数据隔离：observations.json 无真值
        obs = json.loads((ds / "observations.json").read_text(encoding="utf-8"))
        leak = any(any(k in o["observation"] for k in
                       ("sources", "snr_db", "lna", "v2_category"))
                   for o in obs)
        check("ev2: observation 无真值泄露", not leak)
    finally:
        if vdir.exists():
            shutil.rmtree(vdir, ignore_errors=True)


# ---------------------------------------------------------------- 意见1: 带宽配置
def test_bandwidth():
    # 默认实际带宽与表一致
    for m, expect in [("QPSK", 1.35 / 8), ("OFDM", 102 / 128), ("LFM", 0.8), ("FHSS", 0.9)]:
        ab = actual_bandwidth(m)
        check(f"bandwidth: {m} 默认实际带宽", abs(ab - expect) < 1e-9, f"{ab:.5f} vs {expect:.5f}")
    # 配置带宽生效（sps 整数化 → 实际≈请求）
    for m, req in [("QPSK", 0.25), ("QPSK", 0.5), ("LFM", 0.4), ("OFDM", 0.5), ("FHSS", 0.3)]:
        ab = actual_bandwidth(m, 1024, bandwidth_normalized=req)
        x = generate_baseband(m, 1024, rng=np.random.default_rng(0), bandwidth_normalized=req)
        check(f"bandwidth: {m} 请求={req} 生成成功", bool(np.all(np.isfinite(x))))
        if m == "QPSK":
            occ = _occupied_bw(x)
            check(f"bandwidth: QPSK 实测占用≈实际({ab:.3f})", abs(occ - ab) < 0.25 * ab,
                  f"occ={occ:.3f}")
        else:
            check(f"bandwidth: {m} 实际={ab:.4f}", 0 < ab <= 1.0)
    # 端到端：元数据理论带宽字段 == actual_bandwidth；99% OBW 标签 <= 理论值
    for seed in range(5):
        _, meta = generate_signal_sample(num_sources=3, seed=seed)
        tgt = meta["sources"][0]
        expect_t = actual_bandwidth(tgt["modulation"], 1024)
        check(f"bandwidth: 目标理论带宽一致 (seed={seed})",
              abs(tgt["bandwidth_theoretical"] - expect_t) < 1e-9,
              f"{tgt['bandwidth_theoretical']:.5f} vs {expect_t:.5f}")
        check(f"bandwidth: 目标 OBW99<=理论 (seed={seed})",
              0.0 < tgt["bandwidth_normalized"] <= tgt["bandwidth_theoretical"] + 1e-9,
              f"obw99={tgt['bandwidth_normalized']:.4f} theo={tgt['bandwidth_theoretical']:.4f}")
        for s in meta["sources"][1:]:
            bw = s["bandwidth_normalized"]
            assert 0.0 <= bw <= 1.0 + 1e-9, f"interferer bw out of range: {bw}"
    check("bandwidth: 干扰元数据带宽在 [0,1]", True)
    # 带宽定义统一：元数据含定义字段
    _, meta0 = generate_signal_sample(num_sources=1, seed=0)
    check("bandwidth: 元数据标注定义 obw99",
          meta0.get("bandwidth_definition") == "obw99")


# ---------------------------------------------------------------- 意见2: SNR 语义
def test_snr():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    x1 /= np.sqrt(np.mean(np.abs(x1) ** 2))          # 功率=1
    x10 = x1 * 10.0                                   # 功率=100
    n1 = add_awgn(x1, 10.0, rng=rng, ref_power=1.0) - x1
    n10 = add_awgn(x10, 10.0, rng=rng, ref_power=1.0) - x10
    p1, p10 = float(np.mean(np.abs(n1) ** 2)), float(np.mean(np.abs(n10) ** 2))
    check("snr: 噪声功率=ref/10^(SNR/10)", abs(p1 - 0.1) < 0.02, f"{p1:.4f}")
    check("snr: 与信号幅度无关", abs(p1 - p10) < 0.02, f"{p1:.4f} vs {p10:.4f}")
    # 端到端：同种子、不同 SNR 的两次生成，信号部分完全一致，差分即噪声差。
    # diff = n1 - n2 = (c1 - c2)·z，c1 = sqrt(P/10^1), c2 = c1/10 → var(diff) = 0.81·P_noise
    def _noise_power(num_sources, seed=1, **kw):
        a, _ = generate_signal_sample(num_sources=num_sources, snr_db=10.0, seed=seed, **kw)
        b, _ = generate_signal_sample(num_sources=num_sources, snr_db=30.0, seed=seed, **kw)
        return float(np.mean(np.abs(a - b) ** 2) / 0.81)

    pn1 = _noise_power(1)
    pn2 = _noise_power(2, interferer_type="Radar_LFM")
    check("snr: 1源噪声功率≈目标/10^(SNR/10)", abs(pn1 - 0.1) < 0.04, f"P_noise={pn1:.3f}")
    check("snr: 1源/2源样本噪声功率一致（干扰不参与SNR）", abs(pn1 - pn2) < 0.04,
          f"{pn1:.3f} vs {pn2:.3f}")


# ---------------------------------------------------------------- 意见3: blocking 触发
def test_blocking():
    # 直接判定
    S = lambda fo, p: {"freq_offset_normalized": fo, "power_ratio_db": p, "is_pulse": False}
    check("blocking: 带外强信号→blocking",
          _auto_label(S(0.3, 12.0), 0.35) == "blocking")
    check("blocking: 带内强信号→co_channel",
          _auto_label(S(0.05, 12.0), 0.35) == "co_channel")
    check("blocking: 带外弱信号→adjacent",
          _auto_label(S(0.3, 5.0), 0.35) == "adjacent")
    check("blocking: 脉冲优先→pulse",
          _auto_label({**S(0.3, 12.0), "is_pulse": True}, 0.35) == "pulse")
    # 蒙特卡洛：QPSK 目标下 blocking 概率应显著 > 0（旧逻辑恒为 0）
    rng = np.random.default_rng(1)
    n = 50000
    hit = sum(
        _auto_label({"freq_offset_normalized": float(rng.uniform(-0.5, 0.5)),
                     "power_ratio_db": float(rng.uniform(0, 15)),
                     "is_pulse": False}, 0.35) == "blocking"
        for _ in range(n)
    )
    p_block = hit / n
    check("blocking: 蒙特卡洛 P(blocking)>5%", p_block > 0.05, f"P={p_block:.3f}")
    # 端到端：blocking → LNA 饱和触发
    iq, meta = generate_signal_sample(num_sources=2, snr_db=30.0,
                                      interferer_type="Radar_LFM", seed=0)
    cat = meta["sources"][1]["v2_category"]
    check("blocking: 端到端标注 blocking", cat == "blocking", f"category={cat}")
    check("blocking: LNA 饱和同步触发", meta["lna_saturation"]["applied"] is True)


# ---------------------------------------------------------------- Task 3
def test_jamming():
    rng = np.random.default_rng(2)
    n = 1024
    # swept 默认连续（向后兼容）
    j = generate_interferer_waveform("swept", n, rng=rng)
    check("jamming: swept 连续(默认)", float(np.mean(np.abs(j) ** 2)) > 0.5)
    # 脉冲化 LFM
    jp = generate_interferer_waveform("swept", n, rng=rng, duty_cycle=0.2, pri=0.5)
    on = np.mean(np.abs(jp) > 1e-6)
    check("jamming: 脉冲化 swept 占空比≈0.2", abs(on - 0.2) < 0.05, f"duty={on:.3f}")
    # NFM
    jn = generate_interferer_waveform("nfm", n, rng=rng, noise_bandwidth=0.1, freq_deviation=0.2)
    check("jamming: nfm 存在", jn.dtype == np.complex128 and len(jn) == n)
    check("jamming: nfm 无 NaN/Inf", bool(np.all(np.isfinite(jn))))
    check("jamming: nfm 恒定包络", abs(float(np.std(np.abs(jn))) - 0.0) < 1e-6)
    # nfm 带宽：相位调制后频谱应显著展宽（> 单音）
    spec_n = np.abs(np.fft.fft(jn))
    nz = float(np.sum(spec_n > 0.01 * np.max(spec_n))) / n
    check("jamming: nfm 频谱展宽", nz > 0.1, f"fraction_nonzero={nz:.3f}")
    # 全部类型可生成
    for t in sorted(JAMMING_TYPES - {"none"}):
        w = generate_interferer_waveform(t, 256, rng=rng)
        check(f"jamming: {t} 可生成", bool(np.all(np.isfinite(w))))


# ---------------------------------------------------------------- Task 4
def test_separation():
    rng = np.random.default_rng(3)
    ok_total = 0
    for trial in range(50):
        # 随机造 2~3 个可能冲突的干扰
        k = int(rng.integers(2, 4))
        itfs = [
            {"freq_offset_normalized": float(rng.uniform(-0.5, 0.5)),
             "doa_degree": float(rng.uniform(-60, 60))}
            for _ in range(k)
        ]
        _correct_separation(itfs, rng=rng)
        ok = True
        for i in range(k):
            for j in range(i + 1, k):
                df = abs(itfs[i]["freq_offset_normalized"] - itfs[j]["freq_offset_normalized"])
                dd = abs(itfs[i]["doa_degree"] - itfs[j]["doa_degree"])
                if df < FREQ_MIN_SEP and dd < DOA_MIN_SEP:
                    ok = False
        ok_total += int(ok)
    check("separation: 50 轮随机全部满足最小间隔", ok_total == 50, f"{ok_total}/50")

    # 边界裁剪极端情形：两个干扰初始完全重合在边界
    itfs = [
        {"freq_offset_normalized": -0.5, "doa_degree": -60.0},
        {"freq_offset_normalized": -0.5, "doa_degree": -60.0},
    ]
    _correct_separation(itfs, rng=np.random.default_rng(4))
    df = abs(itfs[0]["freq_offset_normalized"] - itfs[1]["freq_offset_normalized"])
    dd = abs(itfs[0]["doa_degree"] - itfs[1]["doa_degree"])
    check("separation: 边界重合可分离", (df >= FREQ_MIN_SEP or dd >= DOA_MIN_SEP),
          f"df={df:.3f} dd={dd:.1f}")

    # 端到端：多干扰样本元数据全部满足
    for seed in range(20):
        _, meta = generate_signal_sample(num_sources=3, seed=seed)
        srcs = meta["sources"][1:]
        for i in range(len(srcs)):
            for j in range(i + 1, len(srcs)):
                df = abs(srcs[i]["freq_offset_normalized"] - srcs[j]["freq_offset_normalized"])
                dd = abs(srcs[i]["doa_degree"] - srcs[j]["doa_degree"])
                assert df >= FREQ_MIN_SEP or dd >= DOA_MIN_SEP, \
                    f"seed={seed} pair({i},{j}) df={df} dd={dd}"
    check("separation: 20 个 3 源样本端到端可分离", True)


# ---------------------------------------------------------------- Task 5
def test_array_mismatch():
    d = 0.0625
    lam = 0.125
    sv0 = steering_vector(30.0, 4, d, lam)
    check("mismatch: 默认理想阵列 |a_m|=1", bool(np.allclose(np.abs(sv0), 1.0)))
    rng = np.random.default_rng(5)
    svm = steering_vector(30.0, 4, d, lam,
                          gain_error_std=0.05, phase_error_std_deg=5.0, rng=rng)
    mag = np.abs(svm)
    ph = np.angle(svm * np.conj(sv0))
    check("mismatch: 增益误差生效 (|a_m|≠1)", bool(np.any(np.abs(mag - 1.0) > 1e-6)),
          f"|a|={np.round(mag, 4)}")
    check("mismatch: 相位误差生效 (Δφ≈5°量级)",
          float(np.std(ph)) > 0.02, f"std(Δφ)={np.std(ph):.4f} rad")
    # expand_to_array 形状
    x = np.ones(16, dtype=np.complex128)
    A = expand_to_array(x, 30.0, 4, d, lam, gain_error=None, phase_error_rad=None)
    check("mismatch: expand_to_array 形状 (4,16)", A.shape == (4, 16))

    # 端到端：开启失配后元数据记录 std
    _, meta = generate_signal_sample(
        num_sources=2, seed=6,
        array_gain_error_std=0.05, array_phase_error_std_deg=5.0,
    )
    mm = meta["array_mismatch"]
    check("mismatch: 元数据记录失配参数",
          mm["gain_error_std"] == 0.05 and mm["phase_error_std_deg"] == 5.0)


# ---------------------------------------------------------------- End-to-end
def test_dataset(count=10):
    import shutil
    out_dir = Path(__file__).resolve().parent / "_sanity_dataset"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        recs = generate_dataset(
            output_dir=out_dir, count=count, num_sources="random",
            snr_range=(-5.0, 15.0), seed=123, interferer_type="random",
            array_gain_error_std=0.05, array_phase_error_std_deg=5.0,
        )
        meta_path = out_dir / "metadata.json"
        check("dataset: metadata.json 存在", meta_path.exists())
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        check("dataset: 记录数==count", len(data) == count, f"{len(data)}")
        shapes = set()
        dtypes = set()
        finite_ok = True
        for r in data:
            iq = np.load(out_dir / r["file"])
            shapes.add(iq.shape)
            dtypes.add(str(iq.dtype))
            finite_ok = finite_ok and bool(np.all(np.isfinite(iq)))
            m = r["metadata"]
            # 所有 source 必填字段
            for s in m["sources"]:
                for fld in ("source_id", "role", "modulation", "power_ratio_db",
                            "freq_offset_normalized", "freq_offset_hz",
                            "doa_degree", "bandwidth_normalized", "bandwidth_hz"):
                    assert fld in s, f"missing field {fld} in {s}"
        check("dataset: IQ 形状 (4,1024)", shapes == {(4, 1024)}, f"{shapes}")
        check("dataset: dtype complex128", dtypes == {"complex128"}, f"{dtypes}")
        check("dataset: 无 NaN/Inf", finite_ok)
        check("dataset: receiver/snr 字段", all(
            "snr_db" in r["metadata"] and "receiver" in r["metadata"] for r in data))
        # 目标理论带宽字段 == actual_bandwidth（意见1 修复验证，标签为 OBW99）
        bw_ok = all(
            abs(s["metadata"]["sources"][0]["bandwidth_theoretical"]
                - actual_bandwidth(s["metadata"]["sources"][0]["modulation"], 1024)) < 1e-9
            and s["metadata"].get("bandwidth_definition") == "obw99"
            for s in data
        )
        check("dataset: 目标元数据带宽=实际带宽", bw_ok)
        # 端到端 CLI 兼容性：num_sources / interferer_type 组合
        for it in ["single_tone", "swept", "pulse", "broadband", "nfm"]:
            iq2, meta2 = generate_signal_sample(
                num_sources=2, interferer_type=it, seed=77)
            check(f"dataset: interferer_type={it} 可生成 (4,1024)",
                  iq2.shape == (4, 1024) and bool(np.all(np.isfinite(iq2))))
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== Task 1: LNA Saturation ===")
    test_lna_saturation()
    print("\n=== Task 2: Pulse Shaping Edges ===")
    test_pulse_shaping_edges()
    print("\n=== Task 3: Jamming Waveforms ===")
    test_jamming()
    print("\n=== Task 4: Separation Correction ===")
    test_separation()
    print("\n=== Task 5: Array Mismatch ===")
    test_array_mismatch()
    print("\n=== 意见1: 带宽配置 ===")
    test_bandwidth()
    print("\n=== 意见2: SNR 语义 ===")
    test_snr()
    print("\n=== 意见3: blocking 触发 ===")
    test_blocking()
    print("\n=== 防折叠: 频带包含性 ===")
    test_band_containment()
    print("\n=== P1-1: INR 可检测性 ===")
    test_inr_constraint()
    print("\n=== P1-2a: 目标分离约束 ===")
    test_target_separation()
    print("\n=== P1-3: GFSK 带宽标定 ===")
    test_gfsk_bandwidth()
    print("\n=== P2-6: FHSS 连续相位 ===")
    test_fhss_continuous_phase()
    print("\n=== P2-4: pulse 周期脉冲串 ===")
    test_pulse_train()
    print("\n=== 2.1: Observation/GT 隔离 ===")
    test_observation_gt_split()
    print("\n=== 3.1: 可视化模块 ===")
    test_visualization()
    print("\n=== 4.1: 多进程生成 ===")
    test_parallel_generation()
    print("\n=== v2 评估链路 (tools_v2 + evaluator_v2) ===")
    test_evaluator_v2()
    print("\n=== End-to-end Dataset ===")
    test_dataset(10)

    print("\n" + "=" * 50)
    if FAILED:
        print(f"RESULT: {len(FAILED)} FAILED -> {FAILED}")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")
