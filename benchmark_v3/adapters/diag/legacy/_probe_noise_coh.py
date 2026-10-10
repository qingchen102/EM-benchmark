"""零 token 噪声探针：Frontend 的噪声开关会怎样影响仪器判据。

复刻 src/cityrf/receiver.py 的噪声数学（共模白噪 / 有色带限 / 稀疏脉冲），
并逐字复制 capture_tools.py 的 _band_coherence，回答两问：

 ① 共模（correlated_noise）噪声的 rho 会不会污染相干门控（PEAK_COH_MIN=0.6）？
    —— 既是"假通过"风险（噪声自己越过 0.6），也是"掩蔽"风险（真源被噪声把相干度拉下去）。
 ② 有色（colored_noise）/ 脉冲（impulse_noise）会不会造出"过 8 dB 门限"的假峰，
    以及它们的相干度是否会被门控挡住 / 会把噪声底抬高多少（掩蔽预算）。

量纲口径（重要）：`cityrf.waveforms.gaussian` = `sqrt(-log u)·exp(2jπv)`，E|g|²=1，
是**单位平均功率**复高斯；`receiver._colored` 也归一到单位功率。本脚本用 `cnorm` 复刻该口径。

不做任何写操作；不依赖平台依赖（不 import cityrf）。
"""
import numpy as np
from scipy.signal import firwin, fftconvolve
from scipy.ndimage import median_filter

FS = 4e6
N = 40000          # 10 ms @ 4 MHz，与采集窗一致
M = 8              # 阵元数
K_B = 1.380649e-23
RX_CUTOFF = 810e3  # gen_dataset RX_FRAC 口径：firwin 截止
PEAK_COH_MIN = 0.6
PEAK_OVER_DB = 8.0
PEAK_FLOOR_WIN_HZ = 300e3
PEAK_COH_WIN_HZ = 20e3
NF_DB = 5.0
N0 = K_B * 290.0 * 10 ** (NF_DB / 10)   # W/Hz

_TAPS = firwin(129, RX_CUTOFF, fs=FS)


# ---- capture_tools._band_coherence 的逐字副本（见 capture_tools.py:61-76） ----
def band_coherence(iq, fs, center_hz, win_hz):
    n = iq.shape[-1]
    f = np.fft.fftfreq(n, 1 / fs)
    m = np.abs(f - center_hz) <= win_hz
    if int(m.sum()) < 2:
        return float("nan")
    x = np.fft.fft(iq, axis=-1)[:, m]
    x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-30)
    r = x @ x.conj().T / x.shape[1]
    w = np.linalg.eigvalsh(r)[::-1].real
    return float(w[0] / (w.sum() + 1e-30))


def cnorm(n, rng):
    """单位平均功率复高斯（复刻 cityrf.waveforms.gaussian 的量纲）。"""
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def rx_filter(x):
    return fftconvolve(x, _TAPS[None, :], mode="same", axes=1)


def bl_noise(n, f0, bw, rng):
    """带限复白噪，归一到单位平均功率。"""
    w = cnorm(n, rng)
    W = np.fft.fft(w)
    f = np.fft.fftfreq(n, 1 / FS)
    W[np.abs(f - f0) > bw / 2] = 0
    y = np.fft.ifft(W)
    return y / np.sqrt(np.mean(np.abs(y) ** 2))


def colored_like(n, bw, rng):
    """复刻 receiver._colored：白噪过 firwin(65, bw/2) 再归单位功率。"""
    white = cnorm(n, rng)
    taps = firwin(65, min(bw / 2, 0.45 * FS), fs=FS)
    y = fftconvolve(white, taps, mode="same")
    return y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-30)


def impulses_like(n, rate_hz, width_us, power_db, n0, rng):
    """复刻 receiver._impulses（幅度相对热噪声功率谱）。"""
    width = max(int(round(width_us * 1e-6 * FS)), 1)
    period = max(int(round(FS / rate_hz)), width + 1)
    start = int(rng.integers(0, period))
    out = np.zeros(n, complex)
    amp = np.sqrt(n0 * FS) * 10 ** (power_db / 20)
    pos = start
    while pos < n:
        end = min(pos + width, n)
        out[pos:end] += amp * cnorm(end - pos, rng)
        pos += period
    return out


def over_floor_profile(x):
    """复刻 _residual_bands 的 over_floor 口径：通带 PSD(阵元平均) - 局部中位底(±150 kHz)。"""
    n = x.shape[-1]
    f = np.fft.fftfreq(n, 1 / FS)
    m = np.abs(f) <= RX_CUTOFF
    psd = (np.abs(np.fft.fft(x, axis=-1)) ** 2)[:, m].mean(axis=0)
    db = 10 * np.log10(psd + 1e-30)
    half = max(int(round(PEAK_FLOOR_WIN_HZ / (FS / n) / 2)), 1)
    floor = median_filter(db, size=2 * half + 1, mode="nearest")
    return f[m], db - floor


def inband_power(x):
    """接收滤波后 ±810 kHz 内的平均功率（阵元平均）。"""
    n = x.shape[-1]
    f = np.fft.fftfreq(n, 1 / FS)
    m = np.abs(f) <= RX_CUTOFF
    return float((np.abs(np.fft.fft(x, axis=-1)) ** 2)[:, m].mean())


def white_arr(n, rng, scale=1.0):
    return scale * np.sqrt(N0 * FS) * cnorm(n, rng)


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


rng = np.random.default_rng(20261009)
print(f"n0={N0:.4e} W/Hz  n0*fs={N0*FS:.4e} W  "
      f"= {10*np.log10(N0*FS/1e-3):.2f} dBm（未滤波）；"
      f"经 1.62 MHz 带通后 {10*np.log10(N0*FS*1620/4000/1e-3):.2f} dBm（= 真值 -106.9）")

hdr("A) 纯共模白噪：rho 与相干度")
print(f"{'rho':>6} {'实测 λ1/Σλ':>12} {'解析 (1+7r)/8':>14} {'过 0.6 门限?':>12}")
for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.543, 0.6, 0.8, 1.0]:
    common = cnorm(N, rng)
    x = np.zeros((M, N), complex)
    for m in range(M):
        local = cnorm(N, rng)
        x[m] = np.sqrt(rho) * common + np.sqrt(max(1 - rho, 0)) * local
    x = rx_filter(x)
    c = band_coherence(x, FS, 0.0, PEAK_COH_WIN_HZ)
    print(f"{rho:>6.3f} {c:>12.4f} {(1+7*rho)/8:>14.4f} {'是 ← 危险' if c >= PEAK_COH_MIN else '否':>12}")

hdr("B) 真源 + 共模噪声：真源还能不能被门控通过")
print("（每阵元 SNR = 源功率/噪声功率；源为 300 kHz 带限点源，rank-1 随机来向）")
print(f"{'rho':>6} " + " ".join(f"{s:>7}dB" for s in [-15, -10, -5, 0, 5, 10]))
for rho in [0.0, 0.2, 0.3, 0.4, 0.8]:
    row = []
    for snr_db in [-15, -10, -5, 0, 5, 10]:
        common = cnorm(N, rng)
        s = bl_noise(N, 0.0, 300e3, rng)
        a = np.exp(1j * rng.uniform(0, 2 * np.pi, M))
        x = np.outer(a, np.sqrt(10 ** (snr_db / 10)) * s)
        for m in range(M):
            local = cnorm(N, rng)
            x[m] += np.sqrt(rho) * common + np.sqrt(max(1 - rho, 0)) * local
        x = rx_filter(x)
        c = band_coherence(x, FS, 0.0, PEAK_COH_WIN_HZ)
        row.append(f"{c:>7.3f}{'*' if c < PEAK_COH_MIN else ' '}")
    print(f"{rho:>6.2f} " + " ".join(row))
print("（* = 低于 0.6 门限 → 真源会被判掉）")

hdr("C) 有色噪声：带内鼓包会不会过 8 dB？相干度能否被门控挡住？")
print("（colored_bandwidth_hz=200 kHz，按 receiver._colored 每阵元独立抽取）")
for cdb in [-10, 0, 10]:
    x = np.zeros((M, N), complex)
    n0c = N0 * 10 ** (cdb / 10)
    for m in range(M):
        x[m] = white_arr(N, rng)
        x[m] += np.sqrt(n0c * FS) * colored_like(N, 200e3, rng)
    x = rx_filter(x)
    _, over = over_floor_profile(x)
    peak = float(over.max())
    coh_bump = band_coherence(x, FS, 0.0, PEAK_COH_WIN_HZ)
    coh_flat = band_coherence(x, FS, 700e3, PEAK_COH_WIN_HZ)
    print(f"colored_noise_db={cdb:>4}: 最大 over_floor={peak:>6.2f} dB "
          f"(过 8 dB? {'是' if peak >= PEAK_OVER_DB else '否'})  "
          f"鼓包相干={coh_bump:.3f} 平区相干={coh_flat:.3f} "
          f"(过 0.6 门控? {'是 ← 危险' if max(coh_bump, coh_flat) >= PEAK_COH_MIN else '否'})")

hdr("D) 脉冲噪声：抬底幅度与相干度")
for pdb in [15, 20, 25, 30]:
    x = np.zeros((M, N), complex)
    for m in range(M):
        x[m] = white_arr(N, rng)
        x[m] += impulses_like(N, 200.0, 5.0, pdb, N0, rng)
    x = rx_filter(x)
    _, over = over_floor_profile(x)
    c = band_coherence(x, FS, 0.0, PEAK_COH_WIN_HZ)
    print(f"impulse_power_db={pdb:>3}: 最大 over_floor={over.max():>6.2f} dB  相干={c:.3f} "
          f"(过 0.6 门控? {'是 ← 危险' if c >= PEAK_COH_MIN else '否'})")

hdr("E) 共模噪声对“真源带”的压制：门控通过的 SNR 下限")
print("（判据：真源带 λ1/Σλ ≥ 0.6；表中给出恰好落到 0.6 的每阵元 SNR）")
for rho in [0.0, 0.2, 0.3, 0.4, 0.54, 0.8]:
    lo, hi = -30.0, 30.0          # 不变量：lo 失败、hi 通过
    for _ in range(40):
        mid = (lo + hi) / 2
        common = cnorm(N, rng)
        s = bl_noise(N, 0.0, 300e3, rng)
        a = np.exp(1j * rng.uniform(0, 2 * np.pi, M))
        x = np.outer(a, np.sqrt(10 ** (mid / 10)) * s)
        for m in range(M):
            local = cnorm(N, rng)
            x[m] += np.sqrt(rho) * common + np.sqrt(max(1 - rho, 0)) * local
        x = rx_filter(x)
        if band_coherence(x, FS, 0.0, PEAK_COH_WIN_HZ) >= PEAK_COH_MIN:
            hi = mid
        else:
            lo = mid
    print(f"rho={rho:>4.2f}: 门控通过的 SNR 下限 ≈ {hi:>6.2f} dB（低于它真源被判掉）")

hdr("F) 噪声开关抬底预算：对弱源的掩蔽幅度")
print("（基准 = 纯白热噪的带内平均功率；'等效倍率' = 额外贡献 / 白热噪）")
base = inband_power(rx_filter(np.stack([white_arr(N, rng) for _ in range(M)])))


def show(tag, extra):
    x = np.stack([white_arr(N, rng) + extra(m) for m in range(M)])
    lvl = inband_power(rx_filter(x))
    rise = 10 * np.log10(lvl / base)
    print(f"{tag:<36}{rise:>8.2f} dB{10 ** (rise / 10) - 1:>14.2f}×")


print(f"{'配置':<36}{'抬底':>11}{'等效倍率':>14}")
for cdb in [-10, 0, 10]:
    n0c = N0 * 10 ** (cdb / 10)
    show(f"colored_noise_db={cdb}",
         lambda m, s=np.sqrt(n0c * FS): s * colored_like(N, 200e3, rng))
for pdb in [15, 25, 30, 40, 50]:
    show(f"impulse_power_db={pdb}",
         lambda m, p=pdb: impulses_like(N, 200.0, 5.0, p, N0, rng))
show("impulse rate=500Hz width=20us pdb=30",
     lambda m: impulses_like(N, 500.0, 20.0, 30.0, N0, rng))
cm = cnorm(N, rng)
xcor = np.stack([np.sqrt(N0 * FS) * (np.sqrt(0.3) * cm + np.sqrt(0.7) * cnorm(N, rng))
                 for _ in range(M)])
rise = 10 * np.log10(inband_power(rx_filter(xcor)) / base)
print(f"{'correlated rho=0.3（白噪改写，非叠加）':<36}{rise:>8.2f} dB{10 ** (rise / 10) - 1:>14.2f}×")
print("（理论：脉冲占空比 = rate*width/fs = 200*20/4e6 = 1e-3 → 等效倍率 = 1e-3·10^(pdb/10) = "
      f"{1e-3*10**(20/10):.3f}(20dB) / {1e-3*10**(30/10):.3f}(30dB)）")
