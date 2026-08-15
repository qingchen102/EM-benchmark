"""可视化导出模块：时频图（Spectrogram）与空间谱（MUSIC/Beamforming）。

用途：为多模态（VLM）Agent 提供视觉模态数据资产 ——
- spectrogram_image:    (4,1024) IQ → 时频图 PNG/RGB 数组
- music_spectrum:       4 元 ULA → MUSIC 空间方位谱（数值）
- spatial_spectrum_image: 空间谱 → 图
- save_sample_visualizations: 一键导出单个样本的两张图

所有绘图函数返回 RGB uint8 数组（(H, W, 3)），path 给定时同时保存 PNG。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# matplotlib 为延迟导入（仅绘图时依赖，纯数值路径不需要）


def _to_single_channel(iq):
    """(num_antennas, num_samples) → 单通道（功率平均）。"""
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"iq must be (num_antennas, num_samples), got {x.shape}")
    # 功率平均合成，保留相位参考（天线 0 的相位）
    power = np.mean(np.abs(x) ** 2, axis=0)
    ref = x[0] / (np.abs(x[0]) + 1e-12)
    return ref * np.sqrt(power)


def spectrogram_image(iq, sampling_rate_hz=20e6, nperseg=128, noverlap=None,
                      dyn_range_db=60.0, title="Spectrogram", path=None,
                      figsize=(10, 5), dpi=120):
    """绘制时频图（dB 尺度，归一化到 [−dyn_range_db, 0]）。

    返回 RGB uint8 数组；path 给定时保存 PNG。
    """
    from scipy.signal import spectrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _to_single_channel(iq)
    noverlap = int(nperseg * 0.75) if noverlap is None else int(noverlap)
    # 复数输入 → 显式双边谱 + fftshift（f 单调 [-fs/2, fs/2]，避免 pcolormesh 警告）
    f, t, S = spectrogram(x, fs=float(sampling_rate_hz), nperseg=int(nperseg),
                          noverlap=noverlap, mode="psd", return_onesided=False)
    f = np.fft.fftshift(f)
    S = np.fft.fftshift(S, axes=0)
    psd = np.abs(S) ** 2 if not np.iscomplexobj(S) else np.abs(S)
    psd_db = 10.0 * np.log10(psd / (np.max(psd) + 1e-300) + 1e-300)
    psd_db = np.clip(psd_db, -float(dyn_range_db), 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(t, f / 1e6, psd_db, shading="auto", cmap="viridis",
                         vmin=-float(dyn_range_db), vmax=0.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    return _fig_to_rgb(fig, path, dpi)


def music_spectrum(iq, num_sources=None, doa_range=(-90.0, 90.0), num_points=361,
                   antenna_spacing_wavelength=0.5):
    """MUSIC 空间方位谱。

    iq: (num_antennas, num_samples)。返回 (doas_deg, spectrum)。
    源数未知时默认 num_sources = M - 1（最小噪声子空间维度 1）。
    """
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"iq must be (num_antennas, num_samples), got {x.shape}")
    m, n = x.shape
    if n < m:
        raise ValueError("num_samples must be >= num_antennas for MUSIC")
    if num_sources is None:
        num_sources = max(1, m - 1)
    num_sources = int(np.clip(num_sources, 1, m - 1))

    r = (x @ x.conj().T) / n
    evals, evecs = np.linalg.eigh(r)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    noise_subspace = evecs[:, num_sources:]          # (M, M - num_sources)
    enen = noise_subspace @ noise_subspace.conj().T   # (M, M)

    doas = np.linspace(float(doa_range[0]), float(doa_range[1]), int(num_points))
    theta = np.deg2rad(doas)
    idx = np.arange(m)
    # a(θ)_k = exp(j·2π·(d/λ)·sinθ·k)，d/λ = antenna_spacing_wavelength
    a = np.exp(1j * 2.0 * np.pi * float(antenna_spacing_wavelength)
               * np.sin(theta[:, None]) * idx[None, :])          # (P, M)
    proj = np.real(np.einsum("pm,mn,pn->p", a, enen, a.conj()))
    spectrum = 1.0 / np.clip(proj, 1e-12, None)
    return doas, spectrum


def spatial_spectrum_image(iq, num_sources=None, title="MUSIC Spatial Spectrum",
                           path=None, figsize=(8, 4), dpi=120,
                           antenna_spacing_wavelength=0.5):
    """绘制 MUSIC 空间谱图。返回 RGB 数组；path 给定时保存 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doas, spectrum = music_spectrum(
        iq, num_sources=num_sources,
        antenna_spacing_wavelength=antenna_spacing_wavelength,
    )
    spec_db = 10.0 * np.log10(spectrum / (np.max(spectrum) + 1e-300) + 1e-300)
    spec_db = np.clip(spec_db, -40.0, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(doas, spec_db)
    ax.set_xlabel("DOA (deg)")
    ax.set_ylabel("Spatial spectrum (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_rgb(fig, path, dpi)


def save_sample_visualizations(iq, sample_path, sampling_rate_hz=20e6,
                               num_sources=None, suffix=""):
    """一键导出样本的两张图：{sample}_spectrogram.png / {sample}_spatial.png。

    返回 (spectrogram_path, spatial_path)。
    """
    sample_path = Path(sample_path)
    base = sample_path.with_suffix("")
    spec_path = sample_path.with_name(f"{base.name}{suffix}_spectrogram.png")
    spat_path = sample_path.with_name(f"{base.name}{suffix}_spatial.png")
    spectrogram_image(iq, sampling_rate_hz=sampling_rate_hz, path=spec_path,
                      title=f"{base.name} Spectrogram")
    spatial_spectrum_image(iq, num_sources=num_sources, path=spat_path,
                           title=f"{base.name} MUSIC Spatial Spectrum")
    return spec_path, spat_path


def _fig_to_rgb(fig, path=None, dpi=120):
    """渲染 fig → RGB uint8 数组（可选保存 PNG）。"""
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    if path is not None:
        Path(path).write_bytes(buf.getvalue())
        buf.seek(0)
    from PIL import Image
    img = Image.open(buf).convert("RGB")
    return np.asarray(img)
