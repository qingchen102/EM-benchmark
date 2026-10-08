"""Receiver effects on the composite antenna-port signal."""
import numpy as np
from scipy.signal import firwin, fftconvolve
from .waveforms import gaussian, sub_seed, uniform

K_B = 1.380649e-23


def _colored(indices, fs, bandwidth_hz, seed):
    """白噪声过低通，得到带限有色底噪，再归一回单位功率。"""
    from scipy.signal import firwin, fftconvolve
    white=gaussian(indices, seed)
    taps=firwin(65, min(bandwidth_hz/2, 0.45*fs), fs=fs)
    y=fftconvolve(white, taps, mode="same")
    rms=np.sqrt(np.mean(abs(y)**2)+1e-30)
    return y/rms


def _impulses(indices, fs, rate_hz, width_us, power_db, n0, seed):
    """稀疏短脉冲。幅度相对热噪声功率谱，宽度和出现率可调。"""
    width=max(int(round(width_us*1e-6*fs)), 1)
    period=max(int(round(fs/rate_hz)), width+1)
    start=int(np.floor(uniform_index(indices[0], seed)*period))
    out=np.zeros(len(indices), complex)
    pos=start
    amp=np.sqrt(n0*fs)*10**(power_db/20)
    while pos < indices[-1]:
        local=pos-indices[0]
        if 0 <= local < len(out):
            burst=gaussian(np.arange(pos, pos+width), sub_seed(seed, f"hit/{pos}"))
            end=min(local+width, len(out))
            out[local:end]+=amp*burst[:end-local]
        pos+=period
    return out


def uniform_index(index, seed):
    return float(uniform(np.array([index]), seed)[0])


def receive(arrival, rx, fs, seed, start_index=0):
    if arrival.ndim != 2 or arrival.shape[0] != rx.antenna.elements:
        raise ValueError("arrival array does not match receiver")
    f=rx.frontend
    x=np.asarray(arrival,dtype=np.complex128).copy()
    n=np.arange(start_index,start_index+x.shape[1])
    # 相位噪声乘在信号上：本振抖动，发生在加性噪声之前。
    if f.phase_noise_enabled and f.phase_noise_deg > 0:
        sigma=np.deg2rad(f.phase_noise_deg)
        step=sigma*np.real(gaussian(n,sub_seed(seed,f"{rx.id}/phase/common")))
        common=np.exp(1j*np.cumsum(step))
        for m in range(x.shape[0]):
            local=sigma*0.05*np.real(gaussian(n,sub_seed(seed,f"{rx.id}/phase/{m}")))
            x[m]*=common*np.exp(1j*np.cumsum(local))
    # White equivalent input noise; two-sided complex PSD N0 W/Hz.
    n0=K_B*f.temperature_k*10**(f.noise_figure_db/10) if f.noise_enabled else 0.
    if n0 or f.colored_noise_enabled or f.impulse_noise_enabled:
        common=gaussian(n,sub_seed(seed,f"{rx.id}/noise/common")) if f.correlated_noise_enabled else None
        rho=float(f.noise_correlation) if f.correlated_noise_enabled else 0.
        for m in range(x.shape[0]):
            if n0:
                local=gaussian(n,sub_seed(seed,f"{rx.id}/noise/{m}"))
                white=np.sqrt(rho)*common+np.sqrt(max(1-rho,0))*local if common is not None else local
                x[m]+=np.sqrt(n0*fs)*white
            if f.colored_noise_enabled:
                colored=_colored(n,fs,f.colored_bandwidth_hz,sub_seed(seed,f"{rx.id}/colored/{m}"))
                n0c=n0*10**(f.colored_noise_db/10) if n0 else K_B*f.temperature_k*10**(f.colored_noise_db/10)
                x[m]+=np.sqrt(n0c*fs)*colored
            if f.impulse_noise_enabled:
                x[m]+=_impulses(n,fs,f.impulse_rate_hz,f.impulse_width_us,
                               f.impulse_power_db,n0 or K_B*f.temperature_k,
                               sub_seed(seed,f"{rx.id}/impulse/{m}"))
    if f.imd3_enabled:
        a3=1/(3*10**((f.iip3_dbm-30)/10))
        x=x-a3*x*np.abs(x)**2
    pressure=float(np.mean(abs(x)**2))
    if f.soft_limit_input_dbm is not None:
        limit=np.sqrt(10**((f.soft_limit_input_dbm-30)/10))
        amplitude=abs(x)
        x*=np.divide(limit*np.tanh(amplitude/limit),amplitude,
                     out=np.ones_like(amplitude),where=amplitude>0)
    x*=np.exp(-2j*np.pi*rx.tuning_offset_hz*n/fs)[None,:]
    taps=firwin(f.filter_taps,rx.bandwidth_hz/2,fs=fs)
    x=fftconvolve(x,taps[None,:],mode="same",axes=1)
    gain=10**(f.gain_db/20)
    x*=gain
    return x.astype(np.complex64),{
        "input_composite_power_w":pressure,
        "noise_psd_w_per_hz":n0,
        "equivalent_noise_bandwidth_hz":float(fs*np.sum(taps*taps)),
        "expected_output_noise_w":float(n0*fs*np.sum(taps*taps)*gain**2),
        "noise_expectation_scope":"linear receiver only; nonlinear setting changes output noise",
        "frontend_model":"linear" if f.soft_limit_input_dbm is None else "pre-channel-filter tanh soft limiter proxy",
        "filter_edge_guard_samples":len(taps)//2,
    }
