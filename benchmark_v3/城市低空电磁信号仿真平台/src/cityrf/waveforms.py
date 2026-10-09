"""Deterministic indexed waveforms; power is active mean power in watts.

These are bandlimited proxy waveforms, not protocol-compliant drone signals.
Every query uses the same fixed normalization reference, independent of window.
"""
import hashlib
from functools import lru_cache
import numpy as np
from scipy.signal import firwin
from .config import Component, Emitter

# ---- 信号源分发：v2 调制 / v3 辐射体（按需加载，缺库则回退平台原生波形） ----
# 自由模式新增的 v2 独有 kind → v2 基带调制名
_V2_KIND_MOD = {"gfsk": "GFSK", "16qam": "16QAM", "64qam": "64QAM",
                "ook": "OOK", "fhss": "FHSS", "lfm": "LFM"}
_SOURCES = None


def _load_sources():
    """按需加载 v2 基带与 v3 辐射体；返回 (generate_baseband, generate_emitter, EMITTERS)。"""
    global _SOURCES
    if _SOURCES is not None:
        return _SOURCES
    import sys
    from pathlib import Path as _P
    here = _P(__file__).resolve()
    _v3sim = here.parents[3] / "simulation"                  # benchmark_v3/simulation
    _v2sim = here.parents[4] / "benchmark_v2" / "simulation"
    for _p in (str(_v3sim), str(_v2sim), str(_v2sim.parent)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from em_signal_simulator.baseband import generate_baseband              # v2
        from em_signal_simulator_v3.emitters import EMITTERS, generate_emitter  # v3
        _SOURCES = (generate_baseband, generate_emitter, EMITTERS)
    except Exception:
        _SOURCES = (None, None, None)
    return _SOURCES


def sub_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


def uniform(indices, seed):
    x = np.asarray(indices, dtype=np.int64).view(np.uint64).copy()
    x ^= np.uint64(seed)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return ((x >> np.uint64(11)).astype(np.float64)+.5) / 2**53


def gaussian(indices, seed):
    u = uniform(indices, seed)
    v = uniform(indices, seed ^ 0xABCDEF12345)
    return np.sqrt(-np.log(u)) * np.exp(2j*np.pi*v)


def _raw(c, fs, n, seed):
    t = n/fs
    if c.kind == "tone":
        return np.ones(len(n), complex)
    if c.kind == "noise":
        return gaussian(n, seed)
    if c.kind in ("bpsk", "qpsk"):
        symbols = np.floor(t*c.symbol_rate_hz).astype(np.int64)
        order = 2 if c.kind == "bpsk" else 4
        return np.exp(2j*np.pi*np.floor(uniform(symbols,seed)*order)/order)
    if c.kind == "fsk":
        sy = np.floor(t*c.symbol_rate_hz).astype(np.int64)
        lo, hi = min(0,int(sy.min())), max(0,int(sy.max()))
        signs = np.where(uniform(np.arange(lo,hi+1),seed) < .5, -1., 1.)
        prefix = np.r_[0., np.cumsum(signs)]
        integral = (prefix[sy-lo]-prefix[-lo])/c.symbol_rate_hz
        integral += signs[sy-lo]*(t-sy/c.symbol_rate_hz)
        return np.exp(2j*np.pi*(c.symbol_rate_hz/4)*integral)
    if c.kind == "chirp":
        u = np.mod(t, c.chirp_period_s)
        return np.exp(2j*np.pi*(-c.bandwidth_hz/2*u +
                    c.bandwidth_hz/(2*c.chirp_period_s)*u*u))
    if c.kind == "ofdm":
        # 64-point IFFT, 16-sample CP; 32 occupied carriers. Fixed random access.
        fft_n, cp = 64, 16
        rate = c.bandwidth_hz*2
        k = np.floor(t*rate).astype(np.int64)
        frames, pos = np.divmod(k, fft_n+cp)
        out = np.empty(len(n), complex)
        bins = np.r_[np.arange(-16,0), np.arange(1,17)]
        for frame in np.unique(frames):
            q = np.floor(uniform(frame*64+np.arange(32), seed)*4)
            spectrum = np.zeros(64,complex)
            spectrum[bins % 64] = np.exp(1j*np.pi/2*q)
            symbol = np.fft.ifft(spectrum)*64/np.sqrt(32)
            mask = frames == frame
            out[mask] = symbol[(pos[mask]-cp) % 64]
        return out
    raise ValueError("unknown waveform")


def _v2_modulation(c, fs, n, seed):
    """自由模式下 v2 独有 kind：用 v2 基带 generate_baseband 生成（已按带宽成形）。"""
    gen, _, _ = _load_sources()
    if gen is None:
        raise ValueError(f"v2 基带不可用，无法生成 kind='{c.kind}' 波形")
    rng = np.random.default_rng(seed)
    return gen(_V2_KIND_MOD[c.kind], len(n), rng=rng,
               bandwidth_normalized=max(c.bandwidth_hz / fs, 1e-6))


def _filtered(c, fs, n, seed, gated=False):
    if c.kind in _V2_KIND_MOD:
        return _v2_modulation(c, fs, n, seed)
    if c.kind == "tone" and not (gated and c.pulse_period_s is not None):
        return _raw(c,fs,n,seed)
    half = 64
    extended = np.arange(n[0]-half,n[-1]+half+1,dtype=np.int64)
    taps = firwin(2*half+1,c.bandwidth_hz/2,fs=fs)
    raw=_raw(c,fs,extended,seed)
    if gated and c.pulse_period_s is not None:
        phase=np.mod((extended/fs-c.pulse_start_s)/c.pulse_period_s+1e-10,1)
        raw*=phase<c.duty_cycle
    return np.convolve(raw,taps,mode="valid")


@lru_cache(maxsize=256)
def _scale(spec_json, fs, seed):
    c = Component.model_validate_json(spec_json)
    ref = _filtered(c,fs,np.arange(16384,dtype=np.int64),seed)
    return np.sqrt(np.mean(abs(ref)**2))


@lru_cache(maxsize=128)
def _composite_scale(tx_json,fs,seed):
    """Calibrate ungated composite to unit reference power, including cross terms."""
    tx=Emitter.model_validate_json(tx_json)
    n=np.arange(16384,dtype=np.int64)
    ref=np.zeros(len(n),complex)
    weight=sum(c.weight for c in tx.components)
    for i,c in enumerate(tx.components):
        key=sub_seed(seed,f"{tx.id}/component/{i}")
        x=_filtered(c,fs,n,key)/_scale(c.model_dump_json(),fs,key)
        ref+=np.sqrt(c.weight/weight)*x*np.exp(1j*(2*np.pi*c.offset_hz*n/fs+c.phase_rad))
    rms=np.sqrt(np.mean(abs(ref)**2))
    if rms<1e-8:
        raise ValueError("composite components cancel; nonzero TX power cannot be assigned")
    return rms


def _generate_preset(tx, fs, n, seed):
    """预设模式：用 v3 辐射体生成 IQ，并把发射源上可调的带宽/频偏/脉冲叠上去。"""
    _, generate_emitter, EMITTERS = _load_sources()
    name = tx.preset
    if generate_emitter is None or not EMITTERS or name not in EMITTERS:
        raise ValueError(f"v3 辐射体 '{name}' 不可用（未找到 v3 辐射体库或名称不存在）")
    import inspect
    import zlib
    rng = np.random.default_rng((seed ^ 0x9E3779B9 ^ zlib.crc32(name.encode())) & 0xFFFFFFFF)
    c = tx.components[0]
    gen = EMITTERS[name][0]
    # 带宽（2026-10-09「生成器带宽补全」）：数据集走的就是这条路径（gen_dataset.py:615
    # 直接调 runner.run_scenario），而它以前只给「受理 bw 形参」的生成器传 bw → 噪声类/
    # 工业脉冲类/微波炉等完全忽略、蓝牙/锯齿/Zigbee 等带外溢出，实测 OBW99 是标称的 4~24
    # 倍且 d2/d3/d4 分别有 12/0/25 个源的频带越出接收通带（benchmark_v3/_an_lna_obw.txt）。
    # `bandwidth_normalized` 由 generate_emitter 统一兜底（软带限），覆盖全部 19 类。
    # 只对 role=interference 生效：目标的公开先验 target_bw_normalized 取自实测表（≈0.352），
    # 一旦给目标也加带限反而会与先验不一致，且会动到参考直减链路的基线。
    bw_norm = None
    if getattr(tx, "role", None) == "interference":
        bw_norm = max(c.bandwidth_hz / fs, 1e-6)
    kw = {"bw": bw_norm} if (bw_norm is not None and "bw" in inspect.signature(gen).parameters) else {}
    out = generate_emitter(name, rng, length=len(n), bandwidth_normalized=bw_norm, **kw)
    iq = np.asarray(out["iq"], dtype=complex)
    if len(iq) != len(n):                    # 防御：个别生成器长度不足时补零/截断
        iq = np.pad(iq, (0, max(0, len(n) - len(iq))))[:len(n)]
    iq = iq * np.exp(1j * (2 * np.pi * c.offset_hz * n / fs + c.phase_rad))
    if c.pulse_period_s is not None:
        phase = np.mod((n / fs - c.pulse_start_s) / c.pulse_period_s + 1e-10, 1)
        iq = iq * (phase < c.duty_cycle)
    return iq


def _power_scale(iq, power_dbm):
    p_w = 10.0 ** ((power_dbm - 30.0) / 10.0)
    rms = float(np.mean(np.abs(iq) ** 2) + 1e-30)
    return float(np.sqrt(p_w / rms))


def generate(tx: Emitter, fs: float, indices, seed: int):
    n = np.asarray(indices,dtype=np.int64)
    if n.ndim != 1 or not len(n) or np.any(np.diff(n) != 1):
        raise ValueError("indices must be a nonempty contiguous sample interval")
    if tx.preset:                      # 预设模式：直接用 v3 辐射体真实时域信号
        iq = _generate_preset(tx, fs, n, seed)
        return iq * _power_scale(iq, tx.power_dbm)
    out = np.zeros(len(n),complex)
    weight = sum(c.weight for c in tx.components)
    for i,c in enumerate(tx.components):
        key = sub_seed(seed,f"{tx.id}/component/{i}")
        x = _filtered(c,fs,n,key,gated=True)/_scale(c.model_dump_json(),fs,key)
        x *= np.exp(1j*(2*np.pi*c.offset_hz*n/fs+c.phase_rad))
        out += np.sqrt(c.weight/weight)*x
    return out*np.sqrt(10**((tx.power_dbm-30)/10))/_composite_scale(tx.model_dump_json(),fs,seed)
