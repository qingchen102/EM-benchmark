"""Produce plots and bounded browser previews exclusively from mixed public IQ."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import welch, spectrogram


def build_preview(capture):
    root = Path(capture)
    public = root / "public"
    obs = json.loads((public / "observations.json").read_text(encoding="utf-8"))
    rows = []
    plots = public / "web_plots"
    plots.mkdir(exist_ok=True)
    for r in obs["receivers"]:
        x = np.load(public / r["iq_file"], mmap_mode="r", allow_pickle=False)
        fs = r["sample_rate_hz"]
        n = min(4096, x.shape[1])
        f, p = welch(x, fs=fs, nperseg=n, noverlap=n//2, detrend=False,
                     return_onesided=False, scaling="density", axis=-1)
        f, p = np.fft.fftshift(f), np.fft.fftshift(p.mean(axis=0))
        win = min(512, x.shape[1])
        ft, t, z = spectrogram(x, fs=fs, window="hann", nperseg=win,
                    noverlap=win//2, detrend=False, return_onesided=False,
                    scaling="density", mode="psd", axis=-1)
        ft, z = np.fft.fftshift(ft), np.fft.fftshift(z.mean(axis=0), axes=0)
        db = 10*np.log10(np.maximum(p, 1e-30))+30
        zd = 10*np.log10(np.maximum(z, 1e-30))+30
        np.savez_compressed(plots / f"{r['receiver_id']}_spectral_data.npz",
                           frequency_hz=f, psd_w_per_hz=p,
                           stft_frequency_hz=ft, time_s=t, stft_psd_w_per_hz=z)
        fi = np.arange(0, len(f), max(1, len(f)//1200))
        ti = np.arange(0, len(t), max(1, int(np.ceil(len(t)/240))))
        zi = np.arange(0, len(ft), max(1, len(ft)//256))
        rows.append(dict(id=r["receiver_id"], channels=x.shape[0], samples=x.shape[1],
            sample_rate_hz=fs, duration_ms=x.shape[1]/fs*1000,
            tuning_offset_hz=r["center_frequency_hz"]-r["reference_frequency_hz"],
            power_dbm=float(10*np.log10(max(float(np.mean(abs(x)**2)),1e-30))+30),
            frequency_khz=(f[fi]/1000).tolist(), psd_dbm_hz=db[fi].tolist(),
            stft_frequency_khz=(ft[zi]/1000).tolist(), time_ms=(t[ti]*1000).tolist(),
            spectrogram_dbm_hz=zd[np.ix_(zi,ti)].tolist(),
            color_range=[float(np.quantile(zd,.05)),float(np.quantile(zd,.999))]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2,figsize=(12,3.6),layout="constrained")
        ax[0].plot(f/1000,db,color="#2563eb",linewidth=.8)
        ax[0].set(title=r["receiver_id"]+" | mean array PSD",xlabel="Baseband offset (kHz)",ylabel="dBm/Hz")
        ax[0].grid(alpha=.15)
        im=ax[1].pcolormesh(t*1000,ft/1000,zd,shading="auto",cmap="viridis")
        ax[1].set(title="Spectrogram",xlabel="Time (ms)",ylabel="Baseband offset (kHz)")
        fig.colorbar(im,ax=ax[1],label="dBm/Hz")
        fig.savefig(plots / f"{r['receiver_id']}_overview.png",dpi=140)
        plt.close(fig)
    preview=dict(stations=rows, units="dBm/Hz", averaging="Mean of per-element power spectra; not coherent IQ averaging",
                 frequency_reference="Receiver tuned center", welch_max_nperseg=4096, spectrogram_max_nperseg=512)
    temporary=root/'preview.tmp'
    temporary.write_text(json.dumps(preview,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    temporary.replace(root/'preview.json')
    return preview
