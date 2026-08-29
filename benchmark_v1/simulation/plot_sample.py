"""Plot one complex-IQ sample in time and frequency domains."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description="Plot time waveform and FFT spectrum")
    p.add_argument("sample", type=Path, help=".npy file")
    p.add_argument("--metadata", type=Path, help="metadata.json (default: sibling file)")
    p.add_argument("--output", type=Path, help="save PNG instead of displaying")
    a = p.parse_args(); x = np.asarray(np.load(a.sample)).reshape(-1)
    meta_path = a.metadata or a.sample.parent / "metadata.json"
    title = a.sample.name
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for row in data:
            gt = row.get("ground_truth", row)
            if row.get("file", gt.get("file")) == a.sample.name:
                title = f"{a.sample.name} | {gt.get('mod_type')} + {gt.get('jamming_type')} | SNR={gt.get('snr_db')} dB"; break
    freq = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0)); spectrum = 20*np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(x))) / len(x), 1e-12))
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    ax[0].plot(np.real(x), label="I", linewidth=.8); ax[0].plot(np.imag(x), label="Q", linewidth=.8, alpha=.8); ax[0].set(title=title, xlabel="Sample", ylabel="Amplitude"); ax[0].legend(); ax[0].grid(alpha=.25)
    ax[1].plot(freq, spectrum, linewidth=.8); ax[1].set(xlabel="Normalized frequency", ylabel="Magnitude (dB)"); ax[1].grid(alpha=.25)
    if a.output: fig.savefig(a.output, dpi=150); print(a.output)
    else: plt.show()
if __name__ == "__main__": main()
