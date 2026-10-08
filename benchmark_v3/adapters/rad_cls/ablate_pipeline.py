"""链路消融（快速版）：波形池只生成一次，各链路复用，GPU 批量推理。"""
import sys
from pathlib import Path
for p in [r'.', r'..', r'../../simulation']:
    rp = str(Path(p).resolve())
    if rp not in sys.path: sys.path.insert(0, rp)
import numpy as np
import torch
import data_gen16 as dg
from model16 import RadCNN

N_PER_CLASS = 10
SEED = 700000

print("== 生成波形池（一次） ==", flush=True)
waves, labels = [], []
for ci, name in enumerate(dg.CLASSES16):
    for j in range(N_PER_CLASS):
        rng = np.random.default_rng(SEED + ci*1009 + j)
        try:
            waves.append(dg.gen_wave16(name, rng).astype(np.complex128))
            labels.append(ci)
        except RuntimeError:
            pass
    print(f"  {name} ok", flush=True)
labels = np.array(labels)
pool_x = np.stack(waves)
pool_y = labels.copy()
print(f"pool={len(waves)}", flush=True)

ck = torch.load('rad_classifier/checkpoint.pt', map_location='cpu', weights_only=False)
model = RadCNN(); model.load_state_dict(ck['model']); model.eval()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev)


def batch_acc(outs, tag):
    xb = torch.from_numpy(np.stack([dg.to_input(o) for o in outs])).to(dev)
    with torch.no_grad():
        pred = model(xb).argmax(dim=1).cpu().numpy()
    r = float(np.mean(pred == labels))
    print(f"{tag}: {r:.3f}", flush=True)
    return r


# 1. 训练同链路（beamformed；全池供兄弟源抽取，主源类=真实标签）
outs = [dg.simulate_beamformed_train(x, int(labels[i]), pool_x, pool_y,
                                     np.random.default_rng(SEED+7+i))
        for i, x in enumerate(waves)]
batch_acc(outs, "beamformed 链路(训练同分布)")

# 2. residual 全链路（oracle clean 口径，含新滤波）
batch_acc([dg.simulate_residual(x, np.random.default_rng(SEED+7+i))
           for i, x in enumerate(waves)], "residual 全链路(含滤波)")

# 3. residual 去滤波（消融：滤波是否是崩点）
save = dg._rx_filter
dg._rx_filter = lambda x: x
batch_acc([dg.simulate_residual(x, np.random.default_rng(SEED+7+i))
           for i, x in enumerate(waves)], "residual 去滤波(消融)")
dg._rx_filter = save

# 4. residual 去多径（隔离多径因素，其余保留含滤波）
def no_mp(x, rng):
    x = x * np.exp(1j*2*np.pi*float(rng.uniform(*dg.FOFF_RANGE_HZ))*np.arange(len(x))/4e6)
    x = x.astype(np.complex128)
    bw = float(rng.uniform(*dg.BW_RANGE))
    spec = np.fft.fft(x); freq = np.fft.fftfreq(len(x), 1/4e6)
    edge = bw*4e6/2
    soft = np.clip((edge-np.abs(freq))/(0.05*4e6)+1, 0, 1)
    x = np.fft.ifft(spec*soft)
    p = float(np.mean(np.abs(x)**2))+1e-30
    snr = float(rng.uniform(*dg.SNR_RANGE))
    x = x + np.sqrt(p*10**(-snr/10)/2)*(rng.standard_normal(len(x))+1j*rng.standard_normal(len(x)))
    x = dg._rx_filter(x)
    x = np.roll(x, int(rng.integers(0, len(x))))
    return (x/(np.sqrt(np.mean(np.abs(x)**2))+1e-12)).astype(np.complex64)
batch_acc([no_mp(x, np.random.default_rng(SEED+7+i)) for i, x in enumerate(waves)],
          "residual 去多径(含滤波)")
