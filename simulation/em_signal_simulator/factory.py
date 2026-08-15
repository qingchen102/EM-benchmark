from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .baseband import generate_baseband, MODULATIONS
from .jamming import inject_jamming, JAMMING_TYPES
from .channel import apply_channel

def generate_signal_sample(mod_type="QPSK", jamming_type="none", snr_db=10.0, num_samples=1024, jamming_to_signal_ratio_db=3.0, cfo=0.0, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    base = generate_baseband(mod_type, num_samples, rng=rng, **kwargs)
    jammed = inject_jamming(base, jamming_type, jamming_to_signal_ratio_db, rng=rng, **kwargs)
    return apply_channel(jammed, snr_db, cfo, rng=rng).astype(np.complex128)

def _ground_truth(params, file_name):
    mod = str(params["mod_type"]).upper(); jam = str(params["jamming_type"]).lower()
    n = int(params["num_samples"])
    freq = None
    if jam == "single_tone": freq = float(params.get("frequency", params.get("center_frequency", .15)))
    elif jam == "swept": freq = [float(params.get("f_start", -.4)), float(params.get("f_stop", .4))]
    return {
        "file": file_name, "mod_type": mod, "jamming_type": jam,
        "has_jamming": jam != "none", "num_samples": n,
        "snr_db": float(params["snr_db"]),
        "jamming_to_signal_ratio_db": float(params["jamming_to_signal_ratio_db"]),
        "jamming_time_range": params.get("time_range", [0, n] if jam in {"single_tone", "swept", "broadband"} else None),
        "jamming_freq_normalized": freq, "cfo_normalized": float(params.get("cfo", 0.0)),
    }

def generate_dataset(output_dir="dataset", count=10, mixed=False, snr_range=(-10.0, 20.0), **params):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); records=[]
    params = {"mod_type": "QPSK", "jamming_type": "none", "snr_db": 10.0,
              "num_samples": 1024, "jamming_to_signal_ratio_db": 3.0, "cfo": 0.0, **params}
    base_seed = params.get("seed", 0)
    for i in range(int(count)):
        rng = np.random.default_rng(None if base_seed is None else int(base_seed) + i)
        p = dict(params); p["seed"] = None if base_seed is None else int(base_seed) + i
        if mixed:
            p["mod_type"] = str(rng.choice(sorted(MODULATIONS)))
            p["jamming_type"] = str(rng.choice(sorted(JAMMING_TYPES)))
            p["snr_db"] = float(rng.uniform(float(snr_range[0]), float(snr_range[1])))
        x = generate_signal_sample(**p); np.save(out / f"sample_{i:05d}.npy", x)
        file_name = f"sample_{i:05d}.npy"
        records.append({"seed": p["seed"], "parameters": {k:v for k,v in p.items() if k != "seed"}, "ground_truth": _ground_truth(p, file_name)})
    (out / "metadata.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records
