"""Post-run artifact and basic physical integrity checks (evaluator side)."""
import json
from pathlib import Path
import numpy as np
from .export import digest
from .analysis import load_public


def audit_run(root):
    root=Path(root).resolve()
    summary=json.loads((root/"run_summary.json").read_text(encoding="utf-8"))
    for name,expected in summary["hashes"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    obs=load_public(root/"public")
    stations=[]
    for r in obs["receivers"]:
        x=np.load(root/"public"/r["iq_file"],allow_pickle=False)
        if x.shape != tuple(r["shape"]) or x.dtype!=np.complex64 or not np.isfinite(x).all():
            raise ValueError("invalid IQ")
        stations.append({"receiver_id":r["receiver_id"],"samples":x.shape[1],
            "channels":x.shape[0],"duration_s":x.shape[1]/r["sample_rate_hz"],
            "port_power_dbw":(10*np.log10(np.maximum(np.mean(abs(x.astype(complex))**2,axis=1),1e-30))).tolist()})
    truth=json.loads((root/"private"/"truth.json").read_text(encoding="utf-8"))
    return {"status":"verified","artifacts_checked":len(summary["hashes"]),"stations":stations,
        "path_counts":{r["receiver_id"]:{s["emitter_id"]:s["path_count"] for s in r["sources"]} for r in truth["stations"]},
        "limitations":["No-path links mean no paths found under this solver configuration, not proven RF invisibility.",
                       "This audit checks consistency, not calibrated real-world accuracy."]}
