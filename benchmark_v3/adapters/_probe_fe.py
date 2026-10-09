import json
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parent.parent
for d in ("d1", "d4"):
    tp = B / f"dataset_{d}" / f"{d}_00000" / "capture" / "private" / "truth.json"
    t = json.loads(tp.read_text(encoding="utf-8"))
    st = t["stations"][0]
    r = st["receiver"]
    ic = r.get("input_composite_power_w")
    print(f"=== {d} station {st['receiver_id']}")
    print("  receiver keys:", list(r.keys()))
    print("  input_composite_power_w:", type(ic).__name__,
          (len(ic) if hasattr(ic, "__len__") else ""),
          "elem0=%.4e W = %.2f dBm" % (ic[0], 10*np.log10(ic[0])+30) if hasattr(ic, "__len__")
          else "%.4e" % ic)
    fm = r.get("frontend_model")
    print("  frontend_model:", json.dumps(fm, ensure_ascii=False)[:600] if not isinstance(fm, list)
          else json.dumps(fm[0], ensure_ascii=False)[:600])
    print("  noise_measurement:", json.dumps(r.get("noise_measurement"), ensure_ascii=False)[:400])
    print("  effective_output_noise_dbm:", r.get("effective_output_noise_dbm"))
