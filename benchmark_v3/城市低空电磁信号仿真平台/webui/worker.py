"""Isolated job worker. No API credentials or Agent required."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from cityrf.config import Scenario
from cityrf.runner import run_scenario
from cityrf.audit import audit_run
from cityrf.export import write_json
from webui.results import build_preview


def main():
    job = Path(sys.argv[1]).resolve()
    s = Scenario.model_validate_json((job/"config.json").read_text(encoding="utf-8"))
    print(f"[1/3] Propagation + IQ: {s.propagation.backend}, {len(s.emitters)} TX / {len(s.receivers)} RX",flush=True)
    run_scenario(s,job/"capture")
    print("[2/3] Verifying hashes, shapes and physical contracts",flush=True)
    write_json(job/"audit.json",audit_run(job/"capture"))
    print("[3/3] Computing station PSD / spectrogram and export",flush=True)
    build_preview(job/"capture")
    print("COMPLETE",flush=True)


if __name__ == "__main__":
    main()
