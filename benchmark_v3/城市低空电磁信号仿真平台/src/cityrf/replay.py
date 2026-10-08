"""Replay a verified private channel snapshot, without invoking RT again."""
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from .audit import audit_run
from .channels import LinkChannel
from .config import Scenario
from .export import digest
from .runner import run_scenario


def replay_scenario(original, output):
    root=Path(original)
    audit_run(root)
    summary=json.loads((root/"run_summary.json").read_text(encoding="utf-8"))
    current={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}
    if current!=summary["versions"]:
        raise ValueError("exact replay requires original Python/NumPy/SciPy versions")
    for name in ("waveforms.py","channels.py","receiver.py","config.py"):
        if summary["source_hashes"].get(name)!=digest(Path(__file__).parent/name):
            raise ValueError(f"exact replay source changed: {name}")
    config=Scenario.model_validate_json((root/"private"/"scenario.json").read_text(encoding="utf-8"))
    truth=json.loads((root/"private"/"truth.json").read_text(encoding="utf-8"))
    links=[]
    for station in truth["stations"]:
        for source in station["sources"]:
            tx,rx=source["emitter_id"],station["receiver_id"]
            data=np.load(root/"private"/f"channel_{tx}_{rx}.npz",allow_pickle=False)
            links.append(LinkChannel(tx,rx,data["coefficients"],data["delays_s"],
                                     source["geometry"],source["channel_provenance"]))
    return run_scenario(config,output,frozen_links=links)
