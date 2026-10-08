"""Public-only prediction entry point; no import of evaluator or private scenario."""
import time
from pathlib import Path
import numpy as np
from ..analysis import load_public
from ..export import digest,write_json
from .detection import detect_bands,frequency_groups
from .spatial import band_covariance,scan_directions
from .localization import localize


def run_baseline(public_dir,output):
    public_dir=Path(public_dir)
    root=Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("baseline output must be empty")
    started=time.perf_counter()
    public=load_public(public_dir)
    stations=[]
    arrays={}
    for obs in public["receivers"]:
        iq=np.load(public_dir/obs["iq_file"],allow_pickle=False,mmap_mode="r")
        if list(iq.shape)!=obs["shape"] or not np.isfinite(iq).all():
            raise ValueError("invalid IQ")
        arrays[obs["receiver_id"]]=iq
        stations.append({"receiver_id":obs["receiver_id"],"center_frequency_hz":obs["center_frequency_hz"],
                         "detection":detect_bands(iq,obs["sample_rate_hz"],obs["receiver_bandwidth_hz"])})
    groups=frequency_groups(stations)
    by_id={r["receiver_id"]:r for r in public["receivers"]}
    for group in groups:
        directions=[]
        for name in group["receiver_ids"]:
            obs=by_id[name]
            low=group["low_rf_hz"]-obs["center_frequency_hz"]
            high=group["high_rf_hz"]-obs["center_frequency_hz"]
            covariance,snapshots=band_covariance(arrays[name],obs["sample_rate_hz"],low,high)
            scan=scan_directions(covariance,obs["element_offsets_world_m"],
                                 obs["reference_frequency_hz"],snapshots)
            directions.append({"receiver_id":name,"position_m":obs["receiver_position_m"],"scan":scan})
        group["directions"]=directions
        group["localization"]={method:localize(directions,method) for method in ("bartlett","music")}
    prediction={"schema_version":"1.0","input_observations_sha256":digest(public_dir/"observations.json"),
                "settings":{"detector":"robust Welch energy, 8 dB, gap closing 8 kHz",
                    "doa":"2-degree full azimuth, elevation -88..88; eight peaks within 6 dB",
                    "rank":"MDL capped at six; rank zero uses one only for detected-band direction scan",
                    "localization":"one top hypothesis per frequency group, all detected sites, positive range <=1000m, height 0..200m"},
                "stations":stations,"groups":groups,"elapsed_seconds":time.perf_counter()-started,
                "scope":"Frequency groups and path directions; no assumption that group count equals physical sources."}
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"prediction.json",prediction)
    write_json(root/"freeze.json",{"prediction_sha256":digest(root/"prediction.json"),
                                  "input_iq_sha256":{r["receiver_id"]:r["sha256"] for r in public["receivers"]},
                                  "source_hashes":{p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}})
    return {"group_count":len(groups),"elapsed_seconds":prediction["elapsed_seconds"],
            "bands":[{"id":g["group_id"],"low_rf_hz":g["low_rf_hz"],"high_rf_hz":g["high_rf_hz"],
                      "stations":g["receiver_ids"],
                      "localizations":{m:len(v["hypotheses"]) for m,v in g["localization"].items()}} for g in groups]}
