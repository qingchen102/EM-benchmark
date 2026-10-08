"""Post-freeze evaluator; this module may read simulation truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..analysis import load_public
from ..audit import audit_run
from ..export import digest,write_json
from ..config import Scenario
from ..channels import LinkChannel,apply_channel,DELAY_HALF
from ..waveforms import generate
from ..receiver import receive


def match_locations(estimates,sources,gate_m=50.):
    matches=[]
    if estimates and sources:
        distances=np.linalg.norm(np.array([e["position_m"] for e in estimates])[:,None,:]-
                                 np.array([s["position_m"] for s in sources])[None,:,:],axis=-1)
        compatible=np.array([[s["id"] in e.get("eligible_source_ids",[v["id"] for v in sources])
                              for s in sources] for e in estimates])
        cost=np.where((distances<=gate_m)&compatible,distances/(gate_m+1),1e6+distances)
        rows,cols=linear_sum_assignment(cost)
        for i,j in zip(rows,cols):
            if distances[i,j]<=gate_m and compatible[i,j]:
                matches.append({"group_id":estimates[i]["group_id"],"source_id":sources[j]["id"],
                                "error_3d_m":float(distances[i,j])})
    return {"gate_m":gate_m,"prediction_count":len(estimates),"source_count":len(sources),
            "matched_sources":len(matches),"false_locations":len(estimates)-len(matches),
            "missed_sources":len(sources)-len(matches),"matches":matches}


def contributors(group,sources,fc,duration):
    ids=[]
    for s in sources:
        for c in s["components"]:
            half=1/duration if c["kind"]=="tone" and c["pulse_period_s"] is None else c["bandwidth_hz"]/2
            low,high=fc+c["offset_hz"]-half,fc+c["offset_hz"]+half
            if max(low,group["low_rf_hz"])<min(high,group["high_rf_hz"]):
                ids.append(s["id"])
                break
    return ids


def _inband_oracle(capture,scenario,truth,groups):
    """Regenerate individual linear contributions from saved channels, evaluator only."""
    if any(r.frontend.soft_limit_input_dbm is not None for r in scenario.receivers):
        raise ValueError("per-source oracle currently requires linear frontend")
    links={}
    for station in truth["stations"]:
        for source in station["sources"]:
            tx,rx=source["emitter_id"],station["receiver_id"]
            with np.load(capture/"private"/f"channel_{tx}_{rx}.npz",allow_pickle=False) as d:
                links[tx,rx]=LinkChannel(tx,rx,d["coefficients"],d["delays_s"],source["geometry"],{})
    maximum=max(float(np.max(l.delays_s,initial=0)) for l in links.values())
    guard=max(r.frontend.filter_taps//2 for r in scenario.receivers)
    history=int(np.ceil(maximum*scenario.sample_rate_hz))+DELAY_HALF+guard+2
    indices=np.arange(-history,scenario.num_samples+guard+DELAY_HALF+2)
    waves={tx.id:generate(tx,scenario.sample_rate_hz,indices,scenario.seed) for tx in scenario.emitters}
    power={}
    for rx in scenario.receivers:
        clean=rx.model_copy(deep=True)
        clean.frontend.noise_enabled=False
        f=np.fft.fftfreq(scenario.num_samples,1/scenario.sample_rate_hz)+scenario.carrier_frequency_hz+rx.tuning_offset_hz
        for tx in scenario.emitters:
            arrival=apply_channel(waves[tx.id],indices[0],links[tx.id,rx.id],scenario.sample_rate_hz,
                                  scenario.num_samples+2*guard,output_start=-guard)
            y,_=receive(arrival,clean,scenario.sample_rate_hz,scenario.seed,start_index=-guard)
            spectrum=abs(np.fft.fft(y[:,guard:guard+scenario.num_samples],axis=-1))**2
            for g in groups:
                mask=(f>=g["low_rf_hz"])&(f<=g["high_rf_hz"])
                power[g["group_id"],rx.id,tx.id]=float(np.mean(spectrum[:,mask].sum(axis=1))/scenario.num_samples**2)
    return power,links


def _angle(u,v):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(u,v),-1,1))))


def evaluate(capture,prediction_dir):
    capture,prediction_dir=Path(capture),Path(prediction_dir)
    frozen=json.loads((prediction_dir/"freeze.json").read_text(encoding="utf-8"))
    if digest(prediction_dir/"prediction.json")!=frozen["prediction_sha256"]:
        raise ValueError("prediction hash changed")
    prediction=json.loads((prediction_dir/"prediction.json").read_text(encoding="utf-8"))
    public=load_public(capture/"public")
    if digest(capture/"public/observations.json")!=prediction["input_observations_sha256"]:
        raise ValueError("prediction/capture mismatch")
    # Only after predictions and inputs are verified is private truth opened.
    audit_run(capture)
    truth=json.loads((capture/"private/truth.json").read_text(encoding="utf-8"))
    scenario=Scenario.model_validate_json((capture/"private/scenario.json").read_text(encoding="utf-8"))
    sources=truth["sources"]
    source_map={s["id"]:s for s in sources}
    powers,links=_inband_oracle(capture,scenario,truth,prediction["groups"])
    groups=[]
    for g in prediction["groups"]:
        ids=contributors(g,sources,scenario.carrier_frequency_hz,scenario.duration_s)
        record={"group_id":g["group_id"],"low_offset_hz":g["low_rf_hz"]-scenario.carrier_frequency_hz,
                "high_offset_hz":g["high_rf_hz"]-scenario.carrier_frequency_hz,
                "nominal_contributor_ids":ids,"station_diagnostics":[],"localization":{}}
        for d in g["directions"]:
            name=d["receiver_id"]
            p={s["id"]:powers[g["group_id"],name,s["id"]] for s in sources}
            dominant=max(p,key=p.get)
            delta=np.array(source_map[dominant]["position_m"])-d["position_m"]
            bearing=delta/np.linalg.norm(delta)
            link=links[dominant,name]
            path=None
            if link.coefficients.shape[1] and link.geometry:
                k=int(np.argmax(np.mean(abs(link.coefficients)**2,axis=0)))
                az,zen=link.geometry["arrival_azimuth_rad"][k],link.geometry["arrival_zenith_rad"][k]
                path=np.array([np.sin(zen)*np.cos(az),np.sin(zen)*np.sin(az),np.cos(zen)])
            diagnostics={"receiver_id":name,"oracle_dominant_source":dominant,
                         "oracle_source_band_power_w":p,"mdl_rank":d["scan"]["mdl_rank"]}
            for method in ("bartlett","music"):
                peaks=d["scan"][method]["peaks"]
                vectors=[np.array(v["unit_direction_world"]) for v in peaks]
                diagnostics[method]={"top_direction_bearing_error_deg":_angle(vectors[0],bearing),
                    "best_candidate_bearing_error_deg":min(_angle(v,bearing) for v in vectors),
                    "best_candidate_strongest_path_error_deg":min(_angle(v,path) for v in vectors) if path is not None else None}
            record["station_diagnostics"].append(diagnostics)
        for method in ("bartlett","music"):
            hypotheses=g["localization"][method]["hypotheses"]
            nearest=[]
            for i,h in enumerate(hypotheses):
                if not ids:
                    continue
                distances={sid:float(np.linalg.norm(np.array(h["position_m"])-source_map[sid]["position_m"])) for sid in ids}
                sid=min(distances,key=distances.get)
                nearest.append({"hypothesis_rank":i+1,"position_m":h["position_m"],"nearest_compatible_source":sid,
                                "distance_m":distances[sid],"angular_rms_deg":h["angular_rms_deg"]})
            record["localization"][method]={"hypotheses":nearest}
        groups.append(record)
    metrics={}
    for method in ("bartlett","music"):
        estimates=[{"group_id":g["group_id"],"position_m":g["localization"][method]["hypotheses"][0]["position_m"],
                    "eligible_source_ids":contributors(g,sources,scenario.carrier_frequency_hz,scenario.duration_s)}
                   for g in prediction["groups"] if g["localization"][method]["hypotheses"]]
        metrics[method]=match_locations(estimates,sources)
    result={"prediction_sha256":frozen["prediction_sha256"],"physical_source_count":len(sources),
            "detected_frequency_group_count":len(groups),"groups":groups,"localization_metrics":metrics,
            "prediction_seconds":prediction["elapsed_seconds"],
            "limits":["One capture, not statistical accuracy or probability of detection.",
                      "Frequency coverage is not independent detection of each cochannel emitter.",
                      "Best-of-candidate angular errors use truth and are diagnostic only.",
                      "Only the top location in each frequency group enters the one-to-one 50m metric.",
                      "City data are ray-traced simulation, not real-world measurements."]}
    write_json(prediction_dir/"evaluation.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--capture",required=True)
    p.add_argument("--predictions",required=True)
    args=p.parse_args()
    r=evaluate(args.capture,args.predictions)
    print(json.dumps({"groups":r["detected_frequency_group_count"],"metrics":r["localization_metrics"]},indent=2))
