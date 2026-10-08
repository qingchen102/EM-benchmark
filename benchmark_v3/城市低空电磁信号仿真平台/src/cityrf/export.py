"""Public observations and evaluator-only truth are separate filesystem planes."""
import hashlib
import json
from pathlib import Path
from .channels import C0, element_offsets, element_offsets_local, rotation
from .config import Antenna


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def public_observation(scenario, rx):
    a=rx.antenna
    rot=rotation(rx.orientation_rad)
    offsets=element_offsets(rx,scenario.carrier_frequency_hz)
    geom={"element_count":a.elements,"local_plane":"yz","local_front_normal":[1,0,0],
          "vertical_spacing_m":a.vertical_spacing_wavelengths*C0/scenario.carrier_frequency_hz,
          "horizontal_spacing_m":a.horizontal_spacing_wavelengths*C0/scenario.carrier_frequency_hz}
    if a.geometry_type=="cross":
        geom.update(geometry_type="cross",horizontal_elements=a.horizontal_elements,
                    vertical_elements=a.vertical_elements,shared_center_element=False,
                    channel_order="horizontal arm +y first; vertical arm -z second",
                    channel_map=[{"channel":i,"arm":"horizontal","arm_index":i} for i in range(a.horizontal_elements)]
                    +[{"channel":a.horizontal_elements+i,"arm":"vertical","arm_index":i} for i in range(a.vertical_elements)])
    else:
        geom.update(geometry_type="ura" if a.num_rows>1 and a.num_cols>1 else "ula" if a.elements>1 else "single",
                    num_rows=a.num_rows,num_cols=a.num_cols,
                    row_direction_local=[0,0,-1],col_direction_local=[0,1,0],
                    channel_order="column_major: channel=col*num_rows+row",
                    channel_map=[{"channel":col*a.num_rows+row,"row":row,"col":col}
                                 for col in range(a.num_cols) for row in range(a.num_rows)])
    return {
        "receiver_id":rx.id,"iq_file":f"iq_{rx.id}.npy",
        "shape":[rx.antenna.elements,scenario.num_samples],"dtype":"complex64",
        "layout":"channels_samples","sample_rate_hz":scenario.sample_rate_hz,
        "center_frequency_hz":scenario.carrier_frequency_hz+rx.tuning_offset_hz,
        "time_origin_s":0.,"duration_s":scenario.duration_s,
        "receiver_position_m":rx.position_m,"orientation_rad":rx.orientation_rad,
        "antenna":rx.antenna.model_dump(exclude_none=True),
        "coordinate_frame":{"id":"scene_cartesian","handedness":"right","length_unit":"m",
                            "z_axis":"up","georeferenced":False},
        "array_pose":{"center_world_m":list(rx.position_m),
                      "orientation_rad":list(rx.orientation_rad),
                      "orientation_convention":"Rz(yaw) Ry(pitch) Rx(roll); local to world",
                      "rotation_local_to_world":rot.tolist(),"normal_world":rot[:,0].tolist()},
        "array_geometry":geom,
        "element_offsets_local_m":element_offsets_local(rx,scenario.carrier_frequency_hz).tolist(),
        "element_offsets_world_m":offsets.tolist(),
        "element_positions_world_m":(offsets+rx.position_m).tolist(),
        "reference_frequency_hz":scenario.carrier_frequency_hz,
        "synchronized":True,"calibrated":True,
        "amplitude_convention":"complex envelope sqrt(W) after fixed receiver gain",
        "receiver_bandwidth_hz":rx.bandwidth_hz,"gain_db":rx.frontend.gain_db,
        "angle_convention":"world azimuth atan2(y,x); zenith from +z; array local plane yz",
    }


def cooperative_sources(scenario):
    """Only task-designated communication emitters; no unknown-source truth."""
    result=[]
    for tx in scenario.emitters:
        links=[t.id for t in scenario.tasks if tx.id in t.desired_emitter_ids]
        if tx.role!="communication" or not links:
            continue
        signals=[]
        for c in tx.components:
            center=scenario.carrier_frequency_hz+c.offset_hz
            signal={"modulation":c.kind,"center_frequency_hz":center,
                    "nominal_bandwidth_hz":c.bandwidth_hz,"frequency_range_hz":[center-c.bandwidth_hz/2,center+c.bandwidth_hz/2],
                    "protocol_kind":"simulation_proxy","protocol_name":None}
            if c.kind=="ofdm":
                rate=2*c.bandwidth_hz
                signal.update(symbol_rate_hz=rate/80,ofdm={"fft_size":64,"cyclic_prefix_samples":16,
                    "occupied_subcarrier_indices":list(range(-16,0))+list(range(1,17)),
                    "baseband_sample_rate_hz":rate,"subcarrier_modulation":"qpsk"})
            elif c.kind in ("bpsk","qpsk","fsk"):
                signal["symbol_rate_hz"]=c.symbol_rate_hz
            if c.pulse_period_s is not None:
                signal.update(pulse_period_s=c.pulse_period_s,duty_cycle=c.duty_cycle,pulse_start_s=c.pulse_start_s)
            signals.append(signal)
        result.append({"emitter_id":tx.id,"platform_id":tx.platform_id,"task_ids":links,
                       "position_m":list(tx.position_m),"coordinate_frame":"scene_cartesian",
                       "position_uncertainty_m":0.,"valid_time_s":[0.,scenario.duration_s],
                       "prior_source":"authorized_simulation_cooperative_configuration",
                       "signals":signals})
    return result


def public_tasks(scenario):
    return [{"id":t.id,"receiver_ids":t.receiver_ids,"instruction":t.instruction,
             "protected_center_frequency_hz":scenario.carrier_frequency_hz+t.protected_center_offset_hz,
             "protected_bandwidth_hz":t.protected_bandwidth_hz,
             "expected_desired_source_count":len(t.desired_emitter_ids)}
            for t in scenario.tasks]


def legacy_manifest(public_dir, observation, task=None):
    """Adapter for existing interference_agent.IQSampleManifest; no truth read."""
    rx=observation
    count=rx["shape"][0]
    antenna=Antenna.model_validate(rx["antenna"])
    spacing=antenna.horizontal_spacing_wavelengths*C0/rx["reference_frequency_hz"]
    horizontal_ula=antenna.num_rows==1 and antenna.num_cols>1
    geometry={"geometry_type":"custom","element_count":count,
              "reference_frequency_hz":rx["reference_frequency_hz"],
              "synchronized":True,"calibration":{"calibrated":True}}
    if horizontal_ula:
        geometry.update(geometry_type="ula",element_spacing_m=spacing,
                        element_positions_m=[(i-(count-1)/2)*spacing for i in range(count)])
    result={
        "sample_id":rx["receiver_id"],"iq_path":str((Path(public_dir)/rx["iq_file"]).resolve()),
        "iq_format":"npy","dtype":"complex64","layout":"channel_first",
        "sample_rate_hz":rx["sample_rate_hz"],"center_frequency_hz":rx["center_frequency_hz"],
        "reference_frequency_hz":rx["reference_frequency_hz"],"synchronized":True,
        "array_geometry":geometry,
        "provenance":{"adapter":"static_city_v1.2",
            "doa_warning":"Legacy MUSIC returns local ULA broadside angle, not global emitter bearing."
                if horizontal_ula else "Legacy Agent has no planar/vertical array DOA estimator; custom geometry disables its spatial tools.",
            "array_geometry":rx.get("array_geometry",{}),
            "array_pose":rx.get("array_pose",{}),
            "element_positions_world_m":rx.get("element_positions_world_m",[])},
    }
    if task:
        result["protected_channel"]={
            "center_offset_hz":task["protected_center_frequency_hz"]-rx["center_frequency_hz"],
            "nominal_bandwidth_hz":task["protected_bandwidth_hz"],
        }
        result["expected_desired_source_count"]=task["expected_desired_source_count"]
    return result
