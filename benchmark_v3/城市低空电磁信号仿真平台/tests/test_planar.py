import json
import numpy as np
import pytest
from cityrf.config import Antenna, Receiver, Scenario
from cityrf.channels import element_offsets, element_offsets_local, rotation, C0
from cityrf.export import public_observation, legacy_manifest


def config(rx=None):
    return Scenario(emitters=[dict(id="tx",position_m=[100,80,60])],
                    receivers=[rx or dict(id="rx",position_m=[10,20,5],antenna=dict(num_rows=2,num_cols=2))])


def test_explicit_rectangle_defaults_to_two_by_two():
    a=Antenna(geometry_type="rectangular")
    assert (a.num_rows,a.num_cols,a.elements)==(2,2,4)


def test_asymmetric_array_channel_order_and_spacing():
    rx=Receiver(id="rx",position_m=[10,20,5],
                antenna=dict(num_rows=2,num_cols=3,vertical_spacing_wavelengths=.4,
                             horizontal_spacing_wavelengths=.6))
    local=element_offsets_local(rx,C0)
    expected=np.array([[0,-.6,.2],[0,-.6,-.2],[0,0,.2],[0,0,-.2],[0,.6,.2],[0,.6,-.2]])
    np.testing.assert_allclose(local,expected,atol=1e-15)


def test_world_geometry_and_metadata():
    rx=Receiver(id="rx",position_m=[10,20,5],orientation_rad=[.3,.2,-.1],
                antenna=dict(num_rows=2,num_cols=3))
    s=config(rx)
    obs=public_observation(s,rx)
    local=element_offsets_local(rx,s.carrier_frequency_hz)
    world=local@rotation(rx.orientation_rad).T
    np.testing.assert_allclose(obs["element_offsets_world_m"],world)
    np.testing.assert_allclose(obs["element_positions_world_m"],world+rx.position_m)
    np.testing.assert_allclose(np.mean(obs["element_positions_world_m"],axis=0),rx.position_m)
    np.testing.assert_allclose(obs["array_pose"]["normal_world"],rotation(rx.orientation_rad)[:,0])
    assert obs["array_geometry"]["channel_map"]==[
        {"channel":0,"row":0,"col":0},{"channel":1,"row":1,"col":0},
        {"channel":2,"row":0,"col":1},{"channel":3,"row":1,"col":1},
        {"channel":4,"row":0,"col":2},{"channel":5,"row":1,"col":2}]
    assert obs["array_geometry"]["geometry_type"]=="ura"
    assert obs["shape"]==[6,50000]


def test_old_antenna_config_stays_linear():
    a=Antenna.model_validate(dict(elements=4,spacing_wavelengths=.7))
    assert (a.num_rows,a.num_cols)==(1,4)
    assert a.horizontal_spacing_wavelengths==.7
    old=config().model_dump()
    old["schema_version"]="1.0"
    old["receivers"][0].pop("antenna")
    assert Scenario.model_validate(old).receivers[0].antenna.num_rows==1


def test_ambiguous_old_new_array_config_rejected():
    with pytest.raises(ValueError):
        Antenna.model_validate(dict(elements=4,num_rows=2,num_cols=2))


def test_array_size_bound():
    with pytest.raises(ValueError):
        Antenna(num_rows=8,num_cols=8)


def test_legacy_manifest_does_not_claim_planar_ula(tmp_path):
    rx=config().receivers[0]
    obs=public_observation(config(),rx)
    manifest=legacy_manifest(tmp_path,obs)
    assert manifest["array_geometry"]["geometry_type"]=="custom"
    assert not manifest["array_geometry"].get("element_positions_m")
    assert not manifest["array_geometry"].get("element_spacing_m")
    assert manifest["provenance"]["array_geometry"]["geometry_type"]=="ura"


def test_legacy_horizontal_ula_unchanged(tmp_path):
    rx=Receiver(id="rx",position_m=[10,20,5],antenna=dict(elements=4))
    obs=public_observation(config(rx),rx)
    manifest=legacy_manifest(tmp_path,obs)
    assert manifest["array_geometry"]["geometry_type"]=="ula"
    assert len(manifest["array_geometry"]["element_positions_m"])==4


def test_six_channel_end_to_end(tmp_path):
    from cityrf.runner import run_scenario
    from cityrf.audit import audit_run
    rx=Receiver(id="rx",position_m=[10,20,5],
                antenna=dict(num_rows=2,num_cols=3),orientation_rad=[.3,.2,-.1])
    run_scenario(config(rx),tmp_path/"run")
    obs=json.loads((tmp_path/"run/public/observations.json").read_text())["receivers"][0]
    x=np.load(tmp_path/"run/public/iq_rx.npy")
    assert x.shape==(6,50000)
    assert len(obs["array_geometry"]["channel_map"])==x.shape[0]
    assert audit_run(tmp_path/"run")["status"]=="verified"
