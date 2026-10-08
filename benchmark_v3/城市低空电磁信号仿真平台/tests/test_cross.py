import json
import os
import numpy as np
import pytest
from cityrf.config import Antenna, Receiver, Scenario
from cityrf.channels import element_offsets_local, element_offsets, rotation, C0
from cityrf.export import public_observation, legacy_manifest
from cityrf.runner import run_scenario


def make():
    return Scenario(emitters=[dict(id="tx",position_m=[100,60,80])],
                    receivers=[dict(id="rx",position_m=[10,20,5],orientation_rad=[.3,.2,-.1])])


def test_default_eight_plus_eight_cross():
    a=Antenna()
    assert a.geometry_type=="cross" and a.elements==16
    assert a.horizontal_elements==8 and a.vertical_elements==8


def test_cross_local_order_unique_centered_and_aperture():
    rx=make().receivers[0]
    p=element_offsets_local(rx,C0)
    assert p.shape==(16,3)
    assert len(np.unique(p,axis=0))==16
    np.testing.assert_allclose(p.mean(axis=0),0,atol=1e-15)
    np.testing.assert_allclose(p[:8,1],np.arange(-3.5,4)*.5)
    np.testing.assert_allclose(p[:8,[0,2]],0)
    np.testing.assert_allclose(p[8:,2],np.arange(3.5,-4,-1)*.5)
    np.testing.assert_allclose(p[8:,:2],0)
    assert not np.any(np.all(p==0,axis=1))
    assert np.ptp(p[:,1])==3.5 and np.ptp(p[:,2])==3.5


def test_cross_pose_and_export(tmp_path):
    s=make()
    r=s.receivers[0]
    obs=public_observation(s,r)
    assert obs["array_geometry"]["geometry_type"]=="cross"
    assert [v["arm"] for v in obs["array_geometry"]["channel_map"]]==["horizontal"]*8+["vertical"]*8
    np.testing.assert_allclose(obs["element_positions_world_m"],
                              element_offsets_local(r,s.carrier_frequency_hz)@rotation(r.orientation_rad).T+r.position_m)
    assert legacy_manifest(tmp_path,obs)["array_geometry"]["geometry_type"]=="custom"


@pytest.mark.parametrize("params",[
    dict(geometry_type="cross",horizontal_elements=7,vertical_elements=8),
    dict(geometry_type="cross",horizontal_elements=10,vertical_elements=8),
    dict(geometry_type="cross",num_rows=2,num_cols=8),
    dict(geometry_type="rectangular",horizontal_elements=8,vertical_elements=8)])
def test_cross_invalid_or_ambiguous_rejected(params):
    with pytest.raises(ValueError):
        Antenna(**params)


@pytest.mark.parametrize("params",[dict(horizontal_elements=0),dict(num_rows=0,num_cols=2),
    dict(elements=4,geometry_type="cross")])
def test_array_zero_and_conflicting_migration_rejected(params):
    with pytest.raises(ValueError):
        Antenna(**params)


@pytest.mark.parametrize("antenna",[{"horizontal_spacing_wavelengths":.6},{"pattern":"iso"}])
def test_partial_v11_config_preserves_rectangular_default(antenna):
    data=make().model_dump()
    data["schema_version"]="1.1"
    data["receivers"][0]["antenna"]=antenna
    a=Scenario.model_validate(data).receivers[0].antenna
    assert a.geometry_type=="rectangular" and a.elements==4
    assert a.horizontal_spacing_wavelengths==antenna.get("horizontal_spacing_wavelengths",.5)


def test_cross_generation_shape(tmp_path):
    run_scenario(make(),tmp_path/"cross")
    x=np.load(tmp_path/"cross/public/iq_rx.npy")
    obs=json.loads((tmp_path/"cross/public/observations.json").read_text())["receivers"][0]
    assert x.shape==(16,50000)
    assert len(obs["element_positions_world_m"])==16
    assert np.isfinite(x).all()


@pytest.mark.rt
@pytest.mark.skipif(os.getenv("RUN_RT_TESTS")!="1",reason="actual RT opt-in")
def test_cross_rotated_elevated_rt_phase():
    from cityrf.sionna_backend import SionnaRTBackend
    from cityrf.channels import FreeSpaceBackend
    s=make()
    s.propagation.scene="empty"
    s.propagation.max_depth=0
    s.propagation.samples_per_src=1000
    actual=SionnaRTBackend().solve(s)[0]
    expected=FreeSpaceBackend().solve(s)[0]
    assert actual.coefficients.shape==(16,1)
    np.testing.assert_allclose(actual.coefficients[:,0]/actual.coefficients[0,0],
                               expected.coefficients[:,0]/expected.coefficients[0,0],atol=2e-4)
