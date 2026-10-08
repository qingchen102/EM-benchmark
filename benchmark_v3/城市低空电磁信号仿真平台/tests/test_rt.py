import os
import numpy as np
import pytest
from cityrf.config import Scenario
from cityrf.sionna_backend import SionnaRTBackend


def test_backend_missing_does_not_fallback(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == "mitsuba":
            raise ImportError("not installed")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    s=Scenario(emitters=[dict(id="tx",position_m=[0,0,10])],
               receivers=[dict(id="rx",position_m=[100,0,10])])
    with pytest.raises(RuntimeError,match="Sionna RT"):
        SionnaRTBackend().solve(s)


@pytest.mark.rt
@pytest.mark.skipif(os.getenv("RUN_RT_TESTS") != "1",reason="opt-in actual RT")
def test_actual_rt_empty_scene_los():
    from cityrf.channels import FreeSpaceBackend
    s=Scenario(emitters=[dict(id="tx",position_m=[0,100,10])],
               receivers=[dict(id="rx",position_m=[0,0,10])],
               propagation=dict(backend="sionna_rt",scene="empty",max_depth=0,samples_per_src=1000))
    actual=SionnaRTBackend().solve(s)[0]
    expected=FreeSpaceBackend().solve(s)[0]
    assert actual.coefficients.shape == (s.receivers[0].antenna.elements,1)
    np.testing.assert_allclose(abs(actual.coefficients),abs(expected.coefficients),rtol=1e-4)
    np.testing.assert_allclose(actual.delays_s,expected.delays_s,rtol=1e-5)
    ratio=actual.coefficients[:,0]/actual.coefficients[0,0]
    want=expected.coefficients[:,0]/expected.coefficients[0,0]
    np.testing.assert_allclose(ratio,want,atol=1e-4)


@pytest.mark.rt
@pytest.mark.skipif(os.getenv("RUN_RT_TESTS") != "1",reason="opt-in actual RT")
def test_oblique_rotated_array_rt_phase_sign():
    from cityrf.channels import FreeSpaceBackend
    s=Scenario(emitters=[dict(id="tx",position_m=[120,80,10])],
               receivers=[dict(id="rx",position_m=[0,0,10],orientation_rad=[.3,0,0])],
               propagation=dict(backend="sionna_rt",scene="empty",max_depth=0,samples_per_src=1000))
    actual=SionnaRTBackend().solve(s)[0]
    expected=FreeSpaceBackend().solve(s)[0]
    ratio=actual.coefficients[:,0]/actual.coefficients[0,0]
    want=expected.coefficients[:,0]/expected.coefficients[0,0]
    np.testing.assert_allclose(ratio,want,atol=1e-4)


@pytest.mark.rt
@pytest.mark.skipif(os.getenv("RUN_RT_TESTS") != "1",reason="opt-in actual RT")
def test_planar_asymmetric_rt_geometry_and_elevation_phase():
    import mitsuba as mi
    mi.set_variant("llvm_ad_mono_polarized")
    import sionna.rt as rt
    from cityrf.channels import FreeSpaceBackend, element_offsets_local, C0
    s=Scenario(emitters=[dict(id="tx",position_m=[120,80,70])],
               receivers=[dict(id="rx",position_m=[0,0,10],orientation_rad=[.3,.2,-.1],
                    antenna=dict(num_rows=2,num_cols=3,vertical_spacing_wavelengths=.4,
                                 horizontal_spacing_wavelengths=.6))],
               propagation=dict(backend="sionna_rt",scene="empty",max_depth=0,samples_per_src=1000))
    array=rt.PlanarArray(num_rows=2,num_cols=3,vertical_spacing=.4,horizontal_spacing=.6,
                        pattern="iso",polarization="V")
    expected_local=element_offsets_local(s.receivers[0],s.carrier_frequency_hz)
    np.testing.assert_allclose(array.positions(C0/s.carrier_frequency_hz).numpy().T,
                               expected_local,atol=1e-8)
    actual=SionnaRTBackend().solve(s)[0]
    expected=FreeSpaceBackend().solve(s)[0]
    assert actual.coefficients.shape==(6,1)
    np.testing.assert_allclose(actual.coefficients[:,0]/actual.coefficients[0,0],
                               expected.coefficients[:,0]/expected.coefficients[0,0],atol=1e-4)
