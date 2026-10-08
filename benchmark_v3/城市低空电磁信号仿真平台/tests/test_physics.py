import numpy as np
import pytest
from cityrf.channels import FreeSpaceBackend, TappedBackend, LinkChannel, apply_channel
from cityrf.config import Scenario, Receiver, Frontend
from cityrf.receiver import receive


def cfg(distance=100):
    return Scenario(emitters=[dict(id="tx", position_m=[0,0,10])],
                    receivers=[dict(id="rx",position_m=[distance,0,10])])


def test_friis_distance_squared():
    a = FreeSpaceBackend().solve(cfg(100))[0]
    b = FreeSpaceBackend().solve(cfg(200))[0]
    assert abs(a.coefficients[0,0])**2/abs(b.coefficients[0,0])**2 == pytest.approx(4, rel=.001)
    assert a.delays_s[0] == pytest.approx(100/299792458)


def test_array_phase_for_endfire():
    s = Scenario(emitters=[dict(id="tx",position_m=[0,100,10])],
                 receivers=[dict(id="rx",position_m=[0,0,10],antenna=dict(elements=4))])
    ch = FreeSpaceBackend().solve(s)[0]
    assert np.angle(ch.coefficients[1,0]/ch.coefficients[0,0]) == pytest.approx(np.pi,abs=.001)


def test_fractional_delay_matches_analytic_tone():
    fs, f, delay = 1e6, 70000, 3.35e-6
    indices = np.arange(-128, 1200)
    x = np.exp(2j*np.pi*f*indices/fs)
    ch = LinkChannel("tx","rx",np.array([[.3+.2j]]),np.array([delay]),{}, {})
    y = apply_channel(x,indices[0],ch,fs,1000)
    expected = (.3+.2j)*np.exp(2j*np.pi*f*(np.arange(1000)/fs-delay))
    np.testing.assert_allclose(y[0],expected,atol=2e-5)


def test_absolute_delay_on_impulse():
    indices=np.arange(-128,300)
    x=(indices==50).astype(complex)
    ch=LinkChannel("tx","rx",np.ones((1,1),complex),np.array([10e-6]),{}, {})
    y=apply_channel(x,-128,ch,1e6,200)
    assert np.argmax(abs(y[0])) == 60


def test_zero_paths_gives_zero():
    ch=LinkChannel("tx","rx",np.empty((4,0),complex),np.empty(0),{}, {})
    assert not apply_channel(np.zeros(200,complex),-50,ch,1e6,100).any()


def test_post_sum_nonlinearity():
    rx=Receiver(id="rx",position_m=[0,0,0],bandwidth_hz=800000,
                frontend=Frontend(noise_enabled=False,soft_limit_input_dbm=0))
    a=np.full((rx.antenna.elements,4096),np.sqrt(.001),complex)
    together,_=receive(2*a,rx,1e6,42)
    separate,_=receive(a,rx,1e6,42)
    assert abs(together[0,2000]) < 2*abs(separate[0,2000])*.8


def test_noise_power_and_station_independence():
    rx=Receiver(id="rx",position_m=[0,0,0],bandwidth_hz=400000)
    y,meta=receive(np.zeros((rx.antenna.elements,200000),complex),rx,1e6,42)
    assert np.mean(abs(y)**2) == pytest.approx(meta["expected_output_noise_w"],rel=.025)
    ry=rx.model_copy(update={"id":"ry"})
    z,_=receive(np.zeros_like(y),ry,1e6,42)
    assert abs(np.mean(y*z.conj())) < .02*np.mean(abs(y)**2)


def test_receiver_tuning():
    rx=Receiver(id="rx",position_m=[0,0,0],bandwidth_hz=200000,tuning_offset_hz=200000,
                frontend=Frontend(noise_enabled=False))
    x=np.tile(np.exp(2j*np.pi*.21*np.arange(10000)),(rx.antenna.elements,1))
    y,_=receive(x,rx,1e6,42)
    frequency=np.angle(np.mean(y[:,501:9000]*y[:,500:8999].conj()))*1e6/(2*np.pi)
    assert frequency == pytest.approx(10000,abs=.1)


def test_fractional_delay_band_edge():
    fs,f=1e6,479000
    indices=np.arange(-256,1500)
    x=np.exp(2j*np.pi*f*indices/fs)
    ch=LinkChannel("tx","rx",np.ones((1,1)),np.array([.5/fs]),{}, {})
    y=apply_channel(x,indices[0],ch,fs,1000)
    expected=np.exp(2j*np.pi*f*(np.arange(1000)/fs-.5/fs))
    np.testing.assert_allclose(y[0],expected,atol=1e-4)
