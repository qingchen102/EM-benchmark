import numpy as np
import pytest
from cityrf.config import Scenario, Component, Emitter, Receiver
from cityrf.waveforms import generate


def scenario(**kw):
    data = dict(emitters=[dict(id="tx", position_m=[0, 0, 10])],
                receivers=[dict(id="rx", position_m=[100, 0, 10])])
    data.update(kw)
    return Scenario.model_validate(data)


def test_nyquist_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        scenario(emitters=[dict(id="tx", position_m=[0,0,10],
            components=[dict(kind="tone", offset_hz=600000)])])


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        scenario(receivers=[dict(id="tx", position_m=[10,0,10])])


def test_missing_protected_source_rejected():
    with pytest.raises(ValueError):
        scenario(tasks=[dict(id="a", receiver_ids=["rx"], desired_emitter_ids=["missing"])])


def test_tone_power_and_frequency():
    tx = Emitter(id="tx", position_m=[0,0,10], power_dbm=10,
                 components=[Component(kind="tone", offset_hz=10000)])
    x = generate(tx, 1e6, np.arange(10000), 17)
    assert np.mean(abs(x)**2) == pytest.approx(.01, rel=1e-6)
    assert np.angle(np.mean(x[1:]*x[:-1].conj()))*1e6/(2*np.pi) == pytest.approx(10000)


def test_pulse_duty_does_not_renormalize():
    tx = Emitter(id="tx", position_m=[0,0,10], power_dbm=0,
                 components=[Component(kind="tone", pulse_period_s=.001, duty_cycle=.2)])
    x = generate(tx, 1e6, np.arange(50000), 17)
    # Unit-DC-gain TX bandlimiting preserves mean envelope; edges remove some energy.
    assert x.mean().real == pytest.approx(np.sqrt(.001)*.2,rel=1e-8)
    assert np.mean(abs(x)**2) == pytest.approx(.001*.2, rel=.02)


@pytest.mark.parametrize("kind", ["tone", "noise", "qpsk", "bpsk", "fsk", "ofdm", "chirp"])
def test_windows_share_exact_source(kind):
    tx = Emitter(id="tx", position_m=[0,0,10], components=[Component(kind=kind)])
    whole = generate(tx, 1e6, np.arange(-64, 10000), 17)
    part = generate(tx, 1e6, np.arange(123, 4123), 17)
    np.testing.assert_allclose(part, whole[187:4187], rtol=1e-12, atol=1e-12)
    assert np.isfinite(whole).all()


def test_coherent_mixture_keeps_configured_total_power():
    tx=Emitter(id="tx",position_m=[0,0,10],power_dbm=0,
               components=[Component(kind="tone"),Component(kind="tone")])
    x=generate(tx,1e6,np.arange(10000),17)
    assert np.mean(abs(x)**2)==pytest.approx(.001,rel=1e-6)


def test_cancelled_composite_is_rejected():
    tx=Emitter(id="tx",position_m=[0,0,10],
               components=[Component(kind="tone"),Component(kind="tone",phase_rad=np.pi)])
    with pytest.raises(ValueError,match="cancel"):
        generate(tx,1e6,np.arange(1000),17)


def test_gated_tone_has_nonzero_configured_bandwidth():
    with pytest.raises(ValueError,match="Nyquist"):
        scenario(emitters=[dict(id="tx",position_m=[0,0,10],components=[
            dict(kind="tone",offset_hz=470000,pulse_period_s=.00001,duty_cycle=.5)])])


def test_gate_is_followed_by_bandlimiting():
    tx=Emitter(id="tx",position_m=[0,0,10],
               components=[Component(kind="tone",bandwidth_hz=50000,
                                     pulse_period_s=.00001,duty_cycle=.5)])
    x=generate(tx,1e6,np.arange(20000),17)
    f=np.fft.fftfreq(len(x),1e-6)
    p=abs(np.fft.fft(x))**2
    assert p[abs(f)>50000].sum()/p.sum() < 1e-5


def test_zero_samples_rejected():
    with pytest.raises(ValueError):
        scenario(duration_s=1e-14)


def test_duplicate_task_sources_rejected():
    with pytest.raises(ValueError):
        scenario(tasks=[dict(id="t",receiver_ids=["rx"],desired_emitter_ids=["tx","tx"])])
