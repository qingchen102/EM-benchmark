import numpy as np
import pytest
from cityrf.baselines.detection import detect_bands
from cityrf.baselines.spatial import scan_directions
from cityrf.baselines.localization import intersect_rays
from cityrf.channels import element_offsets, C0
from cityrf.config import Receiver


def test_energy_detects_two_tones_not_noise():
    rng=np.random.default_rng(123)
    fs=1e6
    n=np.arange(50000)
    noise=(rng.normal(size=(4,len(n)))+1j*rng.normal(size=(4,len(n))))*.02
    x=noise+np.exp(2j*np.pi*100000*n/fs)+.5*np.exp(-2j*np.pi*200000*n/fs)
    out=detect_bands(x,fs,800000)
    for frequency in (100000,-200000):
        assert any(b["low_hz"]<=frequency<=b["high_hz"] for b in out["bands"])
    assert not detect_bands(noise,fs,800000)["bands"]


def test_coplanar_music_retains_mirror_and_correct_sign():
    rx=Receiver(id="rx",position_m=[0,0,0])
    offsets=element_offsets(rx,2.4e9)
    az,el=np.deg2rad([30,20])
    u=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    a=np.exp(2j*np.pi/(C0/2.4e9)*offsets@u)
    cov=a[:,None]*a[None,:].conj()+np.eye(16)*.01
    out=scan_directions(cov,offsets,2.4e9,1000)
    peaks=out["music"]["peaks"]
    vectors=np.array([p["unit_direction_world"] for p in peaks])
    assert np.max(vectors@u) > .9999
    mirror=u.copy();mirror[0]*=-1
    assert np.max(vectors@mirror) > .9999
    assert out["mdl_rank"]==1


def test_intersections_and_degenerate_geometry():
    origins=np.array([[0,0,0],[100,0,0],[0,100,10]],float)
    truth=np.array([30,40,50],float)
    directions=truth-origins
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    result=intersect_rays(origins,directions)
    np.testing.assert_allclose(result["position_m"],truth,atol=1e-9)
    assert result["angular_rms_deg"] < 1e-5
    assert intersect_rays(origins,np.tile([1.,0,0],(3,1))) is None


def test_baseline_public_only_and_frozen(tmp_path,monkeypatch):
    from cityrf.config import Scenario
    from cityrf.runner import run_scenario
    from cityrf.baselines.runner import run_baseline
    from pathlib import Path
    import json
    from cityrf.export import digest
    s=Scenario(emitters=[dict(id="tx",position_m=[30,40,50],
                   components=[dict(kind="tone",offset_hz=100000)])],
        receivers=[dict(id="r1",position_m=[0,0,5]),dict(id="r2",position_m=[100,0,5])])
    run_scenario(s,tmp_path/"capture")
    original=Path.read_text
    def guarded(path,*args,**kwargs):
        assert "private" not in path.parts
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",guarded)
    run_baseline(tmp_path/"capture/public",tmp_path/"baseline")
    frozen=json.loads((tmp_path/"baseline/freeze.json").read_text())
    assert frozen["prediction_sha256"]==digest(tmp_path/"baseline/prediction.json")
    prediction=json.loads((tmp_path/"baseline/prediction.json").read_text())
    assert len(prediction["groups"])==1
    assert prediction["groups"][0]["localization"]["music"]["hypotheses"]


def test_location_matching_one_to_one_with_distance_gate():
    from cityrf.baselines.evaluate import match_locations
    sources=[{"id":"a","position_m":[0,0,0]},{"id":"b","position_m":[100,0,0]}]
    estimates=[{"group_id":"g0","position_m":[3,0,0]},{"group_id":"g1","position_m":[4,0,0]}]
    result=match_locations(estimates,sources,50)
    assert result["matched_sources"]==1
    assert result["false_locations"]==1
    assert result["missed_sources"]==1


def test_evaluator_requires_frozen_prediction_before_truth(tmp_path):
    from cityrf.baselines.evaluate import evaluate
    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path/"no_capture",tmp_path/"unfrozen")


def test_direction_peaks_exclude_broad_lobe_shoulders():
    from cityrf.baselines.spatial import _peaks
    az,el=np.meshgrid(np.arange(-180,180,2),np.arange(-88,89,2))
    a,e=np.deg2rad(az.ravel()),np.deg2rad(el.ravel())
    u=np.stack([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)],axis=1)
    power=np.exp(-((az.ravel()-30)**2+(el.ravel()-20)**2)/400)
    out=_peaks(power,u,az.ravel(),el.ravel())
    assert len(out)==1


def test_location_match_cannot_match_unrelated_frequency_source():
    from cityrf.baselines.evaluate import match_locations
    result=match_locations([{"group_id":"tone","position_m":[0,0,0],"eligible_source_ids":["b"]}],
                           [{"id":"a","position_m":[0,0,0]},{"id":"b","position_m":[100,0,0]}])
    assert result["matched_sources"]==0


def test_changed_prediction_rejected_before_missing_truth(tmp_path):
    from cityrf.baselines.evaluate import evaluate
    from cityrf.export import write_json
    write_json(tmp_path/"freeze.json",{"prediction_sha256":"wrong"})
    write_json(tmp_path/"prediction.json",{})
    with pytest.raises(ValueError,match="hash"):
        evaluate(tmp_path/"missing-capture",tmp_path)
