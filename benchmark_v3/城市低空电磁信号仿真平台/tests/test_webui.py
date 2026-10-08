import json
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from webui import server


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'RUNS',tmp_path/'runs');server.RUNS.mkdir()
    monkeypatch.setattr(server,'SCENARIOS',tmp_path/'scenarios')
    monkeypatch.setattr(server,'RUNTIME',dict(native_rt=False,wsl_rt=False,available_backends=['free_space','tapped']))
    return TestClient(server.app)


def scenario():
    return dict(name='api-test',duration_s=.002,sample_rate_hz=1e6,
        emitters=[dict(id='tx',position_m=[0,0,50],components=[dict(kind='tone',offset_hz=100000)])],
        receivers=[dict(id='rx',position_m=[100,0,60],antenna=dict(geometry_type='rectangular',num_rows=1,num_cols=1))])


def test_schema_presets_and_validation(client):
    b=client.get('/api/bootstrap').json()
    assert len(b['presets'])>=9
    assert b['schema']['$defs']['Component']['properties']['kind']['enum']
    s=scenario();r=client.post('/api/validate',json=s)
    assert r.status_code==200 and r.json()['num_samples']==2000
    s['sample_rate_hz']=10000
    assert client.post('/api/validate',json=s).status_code==422


def test_reject_cross_origin_and_save_roundtrip(client):
    assert client.post('/api/scenarios',json=scenario(),headers={'origin':'https://other.invalid'}).status_code==403
    r=client.post('/api/scenarios',json=scenario());assert r.status_code==200
    saved=client.get('/api/scenarios').json();assert saved[0]['config']['name']=='api-test'
    assert client.get('/api/runs/bad.run').status_code==400


def test_real_free_space_job_preview_and_download(client):
    r=client.post('/api/runs',json=scenario());assert r.status_code==200
    jid=r.json()['id']
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        data=client.get('/api/runs/'+jid).json()
        if data['status'] not in ('running','queued'):break
        time.sleep(.1)
    assert data['status']=='complete',data.get('log')
    row=data['preview']['stations'][0]
    f=np.array(row['frequency_khz']);p=np.array(row['psd_dbm_hz'])
    assert abs(f[p.argmax()]-100)<1
    assert row['samples']==2000 and len(row['spectrogram_dbm_hz'])<=256
    assert data['summary']['backend']=='free_space'
    assert client.get(f'/api/runs/{jid}/files/capture/public/iq_rx.npy').status_code==200
    assert client.get(f'/api/runs/{jid}/files/%2E%2E%2F%2E%2E%2FREADME.md').status_code==404
