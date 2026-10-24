"""Radio map（电磁态势图）：请求口径、网格约定、载荷序列化与 API 契约。

真实的 Sionna RT 求解是 opt-in 的（``RUN_RT_TESTS=1``），与 tests/test_rt.py 一致：
默认 CI 只验证不依赖 GPU/场景加载的那一层。
"""
import json
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cityrf.config import Scenario
from cityrf.radiomap import (MAX_CELLS, METRIC_META, RAMP_STOPS, RadioMapRequest,
                             RadioMapResult, _db, _dbm, _encode, to_payload)
from webui import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNS', tmp_path / 'runs'); server.RUNS.mkdir()
    monkeypatch.setattr(server, 'SCENARIOS', tmp_path / 'scenarios')
    monkeypatch.setattr(server, 'RADIO_MAPS', tmp_path / 'radio_maps')
    monkeypatch.setattr(server, 'RUNTIME', dict(native_rt=False, wsl_rt=False,
                                                available_backends=['free_space', 'tapped']))
    return TestClient(server.app)


def scenario(backend='sionna_rt', scene='munich'):
    return dict(name='rm-test', duration_s=.002, sample_rate_hz=1e6,
                propagation=dict(backend=backend, scene=scene),
                emitters=[dict(id='tx', position_m=[0, 0, 50],
                               components=[dict(kind='tone', offset_hz=100000)])],
                receivers=[dict(id='rx', position_m=[100, 0, 60],
                                antenna=dict(geometry_type='rectangular', num_rows=1, num_cols=1))])


def sample_result(metrics=('path_gain',), **overrides):
    values = {'path_gain': np.array([[-80.0, np.nan], [-95.0, -100.0]]),
              'rss': np.array([[-40.0, np.nan], [-55.0, -60.0]]),
              'sinr': np.array([[12.0, np.nan], [3.0, 1.0]])}
    grid = dict(origin_xy_m=[0.0, 0.0], cell_size_m=5.0, nx=2, ny=2, height_m=60.0,
                x_min_m=2.5, x_max_m=7.5, y_min_m=2.5, y_max_m=7.5)
    fields = dict(
        grid=grid, metrics={name: values[name] for name in metrics},
        association=np.array([[0, -1], [0, 0]], dtype=np.int32), tx_ids=['tx'],
        ranges={name: dict(min=-100.0, max=-80.0, vmin=-100.0, vmax=-80.0,
                           valid_count=3, total_count=4) for name in metrics},
        seconds=1.25, scene='munich', samples_per_tx=10_000_000, max_depth=3,
        metrics_requested=list(metrics))
    fields.update(overrides)
    return RadioMapResult(**fields)


# ---------------------------------------------------------------- 请求口径

def test_request_defaults_and_cleaning():
    default = RadioMapRequest()
    assert default.metrics == ['path_gain', 'rss', 'sinr']
    assert (default.height_m, default.cell_size_m) == (1.5, 5.0)
    assert (default.max_depth, default.samples_per_tx) == (3, 10_000_000)
    assert default.size_m is None and default.center_xy_m is None
    assert default.los and default.specular_reflection
    assert not default.diffuse_reflection and not default.refraction and not default.diffraction
    # 重复指标被去掉，且保持用户给定的顺序
    assert RadioMapRequest(metrics=['rss', 'rss', 'path_gain']).metrics == ['rss', 'path_gain']
    # 低空高度与地面高度都合法
    assert RadioMapRequest(height_m=120.0).height_m == 120.0
    assert RadioMapRequest(height_m=1.5).height_m == 1.5


@pytest.mark.parametrize('kwargs', [
    dict(metrics=[]),               # 至少一个指标
    dict(cell_size_m=0),            # 网格必须为正
    dict(size_m=[100.0, -1.0]),     # 平面尺寸必须为正
    dict(samples_per_tx=10),        # 采样数下界 1000
    dict(max_depth=11),             # 深度上界 10
    dict(height_m=6000.0),          # 高度上界 5000
])
def test_request_rejects_out_of_range(kwargs):
    with pytest.raises(Exception):
        RadioMapRequest(**kwargs)


def test_metric_metadata_and_ramp_are_shared_with_the_frontend():
    assert set(METRIC_META) == {'path_gain', 'rss', 'sinr'}
    assert METRIC_META['path_gain']['unit'] == 'dB'
    assert METRIC_META['rss']['unit'] == 'dBm'
    assert METRIC_META['sinr']['unit'] == 'dB'
    assert all(METRIC_META[name]['label'] for name in METRIC_META)
    assert len(RAMP_STOPS) == 9
    assert all(len(stop) == 3 and all(0.0 <= channel <= 1.0 for channel in stop)
               for stop in RAMP_STOPS)
    assert MAX_CELLS == 300_000


# ---------------------------------------------------------------- 数值口径

def test_db_and_dbm_conversion_and_nan_handling():
    linear = np.array([[1.0, 0.1, 0.0, np.nan]])
    valid = np.array([[True, True, False, True]])
    db = _db(linear, valid)
    assert db[0, 0] == pytest.approx(0.0)
    assert db[0, 1] == pytest.approx(-10.0)
    assert np.isnan(db[0, 2]) and np.isnan(db[0, 3])
    assert _dbm(np.array([[1e-6]]), np.array([[True]]))[0, 0] == pytest.approx(-30.0)


def test_encode_rounds_to_one_decimal_and_turns_nan_into_null():
    assert _encode(np.array([[-84.66, np.nan, 0.0, np.inf]])) == [-84.7, None, 0.0, None]


def test_payload_is_json_safe_and_keeps_grid_convention():
    result = sample_result()
    payload = to_payload(result, Scenario.model_validate(scenario()))
    json.dumps(payload, allow_nan=False)          # NaN 必须已经变成 null
    metric = payload['metrics']['path_gain']
    assert metric['data'] == [-80.0, None, -95.0, -100.0]
    assert metric['label'] == METRIC_META['path_gain']['label']
    assert metric['unit'] == 'dB' and metric['valid_count'] == 3 and metric['total_count'] == 4
    assert payload['association'] == {'tx_ids': ['tx'], 'data': [0, None, 0, 0]}
    assert payload['grid']['ny'] == 2 and payload['grid']['nx'] == 2
    assert payload['grid']['height_m'] == 60.0
    assert payload['scene'] == 'munich'
    assert [tx['id'] for tx in payload['emitters']] == ['tx']
    assert [rx['id'] for rx in payload['receivers']] == ['rx']


def test_payload_only_exposes_requested_metrics():
    payload = to_payload(sample_result(metrics=('rss',)),
                         Scenario.model_validate(scenario()))
    assert set(payload['metrics']) == {'rss'}


# ---------------------------------------------------------------- API 契约

def test_radio_map_api_rejects_wrong_backend_and_bad_request(client):
    assert client.post('/api/radio-maps',
                       json=dict(scenario=scenario('free_space'))).status_code == 422
    assert client.post('/api/radio-maps',
                       json=dict(scenario=scenario(), radio_map=dict(cell_size_m=0))).status_code == 422
    assert client.post('/api/radio-maps',
                       json=dict(scenario=scenario(), radio_map=dict(metrics=[]))).status_code == 422
    assert client.get('/api/radio-maps/bad.id').status_code == 400
    assert client.get('/api/radio-maps/map_missing').status_code == 404


def test_radio_map_job_reports_a_readable_failure_without_rt(client):
    created = client.post('/api/radio-maps',
                          json=dict(scenario=scenario(), radio_map=dict(cell_size_m=50.0)))
    assert created.status_code == 200
    jid = created.json()['id']
    assert jid.startswith('map_')
    deadline = time.monotonic() + 30
    data = {}
    while time.monotonic() < deadline:
        data = client.get('/api/radio-maps/' + jid).json()
        if data['status'] not in ('running', 'queued'):
            break
        time.sleep(.05)
    assert data['status'] == 'failed', data
    assert 'Sionna RT' in data['error']
    # 作业元数据在创建时就写好了，前端不必等求解完成才能显示高度与网格
    assert data['height_m'] == 1.5 and data['cell_size_m'] == 50.0
    assert data['scene'] == 'munich'
    assert any(job['id'] == jid for job in client.get('/api/radio-maps').json())
    # 文件下载只能落在该作业目录内，且后缀受白名单限制
    assert client.get(f'/api/radio-maps/{jid}/files/%2E%2E%2Fjob.json').status_code == 404
    assert client.get(f'/api/radio-maps/{jid}/files/worker.log').status_code == 200
    assert client.get(f'/api/radio-maps/{jid}/files/radiomap.exe').status_code == 404


def test_radio_map_list_is_newest_first(client, tmp_path):
    root = server.RADIO_MAPS
    for name, started in (('map_old', 100.0), ('map_new', 200.0)):
        (root / name).mkdir(parents=True)
        (root / name / 'job.json').write_text(
            json.dumps(dict(id=name, status='complete', name=name, started_at=started)),
            encoding='utf-8')
    assert [job['id'] for job in client.get('/api/radio-maps').json()] == ['map_new', 'map_old']


# ---------------------------------------------------------------- 真实求解（opt-in）

@pytest.mark.skipif(os.getenv('RUN_RT_TESTS') != '1', reason='opt-in actual RT')
def test_real_radio_map_on_munich():
    from cityrf.radiomap import compute_radio_map

    scenario_obj = Scenario.model_validate(scenario())
    request = RadioMapRequest(height_m=60.0, cell_size_m=50.0, samples_per_tx=100_000,
                              max_depth=2, metrics=['path_gain', 'rss', 'sinr'])
    result = compute_radio_map(scenario_obj, request, emit=lambda _line: None)

    assert result.seconds > 0
    assert result.scene == 'munich'
    shape = (result.grid['ny'], result.grid['nx'])
    assert result.association.shape == shape
    gain, rss, sinr = (result.metrics[name] for name in ('path_gain', 'rss', 'sinr'))
    for array in (gain, rss, sinr):
        assert array.shape == shape
    finite = gain[np.isfinite(gain)]
    assert finite.size > 0
    assert finite.max() <= 1e-9                     # 路径增益不会超过 0 dB
    assert np.all(np.isfinite(sinr[np.isfinite(sinr)]) & (sinr[np.isfinite(sinr)] >= 0))
    # 单个发射源：每格要么属于它，要么没有任何路径
    assert set(np.unique(result.association)).issubset({-1, 0})
    assert result.grid['x_min_m'] < result.grid['x_max_m']
    assert result.grid['y_min_m'] < result.grid['y_max_m']
