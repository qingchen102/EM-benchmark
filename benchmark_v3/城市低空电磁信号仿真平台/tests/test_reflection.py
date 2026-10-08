import numpy as np
from cityrf.baselines.reflection import Geometry


def test_wall_hit_reflection_and_miss():
    g=Geometry(np.array([[[0,-20,-20],[0,20,-20],[0,0,20]]],float))
    hit=g.first_hit([2,0,0],[-1,0,0])
    assert np.allclose(hit['point_m'],[0,0,0])
    paths=g.backtrace([2,0,0],[-1,0,0])
    assert len(paths)==2
    assert np.allclose(paths[1]['direction'],[1,0,0])
    assert g.first_hit([2,0,0],[1,0,0]) is None


def test_direct_intersection_and_blocked_segment():
    g=Geometry(np.array([[[0,-20,-20],[0,20,-20],[0,0,20]]],float))
    assert not g.visible([2,0,0],[-2,0,0])
    assert g.visible([2,0,0],[3,1,0])
    stations=[{'receiver_id':'a','position_m':[2,-2,1],'directions':[[0,1,0]]},
              {'receiver_id':'b','position_m':[3,0,1],'directions':[[-1,0,0]]}]
    result=g.localize(stations)
    assert np.allclose(result['hypotheses'][0]['position_m'],[2,0,1])


def test_one_bounce_recovers_source_without_passing_through_wall():
    g=Geometry(np.array([[[0,-20,-20],[0,20,-20],[0,0,20]]],float))
    stations=[{'receiver_id':'a','position_m':[1,-2,1],'directions':[[-3,2,0]]},
              {'receiver_id':'b','position_m':[1,2,1],'directions':[[-3,-2,0]]}]
    result=g.localize(stations)
    assert np.allclose(result['hypotheses'][0]['position_m'],[2,0,1])
    assert all(p['kind']=='single_reflection' for p in result['hypotheses'][0]['paths'])


def test_geometry_export_is_bound_and_frozen(tmp_path,monkeypatch):
    import pytest
    pytest.importorskip("cityrf.light_agent", reason="Agent workspace is outside standalone simulator scope")
    import json
    from cityrf.light_agent import workspace as module
    from cityrf.export import digest
    from cityrf.light_agent.runner import verify_inputs
    public=tmp_path/'public';public.mkdir()
    (public/'observations.json').write_text('{"receivers":[]}',encoding='utf-8')
    monkeypatch.setattr(module,'load_public',lambda _: {'receivers':[]})
    geometry=tmp_path/'geometry';geometry.mkdir()
    np.savez(geometry/'mesh.npz',triangles=np.zeros((1,3,3)))
    meta={'observations_sha256':digest(public/'observations.json'),'mesh_sha256':digest(geometry/'mesh.npz')}
    (geometry/'geometry.json').write_text(json.dumps(meta),encoding='utf-8')
    ws=module.assemble_workspace(public,tmp_path/'run/sandbox',geometry_dir=geometry)
    assert verify_inputs(ws) and ws.resolve('lib/rfkit/reflection.py').is_file()
    meta['mesh_sha256']='bad'
    (geometry/'geometry.json').write_text(json.dumps(meta),encoding='utf-8')
    import pytest
    with pytest.raises(ValueError,match='哈希'):
        module.assemble_workspace(public,tmp_path/'bad/sandbox',geometry_dir=geometry)

