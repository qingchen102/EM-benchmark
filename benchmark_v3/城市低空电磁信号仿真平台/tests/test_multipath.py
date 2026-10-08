import numpy as np
from cityrf.baselines.reflection import Geometry


def test_coherent_ula_smoothing_restores_two_modes():
    from cityrf.baselines.multipath import smooth_ula_covariance, ula_music
    n=np.arange(8);mu=np.array([-.4,.25])
    a=np.exp(2j*np.pi*.5*n[:,None]*mu)
    x=(a@np.array([1,.7j]))[:,None]*np.exp(1j*np.arange(1000)*.17)
    assert np.linalg.matrix_rank(x@x.conj().T)==1
    r=smooth_ula_covariance(x,5)
    assert np.linalg.matrix_rank(r,tol=1e-8)==2
    result=ula_music(r,.5,2)
    assert np.allclose(sorted(result['direction_cosines'][:2]),mu,atol=.003)


def test_cross_pairing_uses_all_channels_and_preserves_mirror():
    from cityrf.baselines.multipath import cross_direction_candidates, fit_path_candidates
    t=(np.arange(8)-3.5)*.5
    off=np.vstack([np.c_[t*0,t,t*0],np.c_[t*0,t*0,-t]])
    mu_y=np.array([-.4,.25]);mu_z=np.array([.55,.15])
    dirs=np.c_[np.sqrt(1-mu_y**2-mu_z**2),mu_y,mu_z]
    a=np.exp(2j*np.pi*off@dirs.T)
    x=(a@np.array([1,.7j]))[:,None]*np.exp(1j*np.arange(200)*.13)
    candidates=cross_direction_candidates(mu_y,mu_z,np.eye(3))
    result=fit_path_candidates(x,off,299792458.,candidates,2)
    assert result[0]['residual_fraction']<1e-10
    recovered=np.array(result[0]['directions'])[:,1:]
    assert all(np.min(np.linalg.norm(recovered-v,axis=1))<1e-10 for v in dirs[:,1:])
    assert sum(r['residual_fraction']<1e-10 for r in result)>=4


def setup_walls():
    # x=0 and y=0, each covered by two triangles.
    square=np.array([[0,-20,-20],[0,20,-20],[0,20,20],[0,-20,20.]])
    tris=np.array([square[[0,1,2]],square[[0,2,3]]])
    other=tris[:,:, [1,0,2]]
    return Geometry(np.concatenate([tris,other]))


def test_single_station_two_reflections_and_path_difference():
    from cityrf.baselines.multipath import localize_multipath
    g=setup_walls();rx=np.array([2.,3.,1.]);src=np.array([5.,6.,2.])
    observations=[];lengths=[]
    for axis in [0,1]:
        image=src.copy();image[axis]*=-1
        observations.append(dict(receiver_id='same_station',position_m=rx.tolist(),
            directions=[((image-rx)/np.linalg.norm(image-rx)).tolist()]))
        lengths.append(np.linalg.norm(image-rx))
    delta=lengths[0]-lengths[1]
    result=localize_multipath(g,observations,path_differences=[dict(i=0,j=1,difference_m=delta,tolerance_m=.01)])
    assert result['hypotheses']
    h=result['hypotheses'][0]
    assert np.allclose(h['position_m'],src,atol=1e-6)
    assert np.allclose(h['total_path_lengths_m'],lengths)
    bad=localize_multipath(g,observations,path_differences=[dict(i=0,j=1,difference_m=delta+1,tolerance_m=.01)])
    assert not bad['hypotheses']
    absolute=[dict(o,total_length_m=lengths[i],length_tolerance_m=.01) for i,o in enumerate(observations)]
    assert np.allclose(localize_multipath(g,absolute)['hypotheses'][0]['position_m'],src)
    absolute[0]['total_length_m']+=1
    assert not localize_multipath(g,absolute)['hypotheses']


def test_parallel_rays_do_not_produce_a_location():
    from cityrf.baselines.multipath import localize_multipath
    g=Geometry(np.array([[[100,-10,-10],[100,10,-10],[100,0,10]]]))
    obs=[dict(receiver_id=str(i),position_m=[0,i,1],directions=[[-1,0,0]]) for i in [0,1]]
    assert not localize_multipath(g,obs)['hypotheses']


def test_exact_wiki_examples(tmp_path, monkeypatch):
    import re, sys, types
    from pathlib import Path
    from cityrf.baselines import multipath
    alias=types.ModuleType('rfkit');alias.__path__=[]
    monkeypatch.setitem(sys.modules,'rfkit',alias)
    monkeypatch.setitem(sys.modules,'rfkit.multipath',multipath)
    pages=Path(__file__).resolve().parents[1]/'docs/methods'
    t=(np.arange(8)-3.5)*.5
    off=np.vstack([np.c_[t*0,t,t*0],np.c_[t*0,t*0,-t]])
    y=np.array([-.4,.25]);z=np.array([.55,.15]);dirs=np.c_[np.sqrt(1-y*y-z*z),y,z]
    x=(np.exp(2j*np.pi*off@dirs.T)@np.array([1,.7j]))[:,None]*np.exp(1j*np.arange(500)*.13)
    env=dict(xband=x,offsets_world_m=off,carrier_frequency_hz=299792458.,rotation_local_to_world=np.eye(3))
    def reference_code(name):
        text=(pages/name).read_text(encoding='utf-8')
        text=re.sub(r'<!-- checked-example:start -->.*?<!-- checked-example:end -->','',text,flags=re.S)
        text=re.sub(r'<!-- handoff-example:start -->.*?<!-- handoff-example:end -->','',text,flags=re.S)
        return re.findall(r'```python\n(.*?)```',text,re.S)
    for code in reference_code('coherent_multipath_directions.md'):exec(code,env)
    assert env['hypotheses'][0]['residual_fraction']<1e-8
    rx=np.array([2.,3.,1.]);src=np.array([5.,6.,2.]);observations=[]
    for axis in [0,1]:
        image=src.copy();image[axis]*=-1
        observations.append(dict(receiver_id='same',position_m=rx.tolist(),directions=[(image-rx).tolist()]))
    env=dict(geometry=setup_walls(),observations=observations,path_differences=[])
    for code in reference_code('multipath_geometry_localization.md'):exec(code,env)
    assert np.allclose(env['result']['hypotheses'][0]['position_m'],src)
