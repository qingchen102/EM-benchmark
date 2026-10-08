import numpy as np
import pytest


def test_signal_ranges_and_provenance():
    from cityrf.baselines.analysis_api import signal_block, select_band, crop, assert_comparable
    x=np.exp(2j*np.pi*.1*np.arange(4096))[None,:]
    raw=signal_block(x,1000,2.4e9,receiver_id='r')
    b=select_band(raw,80,120)
    assert b.band_hz==(80.,120.) and b.selected
    assert np.mean(abs(b.iq)**2)>.9
    with pytest.raises(ValueError):select_band(raw,200,-200)
    with pytest.raises(ValueError):select_band(b,50,150)
    with pytest.raises(ValueError):crop(raw,-.1,.2)
    with pytest.raises(ValueError):assert_comparable(raw,b)
    assert_comparable(b,b)


def test_cp_derives_lags_and_sees_positive_control():
    from cityrf.baselines.analysis_api import signal_block,select_band,cp_structure
    rng=np.random.default_rng(123)
    s=np.fft.ifft(rng.choice([-1,1],(500,64))+1j*rng.choice([-1,1],(500,64)),axis=1)
    x=np.concatenate([s[:,-16:],s],axis=1).reshape(1,-1)
    sig=select_band(signal_block(x,360000,2.4e9),-170000,170000)
    result=cp_structure(sig,nfft=64,ncp=16,internal_fs_hz=360000,center_hz=0)
    assert result['lag_samples']==64 and result['symbol_samples']==80
    assert result['lag_coherence'][0]>.12 and result['cyclic_coherence'][0]>.1
    with pytest.raises(ValueError):cp_structure(sig,nfft=64,ncp=80,internal_fs_hz=360000,center_hz=0)
    with pytest.raises(ValueError):cp_structure(sig,nfft=64,ncp=16,internal_fs_hz=360000,center_hz=300000)


def test_paths_reject_full_band_and_mirror_uses_pose():
    from cityrf.baselines.analysis_api import signal_block,select_band,path_hypotheses,mirror_direction
    n=np.arange(8)-3.5; spacing=299792458/2.4e9/2
    off=np.vstack([np.c_[n*0,n*spacing,n*0],np.c_[n*0,n*0,-n*spacing]])
    yaw=.7; rot=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    world=off@rot.T; u=rot@np.array([.8,.6,0.])
    x=np.exp(2j*np.pi*2.4e9/299792458*world@u)[:,None]*np.exp(2j*np.pi*.1*np.arange(2048))
    raw=signal_block(x,1e6,2.4e9,offsets_world_m=world,rotation_local_to_world=rot,position_m=[0,0,5])
    with pytest.raises(ValueError):path_hypotheses(raw,path_counts=(1,))
    b=select_band(raw,90000,110000)
    out=path_hypotheses(b,path_counts=(1,),subarray_sizes=(5,))
    assert out['models'] and out['models'][0]['validation_residual']<.02
    m=mirror_direction(u,rot[:,0]);assert np.allclose(world@u,world@m)
    assert not np.allclose(m,[-u[0],u[1],u[2]])


def test_geometry_observations_preserve_distinct_paths():
    from cityrf.baselines.analysis_api import signal_block,select_band,path_observations
    s=select_band(signal_block(np.ones((2,128),complex),1000,2.4e9,receiver_id='r',position_m=[1,2,3]),-100,100)
    model={'directions':[[1,0,0],[0,1,0]],'normal_world':[1,0,0], 'signal_signature':s.signature,
           'training_residual':.1,'validation_residual':.1}
    observations=path_observations(s,model,angle_tolerance_deg=3)
    assert len(observations)==2
    assert observations[0]['receiver_id']==observations[1]['receiver_id']=='r'
    assert len(observations[0]['directions'])==2
    assert len(observations[1]['directions'])==1
    with pytest.raises(ValueError):path_observations(s,model,angle_tolerance_deg=0)


def test_features_and_continuous_group_contrast():
    from cityrf.baselines.analysis_api import signal_block,select_band,spectrum,time_frequency,activity,activity_contrast,periodic_decomposition
    fs=100000.;t=np.arange(20000)/fs
    x=((t%.002)<.0006)*np.exp(2j*np.pi*10000*t)
    rng=np.random.default_rng(7);x+=.01*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
    s=select_band(signal_block(x[None,:],fs,2.4e9),5000,15000)
    p=spectrum(s);assert abs(p['peak_hz']-10000)<100
    m=time_frequency(s,(128,512));assert len(m['maps'])==2
    a=activity(s);assert any(abs(v-.002)<30e-6 for v in a['period_candidates_s'])
    c=activity_contrast(s,[[.0021,.0025]],[[.003,.0034]],nperseg=32)
    assert c['difference_psd'].sum()>0
    with pytest.raises(ValueError):activity_contrast(s,[[0,.01]],[[.005,.02]])
    with pytest.raises(ValueError):time_frequency(s,(40000,))
    parts=periodic_decomposition(s,.002)
    assert np.allclose(parts['periodic'].iq+parts['residual'].iq,s.iq)


def test_structure_table_rejects_incompatible_tests():
    from cityrf.baselines.analysis_api import structure_table
    row={'input':{'receiver_id':'a','rf_center_hz':2.4e9},'effective_symbol_s':1e-4,
         'total_symbol_s':1.25e-4,'center_hz':0.,'lag_coherence':np.array([.2]),'cyclic_coherence':np.array([.18])}
    import copy
    other=copy.deepcopy(row);other['input']['receiver_id']='b';other['lag_coherence']=np.array([.01])
    r=structure_table([row,other]);assert len(r['stations'])==2
    other['center_hz']=10000
    with pytest.raises(ValueError):structure_table([row,other])


def test_cp_resampling_and_noise_control():
    from scipy.signal import resample_poly
    from cityrf.baselines.analysis_api import signal_block,select_band,cp_structure
    rng=np.random.default_rng(11);carriers=np.r_[np.arange(1,17),np.arange(48,64)]
    spec=np.zeros((300,64),complex);spec[:,carriers]=rng.choice([-1,1],(300,32))+1j*rng.choice([-1,1],(300,32))
    x=np.fft.ifft(spec,axis=1);x=np.c_[x[:,-16:],x].ravel()
    x=resample_poly(x,50,9);x*=np.exp(-2j*np.pi*250000*np.arange(len(x))/2e6)
    s=select_band(signal_block(x[None,:],2e6,2.4e9),-345000,-155000)
    r=cp_structure(s,nfft=64,ncp=16,internal_fs_hz=360000,center_hz=-250000)
    assert np.isclose(r['original_lag_samples'],355.5555555556)
    assert r['cyclic_coherence'][0]>.12
    noise=rng.normal(size=(1,len(x)))+1j*rng.normal(size=(1,len(x)))
    ns=select_band(signal_block(noise,2e6,2.4e9),-345000,-155000)
    nr=cp_structure(ns,nfft=64,ncp=16,internal_fs_hz=360000,center_hz=-250000)
    assert nr['cyclic_coherence'][0]<.04


def test_signal_roundtrip_and_template_holdout(tmp_path):
    from cityrf.baselines.analysis_api import signal_block,select_band,save_signal,load_signal,template_residual,save_result
    rng=np.random.default_rng(42);t=rng.normal(size=4096)+1j*rng.normal(size=4096)
    s=select_band(signal_block(np.stack([t,2j*t]),10000,2.4e9),-4990,4990)
    save_signal(tmp_path/'s.npz',s);back=load_signal(tmp_path/'s.npz')
    assert back.signature==s.signature and np.array_equal(back.iq,s.iq)
    r=template_residual(s,s.iq[0],max_delay_samples=0)
    assert r['validation_residual']<1e-20
    save_result(tmp_path/'r.json',r)


def test_locate_paths_uses_same_station_two_reflections():
    from test_multipath import setup_walls
    from cityrf.baselines.analysis_api import signal_block,select_band,locate_paths
    rx=np.array([2.,3.,1.]);src=np.array([5.,6.,2.]);dirs=[]
    for axis in [0,1]:
        im=src.copy();im[axis]*=-1;u=im-rx;dirs.append((u/np.linalg.norm(u)).tolist())
    s=select_band(signal_block(np.ones((2,128),complex),1000,2.4e9,receiver_id='r',position_m=rx),-100,100)
    model=dict(directions=dirs,normal_world=[0,0,1],signal_signature=s.signature,training_residual=0.,validation_residual=0.)
    r=locate_paths(setup_walls(),[(s,model)],angle_tolerance_deg=.01,height_range=(0.,10.))
    assert any(np.linalg.norm(np.array(h['position_m'])-src)<1e-6 for h in r['hypotheses'])
    assert len(r['observations'])==2


def test_ridge_frequency_quality_and_joint_evidence():
    from cityrf.baselines.analysis_api import signal_block,select_band,ridge_fit,waveform_features,quality,joint_features
    fs=100000.;t=np.arange(20000)/fs;slope=50000.
    x=np.exp(2j*np.pi*(10000*t+.5*slope*t*t))[None,:]
    b=select_band(signal_block(x,fs,2.4e9),5000,25000)
    r=ridge_fit(b,window_samples=256)
    assert abs(r['slope_hz_per_s']/slope-1)<.03
    assert r['rms_hz']<300
    f=waveform_features(b);assert abs(f['frequency_quantiles_hz'][0][1]-15000)<300
    q=quality(b);assert q['papr'][0]<2
    j=joint_features(b);assert j['parts'][0]['spectrum']['centroid_hz']<j['parts'][1]['spectrum']['centroid_hz']


def test_checked_api_copied_into_clean_workspace(tmp_path):
    pytest.importorskip("cityrf.light_agent", reason="Agent workspace is outside standalone simulator scope")
    from cityrf.light_agent.workspace import assemble_workspace
    public=tmp_path/'public';public.mkdir()
    (public/'observations.json').write_text('{"receivers":[]}',encoding='utf-8')
    ws=assemble_workspace(public,tmp_path/'sandbox')
    for name in ['analysis_api.py','checked_signal.py','checked_features.py','checked_spatial.py']:
        assert (ws.root/'lib/rfkit'/name).is_file()
    # Import through the same rfkit package name used by the sandbox process.
    import subprocess,sys
    proc=subprocess.run([sys.executable,'-c',
        'from rfkit.analysis_api import signal_block,select_band,quality; import numpy as np; '
        's=select_band(signal_block(np.ones((1,128),complex),1000,2.4e9),-100,100); '
        'assert quality(s)["power"][0]>0'],
        cwd=ws.root,env={**__import__('os').environ,'PYTHONPATH':str(ws.root/'lib')},capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr


def test_brief_cannot_masquerade_as_array_and_save_numeric_types(tmp_path):
    import json
    from cityrf.baselines.analysis_api import brief, save_result
    value={'samples':np.arange(40),'count':np.int64(4),'complex':np.complex128(1+2j)}
    summary=brief(value)
    assert isinstance(summary['samples'],str) and 'summary only' in summary['samples']
    assert value['samples'].shape==(40,)
    save_result(tmp_path/'numeric.json',value)
    saved=json.loads((tmp_path/'numeric.json').read_text())
    assert saved['count']==4 and saved['complex']=={'real':1.,'imag':2.}
    assert len(saved['samples'])==40
