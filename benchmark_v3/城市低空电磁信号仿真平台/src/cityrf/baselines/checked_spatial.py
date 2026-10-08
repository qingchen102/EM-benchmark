"""Path hypotheses retain data scope, joint directions and geometric ambiguity."""
import numpy as np
from .checked_signal import require_record, integer, positive
from .multipath import smooth_ula_covariance, ula_music, cross_direction_candidates, fit_path_candidates, localize_multipath
from .spatial import band_covariance, scan_directions

C0=299792458.


def mirror_direction(direction,normal_world):
    u=np.asarray(direction,float);n=np.asarray(normal_world,float)
    if u.shape!=(3,) or n.shape!=(3,) or not np.isfinite([u,n]).all() or np.linalg.norm(u)==0 or np.linalg.norm(n)==0:
        raise ValueError('nonzero finite 3D direction and actual world array normal required')
    u=u/np.linalg.norm(u);n=n/np.linalg.norm(n)
    return u-2*np.dot(u,n)*n


def _array(s):
    require_record(s,True)
    if s.offsets_world_m is None or s.rotation_local_to_world is None:raise ValueError('record needs actual array offsets and rotation')
    return s.offsets_world_m,s.rotation_local_to_world


def _samples(s,mask,max_samples=8192):
    x=s.iq
    if mask is not None:
        mask=np.asarray(mask)
        if mask.dtype!=bool or mask.shape!=(x.shape[1],):raise ValueError('mask must be boolean on original selected continuous samples')
        x=x[:,mask]
    if x.shape[1]<32:raise ValueError('too few samples for path validation')
    if x.shape[1]>max_samples:x=x[:,np.linspace(0,x.shape[1]-1,max_samples,dtype=int)]
    mid=x.shape[1]//2
    return x[:,:mid],x[:,mid:]


def single_path_fit(s,grid_step_deg=2.,mask=None):
    off,rot=_array(s);step=positive(grid_step_deg,'grid_step_deg')
    if step<.5 or step>10:raise ValueError('grid_step_deg must be between .5 and 10; refine supported candidates separately')
    train,test=_samples(s,mask)
    cov=train@train.conj().T/train.shape[1]
    sc=scan_directions(cov,off,s.rf_center_hz,train.shape[1],grid_step_deg=step)
    dirs=[p['unit_direction_world'] for p in sc['bartlett']['peaks']]
    models=_fits(s,train,test,dirs,1,rot[:,0],dict(method='unsmoothed diagnostic grid',grid_step_deg=step))
    return dict(input=s.describe(),models=models,scan=sc,
                meaning='Single physical steering-direction fit on the selected IQ; a high eigenvalue fraction is not this residual.')


def _fits(s,train,test,dirs,k,normal,extra):
    out=[]
    for h in fit_path_candidates(train,s.offsets_world_m,s.rf_center_hz,dirs,k,max_results=8):
        validation=fit_path_candidates(test,s.offsets_world_m,s.rf_center_hz,h['directions'],k,max_results=1)
        if not validation:continue
        out.append(dict(directions=h['directions'],normal_world=np.asarray(normal).tolist(),path_count=k,
                        training_residual=h['residual_fraction'],validation_residual=validation[0]['residual_fraction'],
                        condition_number=h['condition_number'],signal_signature=s.signature,**extra))
    return out


def path_hypotheses(s,path_counts=(1,2,3),subarray_sizes=(4,5,6),mask=None):
    """Equal-arm yz cross: sort each ULA along its actual positive local axis."""
    off,rot=_array(s);m=len(off)
    if m%2 or m<8:raise ValueError('this wrapper requires two equal ULA arms with >=4 elements each')
    arm=m//2;local=off@rot
    y=local[:arm];z=local[arm:]
    if np.max(np.ptp(y[:,[0,2]],axis=0))>1e-6 or np.max(np.ptp(z[:,[0,1]],axis=0))>1e-6:
        raise ValueError('first arm must lie along local y and second along local z; check offsets/order/pose')
    iy=np.argsort(y[:,1]);iz=np.argsort(z[:,2])+arm
    dy=np.diff(local[iy,1]);dz=np.diff(local[iz,2])
    for spacing in (dy,dz):
        if np.min(spacing)<=0 or not np.allclose(spacing,spacing.mean(),rtol=1e-5,atol=1e-8):raise ValueError('each arm must be uniformly spaced with distinct elements')
    electrical=[float(d.mean()*s.rf_center_hz/C0) for d in (dy,dz)]
    if max(electrical)>.500001:raise ValueError('spacing exceeds half wavelength; wrapper does not resolve grating lobes')
    train,test=_samples(s,mask);models=[];cases=[]
    for P in subarray_sizes:
        p=integer(P,'subarray_size',2)
        if p>arm:raise ValueError('subarray exceeds arm length')
        ry=smooth_ula_covariance(train[iy],p);rz=smooth_ula_covariance(train[iz],p)
        for K in path_counts:
            k=integer(K,'path_count')
            if not k<p or arm-p+1<k:raise ValueError('require P>K and M-P+1>=K; choose supported (P,K) hypotheses')
            sy=ula_music(ry,min(electrical[0],.5),k);sz=ula_music(rz,min(electrical[1],.5),k)
            dirs=cross_direction_candidates(sy['direction_cosines'],sz['direction_cosines'],rot)
            cases.append(dict(subarray_size=p,path_count=k,mu_y=sy['direction_cosines'],mu_z=sz['direction_cosines']))
            models.extend(_fits(s,train,test,dirs,k,rot[:,0],dict(method='arm spatial smoothing',subarray_size=p)))
    models.sort(key=lambda h:(h['path_count'],h['validation_residual']))
    return dict(input=s.describe(),models=models,cases=cases,
                meaning='Each model is one joint K-path hypothesis. Mirror alternatives are not additional paths.',
                warning='K ordering is not source counting. Compare complexity, P stability and held-out errors; temporal holdout is not independent propagation.')


def path_observations(s,model,*,angle_tolerance_deg):
    require_record(s,True);tol=positive(angle_tolerance_deg,'angle_tolerance_deg')
    if tol>30:raise ValueError('angle tolerance >30 degrees does not support this ray model')
    if s.position_m is None:raise ValueError('receiver position required')
    if model.get('signal_signature')!=s.signature:raise ValueError('model belongs to a different data scope; do not transplant directions')
    dirs=np.asarray(model.get('directions'),float);normal=model.get('normal_world')
    if dirs.ndim!=2 or dirs.shape[1]!=3 or len(dirs)==0:raise ValueError('one nonempty joint path model required')
    result=[]
    for i,u in enumerate(dirs):
        m=mirror_direction(u,normal);u=u/np.linalg.norm(u)
        # Two members of the same mirror class cannot masquerade as independent paths.
        if any(min(np.linalg.norm(u-np.array(o['directions'][0])),np.linalg.norm(m-np.array(o['directions'][0])))<1e-7 for o in result):
            raise ValueError('joint model repeats a path/mirror equivalence class')
        choices=[u.tolist()]
        if np.linalg.norm(m-u)>1e-8:choices.append(m.tolist())
        result.append(dict(receiver_id=s.receiver_id,position_m=s.position_m.tolist(),directions=choices,
                           path_id=f'{s.receiver_id}:path{i+1}',angle_tolerance_deg=tol,
                           signal_signature=s.signature))
    return result


def locate_paths(geometry,selections,*,angle_tolerance_deg,path_differences=(),height_range=(0.,200.)):
    """Selections are (SignalBlock, one joint model) per station, not hand-written angles."""
    if not selections:raise ValueError('need selected station path models')
    observations=[];seen=set()
    for s,model in selections:
        if s.receiver_id in seen:raise ValueError('choose ONE joint hypothesis per station, not multiple incompatible models')
        seen.add(s.receiver_id)
        observations.extend(path_observations(s,model,angle_tolerance_deg=angle_tolerance_deg))
    if len(observations)<2:raise ValueError('one ray cannot determine a position; need an independent path')
    if not np.isfinite(height_range).all() or len(height_range)!=2 or height_range[0]>=height_range[1]:raise ValueError('ordered finite height range required')
    for row in path_differences:
        i=integer(row['i'],'path difference i',0);j=integer(row['j'],'path difference j',0)
        if i==j or max(i,j)>=len(observations):raise ValueError('invalid path difference indices')
        positive(row['tolerance_m'],'length difference tolerance')
        if not np.isfinite(row['difference_m']):raise ValueError('finite signed length difference required')
    out=localize_multipath(geometry,observations,path_differences=path_differences,
                          max_angle_deg=angle_tolerance_deg,height_range=height_range)
    out['observations']=observations
    out['meaning']='Conditional fixed-angle direct/one-bounce candidates, not automatically accepted transmitter positions.'
    out['warning']='Angle tolerance must come from measurement uncertainty; no automatic relaxation. Same-source association remains an explicit hypothesis.'
    return out
