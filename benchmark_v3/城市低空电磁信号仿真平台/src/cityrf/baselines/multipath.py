"""Explicit coherent-path hypotheses and geometry checks, not an automatic locator."""
from itertools import combinations, product
from math import comb, prod
import numpy as np
from scipy.signal import find_peaks
from .localization import intersect_rays

C0=299792458.


def _iq(iq):
    x=np.asarray(iq,complex)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('finite IQ [channels,samples] required')
    return x


def smooth_ula_covariance(iq, subarray_size, forward_backward=True):
    """IQ must be ordered along ONE uniformly spaced, calibrated array arm."""
    x=_iq(iq);m,n=x.shape;p=subarray_size
    if type(p) is not int or not 2<=p<=m:
        raise ValueError('subarray_size must be an integer in [2,channels]')
    r=sum(x[j:j+p]@x[j:j+p].conj().T/n for j in range(m-p+1))/(m-p+1)
    if forward_backward:
        r=(r+r[::-1,::-1].conj())/2
    return (r+r.conj().T)/2


def ula_music(covariance, spacing_wavelengths, path_count, grid_size=2001):
    """Returns projection on ordered ULA axis, NOT azimuth/elevation or source count."""
    r=np.asarray(covariance,complex);m=len(r);k=path_count
    if r.shape!=(m,m) or not np.isfinite(r).all() or not np.allclose(r,r.conj().T):
        raise ValueError('finite Hermitian covariance required')
    if type(k) is not int or not 1<=k<m or not 0<spacing_wavelengths<=.5:
        raise ValueError('require 1<=path_count<M and spacing in (0,0.5] wavelengths')
    if type(grid_size) is not int or grid_size<3:
        raise ValueError('grid_size must be >=3')
    values,v=np.linalg.eigh(r);mu=np.linspace(-1.,1.,grid_size)
    a=np.exp(2j*np.pi*spacing_wavelengths*np.arange(m)[:,None]*mu)
    power=1/np.maximum(np.sum(abs(v[:,:m-k].conj().T@a)**2,axis=0),1e-20)
    peaks=list(find_peaks(power)[0])
    if power[0]>power[1]:peaks.append(0)
    if power[-1]>power[-2]:peaks.append(grid_size-1)
    chosen=sorted(peaks,key=lambda i:power[i],reverse=True)[:k]
    return dict(direction_cosines=mu[chosen].tolist(),grid=mu,spectrum=power,
                eigenvalues=values[::-1].tolist(),path_count_hypothesis=k)


def cross_direction_candidates(mu_y, mu_z, rotation_local_to_world):
    """Cross array in local yz plane; mu_z is projection on +z, not -z arm order."""
    rot=np.asarray(rotation_local_to_world,float)
    if rot.shape!=(3,3) or not np.isfinite(rot).all() or not np.allclose(rot.T@rot,np.eye(3),atol=1e-6) or np.linalg.det(rot)<0:
        raise ValueError('proper local-to-world rotation required')
    result=[]
    for y,z in product(mu_y,mu_z):
        if not np.isfinite([y,z]).all() or abs(y)>1 or abs(z)>1:
            raise ValueError('direction cosines must be finite in [-1,1]')
        rest=1-y*y-z*z
        if rest < -1e-12:continue
        for sign in [-1.,1.]:
            d=rot@np.array([sign*np.sqrt(max(0.,rest)),y,z])
            if not any(np.linalg.norm(d-v)<1e-8 for v in result):result.append(d)
    return np.asarray(result).reshape(-1,3)


def fit_path_candidates(iq, offsets_world_m, carrier_frequency_hz, directions,
                        path_count, max_combinations=10000, max_results=20):
    """Compare finite direction sets by full-array projection residual; mirrors retained.

    K is a path hypothesis, not an estimated physical source count. Scores are
    in-sample residuals; repeat on held-out windows and vary K/subarray size.
    """
    x=_iq(iq);off=np.asarray(offsets_world_m,float)
    try:
        d=np.asarray(directions,float)
    except (ValueError, TypeError) as e:
        raise ValueError('directions must be a rectangular numeric [D,3] array; do not mix full models and vectors') from e
    if off.shape!=(x.shape[0],3) or d.ndim!=2 or d.shape[1]!=3 or not np.isfinite(off).all() or not np.isfinite(d).all():
        raise ValueError('offsets [M,3] and finite directions [D,3] required')
    k=path_count
    if not np.isfinite(carrier_frequency_hz) or carrier_frequency_hz<=0 or type(k) is not int or not 1<=k<x.shape[0]:
        raise ValueError('positive frequency and 1<=K<M required')
    if len(d)<k:return []
    if comb(len(d),k)>max_combinations:raise ValueError('too many candidate combinations')
    norms=np.linalg.norm(d,axis=1)
    if np.any(norms<1e-12):raise ValueError('zero direction')
    d=d/norms[:,None]
    r=x@x.conj().T/x.shape[1];total=float(np.trace(r).real)
    if total<=0:raise ValueError('zero signal energy')
    steering=np.exp(2j*np.pi*carrier_frequency_hz/C0*off@d.T)
    out=[]
    for ids in combinations(range(len(d)),k):
        a=steering[:,ids]
        condition=float(np.linalg.cond(a))
        if condition>1e6:continue
        projector=a@np.linalg.pinv(a)
        residual=max(0.,float(np.trace(r-projector@r).real)/total)
        out.append(dict(directions=d[list(ids)].tolist(),candidate_indices=list(ids),
                        residual_fraction=residual,condition_number=condition))
    return sorted(out,key=lambda h:h['residual_fraction'])[:max_results]


def localize_multipath(geometry, observations, path_differences=(), max_angle_deg=5.,
                      height_range=(0.,200.), max_distance=2000., max_combinations=10000,
                      max_results=20):
    """One row per distinct observed path. Rows may share a receiver.

    Each row has receiver_id, position_m, directions (world unit candidates).
    Optional total_length_m + length_tolerance_m require calibrated absolute TOA.
    path_differences rows: i,j,difference_m,tolerance_m for L_i-L_j.
    Distances screen/rank fixed-angle intersections; no nonlinear range refinement.
    """
    if len(observations)<2:return dict(status='insufficient_paths',hypotheses=[])
    if max_angle_deg<=0:raise ValueError('positive angular tolerance required')
    for o in observations:
        if 'total_length_m' in o and (not np.isfinite(o['total_length_m']) or o['total_length_m']<0 or
                not np.isfinite(o.get('length_tolerance_m',0)) or o.get('length_tolerance_m',0)<=0):
            raise ValueError('absolute path length requires finite nonnegative value and positive tolerance')
    for c in path_differences:
        if any(type(c[q]) is not int or not 0<=c[q]<len(observations) for q in ['i','j']) or c['i']==c['j'] or not np.isfinite([c['difference_m'],c['tolerance_m']]).all() or c['tolerance_m']<=0:
            raise ValueError('invalid path difference constraint')
    choices=[]
    for index,o in enumerate(observations):
        pos=np.asarray(o['position_m'],float);options=[]
        for direction in o['directions']:
            for branch in geometry.backtrace(pos,direction,max_distance):
                options.append(dict(branch,observation_index=index,receiver_id=o['receiver_id'],
                                    prefix_length_m=float(np.linalg.norm(np.array(branch['origin'])-pos))))
        choices.append(options)
    count=prod(map(len,choices))
    if count>max_combinations:raise ValueError('too many path combinations; reduce hypotheses')
    results=[]
    for paths in product(*choices):
        # Do not count duplicated same-station geometric rays as distinct evidence.
        if any(a['receiver_id']==b['receiver_id'] and
               np.linalg.norm(np.array(a['origin'])-b['origin'])<1e-5 and
               np.dot(a['direction'],b['direction'])>1-1e-8 for a,b in combinations(paths,2)):continue
        fit=intersect_rays([p['origin'] for p in paths],[p['direction'] for p in paths])
        if fit is None:continue
        pos=np.array(fit['position_m'])
        if not height_range[0]<=pos[2]<=height_range[1]:continue
        angles=[];lengths=[];valid=True
        for p in paths:
            delta=pos-np.array(p['origin']);distance=np.linalg.norm(delta)
            angle=float(np.degrees(np.arccos(np.clip(np.dot(delta,p['direction'])/distance,-1,1))))
            if angle>max_angle_deg or distance>p['max_distance_m']+.01 or not geometry.visible(np.array(p['origin'])+np.array(p['direction'])*1e-4,pos):
                valid=False;break
            angles.append(angle);lengths.append(p['prefix_length_m']+distance)
        if not valid:continue
        errors=[]
        for i,o in enumerate(observations):
            if 'total_length_m' in o:errors.append((lengths[i]-o['total_length_m'])/o['length_tolerance_m'])
        for c in path_differences:
            errors.append((lengths[c['i']]-lengths[c['j']]-c['difference_m'])/c['tolerance_m'])
        if any(abs(e)>1 for e in errors):continue
        fit.update(paths=list(paths),total_path_lengths_m=lengths,per_path_angle_deg=angles,
                   normalized_range_errors=errors,
                   score=float(np.mean((np.array(angles)/max_angle_deg)**2)+(np.mean(np.square(errors)) if errors else 0.)))
        results.append(fit)
    results.sort(key=lambda h:h['score'])
    return dict(status='hypotheses' if results else 'no_consistent_candidate',
                hypotheses=results[:max_results],combinations=count,
                note='Fixed-angle direct/one-bounce candidates; scores are not probabilities. No automatic path/source association.')
