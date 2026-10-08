"""Checked numerical evidence, never automatic modulation/source-count verdicts."""
from fractions import Fraction
import numpy as np
from scipy.signal import welch, spectrogram, correlate, resample_poly, find_peaks
from .checked_signal import require_record, integer, positive, crop, _make
from .spatial import mdl_rank


def _result(s,meaning,**values):
    return dict(input=s.describe(),meaning=meaning,**values)


def _window(s,n):
    w=min(2048,s.iq.shape[1]) if n is None else integer(n,'nperseg',4)
    if w>s.iq.shape[1]:raise ValueError('window exceeds continuous record length')
    return w


def quality(s,adc_full_scale=None):
    require_record(s);x=s.iq;p=np.mean(abs(x)**2,axis=1);tiny=np.finfo(float).tiny
    out=_result(s,'Channel quality evidence, not a hardware diagnosis.',power=np.asarray(p),
        dc_fraction=abs(x.mean(axis=1))**2/np.maximum(p,tiny),
        papr=np.max(abs(x)**2,axis=1)/np.maximum(p,tiny),
        amplitude_quantiles=np.quantile(abs(x),[.5,.99,1.],axis=1),zero_fraction=np.mean(x==0,axis=1))
    if adc_full_scale is not None:
        rail=positive(adc_full_scale,'adc_full_scale')
        out['rail_fraction']=np.mean((abs(x.real)>=rail)|(abs(x.imag)>=rail),axis=1)
    return out


def spectrum(s,nperseg=None,power_fraction=.99,threshold_db=8.):
    require_record(s);w=_window(s,nperseg)
    if not 0<power_fraction<1 or not np.isfinite(threshold_db) or threshold_db<=0:raise ValueError('invalid power fraction/threshold')
    f,p=welch(s.iq,fs=s.fs_hz,window='hann',nperseg=w,noverlap=w//2,detrend=False,
              return_onesided=False,scaling='density',axis=-1)
    order=np.argsort(f);f=f[order];p=p[:,order].mean(axis=0)
    m=(f>=s.band_hz[0])&(f<=s.band_hz[1]);f=f[m];p=p[m]
    if len(f)<2 or p.sum()<=0:raise ValueError('no resolvable nonzero spectral energy')
    df=s.fs_hz/w;cs=np.cumsum(p)/p.sum();tail=(1-power_fraction)/2
    lo,hi=np.searchsorted(cs,[tail,1-tail]);lo=min(lo,len(f)-1);hi=min(hi,len(f)-1)
    floor=float(np.quantile(p,.3));active=p>floor*10**(threshold_db/10)
    edges=np.diff(np.r_[False,active,False].astype(int));starts=np.where(edges==1)[0];stops=np.where(edges==-1)[0]
    return _result(s,'PSD and in-scope energy bands; occupied width/centroid are not CFO or device count.',
        frequency_hz=f,psd=p,psd_unit='sample_power/Hz',bin_spacing_hz=df,window_samples=w,
        integrated_power=float(p.sum()*df),peak_hz=float(f[np.argmax(p)]),
        centroid_hz=float(np.sum(f*p)/p.sum()),occupied_band_hz=[float(f[lo]),float(f[hi])],
        occupied_width_hz=float(f[hi]-f[lo]),power_fraction=power_fraction,
        boundary_touched=bool(lo==0 or hi==len(f)-1),floor_estimate=floor,
        candidate_bands_hz=[[float(f[a]),float(f[b-1])] for a,b in zip(starts,stops)],
        warning='Quantile noise floor and threshold are heuristic; mixtures/truncation affect width.')


def time_frequency(s,windows=(256,1024,4096),overlap_fraction=.75):
    require_record(s)
    if not 0<=overlap_fraction<1:raise ValueError('overlap_fraction must be in [0,1)')
    maps=[]
    for n in windows:
        w=_window(s,n)
        f,t,z=spectrogram(s.iq,fs=s.fs_hz,window='hann',nperseg=w,noverlap=int(w*overlap_fraction),
            detrend=False,return_onesided=False,scaling='density',mode='psd',axis=-1)
        idx=np.argsort(f);f=f[idx];z=z[:,idx,:].mean(axis=0);m=(f>=s.band_hz[0])&(f<=s.band_hz[1])
        maps.append(dict(window_samples=w,window_s=w/s.fs_hz,bin_spacing_hz=s.fs_hz/w,
                         frequency_hz=f[m],time_s=t+s.start_s,psd=z[m]))
    return _result(s,'Window-dependent time/frequency evidence; zero padding does not increase resolution.',maps=maps)


def ridge_fit(s,window_samples=256,threshold_db=8.):
    require_record(s,True)
    if not np.isfinite(threshold_db) or threshold_db<=0:raise ValueError('positive ridge contrast threshold required')
    m=time_frequency(s,(window_samples,))['maps'][0];z=m['psd'];f=m['frequency_hz'];t=m['time_s']
    idx=np.argmax(z,axis=0);peak=z[idx,np.arange(z.shape[1])];floor=np.median(z,axis=0)
    keep=peak>np.maximum(floor,np.finfo(float).tiny)*10**(threshold_db/10)
    if keep.sum()<3:raise ValueError('fewer than three ridge frames above contrast; inspect map/change window')
    tt=t[keep];ff=f[idx[keep]];a=np.c_[tt-tt.mean(),np.ones(len(tt))]
    slope,intercept=np.linalg.lstsq(a,ff,rcond=None)[0];pred=a@np.array([slope,intercept])
    return _result(s,'Single selected time-interval ridge hypothesis; repetition/mixing can invalidate a linear fit.',
        time_s=tt,frequency_hz=ff,slope_hz_per_s=float(slope),frequency_at_mean_time_hz=float(intercept),
        mean_time_s=float(tt.mean()),rms_hz=float(np.sqrt(np.mean((ff-pred)**2))),
        boundary_fraction=float(np.mean((idx[keep]==0)|(idx[keep]==len(f)-1))),window_s=m['window_s'])


def waveform_features(s,amplitude_quantile=.1,reference_hz=None):
    require_record(s,True)
    if not 0<=amplitude_quantile<1:raise ValueError('amplitude_quantile must be in [0,1)')
    x=s.iq;amp=abs(x);cut=np.quantile(amp,amplitude_quantile,axis=1)[:,None]
    keep=np.minimum(amp[:,1:],amp[:,:-1])>cut
    inst=np.angle(x[:,1:]*x[:,:-1].conj())*s.fs_hz/(2*np.pi)
    stats=[]
    for i in range(len(x)):
        v=inst[i,keep[i]];stats.append(np.quantile(v,[.1,.5,.9]).tolist() if len(v) else [None]*3)
    out=_result(s,'Phase-increment frequency and envelope clues, not automatic CFO/modulation classification.',
        frequency_quantiles_hz=stats,instantaneous_hz=inst,valid_mask=keep,
        envelope_papr=quality(s)['papr'],amplitude_quantile=amplitude_quantile)
    if reference_hz is not None:
        if not np.isfinite(reference_hz):raise ValueError('reference_hz must be finite')
        out['reference_hz']=float(reference_hz)
        out['median_offset_from_reference_hz']=[None if v[1] is None else v[1]-reference_hz for v in stats]
        out['offset_warning']='Only a dominant unmodulated tone supports interpreting this as a carrier offset.'
    return out


def activity(s,block_s=20e-6,on_db=6.,off_db=3.,baseline_power=None):
    require_record(s,True);positive(block_s,'block_s')
    if not np.isfinite([on_db,off_db]).all() or not on_db>off_db>=0:raise ValueError('require on_db > off_db >= 0')
    n=max(1,round(block_s*s.fs_hz));count=s.iq.shape[1]//n
    if count<4:raise ValueError('need at least four envelope blocks')
    env=(abs(s.iq[:,:count*n])**2).reshape(s.iq.shape[0],count,n).mean(axis=(0,2))
    floor=float(np.quantile(env,.2)) if baseline_power is None else positive(baseline_power,'baseline_power')
    if floor<=0:raise ValueError('zero baseline; supply a positive measured noise baseline')
    flags=np.zeros(count,bool);on=False
    for i,v in enumerate(env):
        on=bool(v>floor*10**((off_db if on else on_db)/10));flags[i]=on
    edges=np.diff(np.r_[False,flags,False].astype(int));st=np.where(edges==1)[0];en=np.where(edges==-1)[0]
    dt=n/s.fs_hz;v=env-env.mean();ac=correlate(v,v,mode='full',method='fft')[len(v)-1:]
    ac=ac/ac[0] if ac[0]>0 else np.zeros_like(ac);pk,_=find_peaks(ac)
    pk=sorted(pk,key=lambda k:ac[k],reverse=True)[:8]
    return _result(s,'Threshold activity candidates inside the selected band; dwell time is not necessarily full pulse width.',
        envelope=env,time_s=s.start_s+np.arange(count)*dt,baseline_power=floor,
        on_db=on_db,off_db=off_db,intervals_s=[[float(a*dt),float(b*dt)] for a,b in zip(st,en)],
        durations_s=(en-st)*dt,start_intervals_s=np.diff(st)*dt,duty_fraction=float(flags.mean()),
        boundary_censored=bool(flags[0] or flags[-1]),lags_s=np.arange(len(ac))*dt,autocorrelation=ac,
        period_candidates_s=[float(k*dt) for k in pk],warning='No activity does not exclude continuous interference; thresholds not calibrated.')


def activity_contrast(s,on_intervals_s,off_intervals_s,nperseg=256):
    require_record(s)
    def estimate(intervals):
        if not intervals:raise ValueError('need nonempty continuous intervals')
        ordered=sorted(intervals)
        if any(ordered[i][1]>ordered[i+1][0] for i in range(len(ordered)-1)):raise ValueError('overlapping intervals within group')
        spectra=[];weights=[]
        for a,b in ordered:
            part=crop(s,a,b);res=spectrum(part,nperseg=nperseg)
            spectra.append(res['psd']);weights.append(1+(part.iq.shape[1]-nperseg)//(nperseg//2))
        return res['frequency_hz'],np.average(spectra,axis=0,weights=weights)
    if any(max(a,c)<min(b,d) for a,b in on_intervals_s for c,d in off_intervals_s):raise ValueError('on and off groups overlap')
    f,on=estimate(on_intervals_s);_,off=estimate(off_intervals_s)
    return _result(s,'Linear PSD state difference, not automatically separated interferer power.',frequency_hz=f,
                   on_psd=on,off_psd=off,difference_psd=on-off,window_samples=nperseg)


def periodic_decomposition(s,period_s):
    require_record(s,True);samples=positive(period_s,'period_s')*s.fs_hz;n=round(samples)
    if not np.isclose(samples,n,rtol=0,atol=1e-6):raise ValueError('period not integer samples; align/resample before coherent folding')
    if n<2 or s.iq.shape[1]//n<4:raise ValueError('need >=4 full periods and >=2 samples per period')
    k=s.iq.shape[1]//n;x=s.iq[:,:k*n];cycles=x.reshape(len(x),k,n)
    fold=cycles[:,::2,:].mean(axis=1)
    component=np.tile(fold,(1,k));residual=x-component
    def block(data,label):
        return _make(data,s.fs_hz,s.rf_center_hz,s.receiver_id,s.band_hz,True,s.start_s,s.position_m,
                     s.offsets_world_m,s.rotation_local_to_world,s.history+(label,))
    held=cycles[:,1::2,:];held_residual=held-fold[:,None,:]
    return dict(input=s.describe(),period_samples=n,periods=k,template=fold,
                heldout_residual_fraction=float(np.sum(abs(held_residual)**2)/max(np.sum(abs(held)**2),np.finfo(float).tiny)),
                periodic=block(component,'coherent periodic projection'),residual=block(residual,'nonperiodic residual'),
                meaning='Template from alternating cycles, validated on other cycles; asynchronous/CFO components may not survive. Residual is not automatically a new source.')


def cp_structure(s,*,nfft,ncp,internal_fs_hz,center_hz):
    require_record(s,True);fft=integer(nfft,'nfft',2);cp=integer(ncp,'ncp')
    target=positive(internal_fs_hz,'internal_fs_hz')
    if cp>=fft:raise ValueError('require 0 < ncp < nfft')
    if not np.isfinite(center_hz) or not s.band_hz[0]<center_hz<s.band_hz[1]:raise ValueError('OFDM center must lie inside selected band')
    if max(abs(s.band_hz[0]-center_hz),abs(s.band_hz[1]-center_hz))>=target/2:
        raise ValueError('selected band exceeds internal-rate Nyquist after shift; narrow correctly, do not silently discard target energy')
    ratio=Fraction(target/s.fs_hz).limit_denominator(10000)
    if not np.isclose(float(ratio),target/s.fs_hz,rtol=1e-10):raise ValueError('sampling ratio not supported accurately')
    t=np.arange(s.iq.shape[1])/s.fs_hz
    shifted=s.iq*np.exp(-2j*np.pi*center_hz*t)
    y=resample_poly(shifted,ratio.numerator,ratio.denominator,axis=1)
    trim=2*(fft+cp)
    if y.shape[1]<12*(fft+cp):raise ValueError('need at least twelve OFDM symbols for trimmed check')
    y=y[:,trim:-trim];lag=fft;period=fft+cp
    def metrics(L):
        a,b=y[:,L:],y[:,:-L];q=a*b.conj()
        den=np.sqrt(np.mean(abs(a)**2,axis=1)*np.mean(abs(b)**2,axis=1))
        if np.any(den<=0):raise ValueError('zero-power channel in CP test')
        return abs(q.mean(axis=1))/den,abs((q*np.exp(-2j*np.pi*np.arange(q.shape[1])/period)).mean(axis=1))/den
    normal,cyclic=metrics(lag);lags=np.arange(max(1,fft-4),fft+5)
    curve=np.array([metrics(int(L))[0] for L in lags])
    return _result(s,'Evidence for CP structure; ordinary autocorrelation alone is not identification and absence needs sensitivity controls.',
        effective_symbol_s=fft/target,total_symbol_s=period/target,analysis_fs_hz=target,
        original_lag_samples=fft/target*s.fs_hz,lag_samples=lag,symbol_samples=period,
        center_hz=float(center_hz),lag_coherence=normal,cyclic_coherence=cyclic,scan_lags=lags,scan_lag_coherence=curve,
        nominal_cp_fraction=cp/period,decision='unclassified; compare cyclic evidence, controls and station differences')


def spatial_evidence(s,remove_mean=False,max_snapshots=8192):
    require_record(s,True);integer(max_snapshots,'max_snapshots',2)
    stride=max(1,int(s.fs_hz/(2*(s.band_hz[1]-s.band_hz[0]))));x=s.iq[:,::stride]
    if x.shape[1]>max_snapshots:x=x[:,np.linspace(0,x.shape[1]-1,max_snapshots,dtype=int)]
    if remove_mean:x=x-x.mean(axis=1,keepdims=True)
    r=x@x.conj().T/x.shape[1];r=(r+r.conj().T)/2;p=np.real(np.diag(r))
    if np.any(p<=0):raise ValueError('zero-power channel; select valid channels explicitly')
    corr=abs(r/np.sqrt(p[:,None]*p[None,:]));values=np.maximum(np.linalg.eigvalsh(r)[::-1],0)
    rank,scores=mdl_rank(values,x.shape[1]);mask=~np.eye(len(x),dtype=bool)
    return _result(s,'Shared response and covariance modes, not number of physical devices or paths.',
        correlation_magnitude=corr,offdiag_median=float(np.median(corr[mask])) if mask.any() else None,
        eigenvalues=values,eigen_fractions=values/values.sum(),mdl_rank=rank,mdl_scores=scores,snapshots=x.shape[1],
        warning='MDL assumes suitable noise and effective independent snapshots; coherent paths may remain rank one.')


def joint_features(s,split_s=None):
    require_record(s,True)
    split=s.iq.shape[1]/s.fs_hz/2 if split_s is None else float(split_s)
    parts=[crop(s,0,split),crop(s,split,s.iq.shape[1]/s.fs_hz)]
    return _result(s,'Matched processing on two contiguous time intervals; the split itself does not establish ON/OFF.',
        parts=[dict(input=p.describe(),spectrum=spectrum(p),spatial=spatial_evidence(p)) for p in parts])


def structure_table(results):
    """Compare identical protocol tests across stations without merging identities."""
    if not results:raise ValueError('need CP results')
    base=results[0];rows=[];seen=set()
    for r in results:
        rid=r['input']['receiver_id']
        if rid in seen:raise ValueError('duplicate station; compare scopes explicitly')
        seen.add(rid)
        same=[r['effective_symbol_s'],r['total_symbol_s'],r['input']['rf_center_hz']+r['center_hz']]
        expected=[base['effective_symbol_s'],base['total_symbol_s'],base['input']['rf_center_hz']+base['center_hz']]
        if not np.allclose(same,expected,rtol=1e-9,atol=1e-12):raise ValueError('different protocol/time/frequency references cannot be treated as matched tests')
        rows.append(dict(receiver_id=rid,lag_coherence_mean=float(np.mean(r['lag_coherence'])),
                         cyclic_coherence_mean=float(np.mean(r['cyclic_coherence'])),input=r['input']))
    return dict(stations=rows,meaning='Station-specific matched tests. A positive station does not establish another station identity.',
                warning='A missing periodic structure does not exclude continuous cochannel components; assess sensitivity and spatial evidence.')


def beamform(s,direction_world):
    require_record(s,True)
    if s.offsets_world_m is None:raise ValueError('actual element offsets required')
    u=np.asarray(direction_world,float)
    if u.shape!=(3,) or not np.isfinite(u).all() or np.linalg.norm(u)==0:raise ValueError('finite nonzero direction required')
    u=u/np.linalg.norm(u);a=np.exp(2j*np.pi*s.rf_center_hz/299792458*s.offsets_world_m@u)
    y=a.conj()[None,:]@s.iq/len(a)
    return _make(y,s.fs_hz,s.rf_center_hz,s.receiver_id,s.band_hz,True,s.start_s,s.position_m,None,None,
                 s.history+(f'conventional beamformer direction {u.tolist()}; sidelobes/mixing remain',))


def template_residual(s,template,max_delay_samples=8):
    """Fit a short complex FIR receive channel on first half, validate on second."""
    require_record(s,True);template=np.asarray(template)
    if template.ndim!=1 or len(template)!=s.iq.shape[1] or not np.iscomplexobj(template) or not np.isfinite(template).all():
        raise ValueError('complex template on the SAME continuous time/frequency grid required')
    p=integer(max_delay_samples,'max_delay_samples',0)
    if p>256 or len(template)<8*(p+1):raise ValueError('unsupported FIR size/record too short')
    a=np.stack([np.roll(template,k) for k in range(p+1)],axis=1)
    a=a[p:];target=s.iq[:,p:].T;mid=len(a)//2
    if np.linalg.matrix_rank(a[:mid])<p+1:raise ValueError('template delay model is rank deficient; reduce taps')
    coeff=np.linalg.lstsq(a[:mid],target[:mid],rcond=None)[0];fit=a@coeff;residual=target-fit
    def fraction(v,x):return float(np.sum(abs(v)**2)/max(np.sum(abs(x)**2),np.finfo(float).tiny))
    out=_make(residual.T,s.fs_hz,s.rf_center_hz,s.receiver_id,s.band_hz,True,s.start_s+p/s.fs_hz,
              s.position_m,s.offsets_world_m,s.rotation_local_to_world,s.history+('FIR template subtraction trained on first half',))
    return dict(input=s.describe(),residual=out,coefficients=coeff,training_residual=fraction(residual[:mid],target[:mid]),
                validation_residual=fraction(residual[mid:],target[mid:]),
                meaning='Linear time-aligned FIR residual, not guaranteed interference; CFO/timing/model mismatch remain possible.')
