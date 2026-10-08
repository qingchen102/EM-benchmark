"""Continuous IQ with explicit scope. Correct metadata is not adversarial security."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class SignalBlock:
    iq: np.ndarray
    fs_hz: float
    rf_center_hz: float
    receiver_id: str
    band_hz: tuple
    selected: bool
    start_s: float
    position_m: np.ndarray | None
    offsets_world_m: np.ndarray | None
    rotation_local_to_world: np.ndarray | None
    history: tuple
    signature: str

    def describe(self):
        return dict(receiver_id=self.receiver_id,shape=list(self.iq.shape),fs_hz=self.fs_hz,
                    rf_center_hz=self.rf_center_hz,band_hz=list(self.band_hz),selected=self.selected,
                    start_s=self.start_s,duration_s=self.iq.shape[1]/self.fs_hz,
                    history=list(self.history),signature=self.signature)


def positive(value,name):
    if not np.isfinite(value) or value<=0: raise ValueError(name+' must be finite and positive')
    return float(value)


def integer(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def signal_block(iq,fs_hz,rf_center_hz,*,receiver_id='unknown',position_m=None,
                 offsets_world_m=None,rotation_local_to_world=None):
    x=np.asarray(iq)
    if x.ndim!=2 or min(x.shape)<1 or x.shape[1]<4 or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError('finite complex IQ [channels,samples>=4] required; do not replace invalid samples')
    fs=positive(fs_hz,'fs_hz');fc=positive(rf_center_hz,'rf_center_hz')
    def arr(a,shape,name):
        if a is None:return None
        z=np.array(a,float,copy=True)
        if z.shape!=shape or not np.isfinite(z).all():raise ValueError(name+' shape/values invalid')
        z.setflags(write=False);return z
    pos=arr(position_m,(3,),'position_m')
    off=arr(offsets_world_m,(x.shape[0],3),'offsets_world_m')
    rot=arr(rotation_local_to_world,(3,3),'rotation_local_to_world')
    if rot is not None and (not np.allclose(rot.T@rot,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rot),1,atol=1e-6)):
        raise ValueError('rotation must be a proper orthogonal local-to-world matrix')
    return _make(x,fs,fc,str(receiver_id),(-fs/2,fs/2),False,0.,pos,off,rot,('raw continuous IQ',))


def _make(x,fs,fc,rid,band,selected,start,pos,off,rot,history):
    band=tuple(float(v) for v in band);history=tuple(history)
    x=np.array(x,copy=True,order='C');x.setflags(write=False)
    h=hashlib.sha256(x.tobytes())
    h.update(repr((x.shape,x.dtype.str,fs,fc,rid,band,selected,start,history)).encode())
    for a in (pos,off,rot):
        if a is not None:h.update(a.tobytes())
    return SignalBlock(x,fs,fc,rid,tuple(band),selected,start,pos,off,rot,tuple(history),h.hexdigest())


def load_record(observations_path,receiver_id):
    p=Path(observations_path);obs=json.loads(p.read_text(encoding='utf-8'))
    matches=[r for r in obs['receivers'] if r['receiver_id']==receiver_id]
    if len(matches)!=1:raise ValueError('receiver_id not unique/present')
    r=matches[0];src=(p.parent/r['iq_file']).resolve()
    if not src.is_relative_to(p.parent.resolve()):raise ValueError('IQ path outside observation directory')
    return signal_block(np.load(src,allow_pickle=False),r['sample_rate_hz'],r['center_frequency_hz'],
        receiver_id=receiver_id,position_m=r['receiver_position_m'],offsets_world_m=r['element_offsets_world_m'],
        rotation_local_to_world=r.get('array_pose',{}).get('rotation_local_to_world'))


def select_band(record,low_hz,high_hz):
    """FFT mask on the continuous record, no concatenation or decimation."""
    require_record(record)
    lo,hi=float(low_hz),float(high_hz)
    if not np.isfinite([lo,hi]).all() or not record.band_hz[0]<=lo<hi<=record.band_hz[1]:
        raise ValueError('band must be ordered and inside the current available band (Hz relative to receiver center)')
    f=np.fft.fftfreq(record.iq.shape[1],1/record.fs_hz);m=(f>=lo)&(f<=hi)
    if m.sum()<2:raise ValueError('band narrower than two FFT bins; use a longer record')
    x=np.fft.ifft(np.fft.fft(record.iq,axis=1)*m,axis=1)
    return _make(x,record.fs_hz,record.rf_center_hz,record.receiver_id,(lo,hi),True,record.start_s,
                 record.position_m,record.offsets_world_m,record.rotation_local_to_world,
                 record.history+(f'continuous FFT band [{lo},{hi}] Hz; brick-wall ringing possible',))


def crop(record,start_s,stop_s):
    require_record(record);a,b=float(start_s),float(stop_s);duration=record.iq.shape[1]/record.fs_hz
    if not np.isfinite([a,b]).all() or not 0<=a<b<=duration:raise ValueError('crop seconds relative to this record, inside its duration')
    i,j=round(a*record.fs_hz),round(b*record.fs_hz)
    if j-i<4:raise ValueError('crop too short')
    return _make(record.iq[:,i:j],record.fs_hz,record.rf_center_hz,record.receiver_id,record.band_hz,record.selected,
                 record.start_s+i/record.fs_hz,record.position_m,record.offsets_world_m,record.rotation_local_to_world,
                 record.history+(f'contiguous crop samples [{i},{j})',))


def require_record(record,selected=False):
    if not isinstance(record,SignalBlock):raise TypeError('SignalBlock required; use load_record/signal_block and select_band')
    if selected and not record.selected:raise ValueError('explicit select_band required; full-band IQ is not a selected component')


def assert_comparable(first,second):
    require_record(first);require_record(second)
    if first.signature!=second.signature:raise ValueError('different input scope/data; rerun both models on the same SignalBlock')


def save_result(path,result):
    def default(x):
        if isinstance(x,SignalBlock):return x.describe()
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,complex):return {'real':x.real,'imag':x.imag}
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(result,default=default,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def brief(value):
    """Small console summary; full arrays remain accessible for plots/save_result."""
    if isinstance(value,SignalBlock):return value.describe()
    if isinstance(value,np.ndarray):
        return value.tolist() if value.size<=32 else f'<ndarray shape={value.shape} dtype={value.dtype}; summary only, use original array or save_result>'
    if isinstance(value,dict):return {k:brief(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [brief(v) for v in value[:8]]+([{'more_items':len(value)-8}] if len(value)>8 else [])
    if isinstance(value,np.generic):return value.item()
    return value


def save_signal(path,s):
    require_record(s)
    meta=s.describe()
    for name in ('position_m','offsets_world_m','rotation_local_to_world'):
        a=getattr(s,name);meta[name]=None if a is None else a.tolist()
    np.savez_compressed(path,iq=s.iq,metadata=np.array(json.dumps(meta)))


def load_signal(path):
    with np.load(path,allow_pickle=False) as d:
        m=json.loads(str(d['metadata']));x=d['iq']
    validated=signal_block(x,m['fs_hz'],m['rf_center_hz'],receiver_id=m['receiver_id'],
        position_m=m['position_m'],offsets_world_m=m['offsets_world_m'],rotation_local_to_world=m['rotation_local_to_world'])
    if not -validated.fs_hz/2<=m['band_hz'][0]<m['band_hz'][1]<=validated.fs_hz/2:raise ValueError('invalid saved band')
    result=_make(validated.iq,validated.fs_hz,validated.rf_center_hz,validated.receiver_id,m['band_hz'],bool(m['selected']),
                 float(m['start_s']),validated.position_m,validated.offsets_world_m,validated.rotation_local_to_world,tuple(m['history']))
    if result.signature!=m['signature']:raise ValueError('saved signal signature mismatch')
    return result
