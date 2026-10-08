"""Narrowband Bartlett and MUSIC using actual 3D element coordinates."""
import numpy as np
from scipy.ndimage import maximum_filter
from ..channels import C0


def band_covariance(iq,fs,low,high,max_snapshots=8192):
    f=np.fft.fftfreq(iq.shape[1],1/fs)
    selected=(f>=low)&(f<=high)
    if selected.sum()<2:
        raise ValueError("frequency band has fewer than two bins")
    spectrum=np.fft.fft(iq,axis=-1)
    spectrum[:,~selected]=0
    x=np.fft.ifft(spectrum,axis=-1)
    stride=max(1,int(fs/(2*(high-low))))
    x=x[:,::stride]
    if x.shape[1]>max_snapshots:
        x=x[:,np.linspace(0,x.shape[1]-1,max_snapshots,dtype=int)]
    # Keep mean: a narrow tone can become nearly DC after decimation.
    covariance=x@x.conj().T/x.shape[1]
    return covariance,int(x.shape[1])


def mdl_rank(eigenvalues, snapshots,max_rank=6):
    values=np.maximum(np.asarray(eigenvalues,float),max(float(eigenvalues[0])*1e-12,1e-30))
    m=len(values)
    scores=[]
    for k in range(min(max_rank,m-1)+1):
        tail=values[k:]
        scores.append(snapshots*(m-k)*(np.log(tail.mean())-np.log(tail).mean())
                      +.5*k*(2*m-k)*np.log(max(snapshots,2)))
    return int(np.argmin(scores)),[float(s) for s in scores]


def _peaks(power,directions,azimuth,elevation,step=8.,max_peaks=8,within_db=6.):
    db=10*np.log10(np.maximum(power,1e-30)/max(float(power.max()),1e-30))
    grid=db.reshape(len(np.unique(elevation)),len(np.unique(azimuth)))
    maxima=(grid>=maximum_filter(grid,size=3,mode=("nearest","wrap"))).ravel()
    peaks=[]
    for i in np.argsort(db)[::-1]:
        if db[i]<-within_db:
            break
        if not maxima[i]:
            continue
        u=directions[i]
        if any(np.dot(u,p["unit_direction_world"])>np.cos(np.deg2rad(step)) for p in peaks):
            continue
        peaks.append({"azimuth_deg":float(azimuth[i]),"elevation_deg":float(elevation[i]),
                      "relative_db":float(db[i]),"unit_direction_world":u.tolist()})
        if len(peaks)>=max_peaks:
            break
    return peaks


def scan_directions(covariance,offsets_world,fc,snapshots,grid_step_deg=2.):
    az,el=np.meshgrid(np.arange(-180,180,grid_step_deg),np.arange(-88,89,grid_step_deg))
    a,e=np.deg2rad(az.ravel()),np.deg2rad(el.ravel())
    directions=np.stack([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)],axis=1)
    steering=np.exp(2j*np.pi*fc/C0*np.asarray(offsets_world)@directions.T)
    values,vectors=np.linalg.eigh(covariance)
    rank,scores=mdl_rank(values[::-1],snapshots)
    used=max(1,rank)
    noise=vectors[:,:len(values)-used]
    music=1/np.maximum(np.sum(abs(noise.conj().T@steering)**2,axis=0),1e-20)
    bartlett=np.maximum(np.real(np.sum(steering.conj()*(covariance@steering),axis=0)),1e-30)/len(values)**2
    return {"mdl_rank":rank,"music_rank_used":used,"mdl_scores":scores,
            "eigenvalues":values[::-1].tolist(),"snapshots":snapshots,"grid_step_deg":grid_step_deg,
            "music":{"peaks":_peaks(music,directions,az.ravel(),el.ravel())},
            "bartlett":{"peaks":_peaks(bartlett,directions,az.ravel(),el.ravel())},
            "warning":"MDL estimates covariance components, not physical transmitters. Coherent multipath and mirror ambiguities remain."}
