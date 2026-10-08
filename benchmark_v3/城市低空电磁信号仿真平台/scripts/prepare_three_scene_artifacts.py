"""Public IQ-only plots and host-only scene geometry validation for three cases."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.signal import welch,spectrogram
from cityrf.baselines.reflection import Geometry
from cityrf.export import digest,write_json
from cityrf.audit import audit_run
base=Path(__file__).resolve().parents[1];root=base/'results/three_scenes_027'
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
geom_source=base/'priors/munich_geometry';g=Geometry.load(geom_source/'mesh.npz')
scene_fig,scene_axes=plt.subplots(1,3,figsize=(17,5.7),layout='constrained')
tri=g.triangles;n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);mask=(abs(n[:,2])>1e-5)&(tri[:,:,2].mean(axis=1)>1)
titles=['多站多目标：异频 OFDM / QPSK / FSK','遮挡传播：A/B NLOS，C LOS','波形层仿冒：两个同频 OFDM']
all_validations=[]
for ci,case in enumerate(['case01','case02','case03']):
    folder=root/case;capture=folder/'capture';public=capture/'public'
    audit=audit_run(capture);write_json(capture/'audit.json',audit)
    obs=json.loads((public/'observations.json').read_text(encoding='utf-8'))
    scenario=json.loads((capture/'private/scenario.json').read_text(encoding='utf-8'))
    truth=json.loads((capture/'private/truth.json').read_text(encoding='utf-8'))
    prior_dir=folder/'priors';(prior_dir/'geometry').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(geom_source/'mesh.npz',prior_dir/'geometry/mesh.npz')
    meta=json.loads((geom_source/'geometry.json').read_text(encoding='utf-8'));meta['observations_sha256']=digest(public/'observations.json')
    write_json(prior_dir/'geometry/geometry.json',meta)
    if case=='case03':write_json(prior_dir/'cooperative.json',{'schema_version':'1.0','observations_sha256':digest(public/'observations.json'),'cooperative_sources':obs['cooperative_sources']})
    validation=[]
    for st in truth['stations']:
        rx=next(r for r in scenario['receivers'] if r['id']==st['receiver_id'])
        for tx_data in st['sources']:
            tx=next(t for t in scenario['emitters'] if t['id']==tx_data['emitter_id'])
            d=np.array(tx['position_m'])-rx['position_m'];length=np.linalg.norm(d);u=d/length
            with np.load(capture/f"private/channel_{tx['id']}_{rx['id']}.npz") as ch:
                powers=np.mean(abs(ch['coefficients'])**2,axis=0);delays=ch['delays_s']
            geo=tx_data['geometry'];az=np.array(geo['arrival_azimuth_rad']);ze=np.array(geo['arrival_zenith_rad'])
            dirs=np.c_[np.sin(ze)*np.cos(az),np.sin(ze)*np.sin(az),np.cos(ze)]
            direct=bool(len(dirs) and np.any((dirs@u>np.cos(np.deg2rad(.05)))&(abs(delays-length/299792458.)<1e-9)))
            visible=bool(g.visible(rx['position_m'],tx['position_m']))
            assert direct==visible,(case,rx['id'],tx['id'],'RT/mesh LoS mismatch')
            assert len(powers)>0,(case,rx['id'],tx['id'],'no received paths')
            validation.append(dict(receiver_id=rx['id'],source_id=tx['id'],mesh_los=visible,rt_has_direct=direct,path_count=len(powers),mean_received_power_dbm=float(10*np.log10(np.mean(tx_data['mean_port_power_w']))+30)))
    if case=='case02':assert [v['mesh_los'] for v in validation]==[False,False,True]
    if case in ['case01','case03']:assert all(v['mesh_los'] for v in validation)
    write_json(folder/'scene_validation.json',{'case':case,'links':validation,'audit':audit['status']})
    all_validations.append(dict(case=case,links=validation))
    plotdir=public/'plots';plotdir.mkdir(exist_ok=True)
    computed=[]
    for rx in obs['receivers']:
        x=np.load(public/rx['iq_file']);fs=rx['sample_rate_hz']
        f,p=welch(x,fs=fs,nperseg=8192,noverlap=4096,return_onesided=False,scaling='density',axis=-1)
        order=np.argsort(f);f=f[order];p=p[:,order].mean(axis=0)
        ft,t,s=spectrogram(x,fs=fs,window='hann',nperseg=1024,noverlap=768,return_onesided=False,scaling='density',mode='psd',axis=-1)
        order=np.argsort(ft);ft=ft[order];s=s[:,order,:].mean(axis=0)
        pdb=10*np.log10(np.maximum(p,1e-30))+30;sdb=10*np.log10(np.maximum(s,1e-30))+30
        np.savez_compressed(plotdir/(rx['receiver_id']+'_spectral_data.npz'),frequency_hz=f,psd_w_per_hz=p,stft_frequency_hz=ft,time_s=t,stft_psd_w_per_hz=s)
        computed.append((rx,f,pdb,ft,t,sdb))
    vmin=min(float(np.quantile(v[5][abs(v[3])<700000],.1)) for v in computed);vmax=max(float(np.quantile(v[5],.999)) for v in computed)
    fig,axes=plt.subplots(3,2,figsize=(13,9),layout='constrained')
    for ri,(rx,f,pdb,ft,t,sdb) in enumerate(computed):
        rid=rx['receiver_id']
        for target_axes in [axes[ri]]:
            target_axes[0].plot(f/1000,pdb,lw=.9,color='#176ca4');target_axes[0].set(title=rid+' | 阵元功率平均频谱',xlabel='基带频率 (kHz)',ylabel='PSD (dBm/Hz)',xlim=(-750,750),ylim=(vmin-4,vmax+5));target_axes[0].grid(alpha=.2)
            im=target_axes[1].pcolormesh(t*1000,ft/1000,sdb,shading='auto',cmap='viridis',vmin=vmin,vmax=vmax)
            target_axes[1].set(title=rid+' | 阵元功率平均时频图',xlabel='时间 (ms)',ylabel='基带频率 (kHz)',ylim=(-750,750));fig.colorbar(im,ax=target_axes[1],label='PSD (dBm/Hz)')
        one,ax=plt.subplots(figsize=(10,3.6),layout='constrained');ax.plot(f/1000,pdb,lw=1);ax.set(title=rid+' | PSD | RF ref 2.4 GHz',xlabel='Baseband frequency (kHz)',ylabel='PSD (dBm/Hz)',xlim=(-750,750),ylim=(vmin-4,vmax+5));ax.grid(alpha=.2);one.savefig(plotdir/(rid+'_spectrum.png'),dpi=150);plt.close(one)
        one,ax=plt.subplots(figsize=(10,3.6),layout='constrained');im=ax.pcolormesh(t*1000,ft/1000,sdb,shading='auto',cmap='viridis',vmin=vmin,vmax=vmax);ax.set(title=rid+' | Spectrogram | RF ref 2.4 GHz',xlabel='Time (ms)',ylabel='Baseband frequency (kHz)',ylim=(-750,750));one.colorbar(im,ax=ax,label='PSD (dBm/Hz)');one.savefig(plotdir/(rid+'_spectrogram.png'),dpi=150);plt.close(one)
    fig.suptitle('接收观测概览 | 2.4 GHz / 2 MS/s / 50 ms / 16 channels',fontsize=14);fig.savefig(plotdir/'all_stations.png',dpi=150);plt.close(fig)
    params={'source':'Only public mixed receiver IQ; no private source labels used for spectral plots','averaging':'Arithmetic mean of per-channel power spectra, not coherent averaging of complex channels','welch':{'nperseg':8192,'noverlap':4096,'window':'hann','resolution_bin_hz':2000000/8192},'spectrogram':{'nperseg':1024,'noverlap':768,'window':'hann','frequency_bin_hz':2000000/1024,'time_hop_s':256/2000000},'units':'W/Hz in NPZ; dBm/Hz in PNG; frequencies relative to receiver RF center','display_frequency_range_hz':[-750000,750000],'color_scale_dbm_per_hz':[vmin,vmax]}
    write_json(plotdir/'plot_parameters.json',params)
    (plotdir/'README_zh.md').write_text('每站提供 _spectrum.png 频谱图、_spectrogram.png 时频图和 _spectral_data.npz 数值。all_stations.png 为三站总览。图像仅由公开混合IQ计算，不提供源数、调制或位置结论。功率为16阵元功率谱平均，未相干平均复数IQ。频率相对接收中心，图例为dBm/Hz；STFT有限分辨率不能排除同频叠加，阵列信息仍在原始IQ中。计算参数见 plot_parameters.json。',encoding='utf-8')
    ax=scene_axes[ci];ax.add_collection(PolyCollection(tri[mask,:,:2],facecolors='#d5d9de',edgecolors='none',alpha=.85));ax.set(xlim=(-230,220),ylim=(-170,170),aspect='equal',xlabel='世界 x (m)',ylabel='世界 y (m)',title=titles[ci]);ax.grid(alpha=.15)
    for tx in scenario['emitters']:
        x,y,z=tx['position_m'];ax.scatter(x,y,c='#cf6932' if tx['role']=='interference' else '#225ea8',s=45,zorder=4);ax.annotate(tx['id']+f'\nz={z:g}m',(x,y),xytext=(-4 if x>100 else 4,-24),ha='right' if x>100 else 'left',textcoords='offset points',fontsize=8)
    for rx in scenario['receivers']:
        x,y,z=rx['position_m'];ax.scatter(x,y,c='#009b79',marker='^',s=55,zorder=5);ax.annotate(rx['id']+f'\nz={z:g}m',(x,y),xytext=(5,7),textcoords='offset points',fontsize=8)
        if case=='case02':
            tx=scenario['emitters'][0];q=tx['position_m'];los=g.visible(rx['position_m'],q);ax.plot([x,q[0]],[y,q[1]],color='#159875' if los else '#d65a58',ls='-' if los else '--',lw=1.5)
scene_fig.savefig(root/'three_scene_layouts.png',dpi=150);plt.close(scene_fig)
write_json(root/'scene_validation.json',all_validations)
html='<meta charset="utf-8"><title>三场景观测图</title><style>body{max-width:1450px;margin:32px auto;background:#f4f6f9;color:#243245;font:17px sans-serif}img{width:100%;background:white;border-radius:8px}section{margin:32px 0}h1,h2{color:#153e63}a{color:#156aab}</style><h1>城市电磁分析：三个独立场景</h1><p>以下频谱和时频图由混合接收 IQ 直接计算；场景布局供实验复核，不向 Agent 暴露未知源位置。</p><img src="three_scene_layouts.png">'
for case,title in zip(['case01','case02','case03'],titles):html+=f'<section><h2>{case} · {title}</h2><img src="{case}/capture/public/plots/all_stations.png"><p><a href="{case}/capture/public/plots/README_zh.md">图像计算说明</a></p></section>'
(root/'index.html').write_text(html,encoding='utf-8')
print(json.dumps(all_validations,ensure_ascii=True,indent=2))
