"""Render reproducible static acceptance figures; private geometry for developer QA."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft
from cityrf.audit import audit_run
from cityrf.export import write_json

p=argparse.ArgumentParser()
p.add_argument("run",type=Path)
args=p.parse_args()
root=args.run
report=audit_run(root)
write_json(root/"audit.json",report)
obs=json.loads((root/"public"/"observations.json").read_text())
truth=json.loads((root/"private"/"truth.json").read_text())
config=json.loads((root/"private"/"scenario.json").read_text())
rows=len(obs["receivers"])
fig,axes=plt.subplots(rows,2,figsize=(13,3.2*rows),squeeze=False,layout="constrained")
for i,r in enumerate(obs["receivers"]):
    x=np.load(root/"public"/r["iq_file"])
    f,t,z=stft(x[0],fs=r["sample_rate_hz"],nperseg=512,noverlap=384,
               return_onesided=False,boundary=None,padded=False)
    f=np.fft.fftshift(f)/1000
    power=20*np.log10(np.maximum(abs(np.fft.fftshift(z,axes=0)),1e-15))
    im=axes[i,0].pcolormesh(t*1000,f,power,shading="auto",cmap="viridis")
    axes[i,0].set(title=r["receiver_id"]+" | channel 0",xlabel="Time (ms)",ylabel="Baseband frequency (kHz)")
    fig.colorbar(im,ax=axes[i,0],label="STFT amplitude (dB sqrt(W))")
    stats=truth["stations"][i]["sources"]
    levels=[10*np.log10(max(np.mean(s["mean_port_power_w"]),1e-30))+30 for s in stats]
    colors=["#2878b5" if s["emitter_id"].startswith("uav") else "#d66a32" for s in stats]
    axes[i,1].bar([s["emitter_id"] for s in stats],np.maximum(levels,-130)+130,
                  bottom=-130,color=colors)
    axes[i,1].set(ylim=(-130,0),ylabel="Mean antenna-port power (dBm)",
                  title="Per-source coherent multipath sum")
    for j,(s,level) in enumerate(zip(stats,levels)):
        axes[i,1].text(j,max(level,-128)+2,f'{s["path_count"]} paths',ha="center",fontsize=9)
fig.suptitle(f"{config['name']} | {len(config['emitters'])} TX / {len(config['receivers'])} RX | {config['duration_s']*1000:g} ms",fontsize=14)
fig.savefig(root/"signal_overview.png",dpi=140)
plt.close(fig)

if config["propagation"]["backend"]=="sionna_rt" and config["propagation"]["scene"]=="munich":
    import mitsuba as mi
    mi.set_variant(config["propagation"]["rt_variant"])
    import sionna.rt as rt
    scene=rt.load_scene(rt.scene.munich)
    grid=np.linspace(-250,250,220)
    xx,yy=np.meshgrid(grid,grid)
    hit=scene.mi_scene.ray_intersect(mi.Ray3f(
        o=mi.Point3f(xx.ravel(),yy.ravel(),np.full(xx.size,200)),
        d=mi.Vector3f(0,0,-1)))
    heights=hit.p.numpy()[2].reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(9,8),layout="constrained")
    im=ax.imshow(heights,extent=[-250,250,-250,250],origin="lower",cmap="Greys",vmin=0,vmax=60)
    for dev in config["emitters"]+config["receivers"]:
        is_rx="antenna" in dev or dev["id"].startswith("station")
        x,y,z=dev["position_m"]
        ax.scatter(x,y,c="#1b9e77" if is_rx else "#d95f02",marker="^" if is_rx else "o",s=75)
        ax.annotate(f'{dev["id"]}\nz={z:g} m',(x,y),xytext=(6,6),textcoords="offset points",fontsize=9)
    ax.set(xlabel="Scene x (m)",ylabel="Scene y (m)",title="Munich study area | actual RT mesh top-surface heights")
    fig.colorbar(im,ax=ax,label="Surface height (m)")
    fig.savefig(root/"scene_overview.png",dpi=140)
    plt.close(fig)
print(json.dumps(report,indent=2))
