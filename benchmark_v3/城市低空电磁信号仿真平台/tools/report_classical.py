"""Generate plots/tables only from frozen predictions and post-freeze evaluation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
cases=[("Sionna RT: LoS only","classical_rt_los_predictions"),
       ("Munich ray tracing","classical_munich_cross16_v2")]
truth=json.loads((root/"results/munich_cross16_verified/private/truth.json").read_text())
public=json.loads((root/"results/munich_cross16_verified/public/observations.json").read_text())
out=root/"results/classical_comparison"
out.mkdir(exist_ok=True)
evaluations=[json.loads((root/"results"/folder/"evaluation.json").read_text()) for _,folder in cases]

fig,axes=plt.subplots(1,2,figsize=(14,6.5),layout="constrained")
for ax,(title,_),evaluation in zip(axes,cases,evaluations):
    for i,s in enumerate(truth["sources"]):
        x,y,_=s["position_m"]
        ax.scatter(x,y,marker="*",s=150,c="black",label="True emitters" if i==0 else None,zorder=5)
        ax.annotate(s["id"],(x,y),xytext=(5,-14),textcoords="offset points",fontsize=9)
    for i,r in enumerate(public["receivers"]):
        x,y,_=r["receiver_position_m"]
        ax.scatter(x,y,marker="^",s=75,c="#19855b",label="Receivers" if i==0 else None,zorder=4)
        ax.annotate(r["receiver_id"].replace("station_","Rx "),(x,y),xytext=(5,6),textcoords="offset points",fontsize=9)
    for method,color,marker in [("bartlett","#2878b5","o"),("music","#d66a32","x")]:
        for i,g in enumerate(evaluation["groups"]):
            hs=g["localization"][method]["hypotheses"]
            if not hs:
                continue
            h=hs[0]
            x,y,_=h["position_m"]
            ax.scatter(x,y,c=color,marker=marker,s=85,label=method.upper() if i==0 else None,zorder=6)
            source=next(s for s in truth["sources"] if s["id"]==h["nearest_compatible_source"])
            tx,ty,_=source["position_m"]
            ax.plot([x,tx],[y,ty],color=color,alpha=.45,ls="--",lw=1)
            ax.annotate(g["group_id"].replace("band_","B"),(x,y),xytext=(5,6 if method=="bartlett" else -10),
                        textcoords="offset points",fontsize=8,color=color)
    ax.set(title=title,xlabel="Scene x (m)",ylabel="Scene y (m)",xlim=(-220,320),ylim=(-220,170),aspect="equal")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left",fontsize=8)
fig.suptitle("Top-one bearing intersections | dashed lines: post-hoc nearest frequency-compatible truth",fontsize=12)
fig.savefig(out/"localization_comparison.png",dpi=150)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,4),layout="constrained")
for ax,(title,_),evaluation in zip(axes,cases,evaluations):
    x=np.arange(len(evaluation["groups"]))
    for offset,method,color in [(-.18,"bartlett","#2878b5"),(.18,"music","#d66a32")]:
        values=[g["localization"][method]["hypotheses"][0]["distance_m"] for g in evaluation["groups"]]
        bars=ax.bar(x+offset,values,width=.34,color=color,label=method.upper())
        ax.bar_label(bars,fmt="%.1f",fontsize=9)
    ax.axhline(50,color="black",ls="--",lw=1,label="50 m gate")
    ax.set_xticks(x,["B00: -250 kHz","B01: +50 kHz","B02: +250 kHz"])
    ax.set(title=title,ylabel="3D distance to nearest compatible source (m)",ylim=(0,400))
    ax.legend(fontsize=8)
fig.savefig(out/"location_errors.png",dpi=150)
plt.close(fig)

evaluation=evaluations[1]
fig,axes=plt.subplots(3,1,figsize=(11,8),layout="constrained")
pred=json.loads((root/"results/classical_munich_cross16_v2/prediction.json").read_text())
for ax,station in zip(axes,pred["stations"]):
    d=station["detection"]
    ax.plot(np.array(d["frequency_hz"])/1000,d["psd_dbw_per_hz"],lw=.8)
    ax.axhline(d["noise_floor_dbw_per_hz"]+d["threshold_db"],ls="--",color="#d66a32",label="Energy threshold")
    for band in d["bands"]:
        ax.axvspan(band["low_hz"]/1000,band["high_hz"]/1000,color="#19855b",alpha=.15)
    ax.set(title=station["receiver_id"],xlabel="Baseband frequency (kHz)",ylabel="PSD (dBW/Hz)",xlim=(-700,700))
    ax.legend(loc="upper right",fontsize=8)
fig.suptitle("Detected frequency regions | green regions are NOT physical-source labels")
fig.savefig(out/"detection_bands.png",dpi=150)
print(out)
