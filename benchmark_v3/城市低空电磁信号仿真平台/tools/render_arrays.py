"""Visualize exported acquisition geometry; only reads public metadata."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cityrf.analysis import load_public

parser=argparse.ArgumentParser()
parser.add_argument("public",type=Path)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
obs=load_public(args.public)
fig=plt.figure(figsize=(13,5.5),layout="constrained")
ax=fig.add_subplot(121)
r=obs["receivers"][0]
local=np.array(r["element_offsets_local_m"])*100
geom=r["array_geometry"]
colors=["#2878b5" if v.get("arm")=="horizontal" else "#d66a32" for v in geom["channel_map"]]
ax.scatter(local[:,1],local[:,2],c=colors,s=60)
for i,p in enumerate(local):
    ax.annotate(str(i),(p[1],p[2]),xytext=(5,5),textcoords="offset points")
ax.axhline(0,color="#cccccc",lw=.8)
ax.axvline(0,color="#cccccc",lw=.8)
ax.scatter([0],[0],marker="+",c="black",s=70,label="Array center (no antenna)")
ax.set(xlabel="Local y (cm)",ylabel="Local z (cm)",title="8 + 8 cross | channel order | local x = 0",aspect="equal")
ax.legend(loc="lower right",fontsize=8)
ax=fig.add_subplot(122,projection="3d")
for index,r in enumerate(obs["receivers"]):
    color=["#2878b5","#d66a32","#1b9e77"][index%3]
    p=np.array(r["element_offsets_world_m"])*100
    ax.scatter(p[:,0],p[:,1],p[:,2],s=18,label=r["receiver_id"],color=color)
    normal=np.array(r["array_pose"]["normal_world"])*25
    ax.quiver(0,0,0,*normal,linewidth=1,color=color)
ax.set(xlim=(-27,27),ylim=(-27,27),zlim=(-27,27))
ax.set_box_aspect((1,1,1))
ax.set(xlabel="World x offset (cm)",ylabel="World y offset (cm)",zlabel="World z offset (cm)",
       title="Station orientations | arrows: array front normals")
ax.legend(loc="upper right",fontsize=8)
fig.suptitle("16 unique elements | half-wavelength spacing | 2.4 GHz")
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=140)
print(args.output)
