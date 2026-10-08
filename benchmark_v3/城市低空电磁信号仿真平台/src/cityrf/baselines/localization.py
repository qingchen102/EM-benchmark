"""Least-squares 3D bearing intersection with explicit mirror hypotheses."""
from itertools import product
import numpy as np


def intersect_rays(origins,directions):
    origins=np.asarray(origins,float)
    directions=np.asarray(directions,float)
    directions=directions/np.linalg.norm(directions,axis=1)[:,None]
    projectors=np.eye(3)[None,:,:]-directions[:,:,None]*directions[:,None,:]
    a=projectors.sum(axis=0)
    condition=float(np.linalg.cond(a))
    if not np.isfinite(condition) or condition>1e5:
        return None
    point=np.linalg.solve(a,np.einsum("nij,nj->i",projectors,origins))
    delta=point-origins
    distances=np.linalg.norm(delta,axis=1)
    if np.min(distances)<1e-6:
        return None
    along=np.sum(delta*directions,axis=1)
    if np.any(along<=0):
        return None
    perpendicular=np.einsum("nij,nj->ni",projectors,delta)
    angles=np.rad2deg(np.arccos(np.clip(np.sum(delta*directions,axis=1)/distances,-1,1)))
    return {"position_m":point.tolist(),"angular_rms_deg":float(np.sqrt(np.mean(angles**2))),
            "perpendicular_rms_m":float(np.sqrt(np.mean(perpendicular**2)*3)),
            "ranges_m":distances.tolist(),"condition_number":condition}


def localize(station_directions,method):
    if len(station_directions)<2:
        return {"status":"insufficient_stations","hypotheses":[]}
    # Bounded experiment: at most four detected stations, chosen by their order.
    stations=station_directions[:4]
    origins=[s["position_m"] for s in stations]
    sets=[s["scan"][method]["peaks"] for s in stations]
    candidates=[]
    for combination in product(*sets):
        result=intersect_rays(origins,[p["unit_direction_world"] for p in combination])
        if result is None:
            continue
        if not 0<=result["position_m"][2]<=200 or max(result["ranges_m"])>1000:
            continue
        result["score"]=result["angular_rms_deg"]+.1*float(np.mean([-p["relative_db"] for p in combination]))
        result["receiver_ids"]=[s["receiver_id"] for s in stations]
        result["chosen_directions"]=[dict(p) for p in combination]
        candidates.append(result)
    accepted=[]
    for r in sorted(candidates,key=lambda r:r["score"]):
        if any(np.linalg.norm(np.array(r["position_m"])-p["position_m"])<20 for p in accepted):
            continue
        accepted.append(r)
        if len(accepted)==6:
            break
    return {"status":"hypotheses" if accepted else "no_valid_intersection",
            "hypotheses":accepted,"ambiguity_count":len(accepted),
            "constraints":{"height_m":[0,200],"max_range_m":1000},
            "note":"Ranked without truth; top one is the operational prediction. Other hypotheses are ambiguity diagnostics."}
