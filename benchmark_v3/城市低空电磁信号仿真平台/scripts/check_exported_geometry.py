"""Geometry-only comparison against Mitsuba ray intersection; no source truth."""
import json
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import sionna.rt as rt
from cityrf.baselines.reflection import Geometry
from pathlib import Path
g=Geometry.load('priors/munich_geometry/mesh.npz')
scene=rt.load_scene(rt.scene.munich)
rng=np.random.default_rng(456)
errors=[]
for origin in ([80,60,5],[0,100,5],[100,0,20]):
    for _ in range(12):
        d=rng.normal(size=3);d/=np.linalg.norm(d)
        a=g.first_hit(origin,d)
        b=scene.mi_scene.ray_intersect(mi.Ray3f(o=mi.Point3f(*origin),d=mi.Vector3f(*map(float,d))))
        valid=bool(b.is_valid().numpy()[0])
        assert (a is not None)==valid
        if valid:errors.append(float(np.linalg.norm(np.asarray(a['point_m'])-b.p.numpy().reshape(3))))
result={'rays':36,'max_hit_error_m':max(errors),'scope':'Geometry only; no simulated source angles or paths used'}
assert max(errors)<.01
Path('priors/munich_geometry/validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
