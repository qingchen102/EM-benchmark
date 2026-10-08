"""Read-only geometry probe used to select reproducible outdoor example positions."""
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
import sionna.rt as rt
import numpy as np
s=rt.load_scene(rt.scene.munich)
for x,y in [(-150,-100),(-80,40),(80,60),(150,-80),(-100,-150),(0,-100),(100,0),(0,100),(200,0),(-200,0)]:
    hit=s.mi_scene.ray_intersect(mi.Ray3f(o=mi.Point3f(x,y,200),d=mi.Vector3f(0,0,-1)))
    print(x,y,hit.p.numpy().reshape(-1).tolist())
