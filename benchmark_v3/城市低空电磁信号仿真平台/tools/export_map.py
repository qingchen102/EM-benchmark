"""Rebuild public Munich mesh and browser assets; requires rt extra."""
from pathlib import Path
import json,hashlib
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import sionna.rt as rt

ROOT=Path(__file__).resolve().parents[1]
scene=rt.load_scene(rt.scene.munich)
triangles=[];object_ids=[];names=[]
for i,shape in enumerate(scene.mi_scene.shapes()):
    p=mi.traverse(shape)
    v=np.asarray(p['vertex_positions']).reshape(-1,3)
    f=np.asarray(p['faces']).reshape(-1,3)
    triangles.append(v[f]);object_ids.append(np.full(len(f),i,dtype=np.int32));names.append(shape.id())
t=np.concatenate(triangles).astype('<f4')
out=ROOT/'priors/munich_geometry';out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'mesh.npz',triangles=t,object_ids=np.concatenate(object_ids))
meta=dict(schema_version=1,scene='sionna.rt.scene.munich',sionna_rt_version='2.1.0',
          coordinate_frame='Native world Cartesian, metres, z up, not georeferenced',objects=names,
          triangle_count=len(t),mesh_sha256=hashlib.sha256((out/'mesh.npz').read_bytes()).hexdigest())
(out/'geometry.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
assets=ROOT/'webui/assets';assets.mkdir(parents=True,exist_ok=True);t.tofile(assets/'munich_mesh.bin')
meta.update(bounds=[t.min(axis=(0,1)).tolist(),t.max(axis=(0,1)).tolist()],format='float32 little endian triangles[N,3,3]',byte_length=t.nbytes)
(assets/'munich_mesh.json').write_text(json.dumps(meta),encoding='utf-8')
print('Exported',len(t),'triangles')
