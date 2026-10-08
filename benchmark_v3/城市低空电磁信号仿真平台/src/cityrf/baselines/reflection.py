"""Geometry-only reverse ray tracing and candidate intersections in world metres."""
from itertools import product
import numpy as np
from .localization import intersect_rays


class Geometry:
    def __init__(self, triangles):
        t=np.asarray(triangles, dtype=float)
        if t.ndim!=3 or t.shape[1:]!=(3,3) or not np.isfinite(t).all():
            raise ValueError('triangles must be finite [T,3,3]')
        self.triangles=t
        self.v=t[:,0]
        self.e1=t[:,1]-t[:,0]
        self.e2=t[:,2]-t[:,0]
        n=np.cross(self.e1,self.e2)
        self.normals=n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)

    @classmethod
    def load(cls, path='input/geometry/mesh.npz'):
        with np.load(path,allow_pickle=False) as data:
            return cls(data['triangles'])

    def first_hit(self, origin, direction, max_distance=5000.):
        o=np.asarray(origin,float);d=np.asarray(direction,float)
        if o.shape!=(3,) or d.shape!=(3,) or not np.isfinite([o,d]).all() or np.linalg.norm(d)==0:
            raise ValueError('finite nonzero 3D ray required')
        d=d/np.linalg.norm(d)
        h=np.cross(d,self.e2);det=np.einsum('ij,ij->i',self.e1,h)
        valid=abs(det)>1e-10
        inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
        s=o-self.v;u=inv*np.einsum('ij,ij->i',s,h)
        q=np.cross(s,self.e1);v=inv*(q@d)
        distance=inv*np.einsum('ij,ij->i',self.e2,q)
        valid &= (u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(distance>1e-5)&(distance<max_distance)
        indices=np.flatnonzero(valid)
        if not len(indices):return None
        k=int(indices[np.argmin(distance[indices])])
        return dict(point_m=(o+distance[k]*d).tolist(), normal=self.normals[k].tolist(),
                    distance_m=float(distance[k]), triangle_id=k)

    def visible(self, a, b):
        delta=np.asarray(b)-np.asarray(a);length=np.linalg.norm(delta)
        return length<1e-5 or self.first_hit(a,delta,max_distance=length-1e-4) is None

    def backtrace(self, origin, direction, max_distance=2000.):
        o=np.asarray(origin,float);d=np.asarray(direction,float);d=d/np.linalg.norm(d)
        hit=self.first_hit(o,d,max_distance)
        paths=[dict(kind='direct',origin=o.tolist(),direction=d.tolist(),
                    max_distance_m=hit['distance_m'] if hit else max_distance, wall=None)]
        if hit:
            n=np.asarray(hit['normal']);reflected=d-2*np.dot(d,n)*n
            p=np.asarray(hit['point_m'])
            next_hit=self.first_hit(p+reflected*1e-4,reflected,max_distance)
            paths.append(dict(kind='single_reflection',origin=p.tolist(),direction=reflected.tolist(),
                max_distance_m=next_hit['distance_m'] if next_hit else max_distance,wall=hit))
        return paths

    def localize(self, stations, max_distance=2000., max_angle_deg=8.,
                 height_range=(0.,200.), max_combinations=10000, max_results=20):
        """stations: [{receiver_id, position_m, directions:[[dx,dy,dz], ...]}].
        Directions point FROM receiver toward arriving path, in world coordinates.
        Enumerates direct and one-bounce candidates; does not handle diffraction.
        """
        import warnings
        warnings.warn('Legacy locator selects one ray per station; for joint paths use '
                      'rfkit.analysis_api.locate_paths with complete path models.',
                      UserWarning, stacklevel=2)
        if len(stations)<2:return dict(hypotheses=[],status='insufficient_stations')
        choices=[]
        for station in stations:
            candidates=[]
            for i,d in enumerate(station['directions']):
                for path in self.backtrace(station['position_m'],d,max_distance):
                    candidates.append(dict(path,receiver_id=station['receiver_id'],direction_index=i))
            choices.append(candidates)
        total=int(np.prod([len(c) for c in choices]))
        if total>max_combinations:
            raise ValueError(f'{total} combinations; narrow signal/direction candidates or explicitly raise max_combinations')
        results=[]
        for combo in product(*choices):
            result=intersect_rays([p['origin'] for p in combo],[p['direction'] for p in combo])
            if result is None or result['angular_rms_deg']>max_angle_deg:continue
            point=result['position_m']
            if not height_range[0]<=point[2]<=height_range[1]:continue
            if any(length>p['max_distance_m']+0.01 for length,p in zip(result['ranges_m'],combo)):continue
            if not all(self.visible(np.asarray(p['origin'])+np.asarray(p['direction'])*1e-4,point) for p in combo):continue
            result['paths']=list(combo)
            result['score']=result['angular_rms_deg']
            results.append(result)
        accepted=[]
        for result in sorted(results,key=lambda x:x['score']):
            if any(np.linalg.norm(np.asarray(result['position_m'])-x['position_m'])<5 for x in accepted):continue
            accepted.append(result)
            if len(accepted)>=max_results:break
        return dict(status='hypotheses' if accepted else 'no_valid_intersection',hypotheses=accepted,
                    combinations=total,model='direct or one specular reflection per station',
                    note='Geometric candidates, not calibrated probabilities. Compare IQ likelihood and angle perturbations.')
