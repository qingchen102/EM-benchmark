"""Sionna RT 2.1 adapter; never falls back to a different propagation model."""
import importlib.metadata
import numpy as np
from .channels import LinkChannel, element_offsets_local, C0


def _numpy(value):
    return value.numpy() if hasattr(value,"numpy") else np.asarray(value)


def _load_scene(scene_name):
    """\"munich\"→内置场景；\"empty\"→空；其他→自定义 Mitsuba 场景文件路径。"""
    import sionna.rt as rt
    if scene_name == "munich":
        return rt.load_scene(rt.scene.munich)
    if scene_name in ("empty", "", None):
        return rt.load_scene(None)
    from pathlib import Path as _P
    cand = _P(scene_name)
    if not cand.is_absolute():
        platform_root = _P(__file__).resolve().parents[2]
        cand = platform_root / cand
    if not cand.is_file():
        raise RuntimeError(f"自定义场景文件不存在: {cand}（Propagation.scene='{scene_name}'）")
    return rt.load_scene(str(cand))


class SionnaRTBackend:
    def solve(self, scenario):
        try:
            import mitsuba as mi
            mi.set_variant(scenario.propagation.rt_variant)
            import sionna.rt as rt
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Sionna RT unavailable: install the rt extra and configure the requested Mitsuba backend") from exc
        if importlib.metadata.version("sionna-rt") != "2.1.0":
            raise RuntimeError("Sionna RT adapter requires version 2.1.0")
        p=scenario.propagation
        scene=_load_scene(p.scene)
        scene.frequency=scenario.carrier_frequency_hz
        scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern="iso",polarization="V")
        spec=scenario.receivers[0].antenna
        pattern=rt.PlanarArray(num_rows=1,num_cols=1,pattern=spec.pattern,
                              polarization=spec.polarization).antenna_pattern
        normalized=element_offsets_local(scenario.receivers[0],C0)
        scene.rx_array=rt.AntennaArray(antenna_pattern=pattern,normalized_positions=mi.Point3f(
            normalized[:,0],normalized[:,1],normalized[:,2]))
        for tx in scenario.emitters:
            scene.add(rt.Transmitter(tx.id,position=tx.position_m,orientation=tx.orientation_rad))
        for rx in scenario.receivers:
            scene.add(rt.Receiver(rx.id,position=rx.position_m,orientation=rx.orientation_rad))
        if p.scene != "empty":
            box=scene.mi_scene.bbox()
            lower,upper=np.array(box.min),np.array(box.max)
            for dev in scenario.emitters+scenario.receivers:
                x,y,z=dev.position_m
                if not(lower[0] < x < upper[0] and lower[1] < y < upper[1] and z>0):
                    raise ValueError(f"{dev.id}: outside scene XY bounds or below ground")
                # Conservative v1 outdoor rule: requires unobstructed upward ray.
                # Rejects interiors AND covered outdoor spaces; not a general solid classifier.
                hit=scene.mi_scene.ray_intersect(mi.Ray3f(o=mi.Point3f(x,y,z),d=mi.Vector3f(0,0,1)))
                if bool(_numpy(hit.is_valid()).reshape(-1)[0]):
                    raise ValueError(f"{dev.id}: endpoint is not sky-visible (interior/roof/overhang)")
        paths=rt.PathSolver()(scene,max_depth=p.max_depth,
            samples_per_src=p.samples_per_src,max_num_paths_per_src=p.max_num_paths_per_src,
            synthetic_array=True,los=p.los,specular_reflection=p.specular_reflection,
            diffuse_reflection=p.diffuse_reflection,refraction=p.refraction,
            diffraction=p.diffraction,seed=scenario.seed)
        a,tau=paths.cir(normalize_delays=False,out_type="numpy")
        valid=_numpy(paths.valid)
        az,zen=_numpy(paths.phi_r),_numpy(paths.theta_r)
        links=[]
        for ri,rx in enumerate(scenario.receivers):
            for ti,tx in enumerate(scenario.emitters):
                mask=valid[ri,ti].astype(bool)
                coefficients=a[ri,:,ti,0,:,0][:,mask]
                delays=tau[ri,ti,mask]
                links.append(LinkChannel(tx.id,rx.id,coefficients,delays,
                    {"arrival_azimuth_rad":az[ri,ti,mask].tolist(),
                     "arrival_zenith_rad":zen[ri,ti,mask].tolist()},
                    {"backend":"sionna_rt","version":"2.1.0","variant":mi.variant(),
                     "scene":p.scene,"synthetic_array":True,"normalize_delays":False,
                     "endpoint_check":"sky-visible outdoor only"}))
        return links
