"""Inspect installed RT shape and scalar LOS conventions without altering runtime."""
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
import sionna.rt as rt
import numpy as np

s=rt.load_scene()
s.frequency=2.4e9
s.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern="iso",polarization="V")
s.rx_array=rt.PlanarArray(num_rows=1,num_cols=4,pattern="iso",polarization="V")
s.add(rt.Transmitter("tx",position=[0,100,10]))
s.add(rt.Receiver("rx",position=[0,0,10]))
p=rt.PathSolver()(s,max_depth=0,refraction=False,samples_per_src=1000,synthetic_array=True)
a,t=p.cir(normalize_delays=False,out_type="numpy")
print("a",a.shape,a.reshape(-1))
print("t",t.shape,t)
for name in ("theta_r","phi_r","valid","vertices","interactions"):
    obj=getattr(p,name)
    arr=obj.numpy() if hasattr(obj,"numpy") else np.asarray(obj)
    print(name,arr.shape,str(arr)[:700])
print("positions",s.rx_array.positions(s.wavelength).numpy())
