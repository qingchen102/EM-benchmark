"""One-shot 50 ms simulation. Large arrays stay outside model-visible metadata."""
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from .config import Scenario
from .channels import FreeSpaceBackend, TappedBackend, apply_channel, DELAY_HALF
from .sionna_backend import SionnaRTBackend
from .waveforms import generate, sub_seed
from .receiver import receive
from .export import write_json, digest, public_observation, public_tasks, cooperative_sources

BACKENDS={"free_space":FreeSpaceBackend,"tapped":TappedBackend,"sionna_rt":SionnaRTBackend}


def estimate_resources(s):
    # Conservative DSP estimate, excluding RT acceleration structure/path allocation.
    ports=max(r.antenna.elements for r in s.receivers)
    samples=s.num_samples+int(s.sample_rate_hz*.001)+2200
    working=samples*(len(s.emitters)*16+ports*16*14+16*8)
    output=s.num_samples*sum(r.antenna.elements for r in s.receivers)*8
    return {"estimated_dsp_working_mb":working/1e6,"iq_output_mb":output/1e6,
            "rt_memory_note":"RT scene/path solver memory is additional; limited by solver settings."}


def run_scenario(config: Scenario, output, *, frozen_links=None):
    s=Scenario.model_validate(config.model_dump())  # revalidate mutable user configs
    estimate=estimate_resources(s)
    if estimate["estimated_dsp_working_mb"] > s.max_working_mb:
        raise ValueError("estimated DSP memory exceeds max_working_mb")
    root=Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("output must be absent or empty; existing experiments are never overwritten")
    started=time.perf_counter()
    links=BACKENDS[s.propagation.backend]().solve(s) if frozen_links is None else frozen_links
    solve_seconds=time.perf_counter()-started
    by_pair={(l.tx_id,l.rx_id):l for l in links}
    expected={(t.id,r.id) for t in s.emitters for r in s.receivers}
    if len(by_pair) != len(links) or set(by_pair) != expected:
        raise ValueError("backend returned missing or duplicate links")
    for link in links:
        rx=next(r for r in s.receivers if r.id==link.rx_id)
        if link.coefficients.shape[0] != rx.antenna.elements:
            raise ValueError("backend antenna count mismatch")
    max_delay=max((float(np.max(l.delays_s,initial=0)) for l in links),default=0)
    if max_delay > .001:
        raise ValueError("path delay exceeds supported 1 ms guard")
    rx_guard=max(r.frontend.filter_taps//2 for r in s.receivers)
    history=int(np.ceil(max_delay*s.sample_rate_hz))+DELAY_HALF+rx_guard+2
    end=s.num_samples+rx_guard+DELAY_HALF+2
    indices=np.arange(-history,end,dtype=np.int64)
    signals={t.id:generate(t,s.sample_rate_hz,indices,s.seed) for t in s.emitters}
    public=root/"public"
    private=root/"private"
    public.mkdir(parents=True,exist_ok=True)
    private.mkdir(parents=True,exist_ok=True)
    write_json(private/"scenario.json",s.model_dump())
    observations=[]
    measurements=[]
    for rx in s.receivers:
        composite=np.zeros((rx.antenna.elements,s.num_samples+2*rx_guard),complex)
        source_stats=[]
        for tx in s.emitters:
            link=by_pair[tx.id,rx.id]
            arrival=apply_channel(signals[tx.id],indices[0],link,s.sample_rate_hz,
                                  composite.shape[1],output_start=-rx_guard)
            rel=np.array(tx.velocity_mps)
            radial=float(np.dot(rel,np.array(rx.position_m)-np.array(tx.position_m))
                         /max(np.linalg.norm(np.array(rx.position_m)-np.array(tx.position_m)),1e-6))
            doppler=radial/299792458.*s.carrier_frequency_hz
            shift=tx.rf_offset_hz+doppler
            if shift:
                t=(np.arange(arrival.shape[1])-rx_guard)/s.sample_rate_hz
                arrival=arrival*np.exp(1j*2*np.pi*shift*t)
            ant=rx.antenna
            if ant.gain_error_std_db or ant.phase_error_std_deg:
                local=np.random.default_rng(sub_seed(s.seed,f"{rx.id}/array"))
                gain=10**(local.normal(0,ant.gain_error_std_db,arrival.shape[0])/20)
                phase=np.exp(1j*np.deg2rad(local.normal(0,ant.phase_error_std_deg,arrival.shape[0])))
                arrival=(gain*phase)[:,None]*arrival
            composite+=arrival
            core=arrival[:,rx_guard:rx_guard+s.num_samples]
            source_stats.append({
                "emitter_id":tx.id,"path_count":link.coefficients.shape[1],
                "mean_port_power_w":np.mean(abs(core)**2,axis=1).tolist(),
                "scope":"linear antenna-port contribution; before receiver",
                "geometry":link.geometry,"channel_provenance":link.provenance})
            np.savez(private/f"channel_{tx.id}_{rx.id}.npz",
                     coefficients=link.coefficients,delays_s=link.delays_s)
        iq,receiver_metrics=receive(composite,rx,s.sample_rate_hz,s.seed,start_index=-rx_guard)
        iq=np.ascontiguousarray(iq[:,rx_guard:rx_guard+s.num_samples])
        if iq.shape != (rx.antenna.elements,s.num_samples) or not np.isfinite(iq).all():
            raise ValueError("invalid output IQ")
        np.save(public/f"iq_{rx.id}.npy",iq,allow_pickle=False)
        obs=public_observation(s,rx)
        obs["sha256"]=digest(public/obs["iq_file"])
        observations.append(obs)
        measurements.append({"receiver_id":rx.id,"sources":source_stats,
                             "receiver":receiver_metrics,
                             "output_port_power_w":np.mean(abs(iq.astype(complex))**2,axis=1).tolist()})
    write_json(public/"observations.json",{
        "schema_version":"1.2","receivers":observations,"tasks":public_tasks(s),
        "cooperative_sources":cooperative_sources(s)})
    write_json(private/"truth.json",{
        "schema_version":"1.2","sources":[t.model_dump() for t in s.emitters],
        "tx_power_convention":"Ungated composite calibrated on fixed 16384-sample reference; gates and bandlimiting are not renormalized.",
        "actual_tx_capture_power_w":{t.id:float(np.mean(abs(signals[t.id][history:history+s.num_samples])**2))
                                     for t in s.emitters},
        "tasks":[t.model_dump() for t in s.tasks],"stations":measurements,
        "notes":["Physical emitter count is distinct from number of propagation paths.",
                 "Received powers are coherent per-source path sums, averaged over capture.",
                 "No automatic blocking or source-observability label is inferred."]})
    hashes={str(p.relative_to(root)).replace("\\","/"):digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()}
    summary={
        "status":"complete","schema_version":"1.2","num_samples":s.num_samples,
        "duration_s":s.duration_s,"sample_rate_hz":s.sample_rate_hz,
        "emitter_count":len(s.emitters),"receiver_count":len(s.receivers),
        "backend":s.propagation.backend,"path_count":sum(l.coefficients.shape[1] for l in links),
        "execution_mode":"new_channel_solve" if frozen_links is None else "frozen_channel_replay",
        "solve_seconds":solve_seconds,"total_seconds":time.perf_counter()-started,
        "source_hashes":{p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "resources":estimate,"versions":{"python":platform.python_version(),
                  "numpy":np.__version__,"scipy":scipy.__version__},
        "hashes":hashes}
    write_json(root/"run_summary.json",summary)  # completion marker written last
    return summary
