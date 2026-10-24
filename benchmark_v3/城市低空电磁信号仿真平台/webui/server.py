"""Loopback-only API; real simulator execution with no implicit backend fallback."""
import argparse
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from cityrf.config import Scenario,Emitter,Receiver,Component,Task
from cityrf.runner import estimate_resources
from cityrf.radiomap import RadioMapRequest
RUNS=ROOT/"runs"
SCENARIOS=ROOT/"user_scenarios"
RADIO_MAPS=ROOT/"radio_maps"
app=FastAPI(title="城市低空电磁信号仿真平台",version="0.3.0")
# 壳页与引导数据一律不缓存：否则浏览器会继续用旧的 index.html / app.js，
# 表现为“切了示例场景但地图和代码还是老的”（2026-10-11 实测到这个现象）。
NO_STORE={"Cache-Control":"no-store, must-revalidate","Pragma":"no-cache"}
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","testserver"])
LOCK=threading.Lock()
ACTIVE=set()
RADIO_ACTIVE=set()
NAMES={"free_space":"自由空间 · 快速验证","munich":"慕尼黑 · 复合干扰",
       "case01":"场景 01 · 多站多目标","case02":"场景 02 · 建筑遮挡",
       "case03":"场景 03 · 同频波形仿冒","A":"复杂 A · 同频噪声与扫频",
       "B":"复杂 B · 六设备复合干扰","C_compressed":"复杂 C · 接收机压缩","C":"复杂 C · 压缩前对照"}

# 自由模式波形下拉（平台 7 + v2 独有 6）
KINDS=[
    dict(name="tone",label="单音 Tone"),dict(name="noise",label="带限噪声"),
    dict(name="bpsk",label="BPSK"),dict(name="qpsk",label="QPSK"),
    dict(name="fsk",label="FSK"),dict(name="ofdm",label="OFDM"),
    dict(name="chirp",label="线性扫频 Chirp"),
    dict(name="gfsk",label="GFSK · v2"),dict(name="16qam",label="16QAM · v2"),
    dict(name="64qam",label="64QAM · v2"),dict(name="ook",label="OOK · v2"),
    dict(name="fhss",label="FHSS · v2"),dict(name="lfm",label="LFM · v2"),
]


def _radiator_list():
    """v3 现有 16 个辐射体，供预设模式下拉（缺库时返回空）。"""
    _v3sim = ROOT.parent / "simulation"                     # benchmark_v3/simulation
    _v2sim = ROOT.parent.parent / "benchmark_v2" / "simulation"
    for _p in (str(_v3sim), str(_v2sim), str(_v2sim.parent)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from em_signal_simulator_v3.emitters import EMITTERS, INTENT
        return [dict(name=n, modulation=m[1], scene=m[2], intent=INTENT.get(n))
                for n, m in EMITTERS.items()]
    except Exception:
        return []


@app.middleware("http")
async def local_writes(request:Request,call_next):
    if request.method in {"POST","PUT","DELETE"}:
        from urllib.parse import urlparse
        origin=request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"Cross-origin writes disabled"},status_code=403)
    return await call_next(request)


def validate(data):
    try:return Scenario.model_validate(data)
    except (ValidationError,ValueError) as e:raise HTTPException(422,detail=str(e))


def job_root(jid):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",jid):raise HTTPException(400,"Invalid run ID")
    p=(RUNS/jid).resolve()
    if not p.is_relative_to(RUNS.resolve()) or not p.is_dir():raise HTTPException(404,"Run not found")
    return p


def radio_root(jid):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",jid):raise HTTPException(400,"Invalid job ID")
    p=(RADIO_MAPS/jid).resolve()
    if not p.is_relative_to(RADIO_MAPS.resolve()) or not p.is_dir():raise HTTPException(404,"Radio map not found")
    return p


def wsl_path(path):
    text=str(Path(path).resolve())
    if not re.match(r"^[A-Za-z]:\\",text):raise RuntimeError("WSL bridge requires a Windows drive path")
    return "/mnt/"+text[0].lower()+text[2:].replace("\\","/")


def runtime_info():
    # WSL availability is confirmed, not assumed from the presence of wsl.exe.
    rt_python=os.environ.get("CITYRF_WSL_PYTHON","/root/.venvs/static-city-rf/bin/python")
    try:native=importlib.metadata.version("sionna-rt")=="2.1.0" and importlib.util.find_spec("mitsuba") is not None
    except importlib.metadata.PackageNotFoundError:native=False
    available=False
    if os.name=="nt":
        try:
            check=subprocess.run(["wsl","--exec",rt_python,"-c","import importlib.metadata;assert importlib.metadata.version('sionna-rt')=='2.1.0';import matplotlib"],capture_output=True,timeout=15)
            available=check.returncode==0
        except (OSError,subprocess.TimeoutExpired):pass
    return dict(native_rt=native,wsl_rt=available,wsl_python=rt_python,python=sys.executable,
                core=str(ROOT/"src/cityrf"),available_backends=["free_space","tapped"]+(["sionna_rt"] if native or available else []))


RUNTIME=None


def _mesh_list():
    """扫描 webui/assets/*_mesh.json，返回可用城市网格（供地图下拉）。"""
    labels={"munich":"慕尼黑","hangzhou":"杭州 · 市中心"}
    out=[]
    for p in sorted((ROOT/"webui/assets").glob("*_mesh.json")):
        try:m=json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError,OSError):continue
        mid=p.name[:-len("_mesh.json")]
        out.append(dict(id=mid,name=labels.get(mid,m.get("name",mid)),
                        triangle_count=m.get("triangle_count",0)))
    out.sort(key=lambda m:m["id"]!="munich")          # 慕尼黑排前作默认
    return out


@app.get("/api/bootstrap")
def bootstrap(response: Response):
    response.headers.update(NO_STORE)
    global RUNTIME
    if RUNTIME is None:RUNTIME=runtime_info()
    presets=[]
    for p in sorted((ROOT/"examples").rglob("*.json")):
        try:s=Scenario.model_validate_json(p.read_text(encoding="utf-8"))
        except (ValueError,OSError):continue
        presets.append(dict(id=p.relative_to(ROOT/"examples").as_posix(),label=NAMES.get(p.stem,p.stem),config=s.model_dump()))
    return dict(runtime=RUNTIME,presets=presets,schema=Scenario.model_json_schema(),
        kinds=KINDS,radiators=_radiator_list(),meshes=_mesh_list(),
        defaults=dict(emitter=Emitter(id="tx_new",position_m=(0,0,60)).model_dump(),
                      receiver=Receiver(id="rx_new",position_m=(120,0,65)).model_dump(),
                      component=Component().model_dump(),task=Task(id="task_new",receiver_ids=["rx_new"],desired_emitter_ids=["tx_new"]).model_dump()))


@app.post("/api/validate")
def check(data:dict):
    s=validate(data)
    resources=estimate_resources(s)
    warnings=[]
    if resources["estimated_dsp_working_mb"]>s.max_working_mb:
        raise HTTPException(422,"预计 DSP 内存超过 max_working_mb，请降低采样率/阵元数或提高预算。")
    if s.propagation.backend!="sionna_rt":warnings.append("当前传播后端不考虑地图建筑；地图仅用于布站显示。")
    else:warnings.append("配置校验未替代 RT 天空可见性检查；屋内或屋檐下端点将在实际求解时拒绝。")
    return dict(valid=True,config=s.model_dump(),resources=resources,warnings=warnings,num_samples=s.num_samples)


@app.post("/api/scenarios")
def save_scenario(data:dict):
    s=validate(data)
    SCENARIOS.mkdir(exist_ok=True)
    sid="scene_"+uuid.uuid4().hex[:12]
    (SCENARIOS/(sid+".json")).write_text(s.model_dump_json(indent=2),encoding="utf-8")
    return dict(id=sid)


@app.get("/api/scenarios")
def saved_scenarios():
    return [dict(id=p.stem,config=json.loads(p.read_text(encoding="utf-8"))) for p in sorted(SCENARIOS.glob("*.json"),reverse=True)]


def run_job(jid,s):
    root=job_root(jid)
    status=dict(id=jid,status="running",name=s.name,started_at=time.time(),backend=s.propagation.backend)
    state=root/"job.json"
    def save():
        tmp=state.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,ensure_ascii=False),encoding="utf-8")
        tmp.replace(state)
    save()
    try:
        info=RUNTIME or runtime_info()
        worker=ROOT/"webui/worker.py"
        if s.propagation.backend=="sionna_rt" and not info["native_rt"]:
            if not info["wsl_rt"]:raise RuntimeError("Sionna RT 环境不可用，请安装 rt 依赖或配置 CITYRF_WSL_PYTHON。")
            command=["wsl","--exec",info["wsl_python"],"-u",wsl_path(worker),wsl_path(root)]
        else:command=[sys.executable,"-u",str(worker),str(root)]
        env=dict(os.environ,PYTHONIOENCODING="utf-8",PYTHONPATH=str(ROOT/"src"))
        with (root/"worker.log").open("w",encoding="utf-8") as log:
            code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if code!=0:raise RuntimeError(f"仿真进程退出码 {code}，请查看运行日志。")
        status["status"]="complete"
    except Exception as e:
        status.update(status="failed",error=str(e))
    finally:
        status["finished_at"]=time.time();save()
        with LOCK:ACTIVE.discard(jid)


@app.post("/api/runs")
def create_run(data:dict):
    check(data)
    s=validate(data)
    with LOCK:
        if ACTIVE:raise HTTPException(409,"已有仿真在运行，请等待完成。")
        jid="run_"+time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
        root=RUNS/jid;root.mkdir(parents=True)
        (root/"config.json").write_text(s.model_dump_json(indent=2),encoding="utf-8")
        (root/"job.json").write_text(json.dumps(dict(id=jid,name=s.name,status="queued",backend=s.propagation.backend,started_at=time.time())),encoding="utf-8")
        ACTIVE.add(jid)
    threading.Thread(target=run_job,args=(jid,s),daemon=True).start()
    return dict(id=jid)


def job_summary(p):
    if (p/"job.json").exists():
        try:state=json.loads((p/"job.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:return dict(id=p.name,name=p.name,status="running")
        if state["status"] in ("running","queued") and p.name not in ACTIVE:state.update(status="interrupted",error="服务曾重启；请检查日志及产物后重新运行。")
        return state
    summary=p/"capture/run_summary.json"
    if summary.exists():
        s=json.loads(summary.read_text(encoding="utf-8"))
        return dict(id=p.name,name=NAMES.get(p.name.replace("example_",""),p.name),status="complete",backend=s["backend"],imported=True,started_at=summary.stat().st_mtime)
    return dict(id=p.name,status="unknown",name=p.name)


@app.get("/api/runs")
def list_runs():
    return sorted([job_summary(p) for p in RUNS.iterdir() if p.is_dir()],key=lambda j:j.get("started_at",0),reverse=True) if RUNS.exists() else []


@app.get("/api/runs/{jid}")
def get_run(jid):
    p=job_root(jid);data=job_summary(p)
    data["log"]=(p/"worker.log").read_text(encoding="utf-8",errors="replace")[-14000:] if (p/"worker.log").exists() else "已导入原项目的已验证仿真结果。"
    if (p/'capture/run_summary.json').exists():
        for key,rel in [("summary","capture/run_summary.json"),("preview","capture/preview.json"),("truth","capture/private/truth.json"),("config","capture/private/scenario.json")]:
            if (p/rel).exists():data[key]=json.loads((p/rel).read_text(encoding="utf-8"))
    data["files"]=[f.relative_to(p).as_posix() for f in p.rglob("*") if f.is_file() and f.suffix in {".json",".npy",".npz",".png",".log"}]
    return data


@app.get("/api/runs/{jid}/files/{filename:path}")
def download(jid,filename):
    root=job_root(jid);p=(root/filename).resolve()
    if not p.is_relative_to(root) or not p.is_file() or p.suffix not in {".json",".npy",".npz",".png",".log"}:raise HTTPException(404)
    return FileResponse(p,filename=p.name)


def run_radio_job(jid,scenario,request):
    root=radio_root(jid)
    status=dict(id=jid,status="running",name=scenario.name,scene=scenario.propagation.scene,
                height_m=request.height_m,cell_size_m=request.cell_size_m,started_at=time.time())
    state=root/"job.json"
    def save():
        tmp=state.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,ensure_ascii=False),encoding="utf-8")
        tmp.replace(state)
    save()
    try:
        info=RUNTIME or runtime_info()
        worker=ROOT/"webui/radiomap_worker.py"
        if info["native_rt"]:command=[sys.executable,"-u",str(worker),str(root)]
        else:
            if not info["wsl_rt"]:raise RuntimeError("电磁态势图需要 Sionna RT 环境，请安装 rt 依赖或配置 CITYRF_WSL_PYTHON。")
            command=["wsl","--exec",info["wsl_python"],"-u",wsl_path(worker),wsl_path(root)]
        env=dict(os.environ,PYTHONIOENCODING="utf-8",PYTHONPATH=str(ROOT/"src"))
        with (root/"worker.log").open("w",encoding="utf-8") as log:
            code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if code!=0:raise RuntimeError(f"态势图进程退出码 {code}，请查看运行日志。")
        if not (root/"radiomap.json").exists():raise RuntimeError("态势图进程未产出结果文件。")
        status["status"]="complete"
        status["seconds"]=json.loads((root/"radiomap.json").read_text(encoding="utf-8")).get("seconds")
    except Exception as e:
        status.update(status="failed",error=str(e))
        with (root/"worker.log").open("a",encoding="utf-8") as log:log.write(f"[failed] {e}\n")
    finally:
        status["finished_at"]=time.time();save()
        with LOCK:RADIO_ACTIVE.discard(jid)


@app.post("/api/radio-maps")
def create_radio_map(data:dict):
    scenario=validate(data.get("scenario"))
    try:request=RadioMapRequest.model_validate(data.get("radio_map") or {})
    except ValidationError as e:raise HTTPException(422,detail=str(e))
    if scenario.propagation.backend!="sionna_rt":
        raise HTTPException(422,"电磁态势图需要 Sionna RT 后端（sionna_rt）；自由空间与抽头后端不利用建筑。")
    with LOCK:
        if RADIO_ACTIVE:raise HTTPException(409,"已有态势图在计算，请等待完成。")
        jid="map_"+time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
        root=RADIO_MAPS/jid;root.mkdir(parents=True)
        (root/"request.json").write_text(json.dumps(dict(scenario=scenario.model_dump(),radio_map=request.model_dump()),ensure_ascii=False,indent=2),encoding="utf-8")
        (root/"job.json").write_text(json.dumps(dict(id=jid,status="queued",name=scenario.name,scene=scenario.propagation.scene,height_m=request.height_m,cell_size_m=request.cell_size_m,started_at=time.time()),ensure_ascii=False),encoding="utf-8")
        RADIO_ACTIVE.add(jid)
    threading.Thread(target=run_radio_job,args=(jid,scenario,request),daemon=True).start()
    return dict(id=jid)


def radio_map_summary(p):
    state=dict(id=p.name,status="unknown",name=p.name)
    if (p/"job.json").exists():
        try:state=json.loads((p/"job.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:return dict(id=p.name,status="running",name=p.name)
        if state["status"] in ("running","queued") and p.name not in RADIO_ACTIVE:
            state.update(status="interrupted",error="服务曾重启；请检查日志后重新计算。")
    return state


@app.get("/api/radio-maps")
def list_radio_maps():
    return sorted([radio_map_summary(p) for p in RADIO_MAPS.iterdir() if p.is_dir()],key=lambda j:j.get("started_at",0),reverse=True) if RADIO_MAPS.exists() else []


@app.get("/api/radio-maps/{jid}")
def get_radio_map(jid):
    p=radio_root(jid);data=radio_map_summary(p)
    data["log"]=(p/"worker.log").read_text(encoding="utf-8",errors="replace")[-14000:] if (p/"worker.log").exists() else ""
    if (p/"radiomap.json").exists():data["payload"]=json.loads((p/"radiomap.json").read_text(encoding="utf-8"))
    data["files"]=[f.relative_to(p).as_posix() for f in p.rglob("*") if f.is_file() and f.suffix in {".json",".npy",".npz",".png",".log"}]
    return data


@app.get("/api/radio-maps/{jid}/files/{filename:path}")
def download_radio_map(jid,filename):
    root=radio_root(jid);p=(root/filename).resolve()
    if not p.is_relative_to(root) or not p.is_file() or p.suffix not in {".json",".npy",".npz",".png",".log"}:raise HTTPException(404)
    return FileResponse(p,filename=p.name)


app.mount("/assets",StaticFiles(directory=ROOT/"webui/assets"),name="assets")
@app.get("/")
def index():return FileResponse(ROOT/"webui/index.html",headers=NO_STORE)
@app.get("/index.html",include_in_schema=False)
def index_alias():return FileResponse(ROOT/"webui/index.html",headers=NO_STORE)


if __name__=="__main__":
    import uvicorn
    parser=argparse.ArgumentParser();parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args();RUNS.mkdir(exist_ok=True)
    uvicorn.run(app,host="127.0.0.1",port=args.port)
