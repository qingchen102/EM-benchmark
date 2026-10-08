"""Generate the three requested image-inspired scenarios with actual Sionna RT."""
from pathlib import Path
import json
from cityrf.config import Scenario
from cityrf.runner import run_scenario
from cityrf.audit import audit_run
from cityrf.export import write_json
base=Path(__file__).resolve().parents[1]
for name in ['case01','case02','case03']:
    out=base/'results/three_scenes_027'/name/'capture'
    config=Scenario.model_validate_json((base/'examples/three_scenes'/(name+'.json')).read_text(encoding='utf-8'))
    print('START',name,flush=True)
    result=run_scenario(config,out)
    audit=audit_run(out)
    write_json(out/'audit.json',audit)
    print('DONE',name,json.dumps(audit,ensure_ascii=True),flush=True)
