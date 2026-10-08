from pathlib import Path
import json
from cityrf.config import Scenario
from cityrf.sionna_backend import SionnaRTBackend
p=Path('results/three_scenes_027/placement_probe.json')
s=Scenario.model_validate_json(p.read_text())
links=SionnaRTBackend().solve(s)
rows=[]
for tx in s.emitters:
    counts={l.rx_id:l.coefficients.shape[1] for l in links if l.tx_id==tx.id}
    row=dict(id=tx.id,position_m=tx.position_m,path_counts=counts)
    rows.append(row);print(row,flush=True)
p.with_name('placement_probe_result.json').write_text(json.dumps(rows,indent=2))
