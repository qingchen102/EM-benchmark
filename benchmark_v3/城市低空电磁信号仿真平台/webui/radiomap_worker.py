"""电磁态势图（radio map）独立作业入口。

与 :mod:`webui.worker` 同样刻意做成独立进程：作业在具备 Sionna RT 的 Python 里
运行，网页服务只负责调度、读状态和发文件。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cityrf.config import Scenario
from cityrf.radiomap import RadioMapRequest, compute_radio_map, to_payload, write_npz
from webui.radiomap_report import render_png


def main():
    job = Path(sys.argv[1]).resolve()
    spec = json.loads((job / "request.json").read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(spec["scenario"])
    request = RadioMapRequest.model_validate(spec.get("radio_map") or {})
    print(f"[0/4] Radio map: scene={scenario.propagation.scene}, "
          f"{len(scenario.emitters)} TX / {len(scenario.receivers)} RX, "
          f"height={request.height_m:g} m, cell={request.cell_size_m:g} m", flush=True)
    result = compute_radio_map(scenario, request, emit=lambda message: print(message, flush=True))
    payload = to_payload(result, scenario)
    print("[4/4] 写出 NPZ / JSON / PNG", flush=True)
    write_npz(result, job / "radiomap.npz")
    (job / "radiomap.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    render_png(result, payload, job / "radiomap.png", scenario.name)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
