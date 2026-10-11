"""把四档基准数据集（D1–D4）的场景导出为平台预设 `examples/benchmark_v3/*.json`。

原料 = `benchmark_v3/dataset_dX/<sid>/capture/private/scenario.json`（生成数据集时的真实配置），
只改 `name`（便于在网页下拉与编辑器标题里读），其余字段逐字保留，保证"预设 = 数据集实际场景"。

样本选择：默认每档 `dX_00000`，**D2 换成 `d2_00017`** —— `d2_00000` 的整组电台落在杭州图幅
西南角的无建筑区（水面/绿地，到最近建筑 91~792 m），地图上看起来"全在城外"；`d2_00017`
整网在建成区（±200 m 建筑密度 2462~6673 格，到最近建筑 ≤28 m），target 在 4 个站上各 4 条径、
两个干扰源 3~6 条径全部可见。样本可用 `diag/_pick_demo_sample.py` 复算与替换。

注意：`webui/server.py` 的 bootstrap 用 `Scenario.model_validate_json` 逐个校验
`examples/**/*.json`，校验失败会**静默丢弃**；所以写完必须回读校验一次（本脚本已内置）。

用法::

    python -X utf8 diag/_make_platform_presets.py            # 写入并逐档回读校验
    python -X utf8 diag/_make_platform_presets.py --dry-run  # 只打印将要写入的内容摘要
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]           # EM-benchmark/
PLAT = ROOT / "benchmark_v3" / "城市低空电磁信号仿真平台"
sys.path.insert(0, str(PLAT / "src"))

from cityrf.config import Scenario  # noqa: E402

NAMES = {
    "d1": ("d1_logistics_airway", "D1 物流航线基线 · 单站"),
    "d2": ("d2_residential_coexistence", "D2 居民区共存 · 四站"),
    "d3": ("d3_urban_canyon", "D3 城市峡谷多径 · 四站"),
    "d4": ("d4_adversarial_contest", "D4 对抗演习 · 四站"),
}
SAMPLES = {"d1": "d1_00000", "d2": "d2_00017", "d3": "d3_00000", "d4": "d4_00000"}
OUT = PLAT / "examples" / "benchmark_v3"


def main() -> int:
    ap = argparse.ArgumentParser(description="导出平台预设（D1–D4）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    args = ap.parse_args()
    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)

    ok = 0
    for d, (stem, label) in NAMES.items():
        sid = SAMPLES[d]
        src = ROOT / "benchmark_v3" / f"dataset_{d}" / sid / "capture" / "private" / "scenario.json"
        obj = json.loads(src.read_text(encoding="utf-8"))
        obj["name"] = f"{label}（样本 {sid}）"
        dst = OUT / f"{stem}.json"
        text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        if args.dry_run:
            print(f"[dry-run] {dst.relative_to(PLAT)}  ← {src.relative_to(ROOT)}  "
                  f"name={obj['name']!r}  {len(text.encode('utf-8'))} B")
            continue
        dst.write_text(text, encoding="utf-8")
        back = Scenario.model_validate_json(dst.read_text(encoding="utf-8"))
        ok += 1
        print(f"{dst.relative_to(PLAT)}  ✅ name={back.name!r} backend={back.propagation.backend} "
              f"scene={back.propagation.scene} tx={len(back.emitters)} rx={len(back.receivers)} "
              f"tasks={len(back.tasks)}")

    if args.dry_run:
        return 0
    print(f"\n--- 回读目录（{ok}/{len(NAMES)} 档已写） ---")
    for p in sorted(OUT.glob("*.json")):
        print(f"  {p.name}  {p.stat().st_size} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
