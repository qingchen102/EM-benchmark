"""D3/D4 achieved 回填：修复生成器只对 D2 回填 RT 实测可见性的遗漏。

背景（2026-09-28）：gen_dataset 当年只对 d2 回填 achieved（RT 实测每站每源到达
功率/径数），D3/D4 漏了 → 评分端 interferer_visible_at 退回公式 INR（无遮挡，
恒 ≥5dB）→ **答案键把大量"该站物理上收不到"的干扰也算进去**（实测 D3 59% /
D4 66% 的干扰-站链路 0 径，目标被楼完全挡住）。考生从 IQ 里根本看不到这些源，
detection 必然系统性失败。

本脚本从平台 truth.json（runner 生成的每源每站实测 mean_port_power_w / path_count
/ arrival_azimuth_rad）回填 achieved，与 D2 的回填完全同口径，其余字段一个字节不动。
幂等：重复运行安全。回填后 eval 答案键变严（不可见干扰排除），离线回放仍应满分。

用法：
    python backfill_achieved.py ../dataset_d3 ../dataset_d4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def backfill(root: Path) -> dict:
    root = Path(root)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    n_samples = n_sources = n_zero = 0
    for s in idx["samples"]:
        sid = s["id"]
        cap = root / sid / "capture" / "private"
        truth_p = cap / "d1_truth.json"
        plat = json.loads((cap / "truth.json").read_text(encoding="utf-8"))
        truth = json.loads(truth_p.read_text(encoding="utf-8"))
        if truth.get("achieved"):
            continue                                  # 已回填，幂等跳过
        kmap = {r["id"]: f"s{i}" for i, r in enumerate(truth["receivers"])}
        ach: dict[str, dict] = {}
        for st in plat["stations"]:
            sk = kmap.get(st["receiver_id"])
            if sk is None:
                continue
            for src in st["sources"]:
                pw = float(np.mean(src["mean_port_power_w"]))
                e = ach.setdefault(src["emitter_id"], {})
                e[sk] = {"pr_dbm": round(10 * np.log10(pw + 1e-30) + 30, 2),
                         "path_count": src["path_count"],
                         "path_azimuths_deg": [round(np.degrees(a), 1)
                                               for a in src["geometry"]["arrival_azimuth_rad"][:6]]}
                n_sources += 1
                if src["path_count"] == 0:
                    n_zero += 1
        truth["achieved"] = ach
        truth_p.write_text(json.dumps(truth, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        n_samples += 1
    return {"samples_backfilled": n_samples, "source_station_links": n_sources,
            "zero_path_links": n_zero}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", help="数据集根目录（可多个）")
    args = ap.parse_args()
    for r in args.roots:
        stat = backfill(Path(r))
        print(f"== {r}: 回填 {stat['samples_backfilled']} 样本, "
              f"{stat['source_station_links']} 源-站链路, "
              f"其中 0 径(该站物理不可见) {stat['zero_path_links']}")


if __name__ == "__main__":
    main()
