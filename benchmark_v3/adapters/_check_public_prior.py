"""一次性审计：公开先验（受害者层 + cooperative_sources）落盘后的题面完整性。

只读。核查四件事：
  1. 200 个样本的 observations.json 都能解析、格式与平台写盘一致（indent=2 + 结尾换行）；
  2. cooperative_sources 非空且字段齐全（preset / nominal_bandwidth_hz / link_quality_per_station）；
  3. 公共路径与 private 回退路径的 `_target_bw_normalized` 逐位相同（口径中性）；
  4. D1/D2 不带 interference_events 键（历史逐字节可比），D3/D4 带且字段 = 公开四元组；
     并用 run_capture_agent 的真实格式化函数打印一份 d4 观测，确认事件日志真的进 Agent 提示。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import run_capture_agent as rca  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TIERS = ["d1", "d2", "d3", "d4"]
PUBLIC = ("event_id", "source_id", "mechanism", "impact_class")


def main() -> int:
    bw = rca._preset_bw_table()
    problems: list[str] = []
    stat: dict[str, dict] = {}
    for tier in TIERS:
        st = {"samples": 0, "coop_ok": 0, "events": 0, "with_key": 0,
              "fmt_ok": 0, "norm_same": 0, "lnq_keys": set()}
        for sd in sorted((ROOT / f"dataset_{tier}").glob(f"{tier}_*")):
            cap = sd / "capture"
            op, priv = cap / "public" / "observations.json", cap / "private"
            if not (op.is_file() and (priv / "scenario.json").is_file()):
                continue
            raw = op.read_text(encoding="utf-8")
            obs = json.loads(raw)
            st["samples"] += 1
            if raw.endswith("\n") and raw.splitlines()[1].startswith('  "'):
                st["fmt_ok"] += 1
            else:
                problems.append(f"{sd.name}: 写盘格式异常")

            coop = (obs.get("cooperative_sources") or [{}])[0]
            if coop.get("preset") and coop.get("nominal_bandwidth_hz"):
                st["coop_ok"] += 1
                st["lnq_keys"] |= set((coop.get("link_quality_per_station") or {}).keys())
            else:
                problems.append(f"{sd.name}: cooperative_sources 为空或字段缺失")

            ev = obs.get("interference_events")
            if ev:
                st["with_key"] += 1
                st["events"] += len(ev)
                bad = sorted({k for e in ev for k in e} - set(PUBLIC))
                if bad:
                    problems.append(f"{sd.name}: 事件出现非公开字段 {bad}")
                if "interference_events_note" not in obs:
                    problems.append(f"{sd.name}: 有事件但缺 note")
            else:
                st["no_key"] = st.get("no_key", 0) + 1

            sc = json.loads((priv / "scenario.json").read_text(encoding="utf-8"))
            tgt = next(e for e in sc["emitters"] if e["id"] == "target")
            fs = float(obs["receivers"][0]["sample_rate_hz"])
            n_pub = rca._target_bw_normalized(tgt["preset"], coop["nominal_bandwidth_hz"], fs, bw)
            n_priv = rca._target_bw_normalized(
                tgt["preset"], (tgt.get("components") or [{}])[0].get("bandwidth_hz"), fs, bw)
            if abs(n_pub - n_priv) < 1e-12:
                st["norm_same"] += 1
            else:
                problems.append(f"{sd.name}: 带宽先验不等价 {n_pub} vs {n_priv}")
        stat[tier] = st

    print(f"{'tier':4} {'样本':>4} {'格式OK':>6} {'先验OK':>6} {'带宽等价':>8} "
          f"{'带事件键':>8} {'事件数':>6}  链路遥测站")
    for tier, st in stat.items():
        print(f"{tier:4} {st['samples']:>4} {st['fmt_ok']:>6} {st['coop_ok']:>6} "
              f"{st['norm_same']:>8} {st['with_key']:>8} {st['events']:>6}  "
              f"{sorted(st['lnq_keys'])}")

    obs = json.loads((ROOT / "dataset_d4" / "d4_00000" / "capture" / "public"
                      / "observations.json").read_text(encoding="utf-8"))
    ids = [r["receiver_id"] for r in obs["receivers"]]
    prior = {"stations": [{"id": i, "position_m": r["receiver_position_m"]}
                          for i, r in zip(ids, obs["receivers"])],
             "allowed_stations": ids, "slevel": "S2", "difficulty": "D4",
             "target_emitter": None, "target_bw_hz": 1e6, "target_bw_normalized": 0.25,
             "events": obs.get("interference_events") or [],
             "events_note": obs.get("interference_events_note"),
             "aoi_center": [0.0, 0.0], "fs": 4e6, "fc": 2.4e9, "num_samples": 40000}
    txt = rca._format_observation_capture(prior)
    print("\n--- d4_00000 的 Agent 观测文本（真实格式化函数，节选末段）---")
    print("\n".join(txt.splitlines()[-6:]))

    print(f"\n问题 {len(problems)} 条")
    for p in problems[:20]:
        print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
