"""一次性脚本：为现有数据集重建 observations.json（补 target_bandwidth_normalized）。

用法：python rebuild_observations.py <dataset_dir>
样本 .npy 与 ground_truth/metadata 不变，只更新 Agent 可见的 observation。
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("用法: python rebuild_observations.py <dataset_dir>")
        return 1
    out = Path(sys.argv[1])
    meta_path = out / "metadata.json"
    obs_path = out / "observations.json"
    if not meta_path.exists() or not obs_path.exists():
        print(f"缺少 {meta_path.name} 或 {obs_path.name}")
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    meta_by_file = {m["file"]: m["metadata"] for m in meta}

    updated = 0
    for rec in obs:
        m = meta_by_file.get(rec["file"])
        if not m:
            continue
        rec["observation"]["target_bandwidth_normalized"] = \
            m["sources"][0]["bandwidth_normalized"]
        updated += 1

    obs_path.write_text(json.dumps(obs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已更新 {updated} 条 observation（补 target_bandwidth_normalized）-> {obs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
