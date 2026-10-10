"""数据集轻量快照：只存 index.json + 各样本 d1_truth.json + scenario.json
（几十 KB/版，可见性/重叠/位置/功率全部统计可复算，250MB IQ 不留）。
用法：python _snapshot_meta.py <dataset_dir> <meta_version_tag>
重生成前必须先留档（2026-10-03 QA1/QA2 被覆盖、+7.5pp 不可复算的教训）。"""
import json
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
tag = sys.argv[2]
dst = Path(str(src) + f"_meta_{tag}")
dst.mkdir(parents=True, exist_ok=True)
if (src / "index.json").exists():
    shutil.copy2(src / "index.json", dst / "index.json")
n = 0
for d in sorted(src.iterdir()):
    if not d.name.startswith("d") or not d.is_dir():
        continue
    cap = d / "capture" / "private"
    out = dst / d.name / "capture" / "private"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("d1_truth.json", "scenario.json"):
        if (cap / f).is_file():
            shutil.copy2(cap / f, out / f)
            n += 1
print(f"snapshot -> {dst} ({n} files)")
