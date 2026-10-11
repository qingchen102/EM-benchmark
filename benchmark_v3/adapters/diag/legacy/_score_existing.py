"""对已有 answers 目录用（修正后的）评分口径重算——不调 API。"""
import json
import sys
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
import numpy as np
from pathlib import Path

for p in ['.', '../simulation', '../城市低空电磁信号仿真平台/src']:
    rp = str(Path(p).resolve())
    if rp not in sys.path:
        sys.path.insert(0, rp)

import run_capture_agent as rca  # noqa: E402  (导入即注册 v3 工具/评分通路)
import eval_dataset as ed  # noqa: E402

root = Path('../../benchmark_v3/dataset_d2')
ans_dir = root / 'answers_S2_agent_full'
agg = {}
n = 0
for f in sorted(ans_dir.glob('d2_*.json')):
    sid = f.name.replace('.json', '')
    a = json.load(open(f, encoding='utf-8'))
    truth = json.load(open(root / sid / 'capture/private/d1_truth.json'))
    r = ed.score_answer(a, truth['interferers'], 'S2')
    n += 1
    for k, v in r.items():
        if isinstance(v, list):          # 命中列表 → 池化
            agg.setdefault(k, []).extend(v)
        elif isinstance(v, (int, float)):
            agg[k] = agg.get(k, 0) + v
import numpy as np
print('=== agent 全量 50 样本（注册口径：FREQ_TOL=0.02 / POS_GATE_M=50 / DOA_TOL=10°）===')
for k in ('detection_hits', 'detection_notgt', 'kind_hits', 'freq_hits',
          'doa_hits', 'doa_notgt', 'pos_hits', 'pos_notgt', 'intent_hits',
          'intent_notgt', 'spoof_hits', 'spoof_notgt'):
    if k in agg:
        vals = agg[k]
        print('  %-16s %.3f  [%d/%d]' % (k, float(np.mean(vals)) if vals else 0,
                                          sum(vals), len(vals)))
print('样本数:', n)
