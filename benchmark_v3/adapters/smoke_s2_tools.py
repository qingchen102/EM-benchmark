"""S2 上下文冒烟：estimate_sources_by_spatial_clustering + extract_source_waveforms
断言不报错（此前 smoke_capture_tools.py 用 S0 单站上下文，聚类直接返回"需要>=3站"，
且不测这两个工具——CLAIM_TOL 类 NameError 被吞的教训）。
用法：python smoke_s2_tools.py [样本数]，默认 3。任何异常 exit 1。"""
import sys
import json
from pathlib import Path

import numpy as np

for p in ['..', '../simulation', '../城市低空电磁信号仿真平台/src', '.']:
    rp = str(Path(p).resolve())
    if rp not in sys.path:
        sys.path.insert(0, rp)

import capture_tools as ct  # noqa: E402
import run_capture_agent as rca  # noqa: E402

# stdout 污染哨兵：工具调用不应往 stdout 打东西（污染 Agent 的工具输出）；
# 重定向 stdout 收集，若非空则在末尾报失败。
import io as _io
_real_stdout = sys.stdout

ROOT = Path('../dataset_d2')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
idx = json.loads((ROOT / 'index.json').read_text(encoding='utf-8'))
fails = []
for s in idx['samples'][:N]:
    sid = s['id']
    cap = ROOT / sid / 'capture'
    obs = json.load(open(cap / 'public/observations.json'))
    refs = rca._build_target_reference(cap / 'private',
                                       obs['receivers'][0]['shape'][1])
    aoi = np.mean([r['receiver_position_m'] for r in obs['receivers']],
                  axis=0).tolist()
    ctx = ct.ToolContext(
        capture_dir=cap / 'public', cache_dir=ROOT / '_agent_cache' / sid,
        allowed_stations=[r['receiver_id'] for r in obs['receivers']],
        fs=obs['receivers'][0]['sample_rate_hz'],
        fc=obs['receivers'][0]['center_frequency_hz'],
        target_bw_normalized=0.25, aoi_center=aoi, target_reference=refs)
    ct.set_context(ctx)
    _cap = _io.StringIO()
    sys.stdout = _cap
    try:
        clu = ct.estimate_sources_by_spatial_clustering(on_residual=True)
        assert 'error' not in clu, f"聚类 error: {str(clu.get('error'))[:60]}"
        out = ct.extract_source_waveforms()
        assert 'error' not in out, f"提取 error: {str(out.get('error'))[:60]}"
        assert out.get('waveform_files'), "无波形缓存"
        n_c, n_w = len(clu.get('candidates', [])), sum(len(v) for v in out['waveform_files'].values())
    except Exception as e:
        sys.stdout = _real_stdout
        fails.append((sid, f"{type(e).__name__}: {str(e)[:60]}"))
        continue
    finally:
        sys.stdout = _real_stdout
    leaked = _cap.getvalue().strip()
    if leaked:
        fails.append((sid, f"stdout 污染: {leaked[:60]!r}"))
    print(f"{sid}: 候选 {n_c} / 波形 {n_w} 份 ✓")
if fails:
    print('FAIL:')
    for f in fails:
        print(' ', f)
    sys.exit(1)
print(f'S2 冒烟通过（{N} 样本，含聚类+提取断言）')
