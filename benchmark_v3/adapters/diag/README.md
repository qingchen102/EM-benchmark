# 诊断脚本区（`benchmark_v3/adapters/diag/`）

一次性**只读**诊断脚本的仓库。这些脚本不是评测链路的一部分——评测链路的入口只有三个：

| 入口 | 作用 |
|---|---|
| `adapters/regen_noise.py` | 用已落盘信道重跑接收链（噪声重放 / 换装），不碰 RT |
| `adapters/eval_dataset.py` | GT 回放（应全 1.0）与对答案目录评分（注册口径） |
| `rad_cls/oracle16.py` | 仪器 gate：clean / GT 切片 / tool-view 三视角 |

`diag/` 里的脚本用于**查证据**：核对某个数字是怎么来的、复现某次归因、审计数据集与工具。绝大多数零 token、不调 API。

## 目录约定（2026-10-11 整理）

```
benchmark_v3/adapters/diag/          # 当代（2026-10-09 之后）；按可用性排序
benchmark_v3/adapters/diag/legacy/   # 旧轮次（2026-09-24 ~ 10-08）；多数依赖已不存在的中间目录
benchmark_v3/evidence/               # 所有脚本的原始输出（.txt / .json）：166 份
```

- **就地运行**：从 `benchmark_v3/adapters/` 下执行，路径参数与脚本内相对路径都以该目录为基准：
  ```powershell
  cd D:\srtp\EM-benchmark\benchmark_v3\adapters
  python -X utf8 -u diag\_probe_targetvis.py            # 例：目标在哪些站是 0 径
  python -X utf8 -u diag\_fix_abstain_flag.py ../dataset_d3 S2 --dry-run
  ```
- **自举注入**：归位时给每个当代脚本顶部插入了两行，使 `import capture_tools as ct` 之类的裸导入仍然可用：
  ```python
  import sys as _sys, pathlib as _pl
  _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
  ```
  没有这两行的旧脚本（尤其是 `legacy/`）需要自己在 `adapters/` 下运行，或手动补同样的两行。
- **输出位置**：当代脚本写盘时统一写到 `benchmark_v3/evidence/`（如 `evidence/_probe_targetvis.txt`），不再在 `benchmark_v3/` 根散落文件。
- **`legacy/` 的可用性**：这些脚本当时引用的中间目录（`_regen_*`、`_pre_*`、`_ab_d2` 等）在 2026-10-11 大清理中多数已删除，因此**跑不通是正常的**；保留它们只为让旧报告里的数字可追溯（对照 `开发日志.md` 与 `benchmark_v3/docs/BENCHMARK_STATUS_*.md`）。
- **不要**把一次性脚本放回 `adapters/` 根——那会让"生产代码"与"查证据的脚本"再次混在一起（这正是本次整理要解决的问题）。

## 有代表性的几个

| 脚本 | 用途 |
|---|---|
| `_probe_targetvis.py` | 逐站目标链路可用性（missing / invisible / ok），解释 gate 的 GT 切片分母漂移 |
| `_probe_refmatch.py` | `matched_energy_fraction`：判断盘上 IQ 与运行时参考是否同源（D1 自伤的根因工具） |
| `_diag_d1_evidence.py` | D1 逐样本并排打印门控峰 / legacy 峰 / GT 数 / 各次 agent 跑的报数与拒答 |
| `_fix_abstain_flag.py` | 离线改正存量答案文件里被误标的 `abstained`（只读+改写答案 JSON，支持 `--dry-run`） |
| `_check_noise_landed.py` | 噪声档位落盘审计（配方、实测底 vs 解析抬升、派生 INR/SNR） |
| `_diag_d2_rank.py` | D2 窄带特征值秩枚举（第九条证伪路线） |
| `_an_abstain_why.py` | 拒答取证：从 `_run_log.json` 的 `raw_output` 里回溯模型原文的负数哨兵 |
