# 电磁干扰分析 Agent Benchmark（EM-Jam Agent Benchmark）

> 面向**智能电磁干扰分析大模型 Agent** 的评估基准：为 LLM Agent 提供多源、多天线 IQ 信号数据集与评估链路，衡量其在干扰检测、参数估计、分类与定位上的能力。

---

## 1. 项目定位与演进

本项目目标是构建一个可复现、可扩展的电磁信号分析基准，用于评估 LLM Agent 的物理层信号分析能力（而非纯文本推理）。项目分阶段演进：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **V1** | 单天线、4 种简单干扰类型，跑通评估链路，验证"Agent 能做信号分类" | ✅ 已完成（历史代码） |
| **V2（当前）** | 4 阵元 ULA、多源（1 目标 + 0~2 干扰）、复杂干扰类型（脉冲化 LFM / NFM / 周期脉冲串等）、物理保真（LNA 饱和 / 阵元失配 / 防折叠 / 带宽统一），完整评估链路（检测 / 分类 / 参数估计 / DOA） | ✅ 完成（500 样本最终基线 ds_run15） |
| **V3（启动中）** | 真实环境：多径衰落、IQ 不平衡、IMD3、真实采集数据 | ⏳ P0 真实 IQ 采集管线启动中 |

**V2 信号模型的核心修复**（详见 `simulation/README.md`）：

- 干扰类别可触发（blocking 由 LNA 饱和真实呈现，而非仅标签）
- 奈奎斯特防折叠约束（干扰完整频带不越界）
- 带宽定义统一为 **99% 占用带宽（OBW99，实测）**，元数据与工具估计直接可比
- SNR = 目标功率 / 噪声功率；干扰可检测性约束（INR ≥ 3 dB）
- 真升余弦成形（符号中心无 ISI）；GFSK 带宽实测标定；OFDM 频谱整形

---

## 2. 目录结构

```text
EM-benchmark/
├── benchmark_v1/                     # V1 历史代码（单天线、4 类干扰）
└── benchmark_v2/                     # V2（当前，本目录）
    ├── dataset/                      # 500 样本冻结数据集（.npy + ground_truth/metadata/observations）
    ├── simulation/                   # 信号模型与数据生成
    │   ├── generate_dataset_v2.py    # 数据集生成 CLI
    │   ├── sanity_check_v2.py        # 全量验证脚本（90+ 项检查）
    │   ├── README.md                 # 信号模型详细文档（架构/算法/参数）
    │   └── em_signal_simulator/      # 信号模型库
    │       ├── baseband.py           # 目标基带波形（9 种调制 + OBW99 测量）
    │       ├── jamming.py            # 干扰波形（5 种类型）
    │       ├── channel.py            # 信道效应（频偏/阵列/LNA 饱和/失配/AWGN）
    │       ├── factory.py            # 样本工厂（多源叠加/分离修正/打标/元数据）
    │       └── visualization.py      # 时频图 + MUSIC 空间谱（VLM 支持）
    ├── tools_v2.py                   # Agent 工具集（频谱/源数/DOA/时域/调制特征）
    ├── evaluator_v2.py               # 评估器（存档/断点续跑/自动重试）
    ├── run_bailian_deepseek.py       # 百炼 deepseek-v3.2 接入封装（enable_thinking=False）
    ├── eval_mod_upperbound.py        # 调制特征上限离线验证（零 token，C40/C42 消融用）
    ├── diag_offline_report.py        # 全量离线诊断（源数/频偏/类别/幽灵，零 token）
    └── ds_run15.json                 # 500 样本最终基线结果
```

> 约定：**信号模型**（数据生成）在 `simulation/em_signal_simulator/`；**评估侧代码**（`evaluator_v2.py`、`tools_v2.py`）位于 `benchmark_v2/` 根目录，与信号模型解耦。

---

## 3. 快速开始

```bash
# 0) 依赖
pip install numpy scipy matplotlib pillow openai tqdm

# 1) 生成数据集（可选——仓库已自带冻结的 500 样本 dataset/，跳过此步可直接评估）
cd simulation
python generate_dataset_v2.py --count 500 --output-dir ../dataset \
    --num-sources random --snr-range -5 15 --num-workers 4

# 2) 数据健康检查
python sanity_check_v2.py

# 3) 评估（回到 benchmark_v2/ 根目录）
cd ..
python evaluator_v2.py dataset --offline                # 离线：验证评分管道（应满分）
python run_bailian_deepseek.py dataset --model deepseek-v3.2 --max-samples 50 --output ds_runX.json
```

---

## 4. 数据集生成（`simulation/generate_dataset_v2.py`）

输出：每个样本 `(4, 1024)` complex128 IQ + 三个 JSON（信息隔离）：

| 文件 | 内容 | 谁用 |
|------|------|------|
| `metadata.json` | 完整记录（含真值） | 内部调试，**不得提供给 Agent** |
| `observations.json` | Agent 可见先验（接收机参数、目标调制、样本长度） | Agent Prompt 输入 |
| `ground_truth.json` | 评估真值（源数、类别、CFO/DOA/带宽/功率/INR、LNA 状态） | 评估系统专用 |

主要 CLI 参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--count` / `--output-dir` | 100 / `dataset_v2` | 数量与输出目录 |
| `--num-sources` | random | 总源数 1/2/3 或随机 |
| `--modulation` | random | 目标调制（9 种） |
| `--interferer-type` | random | 干扰类型（真实调制 5 种 / 波形 5 种） |
| `--snr-range` / `--fixed-snr` | -5~15 | SNR（**= 目标/噪声**，干扰不参与） |
| `--target-bandwidth` | None | 目标占用带宽（归一化，须 <1.0） |
| `--interferer-bandwidth-range` | 0.05~0.6 | 干扰带宽采样范围 |
| `--lna-saturation-db` / `--lna-p` | 6 / 2 | LNA 饱和门限与平滑系数 |
| `--array-gain-error-std` / `--array-phase-error-std-deg` | 0 / 0 | 阵元失配（默认关闭） |
| `--min-inr-db` | 3 | 干扰可检测性下限（INR = 功率比 + SNR） |
| `--blocking-power-threshold-db` / `--blocking-ratio-threshold` | 10 / 0.5 | blocking 判定阈值 |
| `--num-workers` | 1 | 并行生成进程数（结果与串行逐位一致） |
| `--progress` | 关 | 显示 tqdm 进度条（实时查看生成进度） |

信号模型细节、元数据字段定义、物理参数映射：见 **[`simulation/README.md`](simulation/README.md)**。

---

## 5. 评估链路（`tools_v2.py` + `evaluator_v2.py`）

### 5.1 Agent 工具集（`tools_v2.py`）

多天线适配的分析工具，只返回**测量值**，不替 Agent 下结论：

| 工具 | 功能 |
|------|------|
| `analyze_spectrum` | 多天线频谱：纹波合并 + **谱谷分裂**（区分近距双源；目标带保护区防宽带目标自裂）+ 强度地板（-10dB 滤噪声包）后的源峰检测 + **每源 `sources_candidates`**（频偏/带宽/**ratio_to_target_bw**/**相对目标功率**（带内积分 + 4.5dB 总功率标定）/**near_category_boundary** 边界提示，类别判定所需测量由工具算好） |
| `estimate_num_sources` | 源数估计：`num_sources_estimate`（MDL 原始值）与 **`final_suggestion`**（决策树已在工具端预应用：高估看合并峰+稳定峰；低估看"可信组"——强组或**空间相干组**（带限切片 λ1/其余均值 ≥4，真源 p10=7.2 vs 噪声包 p75=3.6），MUSIC 峰经 25dB 显著度过滤） |
| `estimate_doa` | MUSIC 空间谱 DOA 估计（4 元 ULA，d=λ/2）；含跨阶稳定峰 `stable_peaks_deg` 与 3 阶新峰 `peaks_new_at_order3_deg`（区分真实源与过分辨伪峰） |
| `detect_time_domain` | 时域特征：PAPR、脉冲占空比、脉冲边沿数、**削顶比例 `clipping_fraction`**（LNA 饱和压平波峰，blocking 的独立物理证据：blocking 样本中位 0.027 vs 其他 ~0.01） |
| `estimate_modulation_features` | 调制识别特征：频谱平坦度/频率漂移/峰均比/幅度峰度 + 与 **10 种候选模板**（数据集实际干扰池）的带宽自适应特征距离（Hann 切片解旋，模板同带宽生成，仅返回最近 top-3） |

工具设计沿用 v1 多轮测试验证有效的特征（频谱平坦度→宽带、分段频率漂移→扫频、峰均比→单音、PAPR/占空比→脉冲），且**只忠实返回测量值、不含任何结论性文本（无 expert_insight）**——判断完全交给 Agent。类别判定所需的 `ratio`（干扰频偏/目标带宽）、目标参考功率与边界提示由工具直接计算输出，模型读区间即可，无需自行算术。

评估 Prompt 会注入完整先验（采样率、中心频率、半波长阵列间距、**目标调制与目标带宽（OBW99）**、观测长度）并明确归一化约定（频偏/带宽相对采样率、DOA 单位度）；目标带宽先验同时注入三个测量工具用于谱谷保护/切片。

参考精度（40 样本验证）：频偏估计命中率 0.94（±0.02 容差）、DOA 命中率 1.00（±12°，INR≥8 时）、源数估计 0.80。调制识别在多源混合场景下物理偏难（干扰与目标频带重叠时不可分离），模板匹配仅供参考特征。

### 5.2 评估器（`evaluator_v2.py`）

任务形式为**填空/选择题**：

| 任务 | 形式 | 评分指标 |
|------|------|----------|
| 干扰检测（源数） | 选择题 0/1/2 | 源数准确率 |
| 干扰分类（同频/邻道/脉冲/阻塞） | 选择题 | 类别准确率（贪心匹配）+ 按 SNR 分组 |
| 参数估计（频偏 / 带宽） | 填空题 | 容差命中率（±0.02 / ±25%）+ MAE |
| 干扰源定位（DOA） | 填空题 | 容差命中率（±10°）+ MAE |
| 调制类型（可选） | 选择题 | 真实调制类干扰的调制准确率 |

```bash
python evaluator_v2.py dataset --offline               # 验证评分管道（应全满分）
python run_bailian_deepseek.py dataset --model deepseek-v3.2 --output eval_report.json
python evaluator_v2.py dataset --model ... --max-samples 200   # 抽样评估
```

评估器默认显示 **tqdm 进度条**（实时累计源数/类别准确率）；`--verbose` 逐样本打印，`--no-progress` 关闭进度条。

长跑可靠性（全量 500 约 5~6 小时）：
- **定期存档**：每 10 个样本自动写入输出文件（`partial: true`），崩溃/断网最多丢最近 10 个样本；
- **断点续跑**：同命令加 `--resume`，已完成样本直接跳过、不重复调用 API；
- **网络重试**：连接错误/限流/服务端 5xx 指数退避自动重试（5s→60s，最多 5 次），短暂断网不再中断整跑；
- **repair 轮**：模型最终回复缺有效 JSON 时追加一次无工具的强制格式调用（历史上曾单轮丢失 8/50 样本的全部指标）。

**信息隔离**：Agent 只能看到 `observations.json` 先验与 `.npy` 路径；真值仅在评估侧从 `ground_truth.json` 读取。

**调制双口径**：`modulation_accuracy`（仅匹配对）与 `modulation_accuracy_e2e`（分母=全部真实调制 GT 干扰，未匹配计错）。跨 run 对比请以 e2e 为准——匹配数波动会使 matched 口径失真。报告含 `code_fingerprint`（评估侧代码 md5 前 12 位），用于精确归因版本。

### 5.3 云端 API 评估（百炼 deepseek-v3.2 —— 本项目全部基线的实际配置）

```powershell
# 方式一：薄封装（推荐）——自动注入 enable_thinking=False，输出名自动顺延 ds_run(N+1)
python run_bailian_deepseek.py dataset --model deepseek-v3.2 --max-samples 50 --output ds_runX.json

# 方式二：直接评估（v3.2 非推理、默认关思考，可直接调 evaluator）
python evaluator_v2.py dataset --model deepseek-v3.2 --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --max-samples 500 --output ds_run15.json

# 500 样本全量约 5 小时：每 10 样本自动存档（partial: true），中断后同命令加 --resume 续跑
```

**模型选择**：

| 模型名 | 行为 | 建议 |
|---|---|---|
| **`deepseek-v3.2`**（百炼） | 非推理、默认关思考，~18s/样本，工具调用稳定 | ✅ **首选**（v2 全部基线均用它；官方 `deepseek-chat` 别名同期即指向 v3.2，已互验） |
| `deepseek-v4-flash` 等推理模型 | 深度推理 | ⚠️ 工具模式下不限制输出会生成超长文本（单样本可达 2 分钟+），仅用于对比推理上限；可用 `--reasoning-effort low` 降低思考强度（API 不支持时自动忽略） |

- API Key：阿里云百炼（DashScope）控制台创建，放环境变量 `OPENAI_API_KEY` 或写入一键脚本 `run_v32.ps1`（**含 Key，已 gitignore，勿提交远程**）；
- `--resume` / `--max-samples` / `--verbose` 等参数见 `python evaluator_v2.py --help`。

---

## 6. 可视化（VLM 支持）

`simulation/em_signal_simulator/visualization.py` 导出**时频图**与 **MUSIC 空间谱图**（PNG / RGB 数组），支持 VLM 类 Agent 的视觉模态：

```python
import numpy as np
from em_signal_simulator.visualization import spectrogram_image, music_spectrum

iq = np.load("dataset_v2/sample_00000.npy")            # (4, 1024)
spectrogram_image(iq, sampling_rate_hz=20e6, path="spec.png")
doas, spectrum = music_spectrum(iq, num_sources=3)      # MUSIC 空间谱数值
```

---

## 7. 验证

```bash
cd simulation && python sanity_check_v2.py
```

覆盖（90+ 项断言）：LNA 饱和触发、脉冲成形边界/无 ISI、干扰波形、参数分离、阵元失配、带宽配置与 OBW99 一致性、SNR 语义、blocking 触发、防折叠、INR 约束、数据隔离、可视化、评估链路离线满分。（注：多进程一致性测试在受限环境自动 SKIP，请于正常环境验证。）

### 实时进度一览

| 环节 | 进度显示 |
|------|----------|
| 数据集生成（`generate_dataset_v2.py`） | `--progress` 开启 tqdm 进度条 |
| 验证（`sanity_check_v2.py`） | 逐条实时输出 `[PASS]/[FAIL]` |
| 评估（`evaluator_v2.py`） | 默认 tqdm 进度条（实时累计 src_acc/cat_acc）；`--verbose` 逐样本 |

---

## 8. 文档索引

| 文档 | 内容 |
|------|------|
| `simulation/README.md` | 信号模型：架构、算法公式、参数表、元数据字段、物理映射 |
| 本文档 | 整体结构、快速开始、评估链路、CLI 概览 |

## 9. Roadmap（V3 规划）

- 真实信道：多径衰落（瑞利/莱斯）、IQ 不平衡、三阶互调（IMD3）
- 评估升级：参数估计精度曲线、难度分级、经典 DSP 基线对照
- 数据集规模：多进程大规模生成 + 版本化配置快照

---

## 10. 附录：V2 评估结果记录（截至 2026-08）

> 数据集：同 seed 生成，50 样本快速验证集 + 500 样本全量集（1 目标 + 0~2 干扰，SNR -5~15dB）。
> 模型口径：所有 DeepSeek 行统一为百炼 **deepseek-v3.2**（非推理、默认关思考）。早期行标注的官方别名 `deepseek-chat` 当时段即指向 v3.2（已用官方 chat 与百炼 v3.2 互验，指标重合）。
> 评估方式：填空/选择题，Agent 调用 5 个分析工具（频谱/源数/DOA/时域/调制特征）后作答。

### 10.1 各配置精度演进

| 配置 | 模型 | 源数 | 类别 | 频偏 | 带宽 | DOA | 调制(e2e) | 耗时/token |
|---|---|---|---|---|---|---|---|---|
| 早期（4 工具，30 样本） | v4-flash | 0.63 | 0.28 | 0.53 | 0.32 | 0.67 | 0.02 | 40min / 65w |
| 第 1 版（4 工具） | deepseek-v3.2 | 0.60 | 0.34 | 0.43 | 0.30 | 0.67 | 0.00 | 15min / 30w |
| 第 2 版（5 工具 + per-peak 带宽） | deepseek-v3.2 | 0.56 | 0.36 | 0.41 | 0.28 | 0.59 | 0.04 | ~15min |
| 第 3 版（5 工具 + 合并峰修复） | deepseek-v3.2 | 0.64 | 0.34 | 0.41 | 0.44 | 0.56 | 0.14 | ~15min / 30w |
| 第 4 版（repair 轮 + 谱谷分裂/保护/地板 + 决策树相干门，50 样本） | deepseek-v3.2 | **0.72** | **0.48** | 0.59 | 0.46 | **0.64** | 0.32 / **0.19** | ~33min |
| 第 4 版全量基线（500 样本，blocking 功率参考 bug 修复**前**，ds_run13） | deepseek-v3.2 | 0.65 | 0.57 | 0.59 | 0.47 | **0.68** | 0.31 / 0.16 | 4h04m |
| **第 5 版全量最终基线（500 样本，blocking 修复后，ds_run15）** | deepseek-v3.2 | **0.68** | **0.588** | **0.609** | 0.476 | **0.678** | 0.31 / **0.166** | ~5h |

> 第 4 版修复链：①解析失败 repair 轮；②谱谷分裂 + 目标带保护 + 强度地板（候选覆盖 0.39→0.57）；③源数决策树内移工具端 + 空间相干可信门（消灭干净样本幽灵干扰）；④调制模板收窄到实际干扰池。
> **第 5 版（当前）**：blocking 功率参考 bug 修复（旧版以全局主峰为参考，blocking 恒测得 0dB、规则永不触发：r12 匹配对命中 **0/13**）——功率参考改为**目标带内 PSD 积分** + 4.5dB 总功率标定（不依赖分组，同时解决强干扰下目标组被地板滤掉）；新增削顶比例特征（LNA 饱和的独立物理证据）。50 样本验证（r14）：类别 0.48→**0.56**，blocking 匹配对命中 11/14=**79%**；**500 样本最终基线（ds_run15）：类别 0.588（较 r13 +0.019）、源数 0.68、频偏命中 0.609、DOA MAE 3.63°（历史最佳）、调制 e2e 0.166**。
> 已知代价与限制：①假 blocking 约 9 个/50 样本（其中 7 个为标定功率本身 ≥10dB 的测量模糊区，±4dB 精度 vs 10dB 门限的固有权衡）；②压缩特征在全信号/切片层面均无法区分真假 blocking（OOK 等目标频谱泄漏污染），仅作 5~10dB 边界带辅助证据；③宽带干扰候选覆盖 0.39、低 SNR 源数、16QAM@低SNR 幽灵为物理极限，待 V3。

> 工具层参考精度：频偏估计 0.94（±0.02 容差）、DOA 1.00（±12°，INR≥8）、源数（MDL）0.80——即**工具测量本身很准，Agent 综合判断是主要损耗点**。

### 10.2 关键指标细节（最终基线 ds_run15，500 样本）

- **参数估计数值精度**：频偏 MAE 0.0246（容差 ±0.02）、DOA MAE 3.63°（容差 ±10°，命中 0.678）、带宽命中 0.476（相对误差 0.599）；
- **源数（源级匹配统计）**：漏检 231 / 误报 297；低 SNR 组为 MDL 物理极限；平均工具调用 5.06/样本，工具调用率 100%；
- **类别按 SNR**：Low 0.524 / Mid 0.610 / High 0.599（n=104/200/196）——高 SNR 不高于 Mid，说明残余误差含系统性成分（blocking 假阳性、宽带纹波分裂），并非纯物理极限；
- **调制识别**：e2e 0.166 vs 同数据集特征上限 0.266（`eval_mod_upperbound.py` 离线验证）——7 维特征下 QPSK/GFSK ~4%、LFM ~14%，升级方向为 C40/C42 高阶累积量 + 包络统计。

### 10.3 结论

1. **结构性问题已根除**：blocking 永远 0 分的根因（全局主峰功率参考）修复后，类别 0.57→**0.588**@500、blocking 召回 0→**79%**，源数 0.65→0.68；
2. **剩余瓶颈（均已定位）**：blocking 假阳性 ~18%（±4dB 测量精度 vs 10dB 门限的固有权衡）、宽带干扰候选覆盖 0.39（固定间距分组失效）、低 SNR 源数（MDL 物理极限）、调制 e2e 0.166/上限 0.266（7 维特征物理不可分）；
3. **下一步（V3）**：仿真侧收益递减、继续打磨有过拟合仿真器风险——优先真实 IQ 采集管线（P0）与域适配（P1）；调制特征升级（C40/C42/包络统计）已有零 token 离线验证管线（`eval_mod_upperbound.py`），先在仿真侧消融定型。
