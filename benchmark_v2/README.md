# 电磁干扰分析 Agent Benchmark（EM-Jam Agent Benchmark）

> 面向**智能电磁干扰分析大模型 Agent** 的评估基准：为 LLM Agent 提供多源、多天线 IQ 信号数据集与评估链路，衡量其在干扰检测、参数估计、分类与定位上的能力。

---

## 1. 项目定位与演进

本项目目标是构建一个可复现、可扩展的电磁信号分析基准，用于评估 LLM Agent 的物理层信号分析能力（而非纯文本推理）。项目分阶段演进：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **V1** | 单天线、4 种简单干扰类型，跑通评估链路，验证"Agent 能做信号分类" | ✅ 已完成（历史代码） |
| **V2（当前）** | 4 阵元 ULA、多源（1 目标 + 0~2 干扰）、复杂干扰信号体制（真实辐射体简化建模：WiFi/LTE/蓝牙/雷达/GPS + 脉冲串/NFM 等）、物理保真（LNA 饱和 / 阵元失配 / 防折叠 / 带宽统一），完整评估链路（检测 / 分类 / 参数估计 / DOA） | ✅ 完成（v7 学习型体制分类器，500 样本基线 ds_run18） |
| **V3（规划）** | 真实环境建模：多径衰落、IQ 不平衡、IMD3、低空场景（参考 S-ICDF 等低空电磁数据集的建模思路） | ⏳ 规划中 |

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
    ├── spatial_candidates.py         # MVDR 频率×角度候选生成（v6 候选层）
    ├── diag_spatial_ab.py            # 空间 vs 一维候选离线 A/B（零 token）
    ├── mod_classifier/               # 学习型体制分类器（v7：训练/验收管线 + 两级 oracle）
    └── ds_run18.json                 # 500 样本最终基线结果（v7）
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
python run_bailian_deepseek.py dataset --model deepseek-v3.2 --max-samples 50 --output results.json
```

---

## 4. 数据集生成（`simulation/generate_dataset_v2.py`）

输出：每个样本 `(4, 1024)` complex128 IQ + 三个 JSON（信息隔离）：

| 文件 | 内容 | 谁用 |
|------|------|------|
| `metadata.json` | 完整记录（含真值） | 内部调试，**不得提供给 Agent** |
| `observations.json` | Agent 可见先验（接收机参数、目标调制、样本长度） | Agent Prompt 输入 |
| `ground_truth.json` | 评估真值（源数、类别、CFO/DOA/带宽/功率/INR、LNA 状态） | 评估系统专用 |

**干扰池构成与标签轴**：每个干扰由真实辐射体简化建模——用其代表性调制生成信号（WiFi→OFDM、LTE/基站→QPSK、蓝牙→GFSK、雷达→LFM、GPS/遥测→DSSS/BPSK，完整辐射体表见 `simulation/README.md`），另有模拟调频（NFM）与 4 种未调制简单波形（单音/扫频/脉冲串/宽带噪声）。每个源带三个独立标签轴：

| 标签轴 | 回答的问题 | 字段 |
|------|------|------|
| 干扰类别 | 在频谱什么位置、多强、构成什么威胁 | `v2_category`（同频/邻道/脉冲/阻塞/none） |
| 信号体制 | 它是什么信号 | `modulation`（调制方式）+ `waveform_type`（未调制波形） |
| 辐射体来源 | 现实中对应什么设备 | `display_name`（如 Bluetooth_GFSK，仅评估侧可见，信息隔离） |

> 注：当前"调制识别"任务本质是**辐射体溯源的简化代理**——每种辐射体 ↔ 一种代表性调制，一对一。真实世界中调制与设备并非一一对应，跳频（真实蓝牙为 FHSS）、突发包结构等旁证尚未建模，为 V3 扩展方向。

主要 CLI 参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--count` / `--output-dir` | 100 / `dataset_v2` | 数量与输出目录 |
| `--num-sources` | random | 总源数 1/2/3 或随机 |
| `--modulation` | random | 目标调制（9 种） |
| `--interferer-type` | random | 干扰池（真实辐射体 5 种：WiFi/LTE/蓝牙/雷达/GPS/遥测；简单波形 5 种：单音/扫频/脉冲串/宽带噪声/NFM） |
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
| `analyze_spectrum` | 多天线频谱：**默认 MVDR 频率×角度联合候选生成**（每频点空间协方差 + Capon 波束扫描，功率谱上熔住的源按方位角分辨；候选含 freq/bandwidth/**angle_deg**/ratio_to_target_bw/**相对目标功率**（源自身波束内积分）/near_category_boundary）+ `use_spatial=False` 回退旧版一维合并谱找峰（纹波合并 + 谱谷分裂 + 强度地板，兼容字段仍输出） |
| `estimate_num_sources` | 源数估计：`num_sources_estimate`（MDL 原始值）与 **`final_suggestion`**（决策树已在工具端预应用：高估看合并峰+稳定峰；低估看"可信组"——强组或**空间相干组**（带限切片 λ1/其余均值 ≥4，真源 p10=7.2 vs 噪声包 p75=3.6），MUSIC 峰经 25dB 显著度过滤） |
| `estimate_doa` | MUSIC 空间谱 DOA 估计（4 元 ULA，d=λ/2）；含跨阶稳定峰 `stable_peaks_deg` 与 3 阶新峰 `peaks_new_at_order3_deg`（区分真实源与过分辨伪峰） |
| `detect_time_domain` | 时域特征：PAPR、脉冲占空比、脉冲边沿数、**削顶比例 `clipping_fraction`**（LNA 饱和压平波峰，blocking 的独立物理证据：blocking 样本中位 0.027 vs 其他 ~0.01） |
| `estimate_modulation_features` | **信号体制识别**（答案空间 10 类 = 5 种辐射体调制 BPSK/QPSK/GFSK/LFM/OFDM + 模拟调频 NFM + 4 种简单波形）：默认学习型分类器后端（1D-CNN，feature_distance = 1 − softmax 概率，top-3；两级 oracle 验收 clean 0.873 / 切片 0.688），`use_cnn=False` 回退手工特征模板法。调制为辐射体溯源的简化代理（见第 4 节标签轴说明） |

工具设计沿用 v1 验证有效的特征（频谱平坦度→宽带、分段频率漂移→扫频、峰均比→单音、PAPR/占空比→脉冲），且**只忠实返回测量值、不含任何结论性文本**——判断完全交给 Agent。类别判定所需的 `ratio`（干扰频偏/目标带宽）、目标参考功率与边界提示由工具直接计算输出，模型读区间即可，无需自行算术。

评估 Prompt 会注入完整先验（采样率、中心频率、半波长阵列间距、**目标调制与目标带宽（OBW99）**、观测长度）并明确归一化约定（频偏/带宽相对采样率、DOA 单位度）；目标带宽先验同时注入三个测量工具用于谱谷保护/切片。

工具层参考精度（500 样本离线诊断，`diag_offline_report.py` / `diag_spatial_ab.py`）：候选覆盖率 0.87、功率测量 MAE 1.8 dB、频点 MAE 0.005、角度中位误差 0.6°。调制识别为已知短板：多源混叠下手工特征物理受限（同数据集离线上限仅 0.27，`eval_mod_upperbound.py` 可复现），模板匹配仅作参考证据。

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
- **repair 轮**：模型最终回复缺有效 JSON 时追加一次无工具的强制格式调用，避免整样本指标丢失。

**信息隔离**：Agent 只能看到 `observations.json` 先验与 `.npy` 路径；真值仅在评估侧从 `ground_truth.json` 读取。

**调制双口径**：`modulation_accuracy`（仅匹配对）与 `modulation_accuracy_e2e`（分母=全部真实调制 GT 干扰，未匹配计错）。对比不同结果文件时请以 e2e 为准——匹配数波动会使 matched 口径失真。报告含 `code_fingerprint`（评估侧代码 md5 前 12 位），用于精确归因版本。

### 5.3 云端 API 评估（阿里云百炼）

```powershell
# 方式一：薄封装（推荐）——自动注入 enable_thinking=False，保证非推理配置一致
python run_bailian_deepseek.py dataset --model deepseek-v3.2 --max-samples 50 --output results.json

# 方式二：直接评估（deepseek-v3.2 非推理、默认关思考，可直接调 evaluator）
python evaluator_v2.py dataset --model deepseek-v3.2 --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --max-samples 500 --output results.json

# 500 样本全量约 5 小时：每 10 样本自动存档（partial: true），中断后同命令加 --resume 续跑
```

**模型选择**：

| 模型名 | 行为 | 建议 |
|---|---|---|
| **`deepseek-v3.2`**（百炼） | 非推理、默认关思考，~18s/样本，工具调用稳定 | ✅ **首选** |
| `deepseek-v4-flash` 等推理模型 | 深度推理 | ⚠️ 工具模式下不限制输出会生成超长文本（单样本可达 2 分钟+），仅用于对比推理上限；可用 `--reasoning-effort low` 降低思考强度（API 不支持时自动忽略） |

- API Key：在阿里云百炼（DashScope）控制台创建，通过环境变量 `OPENAI_API_KEY` 或 `--api-key` 传入。请勿将 Key 写入任何会提交到仓库的文件；
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
> 模型：deepseek-v3.2（阿里云百炼，非推理、默认关思考），所有 DeepSeek 行同口径。
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
| 第 5 版全量基线（500 样本，blocking 修复后，ds_run15） | deepseek-v3.2 | 0.68 | 0.588 | 0.609 | 0.476 | 0.678 | 0.31 / 0.166 | ~5h |
| 第 6 版全量基线（500 样本，MVDR 频率×角度候选层，ds_run17） | deepseek-v3.2 | 0.696 | 0.754 | 0.836 | 0.745 | 0.868 | 0.27 / 0.201 | 4h50m |
| **第 7 版全量最终基线（500 样本，学习型体制分类器，ds_run18）** | deepseek-v3.2 | **0.714** | **0.752** | **0.836** | **0.740** | **0.876** | 0.45 / **0.341** | 4h02m |

> 第 4 版修复链：①解析失败 repair 轮；②谱谷分裂 + 目标带保护 + 强度地板（候选覆盖 0.39→0.57）；③源数决策树内移工具端 + 空间相干可信门（消灭干净样本幽灵干扰）；④调制模板收窄到实际干扰池。
> **第 5 版（ds_run15）**：blocking 功率参考 bug 修复（旧版以全局主峰为参考，blocking 恒测得 0dB、规则永不触发：r12 匹配对命中 **0/13**）——功率参考改为**目标带内 PSD 积分** + 4.5dB 总功率标定（不依赖分组，同时解决强干扰下目标组被地板滤掉）；新增削顶比例特征（LNA 饱和的独立物理证据）。500 样本：类别 0.588、源数 0.68、DOA MAE 3.63°。
> **第 6 版（ds_run17）**：候选层升级——`analyze_spectrum` 默认改用 **MVDR 频率×角度联合分析**（`spatial_candidates.py`）：分段 STFT 估计每频点空间协方差 + Capon 波束扫描，功率谱上熔住的源按方位角分辨（利用生成约束 Δfreq≥0.15 或 ΔDOA≥15° 保证的角度可分性；旧一维路径把该维度在功率合成一步丢掉，是 v5 候选覆盖 57% 的根因），候选新增 `angle_deg`，功率在源自身波束内测量（目标被零陷，4.5dB 标定常数不再需要）。500 样本离线 A/B（`diag_spatial_ab.py`）：候选覆盖 57%→**87%**（宽带 39%→**72%**）、功率 MAE 4.8→**1.8dB**、干净样本幽灵 49→**8**。端到端：类别 +0.166、DOA MAE ÷3.8、漏检 231→**107**。
> **第 7 版（当前，ds_run18）**：体制识别后端升级——`estimate_modulation_features` 换用**学习型分类器**（`mod_classifier/`，1D-CNN 0.15M 参数；训练集独立配置生成、种子段/带宽/SNR 范围与考卷隔离，冻结 500 不进训练；两级 oracle 预注册验收 clean **0.873** / 切片 **0.688** 过闸），feature_distance = 1 − softmax 概率，Agent 契约不变，`use_cnn=False` 回退旧后端。旧"手工特征+模板匹配"范式经 oracle 实测干净上限仅 0.66（物理封顶）。端到端：调制 e2e 0.201→**0.341**（1.7×）、matched 0.265→**0.452**，其余五项指标零回退。
> 已知代价与限制（v7 复核）：①体制识别剩余差距在候选覆盖与 LLM 选择（仪器口径 10 类切片 0.688 vs e2e 0.341，e2e 分母含未匹配干扰计错）；②LFM↔swept 标签重叠（swept 默认即脉冲化 LFM，同族信号）为已知类别设计问题；③BPSK↔QPSK 低 INR 混淆为物理限制；④blocking/源数/参数估计与 v6 持平。

> 工具层参考精度：频偏估计 0.94（±0.02 容差）、DOA 1.00（±12°，INR≥8）、源数（MDL）0.80——即**工具测量本身很准，Agent 综合判断是主要损耗点**。

### 10.2 关键指标细节（最终基线 ds_run18，500 样本）

- **参数估计数值精度**：频偏 MAE 0.009（容差 ±0.02，命中 0.836）、DOA MAE 0.87°（容差 ±10°，命中 0.876）、带宽命中 0.740；
- **源数（源级匹配统计）**：漏检 102 / 误报 166（v5：231/297）；平均工具调用 5.1/样本，工具调用率 100%；
- **类别按 SNR**：Low 0.663 / Mid 0.788 / High 0.763（n=104/200/196）；
- **调制识别（体制）**：e2e 0.341 / matched 0.452——学习型后端较模板法 1.7×；仪器口径（冻结集 10 类切片）0.688，与端到端的差距来自候选覆盖与 LLM 选择；训练/验收协议与上限实验见 `mod_classifier/`。

### 10.3 结论

1. **两次架构级仪器修复构成 v2 的主收益**：空间候选层（类别 0.588→0.754、DOA MAE 3.63°→0.87°）与学习型体制分类器（调制 e2e 0.201→0.341）——均按"预注册门槛 + 零 token 离线验收 → 50 样本在线确认 → 500 全量基线"的流程完成，七项指标无一项回退；
2. **剩余瓶颈**：低 INR 弱源覆盖（物理极限）、体制识别的候选覆盖/LLM 选择损耗（仪器 0.688 vs 端到端 0.341）、LFM↔swept 类别重叠（设计问题）、源数未用空间证据；
3. **V3 衔接**：体制识别的"波形→学习型分类器"范式即辐射体溯源的直接模板（补充低空辐射体数据重训即可）；信号模型向低空环境建模扩展（S-ICDF 参考）。仿真侧 v2 至此收口。
