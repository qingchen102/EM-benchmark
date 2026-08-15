# 低空电磁安全 AI Agent 基准测试系统 (EM-Agent-Benchmark)

本项目是一个面向低空电磁安全（EM Security）场景的 AI Agent（智能体）基准测试与自动化评估系统。面对低空复杂电磁环境（包含同频干扰、扫频干扰、突发脉冲及低信噪比噪声等），本系统旨在评估与测试大语言模型智能体（LLM Agent）在不直接具备电磁物理层编码器的情况下，能否通过自主调度 Python 信号分析工具，完成从信号特征提取、干扰类型诊断到防干扰处置建议的完整闭环流程。

## 1. 系统总体架构与设计思想

整个系统采用 **“出题 - 工具 - 监考 - 打分”** 的四层解耦架构设计，确保信号仿真物理层与智能体评估逻辑彻底分离。

```text
benchmark/                          # 项目根目录
├── simulation/                     # [模块 1：出题工厂] 电磁信号与干扰仿真生成器
│   ├── em_signal_simulator/        # 物理层生成算法库
│   │   ├── __init__.py             # 模块 API 导出
│   │   ├── baseband.py             # 基带信号生成 (9 种调制)
│   │   ├── jamming.py              # 干扰信号注入 (5 种干扰)
│   │   ├── channel.py              # 信道物理损伤 (CFO 频偏 + AWGN 噪声)
│   │   └── factory.py              # 样本工厂与 Ground Truth 标注构建
│   ├── generate_dataset.py         # 命令行批量生成数据集工具
│   └── plot_sample.py              # 复数 IQ 采样时频域可视化分析工具
│
├── tools.py                        # [模块 2：智能体工具箱] 暴露给 Agent 的信号分析技能 (Tools)
├── evaluator.py                    # [模块 3：监考打分引擎] 驱动 Agent 评估与多维度指标统计
│
├── dataset/                        # [导出数据] 生成的评估测试集 (.npy 数据 + metadata.json)
├── requirements.txt                # 项目依赖库清单
└── README.md                       # 项目说明文档
```

### 1.1 数据与评估处理流水线 (Pipeline)

系统的工作逻辑遵循 **“物理仿真 → 工具感知 → 智能体推理 → 指标评估”** 的全链路闭环：

```text
[物理仿真 (simulation/)]
   └── 产生基带信号 → 叠加干信比 JSR 干扰 → 注入频偏 CFO 与信噪比 SNR 噪声 → 导出 .npy + metadata.json
                                                                            │
[智能体评估 (evaluator.py)]                                                 │
   ├── 1. 读取 Task Prompt 与 .npy 样本文件路径 ◄───────────────────────────┘
   ├── 2. 将 Prompt 发送给 LLM Agent (OpenAI/Qwen/DeepSeek API)
   ├── 3. 拦截 Agent 的 Tool Calling 指令 (例如调用 analyze_spectrum)
   ├── 4. 执行 tools.py 中的 Python 信号计算，将光谱/时域特征文本喂回 Agent
   ├── 5. 循环交互获得 Agent 最终诊断结论与处置建议
   └── 6. 比对 metadata.json 标准答案，自动导出统计报告 (eval_report.json)
```

## 2. 核心模块与代码职责详解

### 2.1 物理层仿真模块 (`simulation/`)

负责批量制造高保真、可控制变量的低空电磁采样数据及标准答案。

- **`simulation/em_signal_simulator/baseband.py`**：基带信号生成器。支持 BPSK, QPSK, 16QAM, 64QAM, GFSK, OOK, OFDM, FHSS, LFM 共 9 种低空常见通信、抗干扰及雷达调制，并完成初始能量归一化。
- **`simulation/em_signal_simulator/jamming.py`**：干扰信号注入器。支持 `none`（无干扰）、`single_tone`（单频/同频干扰）、`swept`（扫频干扰）、`pulse`（突发脉冲干扰）、`broadband`（宽带压制干扰）共 5 种模式。通过干信比 `jsr_db` 参数精准计算并控制叠加的干扰功率。
- **`simulation/em_signal_simulator/channel.py`**：信道物理损伤模型。模拟载波频率偏移（CFO）产生的相位漂移，根据设定信噪比 `snr_db` 注入复高斯白噪声（AWGN），并对输出的 `complex128` 向量进行全局幅度归一化。
- **`simulation/em_signal_simulator/factory.py`**：工厂中枢。组合上述模块提供单样本生成 API，并实现数据集批量生成（`generate_dataset`）。支持开启 `--mixed` 混搭模式，自动构建包含完整 Ground Truth 属性（干扰类型、时频区间、归一化频偏等）的 `metadata.json`。
- **`simulation/generate_dataset.py`**：数据集生成命令行 CLI 入口。
- **`simulation/plot_sample.py`**：图形抽查工具。读取 `.npy` 复数数据并结合 `metadata.json`，绘制上图为 I/Q 幅度时域波形，下图为中心化（`fftshift`）以 dB 为单位的 FFT 频域幅度谱。

### 2.2 智能体工具箱 (`tools.py`)

大语言模型直接理解离散复数 IQ 点阵的能力较弱。本模块将专业数字信号处理（DSP）算法封装为符合 OpenAI Function Calling 规范的 JSON-serializable 工具。

- **`analyze_spectrum(sample_path)`**：频域分析工具。加载 `.npy` 数据做直流偏置消除与全局 FFT 计算，输出归一化峰值频率（主频/频偏点）、峰值功率幅度（dB）、99% 能量占用带宽（Occupied Bandwidth）及估算信噪比。
- **`detect_time_domain_features(sample_path)`**：时域与脉冲分析工具。计算信号的峰均功率比（PAPR），并利用基于中位数绝对偏差（MAD）的自适应动态阈值检测突发脉冲，输出脉冲占空比（Duty Cycle）与激活采样点数，用于辅助识别突发脉冲干扰（Pulse Jamming）。

### 2.3 监考打分引擎 (`evaluator.py`)

自动化评估跑道与指标统计中枢。

- **`OpenAICompatibleAgent`**：通用 Agent 适配器。兼容 OpenAI / Qwen / DeepSeek 等 OpenAI-style API 接口。实现 5 轮以内的多轮 Tool Calling 交互循环，并具备强鲁棒性的正则表达式 Markdown / JSON 解析剥离能力（`_extract_json`）。
- **`evaluate_dataset`**：自动化评估主函数。遍历数据集目录，记录预测结论与 Ground Truth 的对比结果。自动按 SNR 区间（Low: <0 dB，Mid: 0~10 dB，High: >10 dB）对混淆结果进行分层统计，用于评估模型在嘈杂电磁环境下的抗噪衰减能力。

## 3. 环境准备与依赖安装

系统要求 Python 版本 ≥ 3.10。在项目根目录下安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 依赖项包括：

- `numpy`：离散复数向量与矩阵计算
- `scipy`：信号处理与成形滤波预留
- `matplotlib`：频谱图与波形图绘制
- `openai`：LLM Agent 工具调用与 API 通信

## 4. 全流程使用指南

### 步骤 1：批量生成仿真评估数据集

运行 `generate_dataset.py` 脚本出题。推荐使用 `--mixed` 随机混搭模式，让系统自动随机组合调制类型、干扰类型以及在指定 SNR 范围内抽样：

```bash
# 生成 100 个包含 -10dB 到 20dB 混合信噪比的综合评估样本
python simulation/generate_dataset.py --output-dir dataset --count 100 --mixed --snr-range -10 20
```

数据将被导出至 `dataset/` 目录下，包含 `sample_00000.npy` 等数据文件及 `metadata.json` 标准答案。

### 步骤 2：数据抽查与可视化（可选）

为验证仿真数据的时频特征，可使用 `plot_sample.py` 脚本绘制分析图：

```bash
python simulation/plot_sample.py dataset/sample_00000.npy
```

### 步骤 3：运行 Agent 自动化评估

#### (1) 离线管道校验模式 (Offline Validation)

无需消耗 API Key，使用 Ground Truth 直接校验评估流水线的匹配与统计逻辑：

```bash
python evaluator.py dataset --offline --output test_report.json
```

#### (2) 在线评估真实大模型 Agent (Online Evaluation)

配置大模型 API 密钥（以 DeepSeek 或 Qwen 为例）：

```bash
# Linux / macOS 设置环境变量
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"

# 运行评估引擎
python evaluator.py dataset --model "deepseek-chat" --output deepseek_eval_report.json
```

## 5. 评估报告与指标说明

评估完成后，终端会打印核心摘要，并将详细结果保存至 `--output` 指定的 JSON 文件中（如 `eval_report.json`）。导出的评估报告包含以下核心科研指标：

```json
{
  "dataset": "dataset",
  "num_samples": 100,
  "jamming_classification_accuracy": 0.85,
  "accuracy_by_snr_group": {
    "Low SNR": 0.62,
    "Mid SNR": 0.88,
    "High SNR": 0.96
  },
  "tool_call_success_rate": 0.98,
  "total_tool_calls": 200,
  "successful_tool_calls": 196,
  "results": [ ... ]
}
```

- **`jamming_classification_accuracy`**：总体干扰类型识别准确率
- **`accuracy_by_snr_group`**：分信噪比阶梯准确率（用于绘制 SNR 衰减折线图）
  - `Low SNR`：低信噪比 (<0 dB) 准确率
  - `Mid SNR`：中信噪比 (0~10 dB) 准确率
  - `High SNR`：高信噪比 (>10 dB) 准确率
- **`tool_call_success_rate`**：Agent 正确解析并调用 `tools.py` 的成功率
- **`total_tool_calls`** / **`successful_tool_calls`**：总工具调用次数与成功返回有效结果的次数
- **`results`**：逐样本的详细诊断轨迹与工具调用明细

## 6. 项目扩展指南

- **新增基带调制**：在 `simulation/em_signal_simulator/baseband.py` 中的 `MODULATIONS` 集合注册新调制名称，并在 `generate_baseband()` 函数中追加数学实现逻辑。
- **新增干扰类型**：在 `simulation/em_signal_simulator/jamming.py` 中的 `JAMMING_TYPES` 注册，并在 `inject_jamming()` 函数中追加干扰波形叠加算法。
- **扩展 Agent 工具**：在 `tools.py` 中编写新的复数信号分析函数（例如估计到达角 AoA 或电磁指纹），将其加入 `TOOL_FUNCTIONS` 字典与 `TOOL_SCHEMAS` 数组即可。

## 7. 参考与致谢

- 基带与信道仿真：参考公开的数字信号处理定义与 GNU Radio 架构标准。
- 评测体系设计：参考 CVPR 2026 论文 *MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals* 关于电磁信号任务分层与低 SNR 评估的设计思想。
