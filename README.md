# 电磁干扰分析 Agent Benchmark（EM-Jam Agent Benchmark）

> 面向**智能电磁干扰分析大模型 Agent** 的评估基准：为 LLM Agent 提供多源、多天线 IQ 信号数据集与评估链路，衡量其在干扰检测、参数估计、分类与定位上的能力。

---

## 1. 项目定位与演进

本项目目标是构建一个可复现、可扩展的电磁信号分析基准，用于评估 LLM Agent 的物理层信号分析能力（而非纯文本推理）。项目分阶段演进：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **V1** | 单天线、4 种简单干扰类型，跑通评估链路，验证"Agent 能做信号分类" | ✅ 已完成（历史代码） |
| **V2（当前）** | 4 阵元 ULA、多源（1 目标 + 0~2 干扰）、复杂干扰类型（脉冲化 LFM / NFM / 周期脉冲串等）、物理保真（LNA 饱和 / 阵元失配 / 防折叠 / 带宽统一），完整评估链路（检测 / 分类 / 参数估计 / DOA） | ✅ 数据与评估侧均就绪 |
| **V3（规划）** | 真实环境：多径衰落、IQ 不平衡、IMD3、真实采集数据 | ⏳ 未开始 |

**V2 信号模型的核心修复**（详见 `simulation/README.md`）：

- 干扰类别可触发（blocking 由 LNA 饱和真实呈现，而非仅标签）
- 奈奎斯特防折叠约束（干扰完整频带不越界）
- 带宽定义统一为 **99% 占用带宽（OBW99，实测）**，元数据与工具估计直接可比
- SNR = 目标功率 / 噪声功率；干扰可检测性约束（INR ≥ 3 dB）
- 真升余弦成形（符号中心无 ISI）；GFSK 带宽实测标定；OFDM 频谱整形

---

## 2. 目录结构

```text
benchmark_v2/
├── simulation/                       # 信号模型与数据生成
│   ├── generate_dataset_v2.py        # 数据集生成 CLI
│   ├── sanity_check_v2.py            # 全量验证脚本（90+ 项检查）
│   ├── README.md                     # 信号模型详细文档（架构/算法/参数）
│   ├── dataset/                      # 生成的数据集（.npy + 三个 JSON）
│   └── em_signal_simulator/          # 信号模型库
│       ├── baseband.py               # 目标基带波形（9 种调制 + OBW99 测量）
│       ├── jamming.py                # 干扰波形（5 种类型）
│       ├── channel.py                # 信道效应（频偏/阵列/LNA 饱和/失配/AWGN）
│       ├── factory.py                # 样本工厂（多源叠加/分离修正/打标/元数据）
│       └── visualization.py          # 时频图 + MUSIC 空间谱（VLM 支持）
├── tools_v2.py                       # Agent 工具集（多天线频谱/源数/DOA/时域）
├── evaluator_v2.py                   # v2 评估器（填空/选择题 + 评分报告）
└── simulation.zip                    # 历史压缩包（可忽略）
```

> 约定：**信号模型**（数据生成）在 `simulation/em_signal_simulator/`；**评估侧代码**（`evaluator_v2.py`、`tools_v2.py`）直接位于根目录，与信号模型解耦。

---

## 3. 快速开始

```bash
# 0) 依赖
pip install numpy scipy matplotlib pillow openai tqdm

# 1) 生成数据集（从 simulation/ 目录）
cd simulation
python simulation/generate_dataset_v2.py --count 500 --output-dir dataset \
    --num-sources random --snr-range -5 15 --num-workers 4

# 2) 数据健康检查
python simulation/sanity_check_v2.py

# 3) 评估（回到根目录）
cd ..
python evaluator_v2.py dataset_v2 --offline                # 离线：验证评分管道（应满分）
python evaluator_v2.py dataset_v2 --model deepseek-chat --base-url https://api.deepseek.com  # 在线评估
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
| `analyze_spectrum` | 多天线频谱：合并纹波后的源峰检测 + **每源 `sources_candidates`**（频偏/带宽/**ratio_to_target_bw**/**功率近似**，类别判定所需测量由工具算好） |
| `estimate_num_sources` | 源数估计：`num_sources_estimate`（MDL 空间特征值，默认建议）+ 三路证据（MDL/显著谱峰/纹波合并峰）与一致性，Agent 按决策树修正（MDL 高估看合并峰+DOA 稳定峰、低估看高阶新峰） |
| `estimate_doa` | MUSIC 空间谱 DOA 估计（4 元 ULA，d=λ/2）；含跨阶稳定峰 `stable_peaks_deg` 与 3 阶新峰 `peaks_new_at_order3_deg`（区分真实源与过分辨伪峰） |
| `detect_time_domain` | 时域特征：PAPR、脉冲占空比（脉冲类干扰） |
| `estimate_modulation_features` | 调制识别特征：频谱平坦度/频率漂移/峰均比/幅度峰度 + 与 14 种候选模板的特征距离（逐谱峰切片） |

工具设计沿用 v1 多轮测试验证有效的特征（频谱平坦度→宽带、分段频率漂移→扫频、峰均比→单音、PAPR/占空比→脉冲），且**只忠实返回测量值、不含任何结论性文本（无 expert_insight）**——判断完全交给 Agent。类别判定所需的 `ratio`（干扰频偏/目标带宽）与功率近似由工具直接计算输出，模型读区间即可，无需自行算术。

评估 Prompt 会注入完整先验（采样率、中心频率、半波长阵列间距、**目标调制与目标带宽（OBW99）**、观测长度）并明确归一化约定（频偏/带宽相对采样率、DOA 单位度）。

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
python evaluator_v2.py dataset_v2 --offline               # 验证评分管道（应全满分）
python evaluator_v2.py dataset_v2 --model gpt-4o-mini --output eval_report_v2.json
python evaluator_v2.py dataset_v2 --model ... --max-samples 200   # 抽样评估
```

评估器默认显示 **tqdm 进度条**（实时累计源数/类别准确率）；`--verbose` 逐样本打印，`--no-progress` 关闭进度条。

**信息隔离**：Agent 只能看到 `observations.json` 先验与 `.npy` 路径；真值仅在评估侧从 `ground_truth.json` 读取。

### 5.3 云端 API 评估（DeepSeek，推荐）

```powershell
$env:OPENAI_API_KEY = "sk-你的key"      # 或 --api-key 直接传

python evaluator_v2.py dataset \
    --model deepseek-chat \             # 推荐：快（~18s/样本），工具调用稳定
    --base-url https://api.deepseek.com \
    --max-samples 50 \                  # 先抽样试跑（--verbose 看单样本输出）
    --output eval_report_ds.json
```

**模型选择（DeepSeek 官方 API）**：

| 模型名 | 行为 | 单样本耗时 | 单样本 token | 建议 |
|---|---|---|---|---|
| **`deepseek-chat`** | 对话模型，工具调用快 | ~15~18s | ~6 千 | ✅ **首选**（50 样本约 15 分钟 / 30 万 tokens） |
| `deepseek-v4-flash` | 深度推理模型 | ~155s | ~5 万 | ⚠️ 慢且烧 token（50 样本约 2 小时 / 数百万 tokens），仅用于对比"推理能力上限" |

- 推理模型（如 v4-flash）可在命令中加 `--reasoning-effort low` 降低思考强度（API 不支持时自动忽略）；
- API Key 在 platform.deepseek.com 创建（`sk-` 开头）；`--reasoning-effort` / `--max-samples` / `--verbose` 等参数见 `python evaluator_v2.py --help`。

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

> 数据集：同 seed 生成 50 样本（1 目标 + 0~2 干扰，SNR -5~15dB）；模型为 DeepSeek 官方 API。
> 评估方式：填空/选择题，Agent 调用 5 个分析工具（频谱/源数/DOA/时域/调制特征）后作答。

### 10.1 各配置精度演进

| 配置 | 模型 | 源数 | 类别 | 频偏 | 带宽 | DOA | 调制 | 耗时/token |
|---|---|---|---|---|---|---|---|---|
| 早期（4 工具，30 样本） | v4-flash | 0.63 | 0.28 | 0.53 | 0.32 | 0.67 | 0.02 | 40min / 65w |
| 第 1 版（4 工具） | deepseek-chat | 0.60 | 0.34 | 0.43 | 0.30 | 0.67 | 0.00 | 15min / 30w |
| 第 2 版（5 工具 + per-peak 带宽） | deepseek-chat | 0.56 | 0.36 | 0.41 | 0.28 | 0.59 | 0.04 | ~15min |
| **第 3 版（5 工具 + 合并峰修复，当前）** | deepseek-chat | **0.64** | 0.34 | 0.41 | **0.44** | 0.56 | **0.14** | ~15min / 30w |

> 工具层参考精度：频偏估计 0.94（±0.02 容差）、DOA 1.00（±12°，INR≥8）、源数（MDL）0.80——即**工具测量本身很准，Agent 综合判断是主要损耗点**。

### 10.2 关键指标细节（第 3 版）

- **参数估计数值精度**：频偏 MAE 0.028（容差 ±0.02，卡边缘）、DOA MAE 2.9°（容差 ±10°）、带宽相对误差 0.68；
- **类别混淆主模式**：GT adjacent → 预测 pulse（8/15）、GT blocking → none（6/16）——模型存在"误报 pulse / 漏报强干扰"的偏置；
- **按目标调制（源数准确率）**：LFM/FHSS/OOK 1.00/1.00/0.86（简单目标）vs 64QAM/16QAM 0.20/0.33（宽带成型目标，MDL 源数估计失效）；
- **按 SNR**：类别在 Low/Mid/High 为 0.33/0.29/0.40，与 SNR 关系弱（模型综合问题）；源数 Low 0.22（物理极限）。

### 10.3 结论

1. **工具偏置修复有效**：合并频谱纹波峰后，带宽 0.28→0.44、调制 0→0.14、源数 0.56→0.64；
2. **剩余瓶颈**：类别（0.34）与调制（0.14）——分别来自模型规则综合弱（含 pulse 误报偏置）与多源混合下调制识别的物理限制；
3. **建议方向**：a) 模型对比（v4-flash 高推理 vs chat）判断类别瓶颈归属；b) 类别规则/prompt 细化（v1 验证有效）；c) 容差校准（频偏 ±0.02 → 0.03 更贴合物理可达精度）。
