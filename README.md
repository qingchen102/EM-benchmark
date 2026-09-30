# 低空电磁安全大模型 · 智能干扰分析 Benchmark

> 面向 **「低空安全的类脑推理电磁大模型」**（课题四 · 多源电磁干扰智能分析场景）的
> 可复现、可扩展的 **Agent 信号分析评测基准**。
>
> 以「出题 → 工具 → 监考 → 打分」四层解耦的方式，评测大模型 Agent 在智能干扰分析上的能力，
> 并为未来正式电磁大模型保留接入接口。

---

## 1. 项目目标

电磁大模型能否在没有专用物理层编码器的情况下，仅依靠 **信号测量工具 + 自身推理** 完成专业的电磁干扰分析？
本项目要回答的正是这个问题。我们构建一套**可控、可评分、可归因**的评测基座：

- **出题**：用信号模型仿真生成 IQ 数据、场景参数和标准答案（Ground Truth），从而精确控制题目难度与真实性。
- **评测**：让 Agent（当前用通用 LLM + 专业信号工具模拟）调用工具、综合推理、输出结构化答案；
  由评估器对照 Ground Truth 按分类正确 / 数值容差 / 多源匹配打分，并做**损失分解**——定位分数损失来自题目、工具还是 Agent。

设计上这是一个**面向课题四的场景化、模型无关、工具增强 Agent 验证基座**：
它不是电磁大模型本身，而是它的"考场"，为正式电磁大模型的接入保留统一的样本格式 / 答案 schema / 评测接口。

---

## 2. 为什么需要它（借鉴与参考）

- **MERLIN**（*Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals*, 2026）：借鉴其**评测思想**——多任务/分层能力评测、按 SNR 分组看鲁棒性、分类与数值估计采用不同评分、区分感知与推理。我们没有复现其 Signal Encoder / 大规模指令微调 / 蒸馏训练。
- **S-ICDF 类低空电磁数据集**：作为低空场景建模的**参考口径**（更真实的环境建模方向），不当作对比对象。
- **Sionna RT**：物理仿真平台的传播后端，用光线追踪在真实城市几何上解真实多径 / 反射（见 `benchmark_v3/城市低空电磁信号仿真平台/`）。
- 评测方法论：预注册门槛 + 离线验收 + 全量基线的版本化迭代纪律（逐版本封版，防止"刷分/过拟合考卷"）。

---

## 3. 方法：四层解耦 + 迭代纪律

### 3.1 核心四层（概念地图）

| 角色 | 实现 | 说明 |
|---|---|---|
| **信号模型 = 出题** | `simulation/` | 生成 IQ、场景参数与 Ground Truth，决定题目的真实性与难度 |
| **tools = 仪器** | `tools_v*.py` | 给 Agent 的信号测量工具（频谱 / 源数 / DOA / 时域 / 调制特征），只返回测量值与候选证据，不下最终结论 |
| **Agent = 考生** | 通用 LLM + tools | 自主调用工具、理解测量、综合推理、输出结构化答案 |
| **evaluator = 阅卷** | `evaluator_v*.py` | 对照 Ground Truth，按分类 / 数值容差 / 多源匹配评分并做损失分解 |

### 3.2 迭代纪律

每一步改动都按 **预注册门槛 → 0 token 离线验收 → 小样本在线确认 → 全量基线** 的流程走，
并逐版本封版（如 V2 封版于 v8 / ds_run19）。这样版本之间可复现、可归因，避免"为了涨分而刷题"。

### 3.3 关键约定

- **三 JSON 信息隔离**：`metadata.json`（内部完整记录）/ `observations.json`（Agent 可见先验）/ `ground_truth.json`（评测专用）。答案轴只进私有一侧，保证评测公平。
- **OBW99（带宽语义）**：带宽 = 99% 占用带宽；V3 中跳频 / 脉冲类辐射体的 GT 带宽 = **整窗总跨度（实测）**，与 Agent 可测的频谱工具口径一致。
- **可检测性约束**：分离约束（Δfreq ≥ 0.15 或 ΔDOA ≥ 15°）+ 奈奎斯特防折叠 + INR ≥ 3 dB。
- **敌我意图（V3 新轴）**：`malicious` 恶意（压制 / 欺骗 / 电子对抗）／`incidental` 无意（微波炉 / DECT / Zigbee）／`benign` 无害（正常合法信号）。
- **辐射体溯源（V3 新轴）**：由信号判断"是什么设备 / 辐射体"（16 刻度），是"体制 → 调制 → 辐射体"细分溯源的一部分。

---

## 4. 版本演进（进度）

| 版本 | 场景 / 信号模型 | 任务 | 状态 |
|---|---|---|---|
| **V1** | 单天线、4 类简单干扰 | 跑通「Agent 能做信号分类」的完整评估链路 | ✅ 历史封版 |
| **V2** | 4 阵元 ULA、多源（1 目标 + 0~2 干扰）、真实辐射体简化（调制）+ 物理保真（LNA 饱和 / 阵元失配 / 防折叠 / OBW99 统一） | 检测 / 分类 / 参数估计 / DOA / 体制识别 | ✅ 封版（v8，ds_run19） |
| **V3** | **低空环境仿真**：16 个城市低空辐射体 + 多径 / IQ 不平衡 / 相位噪声 / IMD3 信道 + 敌我意图 / 辐射体溯源答案轴；分级数据集 D1–D4 × 观测条件 S0–S2 已生成（各 50 条）；信道后端可接入物理仿真平台 | V2 全部任务 + 辐射体溯源 + 敌我意图 + 分级评测（D1–D4 × S0–S2） | ✅ 生成器闭合 + 评估器就绪 + 四级数据集落地（见 `benchmark_v3/README.md` §7） |
| **V4（规划）** | **真实数据**（真实 SDR 采集） | 同 V3 | ⏳ |

> **V3 的信号模型是"更真实的低空环境仿真"，不是真实采集**——真实采集留到 V4。
> 为获得真实几何反射 / 多径，V3 的信道后端可切换到并入的物理仿真平台（见 §5、§6）。

---

## 5. 仓库结构与脉络

```text
EM-benchmark/                        # 仓库根
├── benchmark_v1/                    # V1 历史代码（单天线、4 类简单干扰）
├── benchmark_v2/                    # V2（已封版）
│   ├── simulation/                  # 信号模型（基带 / 干扰 / 信道 / 样本工厂 / 可视化）
│   ├── tools_v2.py                  # Agent 工具集
│   ├── evaluator_v2.py              # 评估器（存档 / 断点续跑 / 自动重试）
│   └── ds_run19.json                # 500 样本最终基线（V8）
├── benchmark_v3/                    # V3 低空环境仿真
│   ├── simulation/                  # 16 辐射体 + 低空信道 + 样本工厂 + 冒烟测试
│   ├── eval_v3.py                   # V3 评估器（在 V2 基础上加敌我意图 / 溯源评分）
│   ├── adapters/                    # 融合适配器
│   │   └── cityrf_source.py         # V3 源 → 物理仿真平台信道（不改平台源码）
│   └── 城市低空电磁信号仿真平台/     # 并入的物理级射频仿真工作台（Sionna RT + 慕尼黑城市几何）
├── run_v32.ps1                      # （本地运行脚本，含 API key，已 gitignore）
├── projects/                        # 汇报与出图项目（个人使用）
└── study/                           # 学习 / 自学材料（个人使用）
```

**两个同处一仓、面向同一课题四场景但承担不同层的系统：**

| | 含义 | 角色 |
|---|---|---|
| **V3 信号模型** | 自研的低空辐射体 + 环境信道 | **出题 + 阅卷**（自带三 JSON、答案轴、评估器） |
| **物理仿真平台** | Sionna RT 工作台（真实慕尼黑建筑几何、光线追踪多径 / 反射） | **信号后端 / 环境**，可给 V3 提供真实空间信道 |
| **`adapters/cityrf_source.py`** | 运行时把 V3 辐射体当"信号源"送进平台信道，复用平台的信道 / 接收 / 导出 | **桥接层**（只加文件，不改平台源码） |

> V3 与物理平台是**互补**而非替代：V3 负责"考卷 + 答案轴 + 阅卷"，物理平台负责"真实的空间信道 / 环境"。
> V4 真实数据接入时，信号模型会被替换，但考卷、信息隔离、评估接口会沿用。

---

## 6. 快速开始

依赖：`numpy scipy matplotlib pillow openai tqdm`（物理平台运行需 `pydantic`；Sionna RT 需 WSL 环境）。

### V2（已封版，直接评估）

```bash
cd benchmark_v2
python evaluator_v2.py dataset --offline                       # 离线验证评分管道（应满分）
python run_bailian_deepseek.py dataset --model deepseek-v3.2 --max-samples 50 --output results.json
```

### V3（低空环境仿真：生成 + 冒烟）

```bash
cd benchmark_v3
python smoke_test.py                       # 生成样本 + 验证 v2 工具可消费
python eval_v3.py _v3test --offline        # 离线验证 v3 评分管道（应满分）
```

### V3 + 物理平台信道（融合 DEMO）

```bash
cd benchmark_v3
python adapters/cityrf_source.py           # 默认：多源 + 多接收机（自由空间，本机 Windows 可直接跑）
python adapters/cityrf_source.py simple    # 单站回归：1 目标 + 2 干扰

# 真实多径（Sionna RT，须在 WSL 里用平台的 .venv-rt；LLVM 依赖系统 libLLVM-17）
wsl -d Ubuntu -e bash -lc 'export LD_LIBRARY_PATH=/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH; \
  /mnt/d/srtp/benchmark_v2/benchmark_v3/城市低空电磁信号仿真平台/.venv-rt/bin/python \
  /mnt/d/srtp/benchmark_v2/benchmark_v3/adapters/cityrf_source.py rt'
```

自由空间输出在 `benchmark_v3/_fusion_multi/`；真实多径输出在 `benchmark_v3/_fusion_rt/`。`public/` 是给 Agent 的多站观测（不含答案轴），`private/` 是逐个 (tx,rx) 信道 / 真值，外加 `fusion_truth.json`（辐射体名 / 敌我意图）。`rt` 模式用慕尼黑场景里已验证「天顶可见」的坐标（随机布点会被楼挡掉、求解失败）。

### 数据集生成与评分（D1/D2）

按「任务三层 × 难度四级 × 观测条件三档」框架生成可评分数据集：

- **任务层**：T1 感知（源数/频偏/DOA）→ T2 认知（体制/辐射体/干扰类别）→ T3 决策（敌我意图/威胁评估）；
- **难度**：D1 看得见（视距+0/1 简单干扰+仅热噪声）→ D2 挤得慌（城市 RT 多径+2–3 共存辐射体）→ D3 有人捣乱（恶意干扰+低 SNR）→ D4 极限藏匿（欺骗+非线性+时变）。**难度=场景包络，SNR 由几何+功率推算，不是旋钮**；
- **观测条件**：S0 单站 / S1 三站 / S2 四站——同一采集分站评分，可得"多站信息增益曲线"。

```bash
cd benchmark_v3/adapters
python gen_dataset.py --difficulty d1 --count 50     # D1：视距，~250MB
python audit_dataset.py ../dataset_d1                # 对照 v1/v2 坑清单审计（应全绿）
python eval_dataset.py ../dataset_d1                 # GT 回放验证评分管道（应全 1.0）
# D2（城市 RT 多径 + 共存源，需在 WSL 的 .venv-rt 里跑）
python gen_dataset.py --difficulty d2 --count 50
python audit_dataset.py ../dataset_d2 --d2 && python eval_dataset.py ../dataset_d2
```

已生成（各 50 条，审计全绿 + GT 回放满分）：

| 数据集 | 难度 | 场景/传播 | 干扰 | 目标 SNR | 大小 |
|---|---|---|---|---|---|
| `dataset_d1/` | D1 看得见 | 视距（free_space） | 50% 无 / 50% 单音或噪声，全站 INR≥3dB | 8–38 dB | 248 MB |
| `dataset_d2/` | D2 挤得慌 | 杭州 RT 真实建筑（多径/遮挡） | 每条 2–3 个共存辐射体（蓝牙/WiFi/微波炉/Zigbee/DECT），频谱重叠 | 15–41 dB | 251 MB |
| `dataset_d3/` | D3 有人捣乱 | 杭州 RT + NLoS（目标置楼后） | 1–2 恶意压制（扫频/噪声/单音）+50% 共存源 | 1–34 dB | 249 MB |
| `dataset_d4/` | D4 极限藏匿 | 杭州 RT + NLoS | GNSS 欺骗 + 恶意压制 + 50% 共存源；IMD3 非线性 + 相关噪声 + 机动多普勒 | 1–34 dB | 250 MB |

NLoS 低 SNR 有物理出处：D3/D4 的目标被放到楼后（occupancy 占用图做视线遮挡采样），每站 LoS/NLoS 状态写入 GT；D2 中 44% 的干扰-站链路同样被建筑遮挡（0 径）——GT 可见性按 RT 实测逐站判定，评分与答案键自动排除不可见干扰。

### 物理仿真平台独立运行（可选）

```bash
cd benchmark_v3/城市低空电磁信号仿真平台
python -m venv .venv && . .venv/Scripts/activate     # Windows PowerShell: .\.venv\Scripts\activate
pip install -e ".[web,test]"
python -m webui.server --port 8765                    # 网页工作台 http://127.0.0.1:8765
```

---

## 7. 文档索引

| 位置 | 内容 |
|---|---|
| `README.md`（本文件） | 整体目标、方法、进度、借鉴、结构、快速开始、术语 |
| `benchmark_v1/README.md` | V1 历史文档 |
| `benchmark_v2/README.md` | V2：架构、CLI、评估链路、基线结果（ds_run19） |
| `benchmark_v2/simulation/README.md` | V2 信号模型细节（架构 / 算法 / 参数 / 元数据字段） |
| `benchmark_v3/README.md` | V3：设计决策、16 辐射体覆盖矩阵、信道、Roadmap |
| `benchmark_v3/adapters/cityrf_source.py` | V3 源 → 平台信道的融合适配器（含使用说明） |
| `benchmark_v3/城市低空电磁信号仿真平台/README.md` | 物理平台独立文档（启动 / 能力 / 限制） |

---

## 8. Roadmap

- **V3 定稿**：确定辐射体清单、信道参数、溯源粒度终稿；难度分级（信道效应开关消融）。
- **融合样本接回评分**：把 `adapters` 产生的融合样本接回 V3 三 JSON / `eval_v3.py`，让评分闭环直接消费更真实信道数据。
- **物理平台 Sionna RT 后端**：在 WSL + Sionna RT 2.1.0 环境启用 `sionna_rt`，获得真实多径；对比 free_space vs RT 对 DOA / 定位任务的影响。
- **V3 → V4**：真实 SDR 采集，替换仿真信号源为真实捕获；保住考卷 / 答案轴 / 阅卷接口不变。