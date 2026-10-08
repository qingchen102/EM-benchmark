# 静态城市仿真 v0.1 验收记录

日期：2026-09-14。用户批准的静态 50 ms 多站、多目标、多干扰架构已实现为仿真底座。既有 Agent 未修改，未调用 LLM API。

## 测试结果

- Windows Python 3.14.2：`python -m pytest -q -W error --junitxml=docs/tests-windows.xml` → 38 passed, 2 skipped。两个跳过项为显式启用的 RT 集成测试。
- WSL Python 3.12.3：`RUN_RT_TESTS=1 python -m pytest -q -W error --junitxml=docs/tests-wsl.xml` → 40 passed。
- 测试包含：非法配置、零样点、重复 ID、功率、门控带限、窗口重叠一致性、相关组合功率、Friis、时延、近带边分数时延、噪声、非线性顺序、输出隔离、旧 Agent 工具、哈希审计、精确重放。
- 真实 RT 空场景：端射及斜入射旋转阵列，与解析窄带阵列模型交叉验证。

## 真实城市结果

最终目录：`results/munich_verified`。

配置：2.4 GHz，2 MS/s，50 ms，4 个发射设备、3 个四阵元接收站，完整慕尼黑模型。CPU LLVM RT，3 阶交互，镜面反射＋绕射，每源 100000 次采样。

| 接收站 | uav_a 路径 | uav_b 路径 | source_c 路径 | source_d 路径 |
|---|---:|---:|---:|---:|
| station_a | 11 | 19 | 0 | 52 |
| station_b | 21 | 33 | 0 | 70 |
| station_c | 4 | 13 | 9 | 8 |

总路径 240；每站 shape=[4,100000]、complex64、有限数值，三站原始 IQ 9.6 MB。

本次内部计时：RT 求解及加载 1.2324 s，总生成 1.9030 s；不含解释器启动，存在库/JIT 缓存影响，不能外推为任意场景性能。DSP 工作集预算估计 113.37 MB，不是实测峰值，RT 内存额外计算。

0 路径只表示当前求解配置没有找到路径，不能证明现实中完全无信号。结果中不同站对干扰源的可见性不同，未人为保证每站都能分辨所有源。

## 复现与旧 Agent 接入

- `results/munich_replay` 从正式结果保存的信道重放；三个 IQ 文件 SHA256 与正式结果逐个相同。
- `results/munich_verified/legacy_check.json`：三个站均通过旧 Agent 的 inspect_iq 与 compute_spectrum(Welch)，无 LLM 调用。
- `legacy_manifests/` 保存两个任务在三站上的六份旧接口 manifest；其中路径为当前 Windows 绝对路径，搬移后应重新导出。
- `audit.json` 核验 18 个生成产物的哈希、形状、有限值。
- `signal_overview.png` 和 `scene_overview.png` 已实际渲染并目视检查。
- 完整 RT 环境版本在 `requirements-rt.lock.txt`；复现时对比源码哈希及 Python/NumPy/SciPy 版本。

## 独立审查与修复

独立只读审查发现并已修复：

1. 相干同频成分使总 TX 功率翻倍：使用未门控固定组合参考标定，相消组合拒绝。
2. 门控后未经带限导致频谱混叠：门控移至最终 FIR 前，带宽检查覆盖门控单音。
3. 65 抽头分数时延在近 Nyquist 处损失约 1.34 dB：改为 193 抽头，479 kHz / 1 MHz 解析复波形误差测试通过。
4. 公开元数据 ID 构造导出路径：校验 ID、唯一性及输出目录包含关系。
5. 零样点时长、重复任务源引用：严格拒绝。

复核结论：在声明的静态窄带 v1 范围内通过，无剩余重要问题。冻结信道重放在复核之后新增，由回归测试和真实三站哈希一致性验收覆盖。

## 当前范围

已完成场景配置、波形、三类传播后端、接收链、公开／真值导出、确定性概览、旧 Agent 接口、重放与文档。

尚未实现：多站 LLM 协同规划、正式城市 benchmark 标签与评分器、GPU RT 环境、实测协议波形或校准城市材料。几何检查只接受朝天无遮挡的室外点。
