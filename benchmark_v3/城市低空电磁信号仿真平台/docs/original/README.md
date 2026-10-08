# 静态城市多点仿真干扰分析 Agent：仿真底座 v0.2

已实现：静态城市、多发射设备、多接收站、最长 50 ms IQ，组件式波形、真实 Sionna RT、接收前端、真值隔离及旧 Agent 数据适配。

本次交付是仿真与观测接口，不是新增的多站 LLM 协同推理器。没有调用付费模型。原有 Agent 保持不变。

当前默认采用 **8+8 十字阵，共 16 通道**；阵列中心、朝向、阵面法向和所有阵元绝对坐标均写入公开观测。配置示例及通道顺序详见 [十字阵说明](docs/cross-arrays.md)。旧 Agent 的线阵 MUSIC 不用于十字阵，二维测向尚待接入。

已增加独立的经典算法实验模块（尚未接入旧 Agent）：Welch能量检测、Bartlett/二维MUSIC、多站交会。[城市与LoS对照结果](docs/classical-baseline-results.md)记录检测、定位误差和多径失败分析。

## 已运行的样例

正式验收目录：`results/munich_cross16_verified/`。较早的 `munich_cpu/`、`munich_v1/` 是调试阶段产物，不作为最终结果。

- 慕尼黑实际三维地图，研究区域约 x/y ±250 m，传播仍使用完整地图，避免裁剪边缘失真。
- 2 个通信设备：OFDM、QPSK；2 个干扰设备：带限噪声、脉冲扫频＋单音组合。
- 3 个 8+8 十字阵（16 通道）接收站；2.4 GHz 参考载频；2 MS/s；50 ms。
- 每站 IQ 为 complex64 [16,100000]，三站合计约 38.4 MB。
- 真实 Sionna RT 2.1.0，LLVM CPU，反射＋绕射、最大交互深度 3、每源 100000 个射线采样。
- 当前实例求得 240 条有效路径。部分链路未找到路径，不解释为物理上绝对不可见。
- 已保存场景图、时频图、每源接收功率、信道系数、配置、哈希和审计报告。

![研究区域](results/munich_cross16_verified/scene_overview.png)

## 快速运行

在本项目目录打开 PowerShell：

```powershell
python -m pip install -e ".[test]"
python -m cityrf validate --config examples/free_space.json
python -m cityrf run --config examples/free_space.json --output results/my_free_space
python -m cityrf analyze --public results/my_free_space/public --output results/my_free_space/overview.json
python -m pytest -q -W error
```

输出目录必须不存在或为空，避免覆盖已有实验。

当前机器已建立独立 WSL 环境 `/root/.venvs/static-city-rf`：

```powershell
wsl --exec /root/.venvs/static-city-rf/bin/python -m cityrf run --config examples/munich.json --output results/my_munich
wsl --exec /root/.venvs/static-city-rf/bin/python tools/render_report.py results/my_munich
wsl --exec env RUN_RT_TESTS=1 /root/.venvs/static-city-rf/bin/python -m pytest -q -W error
```

其他机器可在 WSL 项目目录运行 `bash tools/setup_rt.sh`。基础 DSP 支持 Windows；RT 使用独立依赖，缺失时直接报错，不回退到自由空间。

当前 WSL 缺少 OptiX，故明确选用 `llvm_ad_mono_polarized`。GPU 环境准备完成后，配置 `propagation.rt_variant=cuda_ad_mono_polarized`；本次没有验证 GPU RT。

## 模块与接口

| 文件 | 职责 |
|---|---|
| `config.py` | 严格配置、ID/引用、采样与带宽约束 |
| `waveforms.py` | 七类代理波形、组合、门控、发射功率；索引确定性 |
| `channels.py` | LinkChannel、自由空间、指定抽头、分数时延卷积 |
| `sionna_backend.py` | Sionna RT 2.1.0 适配、室外位置检查、静态路径求解 |
| `receiver.py` | 混合信号上的噪声、可选软限幅、调谐、滤波、固定增益 |
| `runner.py` | 场景执行、资源预估、保护样点、逐站落盘 |
| `export.py` | 公开观测／私有真值分离、旧 Agent manifest 转换 |
| `analysis.py` | 仅使用公开 IQ 的 Welch 谱峰与功率时间序列 |
| `audit.py` | 实验哈希、形状、有限值与路径统计核验 |
| `replay.py` | 已冻结信道重放，不再求 RT |

扩展一个传播后端：实现 `solve(Scenario) -> list[LinkChannel]`，然后注册到 `runner.BACKENDS`。必须完整返回每个 TX/RX 对，无路径时返回形状 [rx_ports,0] 的信道。

扩展波形：添加 Component.kind 及生成函数，保持相同 seed/id/样点索引得到同一波形，补功率和重叠窗口测试。不要在每个接收站重新随机生成发射数据。

## 配置方法

复制 `examples/munich.json`，编辑：

- `emitters`：位置 m、姿态 rad、名义功率 dBm、role（仅真值）、components。
- `components`：kind、offset_hz、bandwidth_hz、symbol_rate_hz、weight；可配 pulse_period_s、duty_cycle、chirp_period_s。
- `receivers`：位置、姿态、十字阵双臂元素数／矩形行列数与间距、调谐偏移、接收带宽、frontend。
- `tasks`：明确受保护的发射源集合、目标频带、使用哪些站。通信源相对于其他受保护任务也可能成为干扰，不由 role 决定。
- `propagation`：backend、scene、max_depth、samples_per_src、反射／绕射等开关。

本版：单发射端口、理想各向同性 V 极化；所有站共用采样率，同一频段。RT 接收阵列规格必须一致，位置及姿态可不同。几何固定，没有移动、多普勒或长时流式处理。

单个源的多个波形成分仍算一个物理设备；发射源数量与路径数量分别保存。无人机波形是研究代理，不宣称实现任何厂商专有图传协议。

## 功率、时间与滤波约定

1. IQ 是复包络，平方幅度按 W 解释。总发射功率在固定 16384 样点、未门控的组合参考波形上标定；相关成分的交叉项包含在标定中。完全相消的组合拒绝运行。
2. 门控在最终基带带限滤波之前；之后不重新归一化，因此占空比和滤波边缘都会降低实际平均功率。真值另外保存实际 50 ms 发射功率。`bandwidth_hz` 为 FIR 截止边界配置，有有限过渡带，不是协议占用带宽认证。
3. 信道保留传播损耗和绝对延迟，不逐链路归一化。RT 系数包含天线和载频相位，不能重复施加阵列转向向量。
4. 各源到达信号先相加，再经过共同接收前端。可选 tanh 软限幅是前端压缩代理，不等价于完整 LNA/ADC 数字孪生。
5. 默认等效输入白噪声谱密度 `k*T*F`；滤波后预期噪声功率按 FIR 等效噪声带宽计算，不强行设定每站同一 SNR。
6. 内部生成前后保护样点；193 抽头 Kaiser-sinc 实现分数时延，线性卷积后裁取 50 ms；路径延迟上限 1 ms。
7. 十字阵位于局部 yz 面，水平臂沿 +y，垂直臂沿 -z，阵面正面 +x；姿态为 Rz(yaw)Ry(pitch)Rx(roll)。RT 真值为全局方位角、天顶角；旧 MUSIC 输出局部宽侧角，不能直接当源位置方位。
8. 窄带 synthetic-array 近似用于阵列响应。自由空间对照为标量同极化理想天线模型；不用于验证任意极化旋转或近场阵列。

## 数据与 Agent 接入

```text
run/
  public/
    observations.json       公开采集信息、任务先验、IQ 哈希
    iq_station_*.npy         仅混合接收信号
  private/
    scenario.json           完整参数与 seed
    truth.json              发射源、每源各站功率和传播真值
    channel_*.npz           冻结路径系数和绝对时延
  run_summary.json           完成标志、耗时、版本、哈希
```

`public/` 不包含源位置、真实干扰数量、路径或分离源信号。隔离是数据接口边界，不是操作系统权限沙箱；实际给 Agent 配置 IQRepository 时仅授权 public 目录。

```powershell
python -m cityrf legacy-manifests --public results/munich_cross16_verified/public --output results/munich_cross16_verified/legacy_manifests
python tools/check_legacy.py results/munich_cross16_verified/public --output results/munich_cross16_verified/legacy_check.json
```

检查脚本调用旧 Agent 的真实 inspect_iq 和 Welch 工具，不调用模型。旧 Agent 路径默认取相邻“基本干扰分析功能Agent/src”，也可用 --legacy-src 指定。当前 analyze 命令是确定性概览，不冒充 LLM Agent 的自主分析或正式 benchmark 评分。

## 精确重放

独立射线求解可能改变路径顺序及浮点归约顺序；固定 seed 不承诺跨平台或跨版本的比特一致性。精确复现应读取保存的信道：

```powershell
wsl --exec /root/.venvs/static-city-rf/bin/python -m cityrf replay --run results/munich_cross16_verified --output results/my_replay
```

重放先验证产物哈希，并要求相同 Python/NumPy/SciPy 版本与核心 DSP 源码哈希。已验证正式样例三站 IQ 的 SHA256 完全一致。

## 物理边界与后续工作

- 室外端点校验采用朝天射线，要求无遮挡天空；会拒绝廊下等真实室外位置，不是通用实体内外判定。
- 内置慕尼黑材料与建筑是研究模型，未经实测校准。零路径不代表绝对没有信号；需要对重要结论做求解预算与传播开关敏感性实验。
- 保存源与路径真值，但暂不自动推导 blocking、可分辨源数或完整评测标签。非线性情况下不能把逐源线性贡献当成输出可加分解。
- 配置包含 DSP 内存预算估算；RT 的场景和射线中间内存另计。默认算例以 2 MS/s 验证，不承诺所有最大配置都适合本机。
- 多站协同推理、实测波形、异构阵列、更完整接收机和业务评测，可分别在既有模块上扩展。
