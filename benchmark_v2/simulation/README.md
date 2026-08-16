# V2 多源多天线电磁信号仿真模块 (Simulation Module)

> 本文档详细介绍 `benchmark_v2/simulation/` 的架构设计、核心算法、数据流与使用方式，帮助你完全掌握该模块的原理和实现细节。

---

## 1. 模块概述

### 1.1 设计目标

本模块是 V2 版 EM-Agent-Benchmark 的**物理层数据生成引擎**，核心目标是产生**多发射源、多天线接收**的 IQ 复数信号样本，并为每个样本提供完整的参数标注（Ground Truth），供后续 LLM Agent 评估使用。

与 V1 的关键区别：

| 特性 | V1 | V2 |
|------|----|----|
| 天线维度 | 单天线 `(1024,)` | 4 阵元 ULA `(4, 1024)` |
| 源数量 | 1 目标 + 1 干扰 | 1 目标 + 0~2 干扰 |
| 干扰参数 | 仅类型标签 | 频偏、DOA、功率比、带宽、类型 |
| 干扰类别 | `single_tone/swept/pulse/broadband` | `co_channel/adjacent/pulse/blocking/none` |
| 干扰波形 | 4 种类型 | 真实调制 5 种 + 纯波形 5 种（含 NFM、脉冲化 LFM） |
| 带宽参数 | 静态查表（与实测不符） | **可配置**，元数据记录实际占用带宽 |
| SNR 定义 | 混合信号功率（SINR 式） | **目标功率 / 噪声功率**（干扰不参与） |
| 接收机效应 | 仅 AWGN | AWGN + **LNA 饱和**（Rapp 压缩）+ **阵元增益/相位失配**（可选，默认关闭） |
| 输出标注 | 按波形命名 | 每个源的完整参数 + 自动打标 |

### 1.2 核心设计原则

1. **多源独立建模**：每个发射源（目标/干扰）独立生成 1D 基带波形，各自拥有独立的频偏（CFO）和到达角（DOA），在空间域叠加后加噪。这比"干净信号 + 干扰注入"更真实，也更灵活。
2. **物理参数全标注**：每个源记录频偏、DOA、功率比、带宽，便于评估 Agent 的参数估计能力。
3. **物理真实性优先**：v2 重构补上了两类此前缺失的真实信道效应——
   - **LNA 饱和**：强干扰（blocking）不再只是"加法混入"，而是把接收机前端压入非线性区（Rapp 软限幅），真实地抑制/扭曲目标信号；
   - **阵元失配**：ULA 各阵元具有随机增益误差与相位误差（默认关闭，保持向后兼容）。
4. **自动化分类打标**：根据频偏与目标带宽的比值 + 功率比，自动将干扰归类为 `co_channel` / `adjacent` / `blocking` / `pulse` / `none`。
5. **可区分性保证**：当存在 2 个干扰源时，自动迭代校验并修正参数，确保二者在频偏或 DOA 上可区分，避免产生模棱两可的样本。

---

## 2. 目录结构

```text
benchmark_v2/
└── simulation/                        # 模块根目录
    ├── generate_dataset_v2.py         # 命令行入口 (CLI)
    ├── evaluator_v2.py                # v2 评估器（干扰检测/分类/参数估计/DOA 定位）
    ├── sanity_check_v2.py             # 重构后验证脚本（5 大任务 + 端到端 10 样本）
    └── em_signal_simulator/           # 核心算法库
        ├── __init__.py                # 导出 generate_signal_sample, generate_dataset, 可视化接口
        ├── baseband.py                # 目标基带波形生成（9 种调制 + 带宽可配 + 真升余弦 + OBW99 测量）
        ├── jamming.py                 # 干扰波形生成（5 种类型：含脉冲化 LFM、NFM）
        ├── channel.py                 # 信道效应：频偏、阵列响应、LNA 饱和、阵元失配、AWGN
        ├── factory.py                 # 样本工厂：多源叠加、分离修正、自动打标、元数据构建
        ├── tools_v2.py                # v2 Agent 工具集（频谱/源数/DOA/时域）
        └── visualization.py           # 可视化导出：时频图（Spectrogram）+ MUSIC 空间谱（VLM 用）
```

---

## 3. 整体数据流

下图展示了从参数输入到 `.npy` + `metadata.json` 输出的完整流程（含 v2 新增的 LNA 饱和与阵元失配步骤）：

```text
┌─────────────────────────────────────────────────────────┐
│ generate_dataset_v2.py                                  │
│ (命令行参数解析)                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ factory.generate_dataset()                              │
│ (遍历 count，调用 generate_signal_sample)                │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ factory.generate_signal_sample()                        │
│ (每个样本的核心生成函数)                                  │
└─────────────────────────┬───────────────────────────────┘
                          │
  ┌───────────────────────┼───────────────────────────────┐
  │                       │                               │
  ▼                       ▼                               ▼
┌──────────────────┐ ┌──────────────────────────────┐ ┌─────────────────────┐
│ 目标源生成       │ │ 干扰源生成                    │ │ 物理效应            │
│                  │ │                              │ │                     │
│ baseband.py      │ │ _sample_interferer()         │ │ 阵元失配            │
│ generate_baseband│ │ → 真实调制(WiFi/LTE/蓝牙/雷达  │ │ (可选,默认关闭):     │
│ → 基带符号流(1D) │ │   /DSSS) 或纯波形             │ │ gain_error ~ N(0,σg)│
│ (含脉冲成型中段  │ │   (tone/swept/pulse/          │ │ phase_error ~ N(0,σφ)│
│  切片,无边界伪影)│ │    broadband/nfm)             │ │ 每个样本只抽一组,    │
│                  │ │ _correct_separation()        │ │ 所有源共享同一组     │
│ apply_freq_offset│ │ → 迭代重采样确保可区分         │ │                     │
│ → 施加频偏(默认0)│ │ generate_interferer_waveform  │ │                     │
│                  │ │ / generate_baseband → 波形(1D)│ │                     │
│ expand_to_array  │ │                              │ │                     │
│ → 转向矢量×基带   │ │ apply_freq_offset → 各自频偏  │ │                     │
│ → (4, 1024)      │ │ expand_to_array → (4, 1024)  │ │                     │
│                  │ │ 按功率比缩放后叠加到 mixer     │ │                     │
└──────────────────┘ └──────────────────────────────┘ └─────────────────────┘
  │                       │                               │
  └───────────────────────┼───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 存在 blocking 干扰 ? ──是──▶ apply_lna_saturation()      │
│                          │   Rapp 软限幅 (AM-AM 压缩)    │
│                          │   y = x / (1+(|x|/A_sat)^(2p))^(1/(2p))
│                          └──▶ 目标信号被压缩/扭曲          │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ add_awgn()                                              │
│ 噪声功率 = P_target / 10^(SNR/10)（SNR 只含目标，不含干扰）│
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 输出: (4, 1024) IQ 数据 + metadata                      │
│ (每源完整参数 + lna_saturation + array_mismatch)          │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 核心模块详解

### 4.1 `baseband.py` — 目标信号基带生成

**职责**：生成 9 种调制方式的复基带信号，并提供带宽查表供分类使用。

| 调制类型 | 实现方式 | 默认带宽（归一化，实际值） |
|----------|----------|-------------------|
| BPSK / QPSK / 16QAM / 64QAM | QAM 符号映射 + 升余弦成型 | 0.169（(1+0.35)/8） |
| OOK | 0/1 随机序列，升余弦成型 | 0.169 |
| GFSK | 高斯滤波 + 上采样 + 相位积分 | 0.0908（(0.34375+1.09375·BT)/sps 实测标定） |
| OFDM | 128 点 IFFT + 零填充子载波 + CP | 0.797（102/128 激活子载波） |
| FHSS | 跳频序列（±0.45 均匀随机跳变） | 0.9 |
| LFM | 线性调频（±0.4 扫频） | 0.8 |

**带宽可配置（v2 重构重点）**：

带宽是数据集中的**待估计参数**，v2 支持显式配置，且元数据记录**实际生成带宽**：

- `generate_baseband(mod, num_samples, rng, bandwidth_normalized=0.25, **kw)`：给定目标占用带宽（归一化）—— QAM 族/OOK/GFSK 通过调整 `sps`（每符号采样数）实现，LFM 通过 `f0/f1 = ∓bw/2`，OFDM 通过激活子载波数（`carriers = round(bw·nfft)`），FHSS 通过跳频范围；
- `actual_bandwidth(mod, num_samples, bandwidth_normalized=None, **kw)`：返回**实际带宽**（受 sps 整数化/钳位影响，与请求值可能有微小差异），工厂层用它写元数据，保证 `bandwidth_normalized` 与信号频谱一致；
- 不传带宽时使用各调制类型的默认实际带宽（上表）。

**关键函数**：

- `generate_baseband(mod_type, num_samples, rng, bandwidth_normalized=None, **kw)` → `(num_samples,) complex128`
- `actual_bandwidth(mod_type, num_samples, bandwidth_normalized=None, **kw)` → 实际占用带宽
- `default_bandwidth(mod_type)` → 默认带宽查表

**升余弦脉冲成型**：

`_apply_pulse_shaping(symbols, sps, rolloff, num_taps=65, margin_symbols=4, out_length=None)`

- 成型滤波器改为**标准升余弦 (Raised Cosine) FIR**（不再是 `firwin` 普通低通近似），中心抽头归一化为 1：

$$
h(t) = \mathrm{sinc}\!\left(\frac{t}{T_s}\right)\cdot\frac{\cos(\pi\beta t/T_s)}{1-(2\beta t/T_s)^2}
$$

- rolloff（滚降系数）真正控制频谱滚降形状：占用带宽 = `(1+β)/sps`，-6dB 带宽 = `1/(2·sps)`；
- 由于中心抽头 = 1，符号中心采样点恰好恢复星座点（Nyquist 无 ISI，sanity 验证 max_err ≈ 0）；
- 旧实现在 `num_symbols` 个符号上直接跑滤波再截断/补零到 1024 点，首尾有**滤波启动暂态零点**与 **wrap 相位不连续**，FFT 边缘出现高频毛刺。v2 修复：
  1. 符号序列两端**循环扩展** `margin = max(margin_symbols, ceil(delay/sps))` 个符号；
  2. 在扩展序列上滤波并延迟补偿；
  3. 只取**中间对应原始符号的稳定段** `out_length` 点，两端均为稳态波形。

**设计细节**：

- 所有波形最终通过 `_norm()` 进行能量归一化：`x = x / sqrt(mean(|x|²))`，确保各调制类型的功率基准一致。
- GFSK 修复为**符号级高斯滤波 → 零阶保持上采样到 sps 样点/符号 → 相位积分**（旧版 1 符号/样点 + wrap 填充，带宽失真）；GFSK 占用带宽按实测标定 `bw ≈ (0.34375 + 1.09375·BT)/sps`（-35dB 占用，BT=0.35、sps=8 → 0.0908），不再用 `(1+β)/sps` 近似（旧值 0.1875 比实测大一倍）。
- OFDM 改为**大点数 IFFT + 零填充子载波**，占用带宽 = 激活子载波数/nfft，可精确配置（旧版 32 载波平铺几乎占满全带）。
- FHSS 默认**连续相位跳频**（相位跨跳累加，消除跳变边界毛刺，带外泄漏降低约 30%）；传 `continuous_phase=False` 可恢复旧的非连续相位行为（每跳相位从 0 重新开始）。

---

### 4.2 `jamming.py` — 干扰波形生成

**职责**：产生独立的干扰源基带波形（不含目标信号），支持 5 种类型。

与 V1 的 `inject_jamming` 不同，V2 采用**"生成纯干扰波形 → 外部施加频偏/阵列响应"**的分层设计：

```python
# V1 风格：信号 + 干扰一次叠加
jammed = inject_jamming(signal, jam_type, jsr)

# V2 风格：先生成纯干扰波形，再独立处理
jam_wave = generate_interferer_waveform(jam_type, num_samples)
jam_wave = apply_freq_offset(jam_wave, freq_offset)
jam_array = expand_to_array(jam_wave, doa, num_antennas, ...)
mixer += gain * jam_array
```

**支持类型**：

- **`single_tone`**：单位幅度复指数 exp(j·2π·f·t)，频谱为单根谱线
- **`swept`**：默认脉冲化 LFM（`duty_cycle=0.1`），模拟真实雷达/ECM 脉冲串。如需连续扫频，需传入 `duty_cycle=1.0`。
- **`pulse`**：周期脉冲串（传 `pri` 时，PRI 固定 + 随机起始相位，类似雷达脉冲）或伯努利随机门控（不传 `pri`，向后兼容）；幅值为复高斯，频谱呈宽带特征
- **`broadband`**：复高斯白噪声，频域平坦（热噪声型弹幕）
- **`nfm`**：**噪声调频**（NFM，噪声调频主动宽带弹幕）—— 将带限高斯噪声积分进载波相位：

$$
j(t) = \exp\!\left(j\cdot 2\pi f_c t + j\cdot 2\pi K_f \int n_B(t)\,dt\right)
$$

  参数：`noise_bandwidth`（带限噪声带宽，默认 0.1）、`freq_deviation`（归一化频偏，别名 `modulation_index`，默认 0.2）、`center_frequency`。输出恒定包络、频谱显著展宽，比纯 AWGN 更贴近真实压制干扰。

**关键函数**：

- **`generate_interferer_waveform(jamming_type, num_samples, rng, **kw)`** → **`(num_samples,) complex128`**：生成纯干扰波形。
- **`inject_jamming`**：V1 遗留的"信号+干扰一次叠加"接口，仅作兼容保留，V2 流程不再使用。

**设计注意**：`single_tone` / `swept` 等波形内部的基准频率参数默认传 0.0（如 `frequency=0.0`），因为实际频偏由 `factory` 中的 `apply_freq_offset` 统一施加，实现"波形生成"与"频率偏移"的解耦。

---

### 4.3 channel.py — 信道效应与阵列处理

**职责**：实现多天线接收机的物理效应：频偏、阵列响应（转向矢量）、**LNA 饱和**、**阵元失配**、AWGN 加噪。

#### 4.3.1 接收机参数 (make_receiver)

```python
receiver = {
    "center_frequency_hz": 2.4e9,      # 载频
    "sampling_rate_hz": 20e6,          # 采样率
    "wavelength_m": 0.1249,            # λ = c / fc
    "antenna_spacing_m": 0.06245,      # λ/2（半波长间距）
}
```

#### 4.3.2 转向矢量 (`steering_vector`)

均匀线阵（ULA）的转向矢量公式：

\[
a(\theta) = [1, \exp(j\cdot 2\pi \cdot d \cdot \sin(\theta)/\lambda), \exp(j\cdot 4\pi \cdot d \cdot \sin(\theta)/\lambda), \cdots]^T
\]

其中：

- θ：到达角（度），0° 为法线方向，90° 为端射方向
- d：阵元间距，固定为 λ/2
- 输出形状 `(num_antennas,)`

代码实现：

```python
phase = 2 * np.pi * antenna_spacing_m * np.sin(theta) / wavelength_m
steering_vector = np.exp(1j * phase * np.arange(num_antennas))
```

#### 4.3.3 阵元增益/相位失配（v2 新增，默认关闭）

真实阵列各阵元并非理想一致，v2 为转向矢量增加了可选失配：

$$
a_m(\theta) = a_m^{ideal}(\theta) \cdot (1 + \Delta g_m)\, e^{j\Delta\phi_m},\qquad
\Delta g_m \sim \mathcal{N}(0,\sigma_g^2),\quad \Delta\phi_m \sim \mathcal{N}(0,\sigma_\phi^2)
$$

- `steering_vector(..., gain_error_std=0.0, phase_error_std_deg=0.0, rng=None)`：按标准差自动抽样一组失配；
- `steering_vector(..., gain_error=None, phase_error_rad=None)`：显式传入预生成的失配数组 —— 物理上一个阵列只有一组失配，**工厂层每个样本只抽一次，所有源共享同一组**（推荐用法）；
- 默认 `gain_error_std=0`、`phase_error_std_deg=0` 时完全退化为理想阵列，与旧版行为逐位一致。

#### 4.3.4 扩展到多天线 (`expand_to_array`)

将 1D 基带信号与转向矢量做外积：

```python
array_signal = np.outer(steering_vector, baseband_signal)
# 形状: (num_antennas, num_samples)
```

每个天线上的信号 = 基带信号 × 对应的相位延迟（含可选失配）。

#### 4.3.5 频偏施加 (`apply_freq_offset`)

$$
x_{\text{out}}[t] = x_{\text{in}}[t] \cdot \exp(j \cdot 2\pi \cdot f_{\text{offset}} \cdot t)
$$

频偏是**归一化频偏**（相对于采样率），即实际频偏 / `sampling_rate`。范围为 `[-0.5, 0.5]` 对应 `[-10MHz, 10MHz]`（采样率 20MHz 时）。

#### 4.3.6 接收机 LNA 饱和 (`apply_lna_saturation`)

真实接收机前端 LNA 的线性动态范围有限：当强阻塞干扰进入时，LNA 进入**饱和/压缩区**，目标信号随之被抑制和扭曲 —— 这就是真实 "blocking" 效应的物理来源。v2 用 **Rapp 软限幅 AM-AM 模型**实现：

$$
y = \frac{x}{\left(1 + \left(\frac{|x|}{A_{sat}}\right)^{2p}\right)^{1/(2p)}}
$$

- `|x| ≪ A_sat` 时 `y ≈ x`（线性区，无失真）；
- `|x| ≫ A_sat` 时 `y ≈ A_{sat} \cdot x/|x|`（饱和区，幅度被钳制在 `A_sat`）。

```python
def apply_lna_saturation(signal, a_sat, p=2.0):
    mag = np.abs(signal)
    denom = (1.0 + (mag / a_sat) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    return signal / denom
```

#### 4.3.7 AWGN 加噪 (`add_awgn`)

**SNR 定义为「目标功率 / 噪声功率」**（干扰不参与 SNR 计算，v2 修复）：

```python
def add_awgn(signal, snr_db, rng=None, ref_power=None):
    # ref_power=None → 按信号自身功率（v1 兼容，SINR 式）
    # ref_power=目标功率 → 噪声功率 = ref_power / 10^(SNR/10)
```

$$
P_{\text{noise}} = P_{\text{target}} \,/\, 10^{\text{SNR}/10}
$$

工厂层始终传 `ref_power=target_power`。噪声为复高斯白噪声，实虚部独立，方差各为 `P_noise/2`。

---

### 4.4 `factory.py` — 样本工厂（核心编排）

**职责**：协调所有子模块，完成"多源生成 → 分离修正 → 空间叠加 → LNA 饱和 → 加噪 → 元数据构建"的完整流程。

#### 4.4.1 主函数 `generate_signal_sample`

**输入参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_sources` | 2 | 总源数（1 目标 + (N-1) 干扰），支持 1/2/3 或 "random" |
| `num_antennas` | 4 | 天线数（固定 4） |
| `num_samples` | 1024 | 每天线采样点数 |
| `snr_db` | 10.0 | 信噪比 |
| `modulation` | "random" | 目标调制类型（9 种之一或 random） |
| `interferer_type` | "random" | 干扰类型（真实调制 / 纯波形 / random） |
| `seed` | None | 随机种子（保证可复现） |
| `lna_saturation_db` | 6.0 | LNA 饱和门限（相对目标 RMS，dB），见 4.4.5 |
| `lna_p` | 2.0 | Rapp 模型平滑系数 p |
| `array_gain_error_std` | 0.0 | 阵元增益误差标准差（如 0.05 = 5%） |
| `array_phase_error_std_deg` | 0.0 | 阵元相位误差标准差（度，如 5.0） |
| `target_bandwidth_normalized` | None | 目标信号占用带宽（归一化）；None = 调制类型默认带宽 |
| `interferer_bandwidth_range` | (0.05, 0.6) | 真实调制干扰的带宽采样范围（带宽为待估计参数） |
| `blocking_power_threshold_db` | 10.0 | blocking 判定功率阈值（dB，相对目标） |
| `blocking_ratio_threshold` | 0.5 | blocking 判定频偏比阈值（|freq_offset|/目标带宽） |
| `min_inr_db` | 3.0 | 干扰可检测性下限：INR = 功率比 + SNR ≥ 该值（低于下限的干扰不可检测，无意义） |

**返回值**：

- `iq`：形状 `(num_antennas, num_samples)` 的复数数组
- `metadata`：完整参数标注的字典（含 `lna_saturation` / `array_mismatch`）

**核心流程**（对应 3. 数据流）：

1. **阵元失配抽样**（可选）：每个样本抽一组 `(Δg, Δφ)`，所有源共享
2. **目标源生成**：`generate_baseband`（可配带宽）→ `apply_freq_offset(0)` → `expand_to_array(doa=0)`（带失配）
3. **干扰源参数采样**：`_sample_interferer` 随机选类型，采样 `freq_offset` / `doa` / `power_ratio_db` / **带宽**（真实调制类按 `interferer_bandwidth_range` 随机采样）
4. **可区分性校验**：`_correct_separation` 迭代检查并修正（见 4.4.3）
5. **干扰源波形生成**：纯波形走 `generate_interferer_waveform`，真实调制走 `generate_baseband`（带采样带宽）→ `apply_freq_offset` → `expand_to_array`（带失配）
6. **功率配平**：按 `power_ratio_db` 将每个干扰缩放到相对目标的功率水平后叠加
7. **LNA 饱和**：存在 `blocking` 干扰时对混合信号施加 Rapp 压缩（见 4.4.5）
8. **加噪**：`add_awgn(ref_power=target_power)`，SNR 只含目标功率
9. **元数据构建**：记录每个源的全部参数（带宽为实际值）+ 自动打标 + 物理效应信息

#### 4.4.2 干扰源池

`_sample_interferer` 支持两类干扰源：

**A. 真实调制类**（`interferer_type="random"` 时默认随机选择；带宽在 `interferer_bandwidth_range`（默认 0.05~0.6）内**随机采样**，作为待估计参数）：

| 显示名 | 基带调制 | 默认带宽 | 模拟对象 |
|--------|---------|---------|---------|
| `WiFi_OFDM` | OFDM | 0.797 | 802.11 |
| `LTE_QPSK` | QPSK | 0.169 | 4G/5G 控制信道 |
| `Bluetooth_GFSK` | GFSK | 0.0908 | 蓝牙 |
| `Radar_LFM` | LFM | 0.8 | 雷达脉冲 |
| `DSSS_BPSK` | BPSK | 0.169 | GPS / 遥测 |

> 表中"默认带宽"仅为未配置时的基准值；实际生成时每个干扰按配置范围随机采样带宽，元数据记录**实际带宽**。中心频偏在采样带宽后按防折叠约束 `|f_off| ≤ 0.5 − BW/2` 采样（见 4.4.6），宽带干扰的频偏范围自动收窄。

**B. 纯波形类**（走 `jamming.py`，可通过 CLI `--interferer-type` 指定；带宽按波形参数计算，而非查表）：

| 类型 | 带宽语义 | is_pulse | 说明 |
|------|---------|----------|------|
| `single_tone` | 0（单音无带宽） | False | 单音连续波 |
| `swept` | 扫频范围 `|f_stop-f_start|` | 脉冲化时 True | 连续/脉冲化 LFM（默认脉冲化） |
| `pulse` | 1.0（门控噪声≈全带） | True | 周期脉冲串（PRI 0.1~0.5、占空比 0.05~0.3 随机） |
| `broadband` | 1.0（全带） | False | AWGN 宽带噪声 |
| `nfm` | Carson 近似 `2(f_dev+noise_bw)` | False | 噪声调频主动弹幕 |

#### 4.4.3 可区分性校验 (`_correct_separation`)

当存在 2 个干扰源时，若两源的**频偏差 < 0.15 且 DOA 差 < 15°**，则认为它们在物理上"太相似"，LLM 无法区分。v2 改为**迭代式修正**：

1. `while` 循环（最多 20 次）扫描所有源对；
2. 对不满足约束的干扰源，先用 `_sample_away_from` 在**避开所有其他干扰源的合法频偏区间**内重采样；若频偏区间被压缩到无解（边界裁剪所致），则改在合法 DOA 区间内重采样；
3. 全部满足则提前返回；
4. 兜底：20 次仍未满足时做一次确定性微调（`freq_min - df` 的符号偏移 + 钳位），保证不会留下不可分离的干扰源；
5. **目标纳入约束集（P1-2a）**：目标（f=0, doa=0）作为参考源参与校验，每个干扰必须与目标满足同样的最小间隔（`|Δf| ≥ 0.15` 或 `|ΔDOA| ≥ 15°`），避免出现"干扰与目标完全重合 → 不可辨识"的样本；修正只作用于干扰，目标保持固定。

**设计意图**：确保每个样本都有一个"明确正确"的答案，避免因数据本身模糊导致评估不公平。

#### 4.4.4 自动打标 (`_auto_label`)

先判断是否脉冲：`is_pulse=True` 的干扰强制标为 `"pulse"`。其余按频偏与目标带宽的比值 + 功率分类（v2 修复了旧版 blocking 分支因 `ratio≥2` 在频偏上限 ±0.5 内**永远无法触发**的问题）：

```python
ratio = abs(freq_offset) / target_bandwidth
if ratio < 0.5:                     → "co_channel"
if power >= 10dB and ratio >= 0.5:  → "blocking"   # 带外强信号 → LNA 饱和
if ratio < 2.0:                     → "adjacent"
else:                               → "none"
```

- `blocking` 判定阈值可配置：`blocking_power_threshold_db`（默认 10 dB）、`blocking_ratio_threshold`（默认 0.5）；
- `v2_category` 取值集合：`co_channel` / `adjacent` / `blocking` / `pulse` / `none`；
- **`blocking` 会触发 LNA 饱和**（见下），从信号层面真实呈现"阻塞"效应，而不再只是标签；
- 注意：脉冲化 `swept` 因 `is_pulse=True` 会被标为 `pulse` 而非 `blocking`（设计选择）。

#### 4.4.5 LNA 饱和触发逻辑

```python
blocking_present = any(m["v2_category"] == "blocking" for m in sources_meta[1:])
if blocking_present:
    a_sat = target_rms * 10 ** (lna_saturation_db / 20.0)
    mixer = apply_lna_saturation(mixer, a_sat, p=lna_p)
```

- 触发条件：任意干扰被标注为 `blocking`（≥10 dB 且超出目标带宽）；
- 饱和门限：`A_sat = 目标 RMS × 10^(lna_saturation_db/20)`，默认 6 dB → `A_sat = 2× 目标 RMS`，此时目标本身几乎不被压缩，而 10 dB+ 的阻塞干扰（RMS ≈ 3.2× 目标）会被显著压缩并连带扭曲其上的目标信号；
- 元数据 `lna_saturation.applied` 记录是否触发，便于评估端使用。

#### 4.4.6 奈奎斯特防折叠约束（v2 新增）

只限制干扰**中心频偏**在 ±0.5 是不够的：若干扰的**完整频带** `[f_off − BW/2, f_off + BW/2]` 超出奈奎斯特带 `[−0.5, 0.5]`，超出部分会**频谱折叠**回带内，破坏干扰的带宽/中心频率/DOA 等参数。因此：

$$
|f_{off}| \le 0.5 - \frac{BW}{2}
$$

实现要点（`_valid_offset_range`）：

- `_sample_interferer`：**先采样带宽、再在防折叠范围内采样中心频偏**（单音 BW=0 → 全范围 ±0.5；swept/OFDM/LFM 等宽带干扰的频偏范围自动收窄）；
- `pulse`/`broadband`（全带宽弹幕，BW=1.0）→ 中心频偏固定为 0（其频带恰好覆盖整个奈奎斯特带，物理上无法再搬移）；
- NFM 参数保证 `f_dev + noise_bw ≤ 0.45`（Carson 带宽 ≤ 0.9）；
- `_correct_separation` 重采样与兜底微调同样使用**每个干扰自身的防折叠范围**（两个全带宽弹幕频偏无法分离时自动改由 DOA 分离）；
- 目标带宽同样受约束：`target_bandwidth_normalized ≥ 1.0` 时拒绝生成；`interferer_bandwidth_range` 上限被钳位到 0.98。

#### 4.4.7 干扰可检测性约束（INR，v2 新增）

噪声功率由 `SNR = 目标/噪声` 决定，干扰功率由 `power_ratio_db`（相对目标）决定，因此干扰相对噪声的功率为：

$$
\text{INR} = \text{power\_ratio\_db} + \text{SNR\_db}
$$

两者独立采样时可能出现 `INR < 0`（干扰比噪声还弱）—— 这类干扰**物理上不可检测**，样本无意义却仍被标注和计入评估。v2 修复：

- `min_inr_db`（默认 3 dB）：生成时对每个干扰约束 `power_ratio_db + snr_db ≥ min_inr_db`，不满足则在 `[max(0, min_inr_db − SNR), 15]` 内重采样功率比；
- 若 SNR 过低导致下限超出 15 dB（无解），保留原值并在元数据如实记录 `inr_db`（评估端可据此分层，如剔除不可检测样本）；
- 每个干扰源的元数据新增 `inr_db` 字段。

---

## 5. 元数据与数据隔离 (metadata / observations / ground_truth)

`generate_dataset` 每次生成输出**三个 JSON**，把「Agent 可见先验」与「评估真值」严格隔离（v2 新增）：

| 文件 | 内容 | 谁用 |
|------|------|------|
| `metadata.json` | 完整记录（含全部真值） | **仅内部使用/调试**，不得提供给 Agent |
| `observations.json` | Agent 可见先验：接收机参数、目标调制类型、样本长度、阵元失配规格（标准差） | Agent Prompt 输入 |
| `ground_truth.json` | 评估真值：源数、各源类别/CFO/DOA/带宽/功率/INR、LNA 饱和状态等 | 评估系统专用 |

```python
from em_signal_simulator.factory import split_metadata
observation, ground_truth = split_metadata(metadata)   # 编程方式拆分
```

> ⚠️ **隔离原则**：`observations.json` 不含任何干扰信息（无 sources/类别/SNR/LNA 状态/DOA/频偏）；`metadata.json` 因含真值，评估流程中不得暴露给 Agent（sanity 有泄露检查）。

每条记录对应一个 `.npy` 文件，完整结构如下（JSON 示例，来自真实生成样本，含 blocking + 阵元失配）：

```json
{
  "sample_id": "sample_00000",
  "num_sources": 3,
  "num_samples": 1024,
  "snr_db": 10.0,
  "bandwidth_definition": "obw99",
  "receiver": {
    "center_frequency_hz": 2400000000.0,
    "sampling_rate_hz": 20000000.0,
    "wavelength_m": 0.12491352416666666,
    "antenna_spacing_m": 0.06245676208333333
  },
  "lna_saturation": {
    "applied": true,
    "model": "rapp",
    "a_sat": 1.9952623149688795,
    "p": 2.0,
    "saturation_db": 6.0
  },
  "array_mismatch": {
    "gain_error_std": 0.0,
    "phase_error_std_deg": 0.0
  },
  "sources": [
    {
      "source_id": 0,
      "role": "target",
      "modulation": "OOK",
      "power_ratio_db": 0.0,
      "freq_offset_normalized": 0.0,
      "freq_offset_hz": 0.0,
      "doa_degree": 0.0,
      "bandwidth_normalized": 0.125,
      "bandwidth_theoretical": 0.16875,
      "bandwidth_hz": 2500000.0
    },
    {
      "source_id": 1,
      "role": "interferer",
      "modulation": "QPSK",
      "display_name": "LTE_QPSK",
      "waveform_type": null,
      "power_ratio_db": 14.131696657597468,
      "inr_db": 24.131696657597466,
      "freq_offset_normalized": -0.1652665058174827,
      "freq_offset_hz": -3305330.116349654,
      "doa_degree": -16.18677981062057,
      "bandwidth_normalized": 0.34765625,
      "bandwidth_theoretical": 0.45,
      "bandwidth_fold": 0.45,
      "bandwidth_hz": 6953125.0,
      "v2_category": "blocking",
      "is_pulse": false,
      "is_modulated": true,
      "waveform_params": {}
    },
    {
      "source_id": 2,
      "role": "interferer",
      "modulation": "LFM",
      "display_name": "Radar_LFM",
      "waveform_type": null,
      "power_ratio_db": 6.60565732073676,
      "inr_db": 16.60565732073676,
      "freq_offset_normalized": 0.2579972991982109,
      "freq_offset_hz": 5159945.983964218,
      "doa_degree": 54.550859242888464,
      "bandwidth_normalized": 0.34375,
      "bandwidth_theoretical": 0.39600948334684,
      "bandwidth_fold": 0.39600948334684,
      "bandwidth_hz": 6875000.0,
      "v2_category": "none",
      "is_pulse": false,
      "is_modulated": true,
      "waveform_params": {}
    }
  ]
}
```

**字段说明**：

- `receiver`：接收机物理参数，可用于 Agent Prompt 中的先验信息（采样率、波长、天线间距）
- `lna_saturation`（v2 新增）：LNA 饱和是否触发、模型（rapp）、饱和幅度与平滑系数
- `array_mismatch`（v2 新增）：阵元失配标准差（未开启时为 0）
- `sources[0]`：固定为目标信号，`freq_offset=0`，`doa=0`
- `sources[1:]`：干扰源，包含完整参数 + `v2_category`（评估分类时使用）+ `is_pulse` + `is_modulated`；纯波形干扰额外有 `waveform_type` 字段
- `inr_db`：干扰相对噪声的功率（= power_ratio_db + snr_db），评估端可据此剔除不可检测样本
- `waveform_params`：波形级参数真值（swept 的 f_start/f_stop/duty_cycle/pri、pulse 的 pri/duty_cycle、nfm 的 noise_bandwidth/freq_deviation 等），供参数估计任务使用
- `bandwidth_normalized`：**99% 占用带宽（OBW99，实测）** —— benchmark 带宽标签的统一定义（见 `bandwidth_definition` 字段）；所有源统一为可测量值（Hann 分段 PSD 的 99% 功率跨度），与工具估计直接可比
- `bandwidth_theoretical`：理论占用带宽（生成参数推导：QAM 族 (1+β)/sps、GFSK 标定、OFDM 激活子载波跨度、LFM 扫频跨度等），作为参考字段
- `bandwidth_fold`：防折叠带宽（频偏约束用；OFDM 因子载波旁瓣 OBW99 可达理论的 ~1.7 倍，其 fold = 理论 + 0.10 裕量；NFM 为实测标定 4.2×(f_dev+noise_bw)）
- `snr_db`：**目标功率 / 噪声功率**（SNR 语义，干扰不参与）
- `display_name` : 人类可读的干扰源类型名（如 WiFi_OFDM / Radar_LFM），仅真实调制类有

---

## 6. 使用方式

### 6.1 命令行生成数据集

从 `simulation/` 目录运行：

```bash
# 全随机模式（默认）
python generate_dataset_v2.py --count 100

# 指定部分参数，其余随机
python generate_dataset_v2.py --count 50 --modulation QPSK --num-sources 3

# 全指定模式
python generate_dataset_v2.py --count 30 --modulation OFDM --num-sources 2 --interferer-type pulse --fixed-snr 10

# 开启物理效应：NFM 干扰 + 阵元失配 + LNA 饱和
python generate_dataset_v2.py --count 30 --interferer-type nfm \
    --array-gain-error-std 0.05 --array-phase-error-std-deg 5.0 \
    --lna-saturation-db 6 --lna-p 2

# 配置带宽（目标 0.3，干扰 0.05~0.4 随机采样）与 blocking 阈值
python generate_dataset_v2.py --count 30 --target-bandwidth 0.3 \
    --interferer-bandwidth-range 0.05 0.4 \
    --blocking-power-threshold-db 8 --blocking-ratio-threshold 0.5
```

**支持参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-dir` | str | `dataset_v2` | 输出目录 |
| `--count` | int | 100 | 样本数量 |
| `--num-sources` | 1/2/3/random | random | 总源数（含目标） |
| `--modulation` | str | random | 目标调制类型（BPSK/QPSK/16QAM/64QAM/GFSK/OOK/OFDM/FHSS/LFM 或 random） |
| `--interferer-type` | str | random | 干扰类型（single_tone/swept/pulse/broadband/nfm 或 random） |
| `--snr-range MIN MAX` | float | -5~15 | 每个样本 SNR 均匀采样 |
| `--fixed-snr` | float | None | 固定 SNR（与 `--snr-range` 互斥） |
| `--seed` | int | None | 随机种子 |
| `--num-antennas` | int | 4 | 天线数 |
| `--num-samples` | int | 1024 | 每天线采样点数 |
| `--lna-saturation-db` | float | 6.0 | LNA 饱和门限（相对目标 RMS，dB）；有 blocking 时启用 Rapp 压缩 |
| `--lna-p` | float | 2.0 | Rapp 模型平滑系数 p |
| `--array-gain-error-std` | float | 0.0 | 阵元增益误差标准差（如 0.05 = 5%） |
| `--array-phase-error-std-deg` | float | 0.0 | 阵元相位误差标准差（度，如 5.0） |
| `--target-bandwidth` | float | None | 目标信号占用带宽（归一化，须 <1.0）；默认按调制类型标准带宽 |
| `--interferer-bandwidth-range MIN MAX` | float | 0.05~0.6 | 真实调制干扰的带宽采样范围（带宽为待估计参数；上限自动钳位到 0.98 防折叠） |
| `--blocking-power-threshold-db` | float | 10.0 | blocking 判定功率阈值（dB，相对目标） |
| `--blocking-ratio-threshold` | float | 0.5 | blocking 判定频偏比阈值（\|freq_offset\|/目标带宽） |
| `--min-inr-db` | float | 3.0 | 干扰可检测性下限：INR = 功率比 + SNR ≥ 该值（低于下限的干扰不可检测） |
| `--num-workers` | int | 1 | 并行生成进程数（>1 时用进程池；每样本独立种子，结果与串行逐位一致） |

### 6.2 编程方式

```python
from em_signal_simulator.factory import generate_signal_sample, generate_dataset

# 生成单个样本（含 LNA 饱和 + 阵元失配）
iq, metadata = generate_signal_sample(
    num_sources=3,
    snr_db=10.0,
    seed=42,
    lna_saturation_db=6.0,
    array_gain_error_std=0.05,
    array_phase_error_std_deg=5.0,
)

# 批量生成
records = generate_dataset(
    output_dir="dataset_v2",
    count=100,
    num_sources=3,
    snr_range=(-5, 15),
    seed=42,
)
```

### 6.3 运行验证脚本

```bash
# 全量验证：5 大任务功能 + 端到端（形状/dtype/NaN/元数据/隔离/可视化/并行/评估链路）
python sanity_check_v2.py
```

### 6.4 v2 评估链路（tools_v2 + evaluator_v2，v2 新增）

**工具集**（`em_signal_simulator/tools_v2.py`，多天线适配）：

| 工具 | 功能 |
|------|------|
| `analyze_spectrum` | 多天线功率合成频谱：多峰检测（频偏候选）+ OBW99 带宽 |
| `estimate_num_sources` | 干扰源数量：MDL 信息论准则（空间协方差）为主 + 显著谱峰旁证 |
| `estimate_doa` | MUSIC 空间谱 DOA 估计（复用 visualization.music_spectrum） |
| `detect_time_domain` | 时域特征：PAPR、脉冲占空比（脉冲类干扰） |

工具只返回测量值与洞察，不做分类结论——结论由 Agent 综合判断。

**评估器**（`evaluator_v2.py`，填空/选择题）：

```bash
# 离线模式：用真值回放验证评分管道（应全满分）
python evaluator_v2.py dataset/ --offline

# 在线模式：OpenAI 兼容 API
python evaluator_v2.py dataset/ --model gpt-4o-mini --output eval_report_v2.json
```

任务与指标：

| 任务 | 形式 | 指标 |
|------|------|------|
| 干扰检测（源数） | 选择题 0/1/2 | 源数准确率 |
| 干扰分类（同频/邻道/脉冲/阻塞） | 选择题 | 类别准确率（贪心匹配后）+ 按 SNR 分组 |
| 参数估计（频偏/带宽） | 填空题 | 容差命中率（频偏 ±0.02、带宽 ±25%）+ MAE |
| 干扰源定位（DOA） | 填空题 | 容差命中率（±10°）+ MAE |
| 调制类型（可选） | 选择题 | 真实调制类干扰的调制准确率 |

数据隔离：真值只从 `ground_truth.json` 读取；Agent 仅获得 `observations.json` 中的先验与 `.npy` 路径。

### 6.4 可视化导出（VLM 支持，v2 新增）

`em_signal_simulator/visualization.py` 为多模态（VLM）Agent 提供视觉模态：

```python
from em_signal_simulator.visualization import (
    spectrogram_image, music_spectrum, spatial_spectrum_image,
    save_sample_visualizations,
)
import numpy as np

iq = np.load("dataset/sample_00000.npy")            # (4, 1024)

img = spectrogram_image(iq, sampling_rate_hz=20e6, path="spec.png")   # 时频图
doas, spec = music_spectrum(iq, num_sources=3)      # MUSIC 空间谱（数值）
img2 = spatial_spectrum_image(iq, num_sources=3, path="music.png")    # 空间谱图
save_sample_visualizations(iq, "dataset/sample_00000.npy", num_sources=3)  # 一键两张
```

- 时频图：脉冲串 PRI/脉宽、LFM 扫频斜率在时频图上直观可见，比文本特征更适合 VLM；
- MUSIC 空间谱：4 元 ULA（λ/2 间距），源数由调用方传入（评估时取自 ground_truth）；
- 绘图函数返回 RGB uint8 数组，`path` 给定时同时保存 PNG；matplotlib 为延迟导入，纯数值路径不依赖。

---

## 7. 关键物理参数对应关系（供 Agent Prompt 使用）

| 归一化参数 | 物理含义 | 映射关系（采样率 20MHz） |
|-----------|---------|------------------------|
| `freq_offset_normalized` | 频偏 | `实际频偏(Hz) = f × 20MHz` |
| `doa_degree` | 到达角 | 0° = 法线，±60° 扫描范围 |
| `wavelength_m` | 波长 | `λ = c / 2.4GHz = 0.125m` |
| `antenna_spacing_m` | 阵元间距 | `λ/2 ≈ 0.0625m` |
| `bandwidth_normalized` | 占用带宽 | `实际带宽(Hz) = bw × 20MHz`（含滚降，QAM 族 =(1+β)/sps） |
| `power_ratio_db` | 干扰相对目标功率比 | `10^(ratio/10)` 倍功率 |
| `lna_saturation_db` | LNA 饱和门限 | `A_sat = 目标RMS × 10^(dB/20)` |
| `snr_db` | 目标/噪声功率比 | `噪声功率 = P_target / 10^(SNR/10)` |

Agent 需要这些信息来推算 DOA（利用相位差与波长的关系）、判断同频/邻道（利用频偏与带宽的关系），以及理解 blocking 样本中目标信号为何被压缩/扭曲（LNA 饱和）。注意 SNR 只由目标功率定义（干扰再强也不改变噪声底），blocking 样本中的"目标被压制"来自 LNA 饱和而非噪声。
