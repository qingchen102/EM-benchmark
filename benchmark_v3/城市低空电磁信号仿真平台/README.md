# 城市低空电磁信号仿真平台

用于科研的**静态城市、多发射设备、多监测站、多波形成分 IQ 仿真工作台**。本目录从原“静态城市多点仿真干扰分析 Agent”中独立整理，包含仿真引擎、配置、算法基线、测试、慕尼黑建筑几何、三组已验证结果，以及可直接运行的 Web UI。无需 LLM、API key 或原 Agent 项目。

## 1. 启动 Web UI

### 当前这台 Windows 机器

在本目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[web,test]"
.\start_webui.ps1
```

浏览器打开 **http://127.0.0.1:8765**。也可以直接运行：

```powershell
.\.venv\Scripts\python -m webui.server --port 8765
```

不要双击 `webui/index.html`：页面需要本地 API。Three.js 已随项目保存，页面无需 CDN 或地图服务 token。服务只监听本机回环地址，不是多人远程服务。

当前机器的 RT 运行环境已在 WSL：`/root/.venvs/static-city-rf/bin/python`。网页服务在 Windows 运行，Sionna RT 作业自动交给该 WSL Python；自由空间和抽头作业使用网页服务自身的 Python。子进程显式加载**本目录的 src/cityrf**，不会导入原 Agent 的代码。只是复用环境中的已安装依赖，不依赖旧项目目录。

如果使用其他 WSL 环境，启动网页服务前设置：

```powershell
$env:CITYRF_WSL_PYTHON = '/path/to/your/venv/bin/python'
python -m webui.server
```

RT 环境需要 Sionna RT 2.1.0、NumPy、SciPy、Pydantic 和 Matplotlib。缺少 RT 时明确报错，**不会改用自由空间伪装成功**。Windows 与 WSL 环境各自安装依赖；依赖检测结果显示在页面顶栏。

### 新机器 / 独立 Linux、WSL 环境

基础环境要求 Python 3.12+。在 WSL 或 Linux 的本项目目录执行：

```bash
bash tools/setup_rt.sh
source .venv-rt/bin/activate
python -m webui.server --port 8765
```

脚本创建本项目自己的 `.venv-rt`，不修改原 Agent 环境。也可手动：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[web,rt,test]'
bash start_webui.sh
```

LLVM CPU 后端是当前验证路径；CUDA 后端还需要可用的 CUDA/OptiX/Mitsuba 环境，不能因为有 NVIDIA 显卡就假设已可用。当前交付没有宣称完成 GPU RT 验证。`requirements-rt.lock.txt` 是原项目的参考环境记录，不要求直接覆盖本机全部依赖。

## 2. 工作台怎么用

1. **选择示例**：包括自由空间、慕尼黑复合干扰、三组展示场景和四组复杂场景配置。
2. **布置设备**：左侧选择发射源或监测站；可新增、复制、删除。2D 地图可直接拖动设备更改 X/Y；高度 Z 在右侧编辑。
3. **设置波形**：发射源可叠加多个成分。每个成分支持调制、频偏、带宽、符号率、权重、相位和脉冲门控等设置。
4. **设置接收机**：配置阵列中心、Yaw/Pitch/Roll、十字阵或矩形阵、间距、调谐、接收带宽、噪声、增益和软限幅。可将阵列规格同步到所有站。
5. **设置传播**：自由空间、指定抽头或 Sionna RT；RT 可配置场景、CPU/GPU、射线预算、深度以及直达、反射、绕射等。
6. **配置任务**：指定合作源、参与站点和保护频带。这决定公开元数据里允许包含哪些合作源先验。
7. **校验与运行**：校验显示采样数、IQ 大小、DSP 内存估算和约束；运行生成独立目录，界面自动显示进度与日志。目前每个服务串行执行一个作业。
8. **查看结果**：逐站 Welch PSD / 时频图、逐源链路真值、日志和文件下载。修改配置不会改写旧结果；结果区域明确显示所属运行。

“保存配置”保存到 `user_scenarios/`；“导出 JSON”下载到浏览器；“导入 JSON”先经过后端校验。完整 JSON 面板支持所有 schema 字段，点击“校验并应用”后生效。

### 2D / 3D 地图约定

- 建筑来自 `sionna.rt.scene.munich` 导出的实际 38,936 个三角面，不是虚构方块地图。保持原生世界坐标、米单位、Z 向上；不是经纬度地图。
- 3D：左键旋转，滚轮缩放，右键平移；点选设备进入属性面板。2D：顶视、滚轮缩放，直接拖动设备改位置。
- 蓝色通信源，橙色干扰源，绿色监测站，紫色背景源。监测站箭头指向阵面正面（局部 +x）。
- 地图只改变视角，不裁剪 RT 传播场景。即使只看市中心，求解仍使用完整慕尼黑场景。
- “布站连线”是 TX/RX 配对示意，**不是反射径或已验证 LoS**。当前输出保存路径系数、时延和到达方向，没有保存逐跳反射顶点，因此网页不绘制虚构的反射折线。
- 自由空间/抽头后端不利用建筑，页面会明确提示。天空可见性检查在 RT 求解时执行，配置数值校验不能替代这一检查。

## 3. 支持的物理能力与限制

| 项目 | 当前实现 |
|---|---|
| 时间 | 静态快照，0 < 时长 ≤ 50 ms；不含运动、多普勒或长时流式 |
| 源/站数量 | 1–32 个发射设备、1–16 个接收站 |
| 波形 | tone、noise、BPSK、QPSK、FSK、OFDM、chirp；每源 1–16 成分 |
| 发射端 | 单端口，各向同性、垂直极化；位置、朝向、平台 ID、名义功率 |
| 阵列 | 默认 8+8 十字阵，共16通道；矩形阵最多16阵元；十字双臂必须偶数 |
| 接收链 | 热噪声、噪声系数、温度、tanh软限幅、调谐、FIR滤波、固定增益 |
| 信道 | 自由空间、手工抽头、Sionna RT 2.1.0；保留复系数与绝对时延 |
| 采样 | 所有站同一采样率；配置支持10 kS/s–40 MS/s，实际资源取决于场景 |
| 输出 | complex64 `[channels, samples]`，公开观测与私有真值分目录 |
| 重现 | 冻结信道精确重放；核对版本、源码与产物哈希 |

同一 Sionna RT 场景内所有接收站阵列规格必须一致，但位置和朝向可以不同。RT 使用 synthetic-array 窄带近似，不能当作任意宽带近场全波仿真。

OFDM 是 FFT64、CP16、32 个 QPSK 子载波的科研代理，内部采样率为成分带限参数的2倍；`symbol_rate_hz` 对此 OFDM 生成器不生效。FSK 使用符号率/4 的双频偏。BPSK/QPSK 并非完整通信协议实现。发射 `bandwidth_hz` 是 FIR 截止边界参数，不能等同于测得的占用带宽。

IQ 平方幅度按 W 解释，固定接收增益后的功率可直接计算。名义发射功率使用未门控组合参考波形标定，门控和滤波不重新归一化；实际平均功率单独记录。零求解路径只表示本配置下没有求得路径，不证明现实中绝对不可见。城市模型和材料尚未实测校准。

## 4. 不用网页，直接运行

安装后在本目录执行：

```powershell
python simulate.py validate --config examples/free_space.json
python simulate.py run --config examples/free_space.json --output runs/my_free_space/capture
python simulate.py analyze --public runs/my_free_space/capture/public --output runs/my_free_space/overview.json
```

RT 使用具备对应依赖的 Python，例如当前机器：

```powershell
wsl --cd '/mnt/c/ZJU/杂项/项目/低空安全电磁大模型/城市低空电磁信号仿真平台' --exec env 'PYTHONPATH=/mnt/c/ZJU/杂项/项目/低空安全电磁大模型/城市低空电磁信号仿真平台/src' /root/.venvs/static-city-rf/bin/python -m cityrf run --config examples/three_scenes/case02.json --output runs/my_rt/capture
```

请先激活本项目虚拟环境，或将 `python` 换成 `.venv/Scripts/python.exe`。`simulate.py` 强制使用本目录源码，避免系统中已安装的旧 `cityrf` 干扰；在独立虚拟环境中也可使用 `python -m cityrf`。

输出目录必须不存在或为空。重放：

```bash
python simulate.py replay --run runs/my_rt/capture --output runs/my_rt_replay/capture
```

重放需要匹配原记录的 Python/NumPy/SciPy 版本及核心源码哈希。跨 Windows/WSL 直接重放可能因版本不一致被拒绝，应使用原生成环境。

生成 Web 预览（使用已有 IQ，不重新求信道）：

```bash
python -c "from webui.results import build_preview; build_preview('runs/my_rt/capture')"
```

网页自动枚举 `runs/*/capture/run_summary.json`；命令行作业放入这种结构后即可查看。原三场景批量脚本仍在 `scripts/`，它们按历史约定输出到本目录 `results/three_scenes_027/`，不属于网页主入口。`tools/report_classical.py` 是保留的旧实验报告脚本，依赖指定历史分析结果，不用于生成新场景。

## 5. 文件结构与迁移范围

```text
src/cityrf/              仿真核心、公开观测、重放、审计
  baselines/            检测、空间处理、定位、反射/多径及校验接口
  legacy/               已迁入的旧信号算法及 schema
examples/               基础、三场景、复杂场景配置
priors/munich_geometry/  原生慕尼黑三角网格与来源元数据
webui/
  server.py             FastAPI、配置/作业/下载接口
  worker.py             独立仿真作业入口
  results.py            从公开混合 IQ 生成 PSD/STFT
  index.html            页面
  assets/               页面样式、脚本、真实网格与本地 Three.js
runs/example_case*/      三组原有已验证仿真产物
runs/run_*/              网页新作业、配置快照、日志与采集结果
tests/                  物理、数组、算法与 API 测试
tools/                  环境安装、地图导出、诊断脚本
docs/import_manifest.json 复制时的来源与 SHA256
docs/original/          保留原始设计/验收文档，仅作历史参考
docs/methods/           已复制的多径算法说明
```

原项目未修改。本次复制了仿真和信号处理相关代码、示例及测试；没有复制 `.env`、LLM provider、Agent 运行轨迹与原模型权重。原资料中的“agent尚未接入”等描述可能已过时，当前用法以本 README 为准。

`import_manifest.json` 记录复制时的哈希；为独立运行而调整的脚本和测试会与复制时不同，变更清单见 `docs/migration_changes.json`。核心物理代码原样保留。地图来自 Sionna 内置 Munich 数据，传播场景本身通过 Sionna 包加载；随附网格用于展示和几何分析。Three.js 0.180.0 的 MIT 许可见 `webui/assets/vendor/THREE-LICENSE.txt`；其他来源的代码许可及 provenance 随原文件保留。

## 6. 结果解释与 Agent 接入

```text
run_<id>/
  config.json, job.json, worker.log, audit.json
  capture/
    public/observations.json      采集、阵列、合作源先验
    public/iq_<station>.npy       混合接收 IQ
    public/web_plots/             PNG 和完整数值 NPZ
    private/scenario.json         全部仿真设定
    private/truth.json            源与接收贡献等真值
    private/channel_*.npz         冻结信道
    run_summary.json             仿真完成、耗时、版本与哈希
    preview.json                 网页用降采样预览
```

网页是研究者视角，允许下载私有真值；给 Agent/benchmark 被测模型时，只授权 `capture/public/`。网页的频谱图不会自动推断源数、调制或位置。

PSD 使用 Welch，时频图使用 spectrogram，均对阵元**功率谱**平均，不把复数 IQ 相干平均。数值 NPZ 为 W/Hz，图像与网页为 dBm/Hz；频率相对接收机调谐中心。网页时频预览最多约256×240个显示点，完整数值在 NPZ，原始 IQ 不降采样。

## 7. 验证与故障排查

```powershell
python -m pytest -q
```

安装 RT 后在对应环境开启真实 RT 测试：

```bash
RUN_RT_TESTS=1 python -m pytest -q tests/test_rt.py
```

- 配置失败：查看提示中的字段、Nyquist 边界、阵列总数及任务 ID 引用。
- `endpoint is not sky-visible`：端点处于楼内或屋檐下；调整位置或高度。这是当前保守室外校验的限制。
- RT 不可用：检查 WSL Python、Sionna RT 2.1.0 和 Mitsuba variant。修改环境变量后重启服务。
- 有些链路路径数为0：检查遮挡、深度、绕射开关及射线预算；不要直接理解为没有该源。
- 服务运行中关闭：不保证自动终止 WSL 求解；重启后旧活动作业显示“已中断”，保留日志和产物，不自动覆盖或续写。
- 端口占用：`python -m webui.server --port 8766`。
- 地图打不开：浏览器需 WebGL；没有网络不是问题，本地地图与前端依赖已随项目保存。

## 8. 扩展位置

- 新波形：`config.Component` 与 `waveforms.py`，保持 seed/id/样点索引确定性，补功率与边界测试。
- 新信道：实现 `solve(Scenario) -> list[LinkChannel]`，注册到 `runner.BACKENDS`。
- 更大/异构阵列：同时修改配置约束、后端和阵列测试，不能只改网页数字上限。
- 新地图：扩展场景配置和 Sionna 加载方式，重新导出同坐标系几何；当前 UI 不是任意地图上传器。
- 重新导出慕尼黑网格：在 RT 环境运行 `python tools/export_map.py`。
- 新算法：放入 `src/cityrf/baselines/`，与仿真真值隔离；可沿用已有校验接口。

## 9. 预设/自由信号源与自定义城市地图（2026-09 新增）

**信号源**：发射源面板可选两种模式——
- **预设 · v3 辐射体**：16 种低空辐射体（蓝牙跳频/图传/雷达脉冲/GNSS 压制/微波炉…），调制与时域结构由预设给出，功率/频偏/带宽/脉冲仍可调；
- **自由 · 调制+参数**：13 种波形（原生 7 种 + v2 基带 gfsk/16qam/64qam/ook/fhss/lfm），参数任意组合。
分发实现在 `src/cityrf/waveforms.py` 的 `generate()`（`Emitter.preset` 字段命中 v3 辐射体池即用其生成器）。

**接收噪声**：接收机前端除热噪声外，可选有色底噪、脉冲噪声、本振相位噪声（公共本振+路间差）、阵元相关噪声，均可叠加，见 `src/cityrf/receiver.py`。发射源另支持 `velocity_mps`（径向多普勒）与 `rf_offset_hz`（各源自己的射频）。

**自定义城市地图**：
```powershell
python tools/build_city_scene.py --name hangzhou --bbox 30.270,120.120,30.278,120.130
```
从 OSM Overpass 下载建筑轮廓、按 levels×3 m 拉伸成 LoD1 盒子，产出 `priors/<name>_geometry/scene.xml`（RT 用，ITU 混凝土材质）与 `webui/assets/<name>_mesh.*`（地图显示）。网页右上角下拉切换城市网格，**同时把 `Propagation.scene` 切到对应场景文件**。数据 © OpenStreetMap contributors (ODbL)。已内置杭州西湖东 1 km²（181 栋、3104 面）。
